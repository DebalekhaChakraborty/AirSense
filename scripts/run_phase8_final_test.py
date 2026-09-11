"""Locked final-test execution for AirSense V2.

Protocol Phase 8, under the human-approved pre-opening amendment of
``artifacts/phase8_preopening_protocol_amendment.json``.

Two irreversible stages, in the protocol's order and never interleaved:

``STAGE A`` generates every prediction for the three frozen confirmatory
    models over the canonical test universe, in strictly increasing
    forecast-origin order, and writes ``artifacts/phase8_prediction_lock.json``.
    No test target is read and no metric is computed in this stage; the
    scoring oracle is not even constructed.

``STAGE B`` refuses to start until that lock exists, loads the frozen
    prediction arrays back off disk, opens the test actuals, and scores.
    No model executes in this stage.

Nothing is retrained, refitted, calibrated or clipped. The four B3_R2
boosters, the seed-42 GRU_R1 weights and the B0 rule are loaded exactly as
frozen before test access.

Usage:
    .venv-v2/bin/python scripts/run_phase8_final_test.py --stage a
    .venv-v2/bin/python scripts/run_phase8_final_test.py --stage b
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb                                            # noqa: E402
import torch                                                      # noqa: E402
from safetensors.torch import load_file                           # noqa: E402

from src.data.preprocessing import (                              # noqa: E402
    STATIONS, TARGET, index_of, load_training_statistics, station_code,
    timestamp_at)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.data.windowing import StationSeries                      # noqa: E402
from src.evaluation import metrics as M                           # noqa: E402
from src.evaluation.final_test import (                           # noqa: E402
    CONFIRMATORY_MODELS, EXPECTED_TEST_SAMPLES, FinalTestAccessError,
    FinalTestTargetOracle, MaskedStationInputs, RevealLedger, SEVERE_THRESHOLD,
    TEST_END, assert_confirmatory_only, chronological_blocks, load_test_index,
    verify_matrix, wind_onehot_slice)
from src.models import baselines                                  # noqa: E402
from src.models.b3_features import (                              # noqa: E402
    StationFeatureSource, build_matrix, feature_names)
from src.models.gru_forecaster import GRUForecaster               # noqa: E402
from src.models.neural_features import (                          # noqa: E402
    build_static_matrix, build_station_dynamic_matrix,
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import (                          # noqa: E402
    configure_determinism, predict as neural_predict)
from src.models.neural_training import SampleBundle               # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED_DIR = ARTIFACTS / "phase8_predictions"
TABLES = ARTIFACTS / "phase8_final_test_tables"
OPENING_RECEIPT = ARTIFACTS / "final_test_opening_receipt.json"
PREOPENING_RECEIPT = ARTIFACTS / "phase8_preopening_integrity_receipt.json"
PREDICTION_LOCK = ARTIFACTS / "phase8_prediction_lock.json"
RESULTS_LOCK = ARTIFACTS / "phase8_primary_results_lock.json"
FREEZE = ARTIFACTS / "phase7_pretest_freeze.json"
ADDENDUM = ARTIFACTS / "phase7_pretest_freeze_addendum.json"

GRU_WEIGHTS = PROJECT_ROOT / "results/models/neural/GRU_R1.safetensors"
GRU_CONFIG = PROJECT_ROOT / "results/models/neural/GRU_R1_config.json"

START = time.time()


def log(message):
    print("[%7.1fs] %s" % (time.time() - START, message), flush=True)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def b3_selected_models():
    """The four frozen per-horizon R2 boosters, exactly as pre-registered."""
    freeze = json.load(open(str(FREEZE), encoding="utf-8"))
    per_horizon = freeze["selected_configuration_and_artifacts"]["B3_R2"][
        "per_horizon_models"]
    out = OrderedDict()
    for horizon in (1, 6, 12, 24):
        entry = per_horizon["h%d" % horizon]
        path = PROJECT_ROOT / entry["model_file"]
        live = sha256_of_file(path)
        if live != entry["model_sha256"]:
            raise FinalTestAccessError(
                "B3_R2 h%d booster drifted from the pre-test freeze" % horizon)
        out[horizon] = OrderedDict([
            ("candidate_id", entry["candidate_id"]),
            ("model_file", entry["model_file"]),
            ("model_sha256", live),
            ("booster", lgb.Booster(model_file=str(path))),
        ])
    return out


def load_gru():
    """The exact seed-42 Phase-4 GRU_R1 artifact resolved by human review."""
    config = json.load(open(str(GRU_CONFIG), encoding="utf-8"))
    if config["model"] != "GRU_R1" or config["regime"] != "R1":
        raise FinalTestAccessError("GRU config is not GRU_R1/R1")
    if int(config["training"]["seed"]) != 42:
        raise FinalTestAccessError("GRU config is not the seed-42 artifact")
    model = GRUForecaster(
        dynamic_channels=len(dynamic_channel_names("R1")),
        static_channels=len(static_channel_names()),
        hidden_size=int(config["capacity"]),
        head_hidden=int(config["training"]["head_hidden"]),
        dropout=float(config["training"]["dropout"]))
    state = load_file(str(GRU_WEIGHTS))
    model.load_state_dict(state)
    model.eval()
    if model.parameter_count() != int(config["parameter_count"]):
        raise FinalTestAccessError("GRU parameter count disagrees with config")
    return model, config


def build_rolling(statistics, station_stats):
    """Guarded rolling PM2.5, unsealed into the test period for the first time."""
    rolling = OrderedDict()
    for station in STATIONS:
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        rolling[station] = RollingPm25Series(
            station, path, statistics, station_stats[station][TARGET],
            unseal_until=TEST_END)
    return rolling


# ---------------------------------------------------------------------------
# STAGE A - chronological prediction generation
# ---------------------------------------------------------------------------

def stage_a():
    if not OPENING_RECEIPT.is_file():
        raise SystemExit("refusing to open the final test: %s missing"
                         % OPENING_RECEIPT)
    log("opening receipt   %s" % sha256_of_file(OPENING_RECEIPT)[:16])
    log("pre-test freeze   %s" % sha256_of_file(FREEZE)[:16])

    runtime = configure_determinism()
    log("determinism: threads=%s deterministic=%s device=%s"
        % (runtime["num_threads"], runtime["deterministic_algorithms"],
           runtime["device"]))

    index = load_test_index(PROCESSED)
    n = index["target_index"].size
    log("canonical test universe: %d samples" % n)

    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    station_stats = json.load(open(str(
        ARTIFACTS / "preprocessing_statistics.json"),
        encoding="utf-8"))["station_training_medians"]
    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))

    rolling = build_rolling(statistics, station_stats)
    series = OrderedDict((s, StationSeries(PROCESSED, s)) for s in STATIONS)
    log("guarded rolling history unsealed to %s for 12 stations"
        % TEST_END.isoformat())

    ledger = RevealLedger("phase8")
    blocks = chronological_blocks(index["origin_index"])
    log("chronological blocks: %d, ceilings %d -> %d"
        % (len(blocks), blocks[0][0], blocks[-1][0]))

    predictions = OrderedDict((m, np.full(n, np.nan, dtype=np.float64))
                              for m in CONFIRMATORY_MODELS)
    b0_source = np.full(n, -1, dtype=np.int8)

    boosters = b3_selected_models()
    for horizon, entry in boosters.items():
        log("B3_R2 h%-2d %s %s" % (horizon, entry["candidate_id"],
                                   entry["model_sha256"][:16]))
    gru, gru_config = load_gru()
    log("GRU_R1 seed %d, %d params, weights %s"
        % (gru_config["training"]["seed"], gru.parameter_count(),
           sha256_of_file(GRU_WEIGHTS)[:16]))

    r2_names = feature_names("R2")
    wd_slice = wind_onehot_slice("R1")
    static_all = build_static_matrix(
        np.array([station_code(s) for s in index["station"]], dtype=np.int64),
        index["horizon"], index["origin_index"], index["target_index"],
        calendar)

    for block_number, (ceiling, rows) in enumerate(blocks, start=1):
        ledger.advance_to(ceiling)
        block_origins = index["origin_index"][rows]
        block_targets = index["target_index"][rows]

        # ---- B0: guarded causal persistence, one value per origin --------
        for station in STATIONS:
            local = rows[index["station"][rows] == station]
            if not local.size:
                continue
            origins = index["origin_index"][local]
            ledger.note("B0:%s" % station, origins, origins)
            values, sources = baselines.b0_causal_persistence(
                rolling[station], origins)
            predictions["B0"][local] = values
            b0_source[local] = sources

        # ---- B3_R2: frozen per-horizon boosters over masked inputs -------
        for station in STATIONS:
            local = rows[index["station"][rows] == station]
            if not local.size:
                continue
            masked = MaskedStationInputs(series[station],
                                         rolling[station]._scaled, ceiling)
            source = StationFeatureSource(masked, calendar,
                                          masked.pm25_scaled)
            for horizon, entry in boosters.items():
                sel = local[index["horizon"][local] == horizon]
                if not sel.size:
                    continue
                origins = index["origin_index"][sel]
                targets = index["target_index"][sel]
                ledger.note("B3:%s:h%d" % (station, horizon), origins, origins)
                matrix = build_matrix("R2", source, station, origins, targets)
                verify_matrix(matrix, "B3_R2 %s h%d" % (station, horizon))
                predictions["B3_R2"][sel] = entry["booster"].predict(matrix)
            del masked, source

        # ---- GRU_R1: frozen seed-42 weights over masked windows ----------
        dynamic_blocks = []
        for station in STATIONS:
            masked = MaskedStationInputs(series[station],
                                         rolling[station]._scaled, ceiling)
            dynamic_blocks.append(build_station_dynamic_matrix(
                "R1", masked, masked.pm25_scaled))
            del masked
        dynamic = np.stack(dynamic_blocks, axis=0)
        del dynamic_blocks
        ledger.note("GRU:block%d" % block_number, block_origins, block_origins)
        bundle = SampleBundle(
            np.array([station_code(s) for s in index["station"][rows]],
                     dtype=np.int64),
            index["horizon"][rows], block_origins, block_targets,
            dynamic, static_all[rows], labels=None)
        values, _ = neural_predict(gru, bundle)
        predictions["GRU_R1"][rows] = values
        del dynamic, bundle

        log("block %2d/%d ceiling %6d (%s) rows %6d  cumulative %7d/%d"
            % (block_number, len(blocks), ceiling,
               timestamp_at(ceiling).isoformat(), rows.size,
               int(np.isfinite(predictions["B0"]).sum()), n))

    # ---- completeness, alignment and the prediction lock -----------------
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    lock_models = OrderedDict()
    for name in CONFIRMATORY_MODELS:
        assert_confirmatory_only(name)
        array = predictions[name]
        if array.size != EXPECTED_TEST_SAMPLES:
            raise FinalTestAccessError("%s produced %d predictions"
                                       % (name, array.size))
        if not np.isfinite(array).all():
            raise FinalTestAccessError("%s left a sample unpredicted" % name)
        path = PRED_DIR / ("%s.npy" % name)
        np.save(str(path), array.astype(np.float32))
        lock_models[name] = OrderedDict([
            ("prediction_file", str(path.relative_to(PROJECT_ROOT))),
            ("length", int(array.size)),
            ("dtype", "float32"),
            ("bytes", path.stat().st_size),
            ("sha256", sha256_of_file(path)),
        ])
        log("%-7s %d predictions  %s" % (name, array.size,
                                         lock_models[name]["sha256"][:16]))
    np.save(str(PRED_DIR / "B0_source.npy"), b0_source)

    summary = ledger.summary()
    if summary["future_target_access_violations"] != 0:
        raise FinalTestAccessError("future-target access violations: %s"
                                   % summary["violation_detail"])

    write_json(PREDICTION_LOCK, OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_prediction_lock"),
        ("stage", "A_complete_predictions_frozen_before_scoring"),
        ("opening_receipt_sha256", sha256_of_file(OPENING_RECEIPT)),
        ("preopening_integrity_receipt_sha256",
         sha256_of_file(PREOPENING_RECEIPT)),
        ("test_index_sha256",
         sha256_of_file(PROCESSED / "sample_index" / "test.csv")),
        ("test_sample_count", int(n)),
        ("models", lock_models),
        ("model_artifact_sha256", OrderedDict([
            ("B3_R2_h%d" % h, e["model_sha256"])
            for h, e in boosters.items()] + [
            ("GRU_R1.safetensors", sha256_of_file(GRU_WEIGHTS)),
            ("GRU_R1_config.json", sha256_of_file(GRU_CONFIG))])),
        ("b0_source_counts", OrderedDict([
            (baselines.B0_SOURCE_LABELS[code],
             int((b0_source == code).sum()))
            for code in sorted(baselines.B0_SOURCE_LABELS)])),
        ("alignment_verified", True),
        ("chronological_blocks", len(blocks)),
        ("reveal_ledger", summary),
        ("future_target_access_violations", 0),
        ("model_retraining", 0),
        ("predictions_clipped", False),
        ("test_metrics_seen", 0),
        ("prediction_generation_complete", True),
        ("predictions_frozen_before_scoring", True),
        ("final_test_status", "opened_for_locked_evaluation"),
    ]))
    log("prediction lock written %s" % sha256_of_file(PREDICTION_LOCK)[:16])
    return 0


# ---------------------------------------------------------------------------
# STAGE B - scoring, only after the prediction lock exists
# ---------------------------------------------------------------------------

def descriptive(actual, prediction, stations, horizons):
    cells = M.cell_metrics(actual, prediction, stations, horizons)
    out = OrderedDict([
        ("macro_station_horizon_MAE", M.macro_from_cells(cells, "mae")),
        ("micro_MAE", M.mae(actual, prediction)),
        ("macro_station_horizon_RMSE", M.macro_from_cells(cells, "rmse")),
        ("micro_RMSE", M.rmse(actual, prediction)),
        ("macro_station_horizon_R2", M.macro_from_cells(cells, "r2")),
        ("micro_R2", M.r2(actual, prediction)),
        ("cells", len(cells)),
    ])
    out.update(M.severe_metrics(actual, prediction, stations, horizons))
    out.update(M.negative_prediction_summary(prediction))
    return out, cells


def stage_b():
    if not PREDICTION_LOCK.is_file():
        raise SystemExit("refusing to score: %s missing" % PREDICTION_LOCK)
    lock = json.load(open(str(PREDICTION_LOCK), encoding="utf-8"))
    if not lock["predictions_frozen_before_scoring"]:
        raise SystemExit("prediction lock does not certify the stage order")
    log("prediction lock   %s" % sha256_of_file(PREDICTION_LOCK)[:16])

    index = load_test_index(PROCESSED)
    n = index["target_index"].size

    stored = OrderedDict()
    for name in CONFIRMATORY_MODELS:
        path = PROJECT_ROOT / lock["models"][name]["prediction_file"]
        if sha256_of_file(path) != lock["models"][name]["sha256"]:
            raise FinalTestAccessError("%s changed after the prediction lock"
                                       % name)
        stored[name] = np.load(str(path)).astype(np.float64)
        if stored[name].size != n:
            raise FinalTestAccessError("%s length %d" % (name,
                                                         stored[name].size))
    log("three frozen prediction arrays loaded and hash-verified")

    # ---- the first numerical final-test target is read here -------------
    oracle = OrderedDict((s, FinalTestTargetOracle(
        s, next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % s)))) for s in STATIONS)
    actual = np.empty(n, dtype=np.float64)
    for station in STATIONS:
        rows = np.flatnonzero(index["station"] == station)
        actual[rows] = oracle[station].actuals(index["target_index"][rows],
                                               stored["B3_R2"][rows])
    log("test actuals opened for scoring (after predictions were locked)")

    stations, horizons = index["station"], index["horizon"]
    results = OrderedDict()
    cell_tables = OrderedDict()
    for name in CONFIRMATORY_MODELS:
        results[name], cell_tables[name] = descriptive(
            actual, stored[name], stations, horizons)
        log("%-7s macro station-horizon MAE %.6f | severe MAE %.6f (n=%d)"
            % (name, results[name]["macro_station_horizon_MAE"],
               results[name].get("severe_mae", float("nan")),
               results[name].get("severe_n", 0)))

    severe_n = int(results["B3_R2"]["severe_n"])
    write_json(ARTIFACTS / "phase8_final_test_metrics.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_final_test_metrics"),
        ("severe_threshold", SEVERE_THRESHOLD),
        ("severe_rule", "actual > 244.0"),
        ("residual_convention", "actual - prediction; positive is "
                                "under-prediction"),
        ("test_sample_count", int(n)),
        ("severe_sample_count", severe_n),
        ("models", results),
    ]))

    by_horizon, by_station, by_cell, severe_rows, negative_rows = (
        [], [], [], [], [])
    for name in CONFIRMATORY_MODELS:
        pred = stored[name]
        for horizon in sorted(set(horizons.tolist())):
            sel = horizons == horizon
            cells = M.cell_metrics(actual[sel], pred[sel], stations[sel],
                                   horizons[sel])
            by_horizon.append([
                name, horizon, int(sel.sum()),
                round(M.macro_from_cells(cells, "mae"), 6),
                round(M.mae(actual[sel], pred[sel]), 6),
                round(M.macro_from_cells(cells, "rmse"), 6),
                round(M.rmse(actual[sel], pred[sel]), 6),
                round(M.r2(actual[sel], pred[sel]), 6)])
        for station in STATIONS:
            sel = stations == station
            cells = M.cell_metrics(actual[sel], pred[sel], stations[sel],
                                   horizons[sel])
            by_station.append([
                name, station, int(sel.sum()),
                round(M.macro_from_cells(cells, "mae"), 6),
                round(M.mae(actual[sel], pred[sel]), 6),
                round(M.macro_from_cells(cells, "rmse"), 6),
                round(M.rmse(actual[sel], pred[sel]), 6),
                round(M.r2(actual[sel], pred[sel]), 6)])
        for (station, horizon), cell in cell_tables[name].items():
            by_cell.append([name, station, horizon, cell["n"],
                            round(cell["mae"], 6), round(cell["rmse"], 6),
                            round(cell["r2"], 6)])
        row = results[name]
        severe_rows.append([
            name, row["severe_n"], round(row["severe_mae"], 6),
            round(row["severe_rmse"], 6),
            round(row["severe_mean_residual"], 6),
            round(row["severe_underprediction_pct"], 6),
            round(row["severe_macro_station_horizon_mae"], 6),
            row["severe_contributing_cells"], row["severe_total_cells"]])
        negative_rows.append([
            name, row["n_negative"], round(row["pct_negative"], 6),
            round(row["min_prediction"], 6)])

    roles = OrderedDict([
        ("B3_R2", "PRIMARY CONFIRMATORY MODEL"),
        ("GRU_R1", "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL"),
        ("B0", "REFERENCE BENCHMARK"),
    ])
    write_csv(TABLES / "final_model_comparison.csv",
              ["model", "role", "n", "macro_station_horizon_MAE", "micro_MAE",
               "macro_station_horizon_RMSE", "micro_RMSE",
               "macro_station_horizon_R2", "severe_n", "severe_MAE",
               "severe_RMSE", "severe_mean_residual",
               "severe_underprediction_pct", "n_negative", "min_prediction"],
              [[name, roles[name], n,
                round(results[name]["macro_station_horizon_MAE"], 6),
                round(results[name]["micro_MAE"], 6),
                round(results[name]["macro_station_horizon_RMSE"], 6),
                round(results[name]["micro_RMSE"], 6),
                round(results[name]["macro_station_horizon_R2"], 6),
                results[name]["severe_n"],
                round(results[name]["severe_mae"], 6),
                round(results[name]["severe_rmse"], 6),
                round(results[name]["severe_mean_residual"], 6),
                round(results[name]["severe_underprediction_pct"], 6),
                results[name]["n_negative"],
                round(results[name]["min_prediction"], 6)]
               for name in CONFIRMATORY_MODELS])
    write_csv(TABLES / "metrics_by_horizon.csv",
              ["model", "horizon_hours", "n", "macro_station_MAE", "micro_MAE",
               "macro_station_RMSE", "micro_RMSE", "micro_R2"], by_horizon)
    write_csv(TABLES / "metrics_by_station.csv",
              ["model", "station", "n", "macro_horizon_MAE", "micro_MAE",
               "macro_horizon_RMSE", "micro_RMSE", "micro_R2"], by_station)
    write_csv(TABLES / "metrics_by_station_horizon.csv",
              ["model", "station", "horizon_hours", "n", "MAE", "RMSE", "R2"],
              by_cell)
    write_csv(TABLES / "severe_metrics.csv",
              ["model", "severe_n", "severe_MAE", "severe_RMSE",
               "severe_mean_residual", "severe_underprediction_pct",
               "severe_macro_station_horizon_MAE", "contributing_cells",
               "total_cells"], severe_rows)
    write_csv(TABLES / "negative_predictions.csv",
              ["model", "n_negative", "pct_negative", "min_prediction"],
              negative_rows)

    confirmatory = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_confirmatory_endpoints"),
        ("stage", "B_locked_final_test_scored"),
        ("primary", OrderedDict([
            ("model", "B3_R2"),
            ("role", "PRIMARY CONFIRMATORY MODEL"),
            ("endpoint", "macro_station_horizon_MAE"),
            ("value", results["B3_R2"]["macro_station_horizon_MAE"]),
            ("cells", results["B3_R2"]["cells"]),
        ])),
        ("secondary", OrderedDict([
            ("model", "GRU_R1"),
            ("role", "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL"),
            ("endpoint", "severe_MAE_gt_244"),
            ("severe_threshold", SEVERE_THRESHOLD),
            ("value", results["GRU_R1"]["severe_mae"]),
            ("severe_n", results["GRU_R1"]["severe_n"]),
            ("seed", 42),
        ])),
        ("reference", OrderedDict([
            ("model", "B0"),
            ("role", "REFERENCE BENCHMARK"),
            ("macro_station_horizon_MAE",
             results["B0"]["macro_station_horizon_MAE"]),
            ("severe_MAE_gt_244", results["B0"]["severe_mae"]),
        ])),
        ("test_sample_count", int(n)),
        ("severe_sample_count", severe_n),
        ("prediction_lock_sha256", sha256_of_file(PREDICTION_LOCK)),
        ("prediction_sha256", OrderedDict(
            (name, lock["models"][name]["sha256"])
            for name in CONFIRMATORY_MODELS)),
    ])
    write_json(TABLES / "confirmatory_endpoints.json", confirmatory)

    table_hashes = OrderedDict(
        (p.name, sha256_of_file(p)) for p in sorted(TABLES.iterdir()))
    write_json(RESULTS_LOCK, OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_primary_results_lock"),
        ("stage", "B_confirmatory_metrics_frozen"),
        ("prediction_lock_sha256", sha256_of_file(PREDICTION_LOCK)),
        ("opening_receipt_sha256", sha256_of_file(OPENING_RECEIPT)),
        ("test_target_count", int(n)),
        ("observed_target_count", int(np.isfinite(actual).sum())),
        ("primary_model", "B3_R2"),
        ("primary_endpoint", "macro_station_horizon_MAE"),
        ("primary_value", results["B3_R2"]["macro_station_horizon_MAE"]),
        ("secondary_model", "GRU_R1"),
        ("secondary_endpoint", "severe_MAE_gt_244.0"),
        ("secondary_value", results["GRU_R1"]["severe_mae"]),
        ("secondary_severe_n", results["GRU_R1"]["severe_n"]),
        ("reference_model", "B0"),
        ("reference_primary_value",
         results["B0"]["macro_station_horizon_MAE"]),
        ("reference_severe_value", results["B0"]["severe_mae"]),
        ("severe_threshold", SEVERE_THRESHOLD),
        ("metric_table_sha256", table_hashes),
        ("test_metrics_seen", True),
        ("model_selection_after_test", False),
        ("prediction_changes_after_scoring", False),
        ("post_test_exploration_performed", False),
        ("final_test_status", "evaluated"),
    ]))
    log("primary results lock written %s" % sha256_of_file(RESULTS_LOCK)[:16])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    args = parser.parse_args()
    return stage_a() if args.stage == "a" else stage_b()


if __name__ == "__main__":
    sys.exit(main())
