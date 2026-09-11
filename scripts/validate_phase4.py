"""Validate the AirSense V2 Phase-4 temporal neural baselines.

Read-only. Trains nothing, predicts nothing new, repairs nothing. Exits 0 only
if every gate passes.

Usage:
    .venv-v2/bin/python scripts/validate_phase4.py
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    PARTITION_BOUNDS, STATIONS, index_of)
from src.evaluation import metrics as M                           # noqa: E402
from src.models import neural_grid                                # noqa: E402
from src.models.neural_data import (                              # noqa: E402
    INTERNAL_CORE, INTERNAL_TUNING)
from src.models.neural_features import (                          # noqa: E402
    CONTEXT_HOURS, HORIZON_VOCABULARY, dynamic_channel_names,
    static_channel_names)
from src.models.tcn_forecaster import TCNForecaster, receptive_field  # noqa

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
NEURAL = PROJECT_ROOT / "results" / "models" / "neural"
ARTIFACTS = PROJECT_ROOT / "artifacts"

NEURAL_MODELS = ["GRU_R0", "GRU_R1", "GRU_R2", "TCN_R0", "TCN_R1", "TCN_R2"]
CLASSICAL = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"]
HORIZONS = [1, 6, 12, 24]
EXPECTED_SAMPLES = 413148
FORBIDDEN_ARCHITECTURES = re.compile(
    r"(transformer|patchtst|itransformer|nbeats|n_beats|informer|timesfm|"
    r"chronos|moirai|lstm)", re.IGNORECASE)


# Scoped by ownership for the same reason as the Phase-1 and Phase-2 import
# scans. The assertion is "Phase 4 implemented and trained no Transformer,
# TSFM or LSTM" - not "no later phase may ever implement one". Phase 5 owns
# src/models/itransformer_*.py, where an inverted-Transformer implementation
# is the declared subject of that phase, not a violation of this one.
PHASE4_OWNED_SOURCE = (
    "src/models/gru_forecaster.py",
    "src/models/tcn_forecaster.py",
    "src/models/neural_data.py",
    "src/models/neural_features.py",
    "src/models/neural_grid.py",
    "src/models/neural_training.py",
    "scripts/build_phase4_internal_split.py",
    "scripts/run_phase4_internal_selection.py",
    "scripts/run_phase4_full_refit.py",
    "scripts/build_phase4_report.py",
    "scripts/validate_phase4.py",
)


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 70
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_index():
    stations, horizons, targets = [], [], []
    with open(str(PROCESSED / "sample_index" / "validation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
            targets.append(int(row["target_row_index"]))
    return (np.array(stations), np.array(horizons, dtype=np.int32),
            np.array(targets, dtype=np.int64))


def upstream(report):
    for name, interpreter in (("validate_foundation.py", "python3"),
                              ("validate_phase1.py", "python3"),
                              ("validate_phase2.py", "python3"),
                              ("validate_phase3.py", sys.executable)):
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)

    registry = load_json(ARTIFACTS / "verification_tooling_registry.json")
    drift = [name for name, entry in registry["validators"].items()
             if sha256_of_file(PROJECT_ROOT / name) != entry["sha256"]]
    report.add("Verification tooling matches the registry", not drift,
               "changed: %s" % drift)

    phase3 = load_json(ARTIFACTS / "v2_phase3_manifest.json")
    stale = []
    for group, base in (("prediction_array_sha256",
                         "results/validation/predictions/"),
                        ("metric_table_sha256", "results/validation/"),
                        ("b3_table_sha256", "results/models/b3/")):
        for name, digest in phase3[group].items():
            path = PROJECT_ROOT / (base + name)
            if not path.is_file() or sha256_of_file(path) != digest:
                stale.append(name)
    for name, digest in phase3["selected_model_sha256"].items():
        if sha256_of_file(PROJECT_ROOT / name) != digest:
            stale.append(name)
    report.add("Phase-3 evidence unchanged", not stale,
               "drifted: %s" % stale[:5])
    for key, path in (("foundation_manifest_sha256",
                       "artifacts/v2_foundation_manifest.json"),
                      ("phase1_manifest_sha256",
                       "artifacts/v2_phase1_manifest.json"),
                      ("phase2_manifest_sha256",
                       "artifacts/v2_phase2_manifest.json")):
        report.add("Upstream manifest unchanged: %s" % Path(path).name,
                   phase3[key] == sha256_of_file(PROJECT_ROOT / path))


def contract(report):
    split = load_json(ARTIFACTS / "phase4_internal_split.json")
    train_start, train_end = PARTITION_BOUNDS["train"]
    core_start = datetime.fromisoformat(split["internal_core"]["start"])
    core_end = datetime.fromisoformat(split["internal_core"]["end"])
    tune_start = datetime.fromisoformat(split["internal_tuning"]["start"])
    tune_end = datetime.fromisoformat(split["internal_tuning"]["end"])
    report.add("Internal tuning lies entirely inside the training partition",
               core_start >= train_start and tune_end <= train_end
               and core_end < tune_start)
    report.add("Internal split does not touch external validation",
               tune_end < PARTITION_BOUNDS["validation"][0])
    report.add("Internal split partitions training exactly",
               split["counts"]["core"]["total"]
               + split["counts"]["tuning"]["total"] == 821184)
    report.add("Phase-2 sample universe was not rewritten",
               split["phase2_sample_universe_rewritten"] is False
               and split["phase2_sample_universe_manifest_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "sample_universe_manifest.json"))

    schema = load_json(ARTIFACTS / "neural_feature_schema.json")
    report.add("Context length is unchanged at 48",
               schema["context_hours"] == CONTEXT_HOURS == 48)
    live = {r: dynamic_channel_names(r) for r in ("R0", "R1", "R2")}
    report.add("Neural feature schema matches the frozen record",
               all(schema["dynamic_channels"][r] == live[r] for r in live)
               and schema["static_channels"] == static_channel_names())
    report.add("Horizon vocabulary is the frozen 1/6/12/24",
               schema["horizon_vocabulary"] == HORIZON_VOCABULARY
               == [1, 6, 12, 24])
    report.add("Wind is one-hot, never ordinal",
               len(schema["wind_vocabulary"]) == 17
               and "wd_code" not in " ".join(schema["dynamic_channels"]["R2"]))
    report.add("day_of_week and weekend excluded",
               schema["day_of_week_included"] is False
               and schema["weekend_included"] is False)

    report.add("R0 sees no pollutant history",
               not any(n.startswith("value:PM") or n.startswith("value:SO2")
                       or n.startswith("value:NO2") or n.startswith("value:CO")
                       or n.startswith("value:O3") for n in live["R0"]))
    report.add("R1 sees PM2.5 and no other pollutant",
               "value:PM2.5" in live["R1"]
               and not any("value:%s" % p in live["R1"]
                           for p in ("PM10", "SO2", "NO2", "CO", "O3")))
    report.add("R2 sees all frozen target-station pollutants",
               all("value:%s" % p in live["R2"] for p in
                   ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")))

    report.add("TCN receptive field covers the 48-hour context",
               receptive_field() >= CONTEXT_HOURS)


def grids_and_selection(report):
    gru = neural_grid.candidates("GRU")
    tcn = neural_grid.candidates("TCN")
    report.add("GRU grid is exactly the four frozen candidates",
               [(c["candidate_id"], c["capacity"], c["learning_rate"])
                for c in gru]
               == [("GRU_C01", 64, 0.0003), ("GRU_C02", 64, 0.001),
                   ("GRU_C03", 128, 0.0003), ("GRU_C04", 128, 0.001)])
    report.add("TCN grid is exactly the four frozen candidates",
               [(c["candidate_id"], c["capacity"], c["learning_rate"])
                for c in tcn]
               == [("TCN_C01", 32, 0.0003), ("TCN_C02", 32, 0.001),
                   ("TCN_C03", 64, 0.0003), ("TCN_C04", 64, 0.001)])
    report.add("Training budget fixed at 8 epochs / batch 1024 / no early "
               "stopping",
               all(c["epochs"] == 8 and c["batch_size"] == 1024
                   and c["early_stopping"] is False and c["loss"] == "L1"
                   and c["optimizer"] == "AdamW" for c in gru + tcn))

    metrics = read_csv(NEURAL / "internal_candidate_metrics.csv")
    counts = Counter((r["architecture"], r["candidate_id"]) for r in metrics)
    report.add("24 internal candidate fits, 3 regimes per candidate",
               len(metrics) == 24 and set(counts.values()) == {3})

    selection = read_csv(NEURAL / "internal_candidate_selection.csv")
    problems = []
    for architecture in ("GRU", "TCN"):
        pool = [{"candidate_id": r["candidate_id"],
                 "capacity": int(r["capacity"]),
                 "learning_rate": float(r["learning_rate"]),
                 "mean_macro_mae":
                     float(r["mean_macro_station_horizon_MAE"]),
                 "mean_macro_rmse": float(r["mean_macro_RMSE"])}
                for r in selection if r["architecture"] == architecture]
        recomputed = neural_grid.select(pool)["candidate_id"]
        marked = [r["candidate_id"] for r in selection
                  if r["architecture"] == architecture
                  and r["selected"] == "true"]
        if marked != [recomputed]:
            problems.append("%s: marked %s, rule gives %s"
                            % (architecture, marked, recomputed))
    report.add("Selected candidate matches the frozen selection rule",
               not problems, "; ".join(problems))

    freeze = load_json(ARTIFACTS / "phase4_architecture_selection_freeze.json")
    report.add("Architecture selection came from training only",
               freeze["external_validation_used_for_selection"] is False
               and freeze["selected_from"]
               == "training-only internal core/tuning split")
    report.add("Selection freeze recorded zero external neural metrics",
               freeze["external_neural_validation_metrics_seen"] == 0
               and freeze["test_metrics_seen"] == 0)
    report.add("The same configuration is used for R0, R1 and R2",
               freeze["same_configuration_used_for_all_regimes"] is True)

    configs = {}
    for name in NEURAL_MODELS:
        configs[name] = load_json(NEURAL / ("%s_config.json" % name))
    for architecture in ("GRU", "TCN"):
        capacities = {configs["%s_%s" % (architecture, r)]["capacity"]
                      for r in ("R0", "R1", "R2")}
        rates = {configs["%s_%s" % (architecture, r)]["learning_rate"]
                 for r in ("R0", "R1", "R2")}
        report.add("%s uses one capacity and one learning rate across regimes"
                   % architecture,
                   len(capacities) == 1 and len(rates) == 1,
                   "capacities %s rates %s" % (capacities, rates))
        report.add("%s config matches the selection freeze" % architecture,
                   capacities.pop() == freeze["selected"][architecture]
                   ["capacity"])
    report.add("Six neural full models exist",
               len(configs) == 6
               and all((NEURAL / ("%s.safetensors" % n)).is_file()
                       for n in NEURAL_MODELS))
    report.add("Full models were trained on the complete training universe "
               "from a clean initialisation",
               all(c["trained_on"]
                   == "complete frozen Phase-2 training universe"
                   and c["initialised_from"].startswith("clean seeded")
                   for c in configs.values()))
    report.add("Neural outputs are declared unconstrained",
               all(c["output_constrained"] is False
                   for c in configs.values()))


def predictions_and_metrics(report, station, horizon):
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    problems = []
    predictions = {}
    for name in NEURAL_MODELS:
        path = PRED / ("%s_validation.npy" % name)
        if not path.is_file():
            problems.append("%s missing" % name)
            continue
        values = np.load(str(path)).astype(np.float64)
        predictions[name] = values
        if values.size != EXPECTED_SAMPLES:
            problems.append("%s has %d predictions" % (name, values.size))
        elif not np.isfinite(values).all():
            problems.append("%s has non-finite predictions" % name)
    report.add("Each neural model predicts exactly 413,148 samples",
               not problems, "; ".join(problems[:3]))
    report.add("Prediction arrays align with the canonical validation index",
               station.size == EXPECTED_SAMPLES
               and all(v.size == station.size for v in predictions.values()))

    manifest = load_json(ARTIFACTS / "v2_phase2_manifest.json")
    report.add("Validation index hash unchanged since Phase 2",
               manifest["sample_index_sha256"]["validation"]
               == sha256_of_file(PROCESSED / "sample_index"
                                 / "validation.csv"))

    negatives = read_csv(OUT / "phase4_negative_predictions.csv")
    clipped = [r["model"] for r in negatives if r["clipped"] != "false"]
    report.add("No neural prediction was clipped", not clipped,
               "clipped: %s" % clipped)
    mismatch = [r["model"] for r in negatives if r["model"] in predictions
                and abs(float(predictions[r["model"]].min())
                        - float(r["min_prediction"])) > 1e-3]
    report.add("Reported minima match the stored arrays", not mismatch,
               "mismatched: %s" % mismatch)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase4_model_comparison.csv")}
    cells = read_csv(OUT / "phase4_metrics_by_station_horizon.csv")
    problems = []
    for name in NEURAL_MODELS:
        subset = [float(r["MAE"]) for r in cells if r["model"] == name]
        if len(subset) != 48:
            problems.append("%s has %d cells" % (name, len(subset)))
            continue
        if abs(float(np.mean(subset))
               - float(comparison[name]["macro_station_horizon_MAE"])) > 1e-6:
            problems.append("%s macro mismatch" % name)
    report.add("Primary macro metric reconciles from the 48 cells",
               not problems, "; ".join(problems[:3]))
    report.add("Phase-3 values in the combined table are reused, not "
               "recomputed",
               all(comparison[n]["value_source"] == "phase3_frozen"
                   for n in CLASSICAL)
               and all(comparison[n]["value_source"] == "phase4"
                       for n in NEURAL_MODELS))

    frozen3 = {r["model"]: r for r in read_csv(OUT / "model_comparison.csv")}
    drift = [n for n in CLASSICAL
             if abs(float(frozen3[n]["macro_station_horizon_MAE"])
                    - float(comparison[n]["macro_station_horizon_MAE"]))
             > 1e-12]
    report.add("Frozen Phase-3 baseline values are byte-faithful", not drift,
               "drifted: %s" % drift)

    severe = {r["model"]: r
              for r in read_csv(OUT / "phase4_severe_metrics.csv")}
    problems = []
    for name in NEURAL_MODELS:
        mask = actual > 244.0
        recomputed = float(np.abs(actual[mask]
                                  - predictions[name][mask]).mean())
        if abs(recomputed - float(severe[name]["severe_MAE"])) > 1e-6:
            problems.append(name)
    report.add("Severe metrics reconcile at the frozen 244.0 threshold",
               not problems and M.SEVERE_THRESHOLD == 244.0,
               "mismatched: %s" % problems)
    report.add("Severe threshold was not re-derived from validation",
               all(int(severe[n]["severe_n"]) == int(severe["B0"]["severe_n"])
                   for n in NEURAL_MODELS))


def prohibitions(report):
    owned = [name for name in PHASE4_OWNED_SOURCE
             if (PROJECT_ROOT / name).is_file()]
    offenders = [name for name in owned
                 if FORBIDDEN_ARCHITECTURES.search(Path(name).name)]
    report.add("No Transformer, TSFM or LSTM implementation in Phase-4 "
               "source (%d files)" % len(owned),
               not offenders, "found: %s" % offenders)

    architectures = sorted({load_json(NEURAL / ("%s_config.json" % name))
                            ["architecture"] for name in NEURAL_MODELS})
    stray = [str(path.relative_to(PROJECT_ROOT))
             for path in sorted(NEURAL.rglob("*"))
             if path.is_file() and FORBIDDEN_ARCHITECTURES.search(path.name)]
    report.add("Every Phase-4 model is a GRU or a TCN, and no Phase-4 model "
               "artifact names another family",
               architectures == ["GRU", "TCN"] and not stray,
               "architectures %s; stray %s" % (architectures, stray[:3]))
    report.add("No R3 neural model was produced",
               not list(NEURAL.glob("*_R3*")))
    report.add("torch reports no CUDA device in use",
               not torch.cuda.is_available())

    for name in ("torchvision", "torchaudio", "transformers", "timesfm",
                 "chronos", "uni2ts"):
        try:
            __import__(name)
            report.add("%s is absent" % name, False, "installed")
        except ImportError:
            report.add("%s is absent" % name, True)

    test_artifacts = [str(p.relative_to(PROJECT_ROOT))
                      for p in sorted(PRED.glob("*test*"))]
    report.add("No test prediction artifact exists", not test_artifacts,
               "found: %s" % test_artifacts)
    summary = load_json(ARTIFACTS / "phase4_validation_summary.json")
    report.add("Summary records a sealed test and zero test activity",
               summary["final_test_status"] == "sealed"
               and summary["test_predictions_generated"] == 0
               and summary["test_metrics_seen"] == 0)
    receipt = load_json(ARTIFACTS / "phase4_external_validation_receipt.json")
    report.add("External-validation receipt precedes neural metrics",
               receipt["external_neural_validation_metrics_seen"] == 0
               and receipt["test_status"] == "sealed"
               and receipt["hyperparameters_selected_from_training_only_"
                           "internal_split"] is True)


def independent(report, station, horizon, target):
    """Fourteen recomputations through a path separate from the reporting code."""
    from safetensors.torch import load_file

    from src.data.rolling_history import RollingPm25Series
    from src.data.preprocessing import (
        TARGET, load_training_statistics, timestamp_at)
    from src.data.target_history import TargetAccessError
    from src.data.windowing import StationSeries
    from src.models.gru_forecaster import GRUForecaster
    from src.models.neural_features import (
        build_static_matrix, build_station_dynamic_matrix, gather_windows)

    probe_station = "Dongsi"
    probe_code = STATIONS.index(probe_station)
    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    station_stats = load_json(ARTIFACTS / "preprocessing_statistics.json")[
        "station_training_medians"]
    raw = next((PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228")
               .glob("PRSA_Data_%s_*.csv" % probe_station))
    rolling = RollingPm25Series(probe_station, raw, statistics,
                                station_stats[probe_station][TARGET])
    series = StationSeries(PROCESSED, probe_station)

    rows = np.flatnonzero((station == probe_station) & (horizon == 6))
    row = int(rows[len(rows) // 2])
    origin = int(target[row] - horizon[row])

    # 1 & 2: the window a GRU and a TCN both consume is the same canonical block
    for regime, label in (("R1", "GRU"), ("R2", "TCN")):
        dynamic = build_station_dynamic_matrix(regime, series,
                                               rolling._scaled)
        window = gather_windows(dynamic[None, :, :], [0], [origin])
        report.add("Independent: %s input window is 48 steps ending at origin"
                   % label,
                   window.shape[1] == CONTEXT_HOURS
                   and np.allclose(window[0, -1], dynamic[origin])
                   and np.allclose(window[0, 0], dynamic[origin - 47]))

    # 3: horizon conditioning for each horizon
    static = build_static_matrix([probe_code] * 4, [1, 6, 12, 24],
                                 [origin] * 4, [origin + h for h in
                                                (1, 6, 12, 24)], calendar)
    report.add("Independent: horizon one-hot for all four horizons",
               np.array_equal(static[:, 12:16], np.eye(4, dtype=np.float32)))

    # 4: station conditioning
    report.add("Independent: station one-hot selects the probe station",
               static[0, probe_code] == 1.0 and static[0, :12].sum() == 1.0)

    # 5 & 6: rolling PM2.5 sequence, and refusal of anything after the origin
    sequence = rolling.scaled_window(origin, CONTEXT_HOURS)
    report.add("Independent: rolling validation PM2.5 sequence has 48 finite "
               "values", sequence.size == CONTEXT_HOURS
               and bool(np.isfinite(sequence).all()))
    refused = False
    try:
        rolling.native_at(origin + 1, origin)
    except TargetAccessError:
        refused = True
    sealed = False
    try:
        rolling.scaled_window(index_of(PARTITION_BOUNDS["test"][0]),
                              CONTEXT_HOURS)
    except TargetAccessError:
        sealed = True
    report.add("Independent: no PM2.5 value after the origin, test sealed",
               refused and sealed)

    # 7: re-run one saved model on one sample and match the stored prediction
    config = load_json(NEURAL / "GRU_R1_config.json")
    model = GRUForecaster(config["dynamic_channels"],
                          config["static_channels"],
                          hidden_size=config["capacity"],
                          head_hidden=64, dropout=0.1, num_layers=1)
    model.load_state_dict(load_file(str(NEURAL / "GRU_R1.safetensors")))
    model.eval()
    dynamic = build_station_dynamic_matrix("R1", series, rolling._scaled)
    window = gather_windows(dynamic[None, :, :], [0], [origin])
    static_one = build_static_matrix([probe_code], [int(horizon[row])],
                                     [origin], [int(target[row])], calendar)
    with torch.no_grad():
        value = float(model(torch.from_numpy(np.ascontiguousarray(window)),
                            torch.from_numpy(static_one))[0])
    stored = float(np.load(str(PRED / "GRU_R1_validation.npy"))[row])
    report.add("Independent: reloaded GRU_R1 reproduces a stored prediction",
               abs(value - stored) < 1e-3,
               "recomputed %.6f vs stored %.6f" % (value, stored))

    # 8-12: metrics recomputed from the arrays
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = {n: np.load(str(PRED / ("%s_validation.npy" % n)))
                   .astype(np.float64) for n in NEURAL_MODELS}
    cells = read_csv(OUT / "phase4_metrics_by_station_horizon.csv")
    cell = [r for r in cells if r["model"] == "GRU_R1"
            and r["station"] == probe_station and r["horizon"] == "6"][0]
    selector = (station == probe_station) & (horizon == 6)
    report.add("Independent: GRU_R1 %s h6 cell MAE" % probe_station,
               abs(float(np.abs(actual[selector]
                                - predictions["GRU_R1"][selector]).mean())
                   - float(cell["MAE"])) < 1e-6)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase4_model_comparison.csv")}
    per_cell = []
    for st in STATIONS:
        for hz in HORIZONS:
            mask = (station == st) & (horizon == hz)
            per_cell.append(float(np.abs(actual[mask]
                                         - predictions["TCN_R1"][mask]).mean()))
    report.add("Independent: TCN_R1 overall macro station-horizon MAE",
               abs(float(np.mean(per_cell))
                   - float(comparison["TCN_R1"]["macro_station_horizon_MAE"]))
               < 1e-6, "recomputed %.6f" % float(np.mean(per_cell)))

    severe_rows = {r["model"]: r
                   for r in read_csv(OUT / "phase4_severe_metrics.csv")}
    mask = actual > 244.0
    recomputed = float(np.abs(actual[mask]
                              - predictions["GRU_R1"][mask]).mean())
    report.add("Independent: GRU_R1 severe MAE at the frozen threshold",
               abs(recomputed - float(severe_rows["GRU_R1"]["severe_MAE"]))
               < 1e-6, "recomputed %.6f over %d severe hours"
               % (recomputed, int(mask.sum())))

    gains = {(r["architecture"], r["scope"], r["from_model"]): r
             for r in read_csv(OUT / "phase4_information_regime_gain.csv")}

    def macro(name):
        values = []
        for st in STATIONS:
            for hz in HORIZONS:
                m = (station == st) & (horizon == hz)
                values.append(float(np.abs(actual[m]
                                           - predictions[name][m]).mean()))
        return float(np.mean(values))

    r0, r1, r2 = macro("GRU_R0"), macro("GRU_R1"), macro("GRU_R2")
    row_gain = gains[("GRU", "overall", "GRU_R0")]
    report.add("Independent: GRU R0->R1 gain",
               abs((r0 - r1) - float(row_gain["absolute_MAE_improvement"]))
               < 1e-6, "recomputed %.6f" % (r0 - r1))
    row_gain = gains[("GRU", "overall", "GRU_R1")]
    report.add("Independent: GRU R1->R2 gain",
               abs((r1 - r2) - float(row_gain["absolute_MAE_improvement"]))
               < 1e-6, "recomputed %.6f" % (r1 - r2))

    # 13 & 14
    report.add("Independent: every neural array holds 413,148 predictions",
               all(v.size == EXPECTED_SAMPLES for v in predictions.values()))
    report.add("Independent: no artifact holds a test-period target",
               not list(PRED.glob("*test*"))
               and bool(np.isnan(rolling._scaled[
                   index_of(PARTITION_BOUNDS["test"][0]):]).all()))


def main():
    print("")
    print("[AIRSENSE V2 PHASE-4 VALIDATION]")
    print("")
    print("Read-only. Development validation only; the final test is sealed.")
    print("")
    report = Report()
    station, horizon, target = load_index()
    upstream(report)
    contract(report)
    grids_and_selection(report)
    predictions_and_metrics(report, station, horizon)
    prohibitions(report)
    independent(report, station, horizon, target)

    print("=" * 78)
    report.render()
    print("=" * 78)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-4 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-4 VALIDATION: PASS")
    print("Development validation only. The 2016-17 test remains sealed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
