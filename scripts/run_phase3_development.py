"""Run the AirSense V2 classical development baselines.

Protocol Phase 3, Stage B. Requires the pre-validation freeze and the
validation opening receipt to already exist.

Order of operations is the protocol's, not convenience's: every prediction is
produced first, and only then is the validation target read for scoring.
PM2.5 history reaches feature code exclusively through the guarded rolling
interface, which cannot return anything later than the forecast origin. The
locked test partition is never opened.

Usage:
    .venv-v2/bin/python scripts/run_phase3_development.py
"""

import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb                                            # noqa: E402

from src.data.preprocessing import (                              # noqa: E402
    PARTITION_BOUNDS, STATIONS, TARGET, index_of,
    load_training_statistics, station_code)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.data.windowing import StationSeries                      # noqa: E402
from src.evaluation import metrics as M                           # noqa: E402
from src.evaluation.scoring import ValidationTargetOracle         # noqa: E402
from src.models import b3_grid, baselines                         # noqa: E402
from src.models.b3_features import (                              # noqa: E402
    StationFeatureSource, build_matrix, feature_names)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED_DIR = PROJECT_ROOT / "results" / "validation" / "predictions"
MODEL_DIR = PROJECT_ROOT / "results" / "models" / "b3"
FREEZE = PROJECT_ROOT / "artifacts" / "phase3_prevalidation_freeze.json"
RECEIPT = (PROJECT_ROOT / "artifacts"
           / "validation_development_opening_receipt.json")

N_JOBS = min(16, os.cpu_count() or 1)
VALIDATION_END = PARTITION_BOUNDS["validation"][1]


def log(message):
    print("[%7.1fs] %s" % (time.time() - START, message), flush=True)


START = time.time()


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def load_index(partition):
    """Canonical sample index, in file order. Alignment is by row position."""
    stations, horizons, targets = [], [], []
    with open(str(PROCESSED / "sample_index" / ("%s.csv" % partition)),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
            targets.append(int(row["target_row_index"]))
    return (np.array(stations), np.array(horizons, dtype=np.int32),
            np.array(targets, dtype=np.int64))


def main():
    for required in (FREEZE, RECEIPT):
        if not required.is_file():
            raise SystemExit("refusing to open validation: %s missing"
                             % required)
    log("prevalidation freeze  %s" % sha256_of_file(FREEZE)[:16])
    log("validation receipt    %s" % sha256_of_file(RECEIPT)[:16])

    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    station_stats = json.load(open(str(
        PROJECT_ROOT / "artifacts" / "preprocessing_statistics.json"),
        encoding="utf-8"))["station_training_medians"]

    val_station, val_horizon, val_target = load_index("validation")
    tr_station, tr_horizon, tr_target = load_index("train")
    log("validation samples %d | training samples %d"
        % (val_target.size, tr_target.size))

    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))
    calendar_raw = np.load(str(PROCESSED / "calendar" / "calendar_raw.npy"))

    rolling = OrderedDict()
    series = OrderedDict()
    train_targets = OrderedDict()
    for station in STATIONS:
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        rolling[station] = RollingPm25Series(
            station, path, statistics, station_stats[station][TARGET],
            unseal_until=VALIDATION_END)
        series[station] = StationSeries(PROCESSED, station)
        train_targets[station] = np.load(str(
            PROCESSED / "target_series" / ("%s_train_pm25.npy" % station)))
    log("guarded rolling history built for 12 stations (unsealed to "
        "validation end only)")

    # ---- B0 / B1 / B2, predictions first -------------------------------
    n = val_target.size
    b0 = np.empty(n, dtype=np.float64)
    b0_source = np.empty(n, dtype=np.int8)
    b1 = np.empty(n, dtype=np.float64)
    b1_direct = np.zeros(n, dtype=bool)
    b2 = np.empty(n, dtype=np.float64)
    b2_level = np.empty(n, dtype=np.int8)

    with open(str(PROCESSED / "climatology" / "b2_training_climatology.csv"),
              encoding="utf-8") as handle:
        climatology = baselines.TrainingClimatology(list(csv.DictReader(handle)))

    for station in STATIONS:
        selector = np.flatnonzero(val_station == station)
        origins = val_target[selector] - val_horizon[selector]
        predictions, sources = baselines.b0_causal_persistence(
            rolling[station], origins)
        b0[selector] = predictions
        b0_source[selector] = sources
        seasonal, direct = baselines.b1_seasonal_naive(
            rolling[station], origins, val_target[selector], predictions)
        b1[selector] = seasonal
        b1_direct[selector] = direct
        months = calendar_raw[val_target[selector], 2]
        hours = calendar_raw[val_target[selector], 0]
        climate, levels = baselines.b2_climatology(climatology, station,
                                                   months, hours)
        b2[selector] = climate
        b2_level[selector] = levels
    log("B0/B1/B2 predictions generated for every validation sample")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name, array in (("B0", b0), ("B1", b1), ("B2", b2)):
        np.save(str(PRED_DIR / ("%s_validation.npy" % name)),
                array.astype(np.float32))
    np.save(str(PRED_DIR / "B0_source_validation.npy"), b0_source)
    np.save(str(PRED_DIR / "B1_direct_validation.npy"),
            b1_direct.astype(np.uint8))
    np.save(str(PRED_DIR / "B2_level_validation.npy"), b2_level)

    # ---- scoring opens only now ----------------------------------------
    oracle = OrderedDict((s, ValidationTargetOracle(
        s, next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % s)))) for s in STATIONS)
    actual = np.empty(n, dtype=np.float64)
    for station in STATIONS:
        selector = np.flatnonzero(val_station == station)
        actual[selector] = oracle[station].actuals(val_target[selector],
                                                   b0[selector])
    np.save(str(PRED_DIR / "validation_actual.npy"),
            actual.astype(np.float32))
    log("validation actuals read for scoring (after predictions existed)")

    # ---- B3 -------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    grid = b3_grid.candidates(N_JOBS)
    candidate_rows = []
    selected_rows = []
    b3_predictions = {}

    sources = OrderedDict()
    for station in STATIONS:
        sources[station] = StationFeatureSource(
            series[station], calendar, rolling[station]._scaled)

    for regime in b3_grid.REGIMES:
        names = feature_names(regime)
        b3_predictions[regime] = np.empty(n, dtype=np.float64)
        for horizon in b3_grid.HORIZONS:
            tag = "%s h%d" % (regime, horizon)
            tr_sel = np.flatnonzero(tr_horizon == horizon)
            va_sel = np.flatnonzero(val_horizon == horizon)

            blocks, labels = [], []
            for station in STATIONS:
                rows = tr_sel[tr_station[tr_sel] == station]
                targets = tr_target[rows]
                blocks.append(build_matrix(regime, sources[station], station,
                                           targets - horizon, targets))
                labels.append(train_targets[station][targets])
            x_train = np.vstack(blocks)
            y_train = np.concatenate(labels).astype(np.float64)
            del blocks, labels

            blocks = []
            order = []
            for station in STATIONS:
                rows = va_sel[val_station[va_sel] == station]
                targets = val_target[rows]
                blocks.append(build_matrix(regime, sources[station], station,
                                           targets - horizon, targets))
                order.append(rows)
            x_valid = np.vstack(blocks)
            valid_rows = np.concatenate(order)
            del blocks, order
            log("%s matrices: train %s valid %s"
                % (tag, x_train.shape, x_valid.shape))

            results = []
            for candidate in grid:
                started = time.time()
                dataset = lgb.Dataset(x_train, label=y_train,
                                      feature_name=names, free_raw_data=False)
                model = lgb.train(dict(candidate["parameters"]), dataset,
                                  num_boost_round=candidate["num_boost_round"])
                prediction = model.predict(x_valid)
                macro_mae = M.macro_station_mae(
                    actual[valid_rows], prediction, val_station[valid_rows])
                macro_rmse = M.macro_station_rmse(
                    actual[valid_rows], prediction, val_station[valid_rows])
                results.append(OrderedDict([
                    ("candidate_id", candidate["candidate_id"]),
                    ("num_leaves", candidate["num_leaves"]),
                    ("min_child_samples", candidate["min_child_samples"]),
                    ("learning_rate", candidate["learning_rate"]),
                    ("macro_station_mae", macro_mae),
                    ("macro_station_rmse", macro_rmse),
                    ("micro_mae", M.mae(actual[valid_rows], prediction)),
                    ("micro_rmse", M.rmse(actual[valid_rows], prediction)),
                    ("model", model),
                    ("prediction", prediction),
                ]))
                log("  %s %s macro-MAE %.6f (%.0fs)"
                    % (tag, candidate["candidate_id"], macro_mae,
                       time.time() - started))

            winner = b3_grid.select(results)
            for row in results:
                candidate_rows.append([
                    regime, horizon, row["candidate_id"],
                    row["learning_rate"], row["num_leaves"],
                    row["min_child_samples"], 400,
                    round(row["macro_station_mae"], 6),
                    round(row["macro_station_rmse"], 6),
                    round(row["micro_mae"], 6), round(row["micro_rmse"], 6),
                    "true" if row["candidate_id"] == winner["candidate_id"]
                    else "false"])
            b3_predictions[regime][valid_rows] = winner["prediction"]
            model_file = MODEL_DIR / ("b3_%s_h%d_%s.txt"
                                      % (regime, horizon,
                                         winner["candidate_id"]))
            winner["model"].save_model(str(model_file))
            selected_rows.append([
                regime, horizon, winner["candidate_id"],
                round(winner["macro_station_mae"], 6),
                round(winner["macro_station_rmse"], 6),
                str(model_file.relative_to(PROJECT_ROOT)),
                sha256_of_file(model_file)])
            importance = winner["model"].feature_importance("gain")
            write_csv(MODEL_DIR / ("importance_%s_h%d.csv" % (regime, horizon)),
                      ["feature", "gain"],
                      sorted(zip(names, [float(v) for v in importance]),
                             key=lambda kv: (-kv[1], kv[0])))
            log("%s selected %s" % (tag, winner["candidate_id"]))
            del x_train, x_valid, results

        np.save(str(PRED_DIR / ("B3_%s_validation.npy" % regime)),
                b3_predictions[regime].astype(np.float32))

    write_csv(MODEL_DIR / "b3_candidate_metrics.csv",
              ["regime", "horizon", "candidate_id", "learning_rate",
               "num_leaves", "min_child_samples", "n_estimators",
               "macro_station_mae", "macro_station_rmse", "micro_mae",
               "micro_rmse", "selected"], candidate_rows)
    write_csv(MODEL_DIR / "b3_selected_models.csv",
              ["regime", "horizon", "selected_candidate", "selection_mae",
               "selection_rmse", "model_file", "model_sha256"], selected_rows)
    log("B3 complete: %d candidate fits, %d selected models"
        % (len(candidate_rows), len(selected_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
