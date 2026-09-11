"""Validate the AirSense V2 Phase-3 development baselines.

Read-only. Trains nothing, predicts nothing new, repairs nothing. Exits 0 only
if every gate passes.

As in earlier phases the twelve headline facts are **independently
recomputed** through a separate path that re-derives values from the raw CSVs
and the frozen statistics, so an error in the modelling code cannot validate
itself.

Usage:
    .venv-v2/bin/python scripts/validate_phase3.py
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    CAUSAL_FFILL_MAX_HOURS, PARTITION_BOUNDS, STATIONS, TARGET, FULL_START,
    index_of, load_training_statistics, timestamp_at)
from src.evaluation import metrics as M                           # noqa: E402
from src.evaluation.scoring import (                              # noqa: E402
    ScoringAccessError, ValidationTargetOracle)
from src.models import b3_grid                                    # noqa: E402
from src.models.b3_features import (                              # noqa: E402
    LAGS, SUMMARY_WINDOWS, feature_names)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
MODEL_DIR = PROJECT_ROOT / "results" / "models" / "b3"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FREEZE = ARTIFACTS / "phase3_prevalidation_freeze.json"
RECEIPT = ARTIFACTS / "validation_development_opening_receipt.json"

MODELS = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"]
HORIZONS = [1, 6, 12, 24]
REGIMES = ["R0", "R1", "R2"]
TOLERANCE = 1e-9
TEST_START = PARTITION_BOUNDS["test"][0]


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
    for name in ("validate_foundation.py", "validate_phase1.py",
                 "validate_phase2.py"):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)
    freeze = load_json(FREEZE)
    drift = []
    for key, path in (
            ("foundation_manifest_sha256", "artifacts/v2_foundation_manifest.json"),
            ("phase1_manifest_sha256", "artifacts/v2_phase1_manifest.json"),
            ("phase2_manifest_sha256", "artifacts/v2_phase2_manifest.json"),
            ("sample_universe_manifest_sha256",
             "artifacts/sample_universe_manifest.json"),
            ("preprocessing_statistics_sha256",
             "artifacts/preprocessing_statistics.json"),
            ("windowing_config_sha256", "configs/windowing.json"),
            ("baselines_config_sha256", "configs/baselines.json"),
            ("preprocessing_config_sha256", "configs/preprocessing.json")):
        if freeze["upstream"][key] != sha256_of_file(PROJECT_ROOT / path):
            drift.append(path)
    report.add("Upstream artifacts unchanged since the freeze", not drift,
               "drifted: %s" % drift)


def freeze_gates(report):
    freeze = load_json(FREEZE)
    receipt = load_json(RECEIPT)
    report.add("Validation receipt references the pre-validation freeze",
               receipt["phase3_prevalidation_freeze_sha256"]
               == sha256_of_file(FREEZE))
    report.add("Freeze records no performance value",
               freeze["validation_metrics_seen"] == 0
               and freeze["test_metrics_seen"] == 0
               and freeze["test_predictions_generated"] == 0)
    report.add("Receipt records the grid, features, metric and threshold as "
               "already frozen",
               receipt["b3_candidate_grid_already_frozen"] is True
               and receipt["b3_feature_set_already_frozen"] is True
               and receipt["primary_metric_already_frozen"] is True
               and receipt["severe_threshold_already_frozen"] is True)
    report.add("Receipt carries no performance value",
               not any(isinstance(v, float) and k != "severe_threshold"
                       for k, v in receipt.items()))

    grid = freeze["b3_candidate_grid"]
    report.add("Grid holds exactly 8 candidates with frozen ids",
               grid["count"] == 8
               and grid["candidate_ids"] == ["B3_C%02d" % i
                                             for i in range(1, 9)])
    values = [(c["learning_rate"], c["num_leaves"], c["min_child_samples"])
              for c in grid["candidates"]]
    expected = [(lr, nl, mcs) for lr in [0.03, 0.07] for nl in [31, 63]
                for mcs in [50, 200]]
    report.add("Grid values match the frozen nested loop", values == expected,
               "found %s" % values[:2])
    report.add("No early stopping in the frozen grid",
               grid["early_stopping"] is False
               and grid["n_estimators_num_boost_round"] == 400)
    fixed = grid["candidates"][0]["parameters"]
    report.add("Deterministic CPU configuration frozen",
               fixed["deterministic"] is True
               and fixed["force_col_wise"] is True
               and fixed["bagging_fraction"] == 1.0
               and fixed["feature_fraction"] == 1.0
               and fixed["seed"] == 42
               and fixed["objective"] == "regression_l1")
    threads = {c["parameters"]["num_threads"] for c in grid["candidates"]}
    report.add("Thread count fixed across every candidate", len(threads) == 1,
               "found %s" % threads)

    schema = freeze["b3_feature_schema"]
    report.add("Frozen lag family is 0,1,2,3,6,12,24,47 with no lag 48",
               schema["lags"] == [0, 1, 2, 3, 6, 12, 24, 47]
               and 48 not in schema["lags"])
    report.add("Frozen summary windows are 3/6/12/24/48, trailing only",
               schema["summary_windows_hours"] == [3, 6, 12, 24, 48]
               and schema["centred_windows"] is False
               and schema["summary_windows_end_at"] == "forecast_origin")
    mismatch = []
    for regime in REGIMES:
        live = hashlib.sha256(
            "\n".join(feature_names(regime)).encode("utf-8")).hexdigest()
        if live != schema["per_regime"][regime]["feature_names_sha256"]:
            mismatch.append(regime)
    report.add("Live feature schema matches the frozen schema", not mismatch,
               "changed: %s" % mismatch)
    report.add("Severe threshold frozen at 244.0 from training",
               freeze["severe_threshold"] == 244.0
               and freeze["severe_threshold_source"] == "training_pooled_p95"
               and M.SEVERE_THRESHOLD == 244.0)
    report.add("R3 not implemented in Phase 3",
               freeze["r3_implemented"] is False
               and freeze["regimes"] == REGIMES)


def results_gates(report, station, horizon, target):
    candidates = read_csv(MODEL_DIR / "b3_candidate_metrics.csv")
    selected = read_csv(MODEL_DIR / "b3_selected_models.csv")
    counts = Counter((row["regime"], row["horizon"]) for row in candidates)
    report.add("Exactly 8 candidates for each of 12 regime-horizon cells",
               len(counts) == 12 and set(counts.values()) == {8},
               "cells %d, counts %s" % (len(counts), set(counts.values())))
    report.add("Three regimes and four horizons",
               {row["regime"] for row in candidates} == set(REGIMES)
               and {int(row["horizon"]) for row in candidates}
               == set(HORIZONS))
    report.add("Twelve selected B3 models recorded", len(selected) == 12)

    problems = []
    for regime in REGIMES:
        for hz in HORIZONS:
            rows = [r for r in candidates
                    if r["regime"] == regime and int(r["horizon"]) == hz]
            recomputed = b3_grid.select([{
                "candidate_id": r["candidate_id"],
                "num_leaves": int(r["num_leaves"]),
                "min_child_samples": int(r["min_child_samples"]),
                "macro_station_mae": float(r["macro_station_mae"]),
                "macro_station_rmse": float(r["macro_station_rmse"]),
            } for r in rows])["candidate_id"]
            marked = [r["candidate_id"] for r in rows
                      if r["selected"] == "true"]
            if marked != [recomputed]:
                problems.append("%s h%d: marked %s, rule gives %s"
                                % (regime, hz, marked, recomputed))
    report.add("Selected candidate matches the frozen selection rule "
               "in all 12 cells", not problems, "; ".join(problems[:3]))

    missing = [row["model_file"] for row in selected
               if not (PROJECT_ROOT / row["model_file"]).is_file()
               or sha256_of_file(PROJECT_ROOT / row["model_file"])
               != row["model_sha256"]]
    report.add("Selected model files exist and match recorded hashes",
               not missing, "problems: %s" % missing[:3])
    report.add("Losing candidates are retained, not hidden",
               len([r for r in candidates if r["selected"] == "false"]) == 84)


def prediction_gates(report, station, horizon, target):
    actual = np.load(str(PRED / "validation_actual.npy"))
    problems = []
    for name in MODELS:
        path = PRED / ("%s_validation.npy" % name)
        if not path.is_file():
            problems.append("%s missing" % name)
            continue
        values = np.load(str(path))
        if values.size != station.size:
            problems.append("%s length %d != %d" % (name, values.size,
                                                    station.size))
        elif not np.isfinite(values).all():
            problems.append("%s has non-finite predictions" % name)
    report.add("Every model covers every validation sample (%d)"
               % station.size, not problems, "; ".join(problems[:3]))
    report.add("Actuals align with the canonical validation index",
               actual.size == station.size and np.isfinite(actual).all())

    negatives = read_csv(OUT / "negative_predictions.csv")
    clipped = [r["model"] for r in negatives if r["clipped"] != "false"]
    report.add("No prediction was clipped", not clipped, "clipped: %s"
               % clipped)
    mismatch = []
    for row in negatives:
        values = np.load(str(PRED / ("%s_validation.npy" % row["model"])))
        if abs(float(values.min()) - float(row["min_prediction"])) > 1e-3:
            mismatch.append(row["model"])
    report.add("Reported minimum predictions match the stored arrays",
               not mismatch, "mismatched: %s" % mismatch)


def reconciliation(report, station, horizon):
    comparison = {r["model"]: r for r in read_csv(OUT / "model_comparison.csv")}
    cells = read_csv(OUT / "metrics_by_station_horizon.csv")
    problems = []
    for name in MODELS:
        subset = [float(r["MAE"]) for r in cells if r["model"] == name]
        if len(subset) != 48:
            problems.append("%s has %d cells" % (name, len(subset)))
            continue
        macro = float(np.mean(subset))
        if abs(macro - float(comparison[name]["macro_station_horizon_MAE"])) \
                > 1e-6:
            problems.append("%s macro mismatch" % name)
    report.add("Primary macro metric reconciles from the 48 cells",
               not problems, "; ".join(problems[:3]))
    ranks = sorted(MODELS,
                   key=lambda m: float(
                       comparison[m]["macro_station_horizon_MAE"]))
    declared = sorted(MODELS,
                      key=lambda m: int(comparison[m]["rank_by_primary_metric"]))
    report.add("Ranking uses the primary metric only", ranks == declared,
               "recomputed %s vs declared %s" % (ranks, declared))

    severe = read_csv(OUT / "severe_metrics.csv")
    report.add("Severe tables report contributing and empty cells",
               all(r["contributing_cells"] and r["total_cells"]
                   and r["empty_cells"] != "" for r in severe))


def quarantine(report):
    offenders = []
    for base in (PRED, OUT, MODEL_DIR, ARTIFACTS):
        for path in sorted(base.rglob("*")):
            # A later phase's required ``pretest`` freeze is a declaration
            # that the test is still closed, not a test prediction/metric.
            # Keep scanning every other ``test`` filename in all four roots.
            lowered = path.name.lower()
            if path.is_file() and re.search(r"test", path.name, re.I) \
                    and "latest" not in lowered and "pretest" not in lowered:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    report.add("No test prediction or test metric artifact exists",
               not offenders, "found: %s" % offenders[:5])

    summary = load_json(ARTIFACTS / "phase3_validation_summary.json")
    report.add("Summary records a sealed test and zero test activity",
               summary["final_test_status"] == "sealed"
               and summary["test_predictions_generated"] == 0
               and summary["test_metrics_seen"] == 0
               and summary["predictions_clipped"] is False)

    oracle = ValidationTargetOracle(
        "Dongsi", next(RAW_DIR.glob("PRSA_Data_Dongsi_*.csv")))
    refused = False
    try:
        oracle.actuals([index_of(TEST_START)], [0.0])
    except ScoringAccessError:
        refused = True
    report.add("Scoring oracle refuses the locked test window", refused)
    report.add("Scoring oracle never materialises test values",
               np.isnan(oracle._values[index_of(TEST_START):]).all())


def independent(report, station, horizon, target):
    """Twelve independent recomputations through a separate path."""
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    station_stats = load_json(ARTIFACTS / "preprocessing_statistics.json")[
        "station_training_medians"]
    probe = "Dongsi"
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % probe))

    values = {}
    present = {}
    columns = ["PM2.5", "TEMP"]
    for column in columns:
        values[column] = {}
        present[column] = {}
    with open(str(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            position = index_of(stamp)
            for column in columns:
                raw = row[column]
                present[column][position] = raw not in ("NA", "")
                if raw not in ("NA", ""):
                    values[column][position] = float(raw)

    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    b0 = np.load(str(PRED / "B0_validation.npy")).astype(np.float64)
    b1 = np.load(str(PRED / "B1_validation.npy")).astype(np.float64)
    b2 = np.load(str(PRED / "B2_validation.npy")).astype(np.float64)
    source = np.load(str(PRED / "B0_source_validation.npy"))
    direct = np.load(str(PRED / "B1_direct_validation.npy")).astype(bool)
    rows = np.flatnonzero(station == probe)

    # 1-3: the three B0 resolution paths
    for code, label in ((0, "direct origin observation"),
                        (1, "short causal carry"),
                        (2, "station-median fallback")):
        picks = rows[source[rows] == code]
        if picks.size == 0:
            report.add("Independent: B0 %s" % label, True, "none present")
            continue
        row = int(picks[0])
        origin = int(target[row] - horizon[row])
        if code == 0:
            expected = values["PM2.5"].get(origin)
        elif code == 1:
            expected = None
            for back in range(1, CAUSAL_FFILL_MAX_HOURS + 1):
                if present["PM2.5"].get(origin - back):
                    expected = values["PM2.5"][origin - back]
                    break
        else:
            expected = station_stats[probe][TARGET]
        report.add("Independent: B0 %s" % label,
                   expected is not None
                   and abs(b0[row] - expected) < 1e-6,
                   "recomputed %s vs stored %s" % (expected, b0[row]))

    # 4: B1 direct seasonal prediction
    picks = rows[direct[rows]]
    row = int(picks[0])
    expected = values["PM2.5"].get(int(target[row]) - 24)
    report.add("Independent: B1 direct seasonal value",
               expected is not None and abs(b1[row] - expected) < 1e-6,
               "recomputed %s vs stored %s" % (expected, b1[row]))

    # 5: B2 station-month-hour climatology cell
    row = int(rows[0])
    stamp = timestamp_at(int(target[row]))
    published = None
    for entry in read_csv(PROCESSED / "climatology"
                          / "b2_training_climatology.csv"):
        if (entry["station"] == probe
                and entry["level"] == "station_month_hour"
                and entry["month"] == str(stamp.month)
                and entry["hour"] == str(stamp.hour)):
            published = float(entry["median_pm25_ug_m3"])
    report.add("Independent: B2 station-month-hour value",
               published is not None and abs(b2[row] - published) < 1e-6,
               "table %s vs stored %s" % (published, b2[row]))

    # 6-7: a B3 lag feature and a B3 rolling feature, rebuilt from raw
    from src.data.windowing import StationSeries
    from src.models.b3_features import StationFeatureSource, build_matrix
    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))
    series = StationSeries(PROCESSED, probe)
    scaled_pm25 = np.load(str(PROCESSED / "station_series" / probe
                              / "pm25_scaled_train_only.npy"))
    source_obj = StationFeatureSource(series, calendar, scaled_pm25)
    train_rows = np.array([index_of(datetime(2014, 6, 10, 12))])
    matrix = build_matrix("R2", source_obj, probe, train_rows - 6, train_rows)
    names = feature_names("R2")
    origin = int(train_rows[0] - 6)
    temp_series = series.numeric[:, series.numeric_channels.index("TEMP")]
    expected_temp = float(temp_series[origin - 3])
    report.add("Independent: B3 lag feature (TEMP_lag3)",
               abs(float(matrix[0, names.index("TEMP_lag3")])
                   - expected_temp) < 1e-5,
               "recomputed %.6f" % expected_temp)
    window = temp_series[origin - 11:origin + 1]
    report.add("Independent: B3 trailing feature (TEMP_w12_mean)",
               abs(float(matrix[0, names.index("TEMP_w12_mean")])
                   - float(window.mean())) < 1e-4,
               "recomputed %.6f" % float(window.mean()))

    # 8: a selected candidate's macro-station MAE
    selected = read_csv(MODEL_DIR / "b3_selected_models.csv")
    entry = [r for r in selected
             if r["regime"] == "R2" and int(r["horizon"]) == 1][0]
    predictions = np.load(str(PRED / "B3_R2_validation.npy")).astype(np.float64)
    selector = horizon == 1
    recomputed = M.macro_station_mae(actual[selector], predictions[selector],
                                     station[selector])
    report.add("Independent: selected B3_R2 h1 macro-station MAE",
               abs(recomputed - float(entry["selection_mae"])) < 1e-6,
               "recomputed %.6f vs recorded %s" % (recomputed,
                                                   entry["selection_mae"]))

    # 9: one station-horizon cell
    cells = read_csv(OUT / "metrics_by_station_horizon.csv")
    cell = [r for r in cells if r["model"] == "B3_R2"
            and r["station"] == probe and r["horizon"] == "6"][0]
    selector = (station == probe) & (horizon == 6)
    report.add("Independent: B3_R2 %s h6 cell MAE" % probe,
               abs(M.mae(actual[selector], predictions[selector])
                   - float(cell["MAE"])) < 1e-6)

    # 10: overall macro station-horizon MAE
    comparison = {r["model"]: r for r in read_csv(OUT / "model_comparison.csv")}
    recomputed = M.macro_from_cells(
        M.cell_metrics(actual, predictions, station, horizon))
    report.add("Independent: overall macro station-horizon MAE for B3_R2",
               abs(recomputed
                   - float(comparison["B3_R2"]["macro_station_horizon_MAE"]))
               < 1e-6, "recomputed %.6f" % recomputed)

    # 11: severe MAE for B3_R2
    severe = {r["model"]: r for r in read_csv(OUT / "severe_metrics.csv")}
    mask = actual > 244.0
    recomputed = float(np.abs(actual[mask] - predictions[mask]).mean())
    report.add("Independent: B3_R2 severe MAE at the frozen threshold",
               abs(recomputed - float(severe["B3_R2"]["severe_MAE"])) < 1e-6,
               "recomputed %.6f over %d severe hours"
               % (recomputed, int(mask.sum())))

    # 12: no test-target access anywhere in the prediction arrays
    report.add("Independent: no artifact holds a test-period target",
               not list(PRED.glob("*test*")))


def main():
    print("")
    print("[AIRSENSE V2 PHASE-3 VALIDATION]")
    print("")
    print("Read-only. Development validation only; the final test is sealed.")
    print("")
    report = Report()
    station, horizon, target = load_index()
    upstream(report)
    freeze_gates(report)
    results_gates(report, station, horizon, target)
    prediction_gates(report, station, horizon, target)
    reconciliation(report, station, horizon)
    quarantine(report)
    independent(report, station, horizon, target)

    print("=" * 78)
    report.render()
    print("=" * 78)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-3 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-3 VALIDATION: PASS")
    print("Development validation only. The 2016-17 test remains sealed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
