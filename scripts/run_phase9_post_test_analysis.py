"""Post-test exploratory error analysis for AirSense V2.

Protocol Phase 9, under the frozen plan in
``artifacts/phase9_analysis_freeze.json``.

**Classification: POST-TEST EXPLORATORY. Confirmatory status: NONE.**

This script loads no model, runs no inference and generates no prediction. It
consumes the three frozen Phase-8 prediction arrays, the canonical test sample
index, the final-test truth read from the raw station CSVs, and training-only
statistics. Every definition it applies - severe rule, concentration
boundaries, ACF lags, event rule, seasons, hours, complementarity statistics -
was frozen before this file produced a single number.

Nothing here may change a Phase-8 result. If a reconciliation disagrees, the
script stops rather than adjusting anything.

Usage:
    .venv-v2/bin/python scripts/run_phase9_post_test_analysis.py
"""

import csv
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    STATIONS, TARGET, index_of, timestamp_at)
from src.evaluation.final_test import (                           # noqa: E402
    TEST_END, TEST_START, load_test_index)

ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED_DIR = ARTIFACTS / "phase8_predictions"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"

MODELS = ["B3_R2", "GRU_R1", "B0"]
HORIZONS = [1, 6, 12, 24]
SEVERE_THRESHOLD = 244.0
TRAINING_MEDIAN = 61.0
ACF_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
MIN_PAIRS = 30
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM",
                   5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON",
                   10: "SON", 11: "SON"}
STRATUM_LABEL = "POST-TEST EXPLORATORY STRATA"

# Frozen development values, read from Phase-3/Phase-7 evidence. GRU_R1's
# analogue is its **seed-42** development row, because seed 42 is the artifact
# the locked test evaluated; the five-seed mean is carried alongside as context.
DEVELOPMENT = {
    "B3_R2": {"macro_station_horizon_MAE": 30.111722, "severe_MAE": 136.611861,
              "severe_mean_residual": 132.017957,
              "severe_underprediction_pct": 88.286113},
    "GRU_R1": {"macro_station_horizon_MAE": 30.912193,
               "severe_MAE": 118.964535, "severe_mean_residual": 116.404274,
               "severe_underprediction_pct": 93.555484},
    "B0": {"macro_station_horizon_MAE": 33.127699, "severe_MAE": 101.273235,
           "severe_mean_residual": 70.343743,
           "severe_underprediction_pct": 71.624812},
}
DEVELOPMENT_NOTE = {
    "B3_R2": "deterministic; Phase-3 frozen development validation",
    "GRU_R1": "seed-42 development row, the artifact the locked test "
              "evaluated (five-seed development means: macro 30.75906, "
              "severe 115.793)",
    "B0": "deterministic; Phase-3 frozen development validation",
}
LOCKED = {
    "primary_B3_R2_macro_station_horizon_MAE": 31.98722559774534,
    "secondary_GRU_R1_severe_MAE_gt_244": 109.3328897571946,
    "severe_n": 20336,
    "reference_B0_macro_station_horizon_MAE": 36.138513538776856,
    "reference_B0_severe_MAE_gt_244": 93.22447875688434,
}

START = time.time()


def log(message):
    print("[%6.1fs] %s" % (time.time() - START, message), flush=True)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name, header, rows):
    path = ARTIFACTS / ("phase9_%s.csv" % name)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    log("%-40s %6d rows  %s" % (path.name, len(rows),
                                sha256_of_file(path)[:16]))
    return path


def r6(value):
    return "" if value is None else round(float(value), 6)


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------

def load_truth():
    low, high = index_of(TEST_START), index_of(TEST_END)
    by_station = OrderedDict()
    for station in STATIONS:
        series = np.full(high + 1, np.nan, dtype=np.float64)
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        with open(str(path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = index_of(stamp)
                if position < low or position > high:
                    continue
                if row[TARGET] in ("NA", ""):
                    continue
                series[position] = float(row[TARGET])
        by_station[station] = series
    return by_station, low, high


def metrics(actual, prediction):
    residual = actual - prediction
    absolute = np.abs(residual)
    out = OrderedDict([
        ("n", int(actual.size)),
        ("MAE", float(absolute.mean()) if actual.size else None),
        ("RMSE", float(np.sqrt((residual ** 2).mean())) if actual.size
         else None),
        ("mean_residual", float(residual.mean()) if actual.size else None),
        ("underprediction_rate_pct",
         float(100.0 * (residual > 0).sum() / actual.size) if actual.size
         else None),
    ])
    if actual.size > 1:
        total = ((actual - actual.mean()) ** 2).sum()
        out["R2"] = (float(1.0 - (residual ** 2).sum() / total)
                     if total > 0 else None)
    else:
        out["R2"] = None
    return out


def paired_acf(series, lag):
    """Pearson correlation between x[t] and x[t+lag] over complete pairs."""
    if lag >= series.size:
        return None, 0
    left, right = series[:-lag], series[lag:]
    mask = np.isfinite(left) & np.isfinite(right)
    n = int(mask.sum())
    if n < MIN_PAIRS:
        return None, n
    a, b = left[mask], right[mask]
    if a.std() == 0 or b.std() == 0:
        return None, n
    return float(np.corrcoef(a, b)[0, 1]), n


def main():
    freeze = json.load(open(str(ARTIFACTS / "phase9_analysis_freeze.json"),
                            encoding="utf-8"))
    if freeze["phase9_status"] != "POST_TEST_EXPLORATORY":
        raise SystemExit("analysis freeze is not the exploratory plan")
    log("analysis freeze %s" % sha256_of_file(
        ARTIFACTS / "phase9_analysis_freeze.json")[:16])

    index = load_test_index(PROCESSED)
    stations = index["station"]
    horizons = index["horizon"]
    targets = index["target_index"]
    truth, low, high = load_truth()
    actual = np.empty(targets.size, dtype=np.float64)
    for station in STATIONS:
        rows = np.flatnonzero(stations == station)
        actual[rows] = truth[station][targets[rows]]
    if not np.isfinite(actual).all():
        raise SystemExit("a canonical target is missing")

    stored = OrderedDict()
    lock = json.load(open(str(ARTIFACTS / "phase8_prediction_lock.json"),
                          encoding="utf-8"))
    for model in MODELS:
        entry = lock["models"][model]
        path = PROJECT_ROOT / entry["prediction_file"]
        if sha256_of_file(path) != entry["sha256"]:
            raise SystemExit("%s drifted from the prediction lock" % model)
        stored[model] = np.load(str(path)).astype(np.float64)
    log("frozen predictions loaded and hash-verified; no model was loaded")

    months = np.array([timestamp_at(t).month for t in targets])
    hours = np.array([timestamp_at(t).hour for t in targets])
    seasons = np.array([SEASON_OF_MONTH[m] for m in months])
    severe = actual > SEVERE_THRESHOLD
    boundary = int((actual == SEVERE_THRESHOLD).sum())

    # Row groups, computed once. The station column is an array of Python
    # strings, so comparing it inside a per-event loop costs a full 411k-row
    # scan every time; the event analysis alone would issue tens of thousands
    # of those. Grouping once and binary-searching the sorted target index
    # turns each event lookup into O(log n).
    station_rows = OrderedDict(
        (station, np.flatnonzero(stations == station)) for station in STATIONS)
    group_rows, group_targets = OrderedDict(), OrderedDict()
    for station in STATIONS:
        rows = station_rows[station]
        for horizon in HORIZONS:
            local = rows[horizons[rows] == horizon]
            order = np.argsort(targets[local], kind="stable")
            local = local[order]
            group_rows[(station, horizon)] = local
            group_targets[(station, horizon)] = targets[local]

    def rows_in_window(station, horizon, start, end):
        """Rows for one cell whose target falls in [start, end]."""
        sorted_targets = group_targets[(station, horizon)]
        low_bound = np.searchsorted(sorted_targets, start, side="left")
        high_bound = np.searchsorted(sorted_targets, end, side="right")
        return group_rows[(station, horizon)][low_bound:high_bound]

    # ---- reconciliation against the locked values ----------------------
    cells = OrderedDict()
    for position in range(actual.size):
        cells.setdefault((stations[position], int(horizons[position])),
                         []).append(position)

    def macro(model):
        return float(np.mean([
            float(np.abs(actual[rows] - stored[model][rows]).mean())
            for rows in cells.values()]))

    checks = [
        ("B3_R2 macro", macro("B3_R2"),
         LOCKED["primary_B3_R2_macro_station_horizon_MAE"]),
        ("B0 macro", macro("B0"),
         LOCKED["reference_B0_macro_station_horizon_MAE"]),
        ("GRU_R1 severe", float(np.abs(actual[severe]
                                       - stored["GRU_R1"][severe]).mean()),
         LOCKED["secondary_GRU_R1_severe_MAE_gt_244"]),
        ("B0 severe", float(np.abs(actual[severe]
                                   - stored["B0"][severe]).mean()),
         LOCKED["reference_B0_severe_MAE_gt_244"]),
    ]
    for label, value, expected in checks:
        if abs(value - expected) >= 1e-9:
            raise SystemExit("STOP FOR HUMAN REVIEW: %s recomputed %.12f vs "
                             "locked %.12f" % (label, value, expected))
    if int(severe.sum()) != LOCKED["severe_n"]:
        raise SystemExit("STOP FOR HUMAN REVIEW: severe n disagrees")
    log("locked Phase-8 values reconcile; severe n %d, targets exactly at "
        "244.0 excluded: %d" % (int(severe.sum()), boundary))

    # ---- A: development -> locked-test generalization -------------------
    rows = []
    for model in MODELS:
        locked_values = {
            "macro_station_horizon_MAE": macro(model),
            "severe_MAE": float(np.abs(actual[severe]
                                       - stored[model][severe]).mean()),
            "severe_mean_residual": float((actual[severe]
                                           - stored[model][severe]).mean()),
            "severe_underprediction_pct": float(
                100.0 * ((actual[severe] - stored[model][severe]) > 0).sum()
                / severe.sum()),
        }
        for metric, test_value in locked_values.items():
            dev = DEVELOPMENT[model].get(metric)
            if dev is None:
                rows.append([model, metric, "NA", r6(test_value), "NA", "NA",
                             "no exact frozen development analogue"])
                continue
            rows.append([model, metric, r6(dev), r6(test_value),
                         r6(test_value - dev),
                         r6(100.0 * (test_value - dev) / dev),
                         DEVELOPMENT_NOTE[model]])
    write_csv("generalization_gap",
              ["model", "metric", "development_value", "locked_test_value",
               "absolute_change", "relative_change_pct", "note"], rows)

    # ---- B: horizon structure -------------------------------------------
    b0_mae = {}
    for horizon in HORIZONS:
        selector = horizons == horizon
        b0_mae[horizon] = float(np.abs(actual[selector]
                                       - stored["B0"][selector]).mean())
    rows = []
    for model in MODELS:
        for horizon in HORIZONS:
            selector = horizons == horizon
            row = metrics(actual[selector], stored[model][selector])
            difference = row["MAE"] - b0_mae[horizon]
            rows.append([model, horizon, row["n"], r6(row["MAE"]),
                         r6(row["RMSE"]), r6(row["R2"]),
                         r6(row["mean_residual"]),
                         r6(row["underprediction_rate_pct"]), r6(difference),
                         r6(-100.0 * difference / b0_mae[horizon])])
    write_csv("horizon_error_analysis",
              ["model", "horizon_hours", "n", "MAE", "RMSE", "R2",
               "mean_residual", "underprediction_rate_pct", "mae_diff_vs_B0",
               "mae_rel_improvement_vs_B0_pct"], rows)

    # ---- C: station and station x horizon -------------------------------
    rows, cell_rows = [], []
    for model in MODELS:
        for station in STATIONS:
            selector = stations == station
            row = metrics(actual[selector], stored[model][selector])
            local_severe = selector & severe
            severe_mae = (float(np.abs(actual[local_severe]
                                       - stored[model][local_severe]).mean())
                          if local_severe.any() else None)
            rows.append([model, station, row["n"], r6(row["MAE"]),
                         r6(row["RMSE"]), r6(row["R2"]),
                         r6(row["mean_residual"]),
                         r6(row["underprediction_rate_pct"]),
                         int(local_severe.sum()), r6(severe_mae)])
            for horizon in HORIZONS:
                local = selector & (horizons == horizon)
                cell = metrics(actual[local], stored[model][local])
                cell_rows.append([model, station, horizon, cell["n"],
                                  r6(cell["MAE"]), r6(cell["RMSE"]),
                                  r6(cell["R2"]), r6(cell["mean_residual"]),
                                  r6(cell["underprediction_rate_pct"])])
    write_csv("station_error_analysis",
              ["model", "station", "n", "MAE", "RMSE", "R2", "mean_residual",
               "underprediction_rate_pct", "severe_n", "severe_MAE"], rows)
    write_csv("station_horizon_error_analysis",
              ["model", "station", "horizon_hours", "n", "MAE", "RMSE", "R2",
               "mean_residual", "underprediction_rate_pct"], cell_rows)

    # ---- D: residual temporal structure ---------------------------------
    span = high + 1
    acf_rows, summary_rows = [], []
    per_model_horizon = OrderedDict()
    for model in MODELS:
        for station in STATIONS:
            for horizon in HORIZONS:
                local = np.flatnonzero((stations == station)
                                       & (horizons == horizon))
                series = np.full(span, np.nan, dtype=np.float64)
                series[targets[local]] = (actual[local]
                                          - stored[model][local])
                for lag in ACF_LAGS:
                    value, pairs = paired_acf(series, lag)
                    acf_rows.append([model, station, horizon, lag,
                                     r6(value) if value is not None else "NA",
                                     pairs])
                    per_model_horizon.setdefault(
                        (model, horizon, lag), []).append(value)
    write_csv("residual_acf",
              ["model", "station", "horizon_hours", "lag_hours", "acf",
               "n_pairs"], acf_rows)
    for (model, horizon, lag), values in per_model_horizon.items():
        present = [v for v in values if v is not None]
        if not present:
            summary_rows.append([model, horizon, lag] + ["NA"] * 7 + [0])
            continue
        array = np.array(present)
        q25, q75 = float(np.percentile(array, 25)), float(
            np.percentile(array, 75))
        summary_rows.append([model, horizon, lag, r6(array.mean()),
                             r6(np.median(array)), r6(array.min()),
                             r6(array.max()), r6(q25), r6(q75), r6(q75 - q25),
                             len(present)])
    write_csv("residual_acf_summary",
              ["model", "horizon_hours", "lag_hours", "mean", "median", "min",
               "max", "q25", "q75", "iqr", "n_stations"], summary_rows)

    # ---- target autocorrelation context ---------------------------------
    rows = []
    for station in STATIONS:
        series = np.full(span, np.nan, dtype=np.float64)
        series[low:high + 1] = truth[station][low:high + 1]
        for lag in ACF_LAGS:
            value, pairs = paired_acf(series, lag)
            rows.append([station, lag,
                         r6(value) if value is not None else "NA", pairs])
    write_csv("test_target_acf", ["station", "lag_hours", "acf", "n_pairs"],
              rows)

    # ---- E: training-frozen concentration regimes -----------------------
    strata = OrderedDict([
        ("LOW", actual <= TRAINING_MEDIAN),
        ("ELEVATED", (actual > TRAINING_MEDIAN) & (actual
                                                   <= SEVERE_THRESHOLD)),
        ("SEVERE", severe),
    ])
    rows = []
    for model in MODELS:
        for name, selector in strata.items():
            row = metrics(actual[selector], stored[model][selector])
            absolute = np.abs(actual[selector] - stored[model][selector])
            rows.append([model, name, STRATUM_LABEL, row["n"], r6(row["MAE"]),
                         r6(row["RMSE"]), r6(row["mean_residual"]),
                         r6(row["underprediction_rate_pct"]),
                         r6(np.median(absolute)),
                         r6(np.percentile(absolute, 90))])
    write_csv("concentration_regime_analysis",
              ["model", "regime", "stratum_label", "n", "MAE", "RMSE",
               "mean_residual", "underprediction_rate_pct",
               "median_abs_error", "p90_abs_error"], rows)

    # ---- F: severe tail --------------------------------------------------
    rows = []
    for model in MODELS:
        for horizon in HORIZONS:
            selector = severe & (horizons == horizon)
            residual = actual[selector] - stored[model][selector]
            absolute = np.abs(residual)
            b0_absolute = np.abs(actual[selector] - stored["B0"][selector])
            b3_absolute = np.abs(actual[selector] - stored["B3_R2"][selector])
            rows.append([model, horizon, int(selector.sum()),
                         r6(absolute.mean()),
                         r6(np.sqrt((residual ** 2).mean())),
                         r6(residual.mean()), r6(np.median(residual)),
                         r6(100.0 * (residual > 0).sum() / selector.sum()),
                         r6(np.median(absolute)),
                         r6(np.percentile(absolute, 90)),
                         r6(np.percentile(absolute, 95)),
                         r6((absolute - b0_absolute).mean()),
                         r6((absolute - b3_absolute).mean())])
    write_csv("severe_tail_analysis",
              ["model", "horizon_hours", "severe_n", "MAE", "RMSE",
               "mean_residual", "median_residual", "underprediction_pct",
               "median_abs_error", "p90_abs_error", "p95_abs_error",
               "mean_abs_error_diff_vs_B0",
               "mean_abs_error_diff_vs_B3_R2"], rows)

    # ---- G: strict severe events on the unique hourly truth series ------
    event_rows, event_model_rows = [], []
    for station in STATIONS:
        series = truth[station]
        flags = np.zeros(span, dtype=bool)
        window = np.arange(low, high + 1)
        flags[window] = series[window] > SEVERE_THRESHOLD
        position, event_id = low, 0
        while position <= high:
            if not flags[position]:
                position += 1
                continue
            start = position
            while position + 1 <= high and flags[position + 1]:
                position += 1
            end = position
            # Advance past the run before the next scan. Leaving `position`
            # on the final severe hour re-detects the same event forever.
            position = end + 1
            event_id += 1
            hours_index = np.arange(start, end + 1)
            values = series[hours_index]
            peak_offset = int(np.argmax(values))
            peak_index = int(hours_index[peak_offset])
            event_rows.append([station, event_id,
                               timestamp_at(start).isoformat(),
                               timestamp_at(end).isoformat(),
                               int(end - start + 1), r6(values.mean()),
                               r6(values.max()),
                               timestamp_at(peak_index).isoformat()])
            for model in MODELS:
                for horizon in HORIZONS:
                    local = rows_in_window(station, horizon, start, end)
                    if not local.size:
                        event_model_rows.append(
                            [station, event_id, model, horizon, 0] + ["NA"] * 8)
                        continue
                    residual = actual[local] - stored[model][local]
                    at_peak = np.flatnonzero(targets[local] == peak_index)
                    first = np.flatnonzero(targets[local] == start)
                    peak_prediction = (float(stored[model][local[at_peak[0]]])
                                       if at_peak.size else None)
                    peak_actual = (float(actual[local[at_peak[0]]])
                                   if at_peak.size else None)
                    event_model_rows.append([
                        station, event_id, model, horizon, int(local.size),
                        r6(np.abs(residual).mean()), r6(residual.mean()),
                        r6(peak_prediction) if at_peak.size else "NA",
                        r6(peak_actual - peak_prediction) if at_peak.size
                        else "NA",
                        r6(abs(peak_actual - peak_prediction)) if at_peak.size
                        else "NA",
                        ("true" if peak_actual > peak_prediction else "false")
                        if at_peak.size else "NA",
                        ("true" if stored[model][local[first[0]]]
                         > SEVERE_THRESHOLD else "false") if first.size
                        else "NA",
                        int((stored[model][local] > SEVERE_THRESHOLD).sum())])
    write_csv("severe_events",
              ["station", "event_id", "start_timestamp", "end_timestamp",
               "duration_hours", "actual_mean", "actual_peak",
               "peak_timestamp"], event_rows)
    write_csv("severe_event_model_analysis",
              ["station", "event_id", "model", "horizon_hours", "severe_hours",
               "event_MAE", "event_mean_residual", "prediction_at_peak",
               "residual_at_peak", "abs_error_at_peak", "underpredicted_peak",
               "predicted_severe_at_first_severe_hour",
               "predicted_severe_hours"], event_model_rows)

    # ---- severe detection diagnostic ------------------------------------
    rows = []
    for model in MODELS:
        for horizon in HORIZONS:
            selector = horizons == horizon
            truth_flag = severe[selector]
            predicted = stored[model][selector] > SEVERE_THRESHOLD
            tp = int((truth_flag & predicted).sum())
            fp = int((~truth_flag & predicted).sum())
            tn = int((~truth_flag & ~predicted).sum())
            fn = int((truth_flag & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision and recall else None)
            specificity = tn / (tn + fp) if tn + fp else None
            rows.append([model, horizon, tp, fp, tn, fn, r6(precision),
                         r6(recall), r6(f1), r6(specificity)])
    write_csv("severe_detection_analysis",
              ["model", "horizon_hours", "TP", "FP", "TN", "FN", "precision",
               "recall", "F1", "specificity"], rows)

    # ---- H: seasonal ------------------------------------------------------
    rows = []
    for model in MODELS:
        for season in ("DJF", "MAM", "JJA", "SON"):
            for horizon in HORIZONS:
                selector = (seasons == season) & (horizons == horizon)
                row = metrics(actual[selector], stored[model][selector])
                local_severe = selector & severe
                severe_mae = (float(np.abs(
                    actual[local_severe] - stored[model][local_severe]).mean())
                    if local_severe.any() else None)
                rows.append([model, season, horizon, row["n"], r6(row["MAE"]),
                             r6(row["RMSE"]), r6(row["mean_residual"]),
                             r6(row["underprediction_rate_pct"]),
                             int(local_severe.sum()), r6(severe_mae)])
    write_csv("seasonal_error_analysis",
              ["model", "season", "horizon_hours", "n", "MAE", "RMSE",
               "mean_residual", "underprediction_rate_pct", "severe_n",
               "severe_MAE"], rows)

    # ---- I: hour of day ---------------------------------------------------
    rows = []
    for model in MODELS:
        for horizon in HORIZONS:
            for hour in range(24):
                selector = (horizons == horizon) & (hours == hour)
                row = metrics(actual[selector], stored[model][selector])
                rows.append([model, horizon, hour, row["n"], r6(row["MAE"]),
                             r6(row["mean_residual"]),
                             r6(row["underprediction_rate_pct"])])
    write_csv("hour_of_day_error_analysis",
              ["model", "horizon_hours", "target_hour", "n", "MAE",
               "mean_residual", "underprediction_rate_pct"], rows)

    # ---- J: error complementarity (no ensemble is constructed) ----------
    absolute = OrderedDict(
        (model, np.abs(actual - stored[model])) for model in MODELS)
    rows = []
    for stratum, selector in (("overall", np.ones(actual.size, dtype=bool)),
                              ("severe", severe)):
        for i, first in enumerate(MODELS):
            for second in MODELS[i + 1:]:
                a, b = absolute[first][selector], absolute[second][selector]
                difference = a - b
                rows.append([
                    stratum, first, second, int(selector.sum()),
                    r6(np.corrcoef(a, b)[0, 1]),
                    r6(scipy_stats.spearmanr(a, b).statistic),
                    r6(difference.mean()), r6(np.median(difference)),
                    r6(100.0 * (a < b).sum() / selector.sum()),
                    r6(100.0 * (a > b).sum() / selector.sum()),
                    r6(100.0 * (a == b).sum() / selector.sum())])
        stack = np.vstack([absolute[m][selector] for m in MODELS])
        minimum = stack.min(axis=0)
        tied = (stack == minimum).sum(axis=0) > 1
        for position, model in enumerate(MODELS):
            strictly_best = (stack[position] == minimum) & ~tied
            rows.append([stratum, model, "BEST_OF_THREE", int(selector.sum()),
                         "NA", "NA", "NA", "NA",
                         r6(100.0 * strictly_best.sum() / selector.sum()),
                         "NA", r6(100.0 * tied.sum() / selector.sum())])
    write_csv("error_complementarity",
              ["stratum", "model_a", "model_b", "n", "pearson_abs_error",
               "spearman_abs_error", "mean_paired_abs_error_diff",
               "median_paired_abs_error_diff", "fraction_a_strictly_better",
               "fraction_b_strictly_better", "fraction_tied"], rows)

    # ---- K: negative predictions -----------------------------------------
    rows = []
    for model in MODELS:
        prediction = stored[model]
        negative = prediction < 0
        rows.append([model, "overall", "all", int(prediction.size),
                     int(negative.sum()),
                     r6(100.0 * negative.sum() / prediction.size),
                     r6(prediction.min())])
        for horizon in HORIZONS:
            selector = horizons == horizon
            local = prediction[selector]
            rows.append([model, "horizon", horizon, int(local.size),
                         int((local < 0).sum()),
                         r6(100.0 * (local < 0).sum() / local.size),
                         r6(local.min())])
        for station in STATIONS:
            selector = stations == station
            local = prediction[selector]
            rows.append([model, "station", station, int(local.size),
                         int((local < 0).sum()),
                         r6(100.0 * (local < 0).sum() / local.size),
                         r6(local.min())])
    write_csv("negative_prediction_analysis",
              ["model", "scope", "key", "n", "n_negative", "pct_negative",
               "min_prediction"], rows)

    log("all Phase-9 tables written; models loaded 0, predictions generated 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
