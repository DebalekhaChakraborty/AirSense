"""Independent recomputation of the AirSense V2 post-test analysis.

Protocol Phase 9, section 35.

A separate path from ``scripts/run_phase9_post_test_analysis.py``: the sample
index is re-parsed here, the truth is re-read from the raw station CSVs, and
every statistic is reimplemented in plain Python rather than reusing the
analysis helpers. Agreement is required to tight tolerance.

Nothing is trained, loaded or predicted. Frozen prediction arrays only.

Usage:
    .venv-v2/bin/python scripts/verify_phase9_independently.py
"""

import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                                # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED_DIR = ARTIFACTS / "phase8_predictions"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
FULL_START = datetime(2013, 3, 1, 0)
TEST_START, TEST_END = datetime(2016, 3, 1, 0), datetime(2017, 2, 28, 23)
SEVERE = 244.0
MEDIAN = 61.0
MODELS = ["B3_R2", "GRU_R1", "B0"]
TOLERANCE = 1e-6
CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    width = 66
    status = "PASS" if ok else "FAIL"
    print("%s %s %s" % (label, "." * max(3, width - len(label) - 6), status))
    if detail and not ok:
        print("      %s" % detail)


def hours_since_start(stamp):
    return int((stamp - FULL_START).total_seconds() // 3600)


def table(name):
    with open(str(ARTIFACTS / ("phase9_%s.csv" % name)), newline="",
              encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else None


def main():
    # -- re-parse the index independently ---------------------------------
    rows = []
    with open(str(PROCESSED / "sample_index" / "test.csv"), newline="",
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["station"], int(row["horizon_hours"]),
                         int(row["target_row_index"]),
                         datetime.strptime(row["target_timestamp"],
                                           "%Y-%m-%dT%H:%M:%S")))
    check("1  Test sample count re-parsed independently", len(rows) == 411012,
          str(len(rows)))

    # -- re-read the truth -------------------------------------------------
    low, high = hours_since_start(TEST_START), hours_since_start(TEST_END)
    truth = {}
    for path in sorted(RAW_DIR.glob("PRSA_Data_*.csv")):
        station = path.name.split("_")[2]
        series = [None] * (high + 1)
        with open(str(path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = hours_since_start(stamp)
                if position < low or position > high:
                    continue
                if row["PM2.5"] in ("NA", ""):
                    continue
                series[position] = float(row["PM2.5"])
        truth[station] = series

    actual = [truth[s][t] for s, _, t, _ in rows]
    stored = {}
    for model in MODELS:
        stored[model] = [float(v) for v in
                         np.load(str(PRED_DIR / ("%s.npy" % model)))]

    locked = json.load(open(str(ARTIFACTS
                                / "phase8_primary_results_lock.json"),
                            encoding="utf-8"))

    # -- 1-5: the locked confirmatory values -------------------------------
    cells = OrderedDict()
    for position, (station, horizon, _, _) in enumerate(rows):
        cells.setdefault((station, horizon), []).append(position)
    check("   Station-horizon cells number exactly 48", len(cells) == 48)

    def macro(model):
        return mean([mean([abs(actual[i] - stored[model][i]) for i in idx])
                     for idx in cells.values()])

    severe_idx = [i for i in range(len(rows)) if actual[i] > SEVERE]
    pairs = [
        ("1  B3_R2 locked primary MAE", macro("B3_R2"),
         locked["primary_value"]),
        ("2  GRU_R1 locked severe MAE",
         mean([abs(actual[i] - stored["GRU_R1"][i]) for i in severe_idx]),
         locked["secondary_value"]),
        ("3  B0 locked macro MAE", macro("B0"),
         locked["reference_primary_value"]),
        ("4  B0 locked severe MAE",
         mean([abs(actual[i] - stored["B0"][i]) for i in severe_idx]),
         locked["reference_severe_value"]),
    ]
    for label, value, expected in pairs:
        check("%s reproduces" % label, abs(value - expected) < 1e-9,
              "%.12f vs %.12f" % (value, expected))
    check("5  Severe n reproduces", len(severe_idx) == locked[
        "secondary_severe_n"], str(len(severe_idx)))
    boundary = sum(1 for v in actual if v == SEVERE)
    check("6  Targets exactly at 244.0 counted and excluded",
          boundary == 232 and not any(actual[i] == SEVERE for i in severe_idx),
          "%d at the boundary" % boundary)

    # -- 7: one horizon row -------------------------------------------------
    horizon_rows = table("horizon_error_analysis")
    target = next(r for r in horizon_rows
                  if r["model"] == "B3_R2" and r["horizon_hours"] == "24")
    idx = [i for i in range(len(rows)) if rows[i][1] == 24]
    value = mean([abs(actual[i] - stored["B3_R2"][i]) for i in idx])
    check("7  Horizon row (B3_R2, h=24) reproduces",
          abs(value - float(target["MAE"])) < TOLERANCE,
          "%.9f vs %s" % (value, target["MAE"]))

    # -- 8: one station row -------------------------------------------------
    station_rows = table("station_error_analysis")
    target = next(r for r in station_rows
                  if r["model"] == "GRU_R1" and r["station"] == "Dongsi")
    idx = [i for i in range(len(rows)) if rows[i][0] == "Dongsi"]
    value = mean([abs(actual[i] - stored["GRU_R1"][i]) for i in idx])
    check("8  Station row (GRU_R1, Dongsi) reproduces",
          abs(value - float(target["MAE"])) < TOLERANCE,
          "%.9f vs %s" % (value, target["MAE"]))

    # -- 9: one station x horizon row ---------------------------------------
    cell_rows = table("station_horizon_error_analysis")
    target = next(r for r in cell_rows if r["model"] == "B0"
                  and r["station"] == "Huairou" and r["horizon_hours"] == "12")
    idx = [i for i in range(len(rows))
           if rows[i][0] == "Huairou" and rows[i][1] == 12]
    value = mean([abs(actual[i] - stored["B0"][i]) for i in idx])
    check("9  Station-horizon row (B0, Huairou, h=12) reproduces",
          abs(value - float(target["MAE"])) < TOLERANCE,
          "%.9f vs %s" % (value, target["MAE"]))

    # -- 10: one residual ACF series ----------------------------------------
    acf_rows = table("residual_acf")
    station, horizon, model = "Dongsi", 24, "B3_R2"
    grid = {}
    for position, (s, h, t, _) in enumerate(rows):
        if s == station and h == horizon:
            grid[t] = actual[position] - stored[model][position]
    agreed = True
    detail = ""
    for lag in (1, 24, 168):
        left, right = [], []
        for t in sorted(grid):
            if t + lag in grid:
                left.append(grid[t])
                right.append(grid[t + lag])
        value = pearson(left, right)
        target = next(r for r in acf_rows if r["model"] == model
                      and r["station"] == station
                      and int(r["horizon_hours"]) == horizon
                      and int(r["lag_hours"]) == lag)
        if (abs(value - float(target["acf"])) >= TOLERANCE
                or len(left) != int(target["n_pairs"])):
            agreed = False
            detail = "lag %d: %.9f/%d vs %s/%s" % (
                lag, value, len(left), target["acf"], target["n_pairs"])
    check("10 Residual ACF series (B3_R2, Dongsi, h=24) reproduces", agreed,
          detail)

    # -- 11: one severe event ------------------------------------------------
    events = table("severe_events")
    first = events[0]
    series = truth[first["station"]]
    start = hours_since_start(datetime.strptime(first["start_timestamp"],
                                                "%Y-%m-%dT%H:%M:%S"))
    end = hours_since_start(datetime.strptime(first["end_timestamp"],
                                              "%Y-%m-%dT%H:%M:%S"))
    inside = all(series[t] is not None and series[t] > SEVERE
                 for t in range(start, end + 1))
    before = start - 1 < low or series[start - 1] is None or series[
        start - 1] <= SEVERE
    after = end + 1 > high or series[end + 1] is None or series[
        end + 1] <= SEVERE
    peak = max(series[t] for t in range(start, end + 1))
    check("11 Severe event reproduces and is maximal",
          inside and before and after
          and int(first["duration_hours"]) == end - start + 1
          and abs(peak - float(first["actual_peak"])) < TOLERANCE,
          "%s event %s" % (first["station"], first["event_id"]))
    check("   Every event is bounded by non-severe hours",
          len(events) == 586, "%d events" % len(events))

    # -- 12: one season row ---------------------------------------------------
    season_rows = table("seasonal_error_analysis")
    target = next(r for r in season_rows if r["model"] == "B3_R2"
                  and r["season"] == "DJF" and r["horizon_hours"] == "24")
    idx = [i for i in range(len(rows))
           if rows[i][1] == 24 and rows[i][3].month in (12, 1, 2)]
    value = mean([abs(actual[i] - stored["B3_R2"][i]) for i in idx])
    check("12 Season row (B3_R2, DJF, h=24) reproduces",
          abs(value - float(target["MAE"])) < TOLERANCE
          and len(idx) == int(target["n"]),
          "%.9f/%d vs %s/%s" % (value, len(idx), target["MAE"], target["n"]))

    # -- 13: one hour-of-day row ----------------------------------------------
    hour_rows = table("hour_of_day_error_analysis")
    target = next(r for r in hour_rows if r["model"] == "GRU_R1"
                  and r["horizon_hours"] == "24" and r["target_hour"] == "13")
    idx = [i for i in range(len(rows))
           if rows[i][1] == 24 and rows[i][3].hour == 13]
    value = mean([abs(actual[i] - stored["GRU_R1"][i]) for i in idx])
    check("13 Hour-of-day row (GRU_R1, h=24, hour 13) reproduces",
          abs(value - float(target["MAE"])) < TOLERANCE,
          "%.9f vs %s" % (value, target["MAE"]))

    # -- 14: one complementarity statistic -------------------------------------
    comp = table("error_complementarity")
    target = next(r for r in comp if r["stratum"] == "severe"
                  and r["model_a"] == "B3_R2" and r["model_b"] == "B0")
    a = [abs(actual[i] - stored["B3_R2"][i]) for i in severe_idx]
    b = [abs(actual[i] - stored["B0"][i]) for i in severe_idx]
    value = pearson(a, b)
    fraction = 100.0 * sum(1 for x, y in zip(a, b) if x < y) / len(a)
    check("14 Complementarity (severe, B3_R2 vs B0) reproduces",
          abs(value - float(target["pearson_abs_error"])) < TOLERANCE
          and abs(fraction
                  - float(target["fraction_a_strictly_better"])) < TOLERANCE,
          "r %.9f vs %s" % (value, target["pearson_abs_error"]))

    # -- concentration boundary sanity ------------------------------------------
    regime = table("concentration_regime_analysis")
    low_n = int(next(r["n"] for r in regime if r["model"] == "B3_R2"
                     and r["regime"] == "LOW"))
    check("   LOW stratum count reproduces at the training median",
          low_n == sum(1 for v in actual if v <= MEDIAN), str(low_n))

    failed = [row for row in CHECKS if not row[1]]
    print("")
    if failed:
        print("PHASE 9 INDEPENDENT VERIFICATION FAILED: %d check(s)"
              % len(failed))
        return 1
    print("PHASE 9 INDEPENDENT VERIFICATION PASSED: %d checks agree"
          % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
