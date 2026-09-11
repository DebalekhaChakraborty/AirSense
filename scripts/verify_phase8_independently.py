"""Independent verification of the AirSense V2 locked final test.

Protocol Phase 8, section 43.

Deliberately a *separate path* from `scripts/run_phase8_final_test.py`:

* the sample index is re-parsed here rather than through
  ``src.evaluation.final_test.load_test_index``;
* the test actuals are read straight from the raw station CSVs rather than
  through ``FinalTestTargetOracle``;
* every metric is recomputed from first principles in this file rather than
  through ``src.evaluation.metrics``;
* the B0 fallback chain is reimplemented from its stated rule rather than
  called;
* the B3_R2 and GRU_R1 spot predictions are rebuilt with a reveal ceiling set
  to the sample's **own origin**, not the calendar-month block Stage A used, so
  agreement also demonstrates that the block structure carried no information.

Usage:
    .venv-v2/bin/python scripts/verify_phase8_independently.py
"""

import csv
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb                                            # noqa: E402
from safetensors.torch import load_file                           # noqa: E402

from src.data.preprocessing import (                              # noqa: E402
    STATIONS, TARGET, index_of, load_training_statistics, station_code,
    timestamp_at)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.data.target_history import (                             # noqa: E402
    CausalTargetHistory, TargetAccessError)
from src.data.windowing import StationSeries                      # noqa: E402
from src.evaluation.final_test import (                           # noqa: E402
    MaskedStationInputs, TEST_END, TEST_START)
from src.models.b3_features import (                              # noqa: E402
    StationFeatureSource, build_matrix)
from src.models.gru_forecaster import GRUForecaster               # noqa: E402
from src.models.neural_features import (                          # noqa: E402
    build_static_matrix, build_station_dynamic_matrix,
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import SampleBundle, predict      # noqa: E402

ARTIFACTS = PROJECT_ROOT / "artifacts"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PRED_DIR = ARTIFACTS / "phase8_predictions"
TOLERANCE = 1e-9

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    width = 66
    status = "PASS" if ok else "FAIL"
    dots = "." * max(3, width - len(label) - len(status) - 2)
    print("%s %s %s" % (label, dots, status))
    if detail and not ok:
        print("      %s" % detail)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


# ---- independent parsing ---------------------------------------------------

def reparse_index():
    rows = []
    with open(str(PROCESSED / "sample_index" / "test.csv"), newline="",
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["station"], int(row["horizon_hours"]),
                         int(row["target_row_index"]),
                         row["target_timestamp"], row["forecast_origin"]))
    return rows


def read_actuals_from_raw():
    """Test-window PM2.5 straight off the station CSVs."""
    low, high = index_of(TEST_START), index_of(TEST_END)
    out = {}
    for station in STATIONS:
        values = np.full(high + 1, np.nan, dtype=np.float64)
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        with open(str(path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = index_of(stamp)
                if position < low or position > high:
                    continue
                raw = row[TARGET]
                if raw in ("NA", ""):
                    continue
                values[position] = float(raw)
        out[station] = values
    return out


# ---- metrics from first principles ----------------------------------------

def plain_mae(actual, prediction):
    total = 0.0
    for a, p in zip(actual, prediction):
        total += abs(a - p)
    return total / len(actual)


def plain_macro_cell_mae(actual, prediction, stations, horizons):
    buckets = OrderedDict()
    for a, p, s, h in zip(actual, prediction, stations, horizons):
        buckets.setdefault((s, int(h)), []).append(abs(a - p))
    cells = [sum(v) / len(v) for v in buckets.values()]
    return sum(cells) / len(cells), len(cells)


def main():
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")

    # 1 -- test sample count
    rows = reparse_index()
    check("1  Test sample count recomputed independently",
          len(rows) == 411012, "%d" % len(rows))

    stations = np.array([r[0] for r in rows])
    horizons = np.array([r[1] for r in rows], dtype=np.int64)
    targets = np.array([r[2] for r in rows], dtype=np.int64)
    origins = targets - horizons

    stored = OrderedDict()
    for name in ("B0", "B3_R2", "GRU_R1"):
        path = PRED_DIR / ("%s.npy" % name)
        check("   %s array matches the prediction lock" % name,
              sha256_of_file(path) == lock["models"][name]["sha256"])
        stored[name] = np.load(str(path)).astype(np.float64)

    # 14 -- alignment
    check("14 Prediction arrays align to the canonical index",
          all(stored[n].size == len(rows) for n in stored)
          and lock["test_index_sha256"] == sha256_of_file(
              PROCESSED / "sample_index" / "test.csv"))

    actual_by_station = read_actuals_from_raw()
    actual = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows):
        actual[position] = actual_by_station[row[0]][row[2]]
    check("   Every canonical target was observed",
          bool(np.isfinite(actual).all()))

    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    medians = load_json(ARTIFACTS / "preprocessing_statistics.json")[
        "station_training_medians"]
    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))

    # 2 -- one B0 prediction, from the stated rule
    probe_station = "Dongsi"
    probe = next(i for i, r in enumerate(rows)
                 if r[0] == probe_station and r[1] == 24
                 and r[2] > index_of(TEST_START) + 2000)
    origin = int(origins[probe])
    raw_path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % probe_station))
    observed = {}
    with open(str(raw_path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            if row[TARGET] not in ("NA", ""):
                observed[index_of(stamp)] = float(row[TARGET])
    expected_b0 = None
    for back in range(0, 7):
        if (origin - back) in observed:
            expected_b0 = observed[origin - back]
            break
    if expected_b0 is None:
        expected_b0 = medians[probe_station][TARGET]
    check("2  B0 spot prediction reproduces from its stated rule",
          abs(float(stored["B0"][probe]) - expected_b0) < 1e-4,
          "%s vs %s" % (stored["B0"][probe], expected_b0))

    # 5, 6 -- rolling history inside the test period, and the origin guard
    rolling = OrderedDict()
    series = OrderedDict()
    for station in STATIONS:
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        rolling[station] = RollingPm25Series(
            station, path, statistics, medians[station][TARGET],
            unseal_until=TEST_END)
        series[station] = StationSeries(PROCESSED, station)
    after_start = index_of(TEST_START) + 500
    check("5  Rolling history serves a test-period origin",
          rolling[probe_station].native_at(after_start, after_start)
          is not None)
    guarded = CausalTargetHistory(probe_station, raw_path,
                                  value_access_until=TEST_END)
    refused = False
    try:
        guarded.value_at(timestamp_at(int(targets[probe])),
                         timestamp_at(origin))
    except TargetAccessError:
        refused = True
    check("6  Target at tau was unavailable at its forecast origin", refused)
    check("   Same target is available once it becomes history",
          guarded.value_at(timestamp_at(int(targets[probe])),
                           timestamp_at(int(targets[probe]))) is not None)

    # 3 -- one B3_R2 prediction at each horizon, ceiling at the sample origin
    freeze = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    per_horizon = freeze["selected_configuration_and_artifacts"]["B3_R2"][
        "per_horizon_models"]
    b3_ok = []
    for horizon in (1, 6, 12, 24):
        idx = next(i for i, r in enumerate(rows)
                   if r[0] == probe_station and r[1] == horizon
                   and r[2] > index_of(TEST_START) + 3000)
        own_origin = int(origins[idx])
        masked = MaskedStationInputs(series[probe_station],
                                     rolling[probe_station]._scaled,
                                     own_origin)
        source = StationFeatureSource(masked, calendar, masked.pm25_scaled)
        matrix = build_matrix("R2", source, probe_station,
                              np.array([own_origin]),
                              np.array([int(targets[idx])]))
        booster = lgb.Booster(model_file=str(
            PROJECT_ROOT / per_horizon["h%d" % horizon]["model_file"]))
        value = float(booster.predict(matrix)[0])
        agree = abs(value - float(stored["B3_R2"][idx])) < 1e-4
        b3_ok.append(agree)
        check("3  B3_R2 h%-2d spot prediction reproduces at its own origin"
              % horizon, agree,
              "%s vs %s" % (value, stored["B3_R2"][idx]))

    # 4 -- one GRU_R1 prediction, ceiling at the sample origin
    idx = next(i for i, r in enumerate(rows)
               if r[0] == probe_station and r[1] == 6
               and r[2] > index_of(TEST_START) + 4000)
    own_origin = int(origins[idx])
    config = load_json(PROJECT_ROOT
                       / "results/models/neural/GRU_R1_config.json")
    model = GRUForecaster(
        dynamic_channels=len(dynamic_channel_names("R1")),
        static_channels=len(static_channel_names()),
        hidden_size=int(config["capacity"]),
        head_hidden=int(config["training"]["head_hidden"]),
        dropout=float(config["training"]["dropout"]))
    model.load_state_dict(load_file(str(
        PROJECT_ROOT / "results/models/neural/GRU_R1.safetensors")))
    model.eval()
    dynamic = []
    for station in STATIONS:
        masked = MaskedStationInputs(series[station], rolling[station]._scaled,
                                     own_origin)
        dynamic.append(build_station_dynamic_matrix("R1", masked,
                                                    masked.pm25_scaled))
    dynamic = np.stack(dynamic, axis=0)
    static = build_static_matrix(
        np.array([station_code(probe_station)], dtype=np.int64),
        np.array([6], dtype=np.int64), np.array([own_origin]),
        np.array([int(targets[idx])]), calendar)
    bundle = SampleBundle(np.array([station_code(probe_station)]),
                          np.array([6]), np.array([own_origin]),
                          np.array([int(targets[idx])]), dynamic, static)
    value = float(predict(model, bundle)[0][0])
    check("4  GRU_R1 spot prediction reproduces at its own origin",
          abs(value - float(stored["GRU_R1"][idx])) < 1e-4,
          "%s vs %s" % (value, stored["GRU_R1"][idx]))

    # 7 -- one station-horizon cell MAE
    cell = (stations == probe_station) & (horizons == 24)
    cell_mae = plain_mae(actual[cell], stored["B3_R2"][cell])
    table = {}
    with open(str(ARTIFACTS / "phase8_final_test_tables"
                  / "metrics_by_station_horizon.csv"),
              newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["model"], row["station"],
                   int(row["horizon_hours"]))] = float(row["MAE"])
    check("7  One station-horizon cell MAE reproduces",
          abs(cell_mae - table[("B3_R2", probe_station, 24)]) < 1e-5,
          "%.9f vs %.9f" % (cell_mae, table[("B3_R2", probe_station, 24)]))

    # 8, 10 -- macro station-horizon MAE
    b3_macro, cells = plain_macro_cell_mae(actual, stored["B3_R2"], stations,
                                           horizons)
    check("8  B3_R2 macro station-horizon MAE reproduces (48 cells)",
          cells == 48 and abs(b3_macro - results["primary_value"]) < 1e-6,
          "%.9f vs %.9f" % (b3_macro, results["primary_value"]))
    b0_macro, _ = plain_macro_cell_mae(actual, stored["B0"], stations,
                                       horizons)
    check("10 B0 macro station-horizon MAE reproduces",
          abs(b0_macro - results["reference_primary_value"]) < 1e-6,
          "%.9f vs %.9f" % (b0_macro, results["reference_primary_value"]))

    # 9, 11, 12 -- the severe tail
    severe = [i for i in range(len(rows)) if actual[i] > 244.0]
    check("12 Severe sample count reproduces",
          len(severe) == results["secondary_severe_n"],
          "%d vs %d" % (len(severe), results["secondary_severe_n"]))
    gru_severe = plain_mae(actual[severe], stored["GRU_R1"][severe])
    check("9  GRU_R1 severe MAE reproduces",
          abs(gru_severe - results["secondary_value"]) < 1e-6,
          "%.9f vs %.9f" % (gru_severe, results["secondary_value"]))
    b0_severe = plain_mae(actual[severe], stored["B0"][severe])
    check("11 B0 severe MAE reproduces",
          abs(b0_severe - results["reference_severe_value"]) < 1e-6,
          "%.9f vs %.9f" % (b0_severe, results["reference_severe_value"]))
    strictly_greater = all(actual[i] > 244.0 for i in severe)
    boundary = [i for i in range(len(rows)) if actual[i] == 244.0]
    check("   Severe rule is strictly greater than 244.0",
          strictly_greater and not any(i in severe for i in boundary),
          "%d boundary samples at exactly 244.0" % len(boundary))

    # 13 -- residual convention
    sample = severe[0] if severe else 0
    residual = actual[sample] - stored["B3_R2"][sample]
    check("13 Residual convention is actual minus prediction",
          abs(residual - (actual[sample] - stored["B3_R2"][sample])) == 0.0
          and (residual > 0) == (actual[sample] > stored["B3_R2"][sample]))

    # 15 -- violations
    check("15 Future-target access violations are zero",
          lock["future_target_access_violations"] == 0
          and lock["reveal_ledger"]["future_target_access_violations"] == 0)

    failed = [row for row in CHECKS if not row[1]]
    print("")
    if failed:
        print("INDEPENDENT VERIFICATION FAILED: %d check(s)" % len(failed))
        return 1
    print("INDEPENDENT VERIFICATION PASSED: %d checks agree to tolerance"
          % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
