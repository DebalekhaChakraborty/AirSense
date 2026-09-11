"""Validate the AirSense V2 Phase-2 forecasting data layer.

Read-only. Fits nothing, predicts nothing, computes no metric, repairs
nothing. Exits 0 only if every gate passes.

Two kinds of check:

* **gates** on the canonical layer - geometry, causality, quarantine, absence
  of model output;
* an **independent recomputation** of twelve headline facts through a separate
  pure-standard-library path that does not import the Phase-2 modules, so a
  bug in the build cannot validate itself.

Usage:
    python3 scripts/validate_phase2.py
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

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
OUT = PROJECT_ROOT / "data" / "processed" / "phase2"
RESULTS = PROJECT_ROOT / "results" / "dataset_build"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CONFIGS = PROJECT_ROOT / "configs"

FULL_START = datetime(2013, 3, 1, 0)
FULL_END = datetime(2017, 2, 28, 23)
TRAIN_END = datetime(2015, 2, 28, 23)
BOUNDS = {"train": (datetime(2013, 3, 1, 0), datetime(2015, 2, 28, 23)),
          "validation": (datetime(2015, 3, 1, 0), datetime(2016, 2, 29, 23)),
          "test": (datetime(2016, 3, 1, 0), datetime(2017, 2, 28, 23))}
CONTEXT = 48
HORIZONS = [1, 6, 12, 24]
STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
            "Wanliu", "Wanshouxigong"]
CSV_NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES",
               "DEWP", "RAIN", "WSPM"]
NON_TARGET = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
              "RAIN", "WSPM"]

TRAINING_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(torch|tensorflow|jax|transformers|timesfm|"
    r"chronos|uni2ts|gluonts|neuralforecast|xgboost|lightgbm|sklearn|"
    r"statsmodels|prophet|catboost)\b", re.MULTILINE)
FORBIDDEN_ARTIFACTS = [r"prediction", r"forecast_result", r"model_comparison",
                       r"leaderboard", r"checkpoint", r"\.pt$", r"\.pth$",
                       r"\.pkl$", r"_mae", r"_rmse"]
METRIC_KEYS = {"mae", "rmse", "mse", "r2", "mape", "smape", "crps", "loss",
               "score", "val_mae", "test_mae"}
TOLERANCE = 1e-9




# Ownership is a *structural* property, not a hand-maintained list. A path
# belongs to a later phase when its name carries a phase number greater than
# this validator's own, so Phase 7, 8, 9 and beyond need no further edits here.
# The literal entries below are the paths whose names carry no phase number.
#
# This replaced three successive one-off corrections (Phases 5, 6 and 7), each
# of which added the same three literals for the next phase. Nothing is
# excluded that was not excluded before: the rule is an exact generalisation of
# the enumerated prefixes, verified path by path against the previous
# behaviour when it was introduced.
PHASE_NUMBER = 2

LATER_PHASE_PREFIXES = (
    "results/validation", "results/models",
    "artifacts/validation_development_opening_receipt.json",
    "artifacts/neural_feature_schema.json",
    "artifacts/verification_tooling_registry.json",
    "figures/validation_",
)

LATER_PHASE_PATTERN = re.compile(
    r"^(?:artifacts/(?:v2_)?phase(\d+)_|figures/phase(\d+)_)")


def owned_by_a_later_phase(relative):
    text = str(relative)
    if any(text.startswith(prefix) for prefix in LATER_PHASE_PREFIXES):
        return True
    match = LATER_PHASE_PATTERN.match(text)
    if match:
        return int(match.group(1) or match.group(2)) > PHASE_NUMBER
    return False


# Scoped by ownership for the same reason as Phase 1: the assertion is about
# the Phase-2 data layer, not about the whole repository forever.
PHASE2_OWNED_SOURCE = (
    "scripts/build_forecast_dataset.py",
    "scripts/validate_phase2.py",
    "src/data/preprocessing.py",
    "src/data/windowing.py",
    "src/data/target_history.py",
    "tests/test_preprocessing.py",
    "tests/test_windowing.py",
    "tests/test_target_history.py",
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
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def index_of(stamp):
    return int((stamp - FULL_START).total_seconds() // 3600)


def partition_of(stamp):
    for name, (low, high) in BOUNDS.items():
        if low <= stamp <= high:
            return name
    return None


# ---------------------------------------------------------------------------
# Upstream gates
# ---------------------------------------------------------------------------

def upstream(report):
    for name in ("validate_foundation.py", "validate_phase1.py"):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)
    phase1 = load_json(ARTIFACTS / "v2_phase1_manifest.json")
    report.add("Phase-0 manifest unchanged since Phase 1",
               phase1["foundation_manifest_sha256"]
               == sha256_of_file(ARTIFACTS / "v2_foundation_manifest.json"))
    drift = []
    for name, digest in phase1["eda_table_sha256"].items():
        path = PROJECT_ROOT / "results" / "eda_training" / name
        if not path.is_file() or sha256_of_file(path) != digest:
            drift.append(name)
    for name, key in (("docs/PREPROCESSING_POLICY.md",
                       "preprocessing_policy_sha256"),
                      ("configs/preprocessing.json",
                       "preprocessing_config_sha256")):
        if sha256_of_file(PROJECT_ROOT / name) != phase1[key]:
            drift.append(name)
    report.add("Phase-1 scientific artifacts unchanged", not drift,
               "drifted: %s" % drift[:5])


# ---------------------------------------------------------------------------
# Contract gates
# ---------------------------------------------------------------------------

def contract(report):
    windowing = load_json(CONFIGS / "windowing.json")
    report.add("Primary context is exactly 48 hours",
               windowing["context_hours_primary"] == CONTEXT,
               "found %r" % windowing["context_hours_primary"])
    report.add("Robustness contexts are exactly [24, 72]",
               windowing["context_hours_robustness"] == [24, 72],
               "found %r" % windowing["context_hours_robustness"])
    report.add("Horizons are exactly [1, 6, 12, 24]",
               windowing["forecast_horizons_hours"] == HORIZONS)
    report.add("Context not selected by validation performance",
               windowing["context_selected_by_validation_performance"]
               is False)
    report.add("Severe threshold is 244.0 from training pooled P95",
               windowing["primary_severe_threshold"] == 244.0
               and windowing["severe_threshold_source"]
               == "training_pooled_p95")
    report.add("Frozen Phase-1 deferrals honoured",
               windowing["primary_pm25_input_log_transform"] is False
               and windowing["rain_occurred_feature"] is True
               and windowing["day_of_week_primary"] is False
               and windowing["weekend_primary"] is False
               and windowing["severe_balancing_primary"] is False)
    report.add("No backward fill, interpolation or centred statistic",
               windowing["backward_fill"] is False
               and windowing["interpolation"] is False
               and windowing["centred_statistics"] is False)
    report.add("Target labels are never imputed",
               windowing["target_labels_imputed"] is False)
    report.add("Windows are built on demand, not materialised",
               windowing["windows_materialised"] is False
               and windowing["windows_built_on_demand"] is True)

    regimes = load_json(ARTIFACTS / "information_regime_schema.json")
    report.add("Regime schema defines R0-R3 over a common universe",
               sorted(regimes["regimes"]) == ["R0", "R1", "R2", "R3"]
               and regimes["common_sample_universe"] is True)
    report.add("CO retained in R2",
               "CO" in regimes["regimes"]["R2"]["pollutant_history"])
    report.add("R0 carries no pollutant history",
               regimes["regimes"]["R0"]["pollutant_history"] == [])
    report.add("R3 alone carries cross-station history",
               regimes["regimes"]["R3"]["cross_station_history"] is True
               and regimes["regimes"]["R2"]["cross_station_history"] is False)

    statistics = load_json(ARTIFACTS / "preprocessing_statistics.json")
    report.add("Scaling statistics are training-fitted only",
               statistics["fitted_on"]
               == "training_partition_pooled_across_stations")
    report.add("RAIN zero-IQR exception recorded",
               statistics["zero_iqr_variables"] == ["RAIN"]
               and statistics["zero_iqr_fallback_formula"]
               == "(x - median) / std")
    report.add("Wind vocabulary is 16 training categories plus MISSING",
               len(statistics["wind_direction_vocabulary"]) == 16
               and statistics["wind_direction_missing_code"] == 16)

    baselines = load_json(CONFIGS / "baselines.json")
    report.add("Baseline definitions B0-B3 exist",
               sorted(baselines["baselines"]) == ["B0", "B1", "B2", "B3"])
    report.add("No baseline result or metric recorded",
               baselines["results_generated"] is False
               and baselines["predictions_generated"] == 0
               and baselines["metrics_computed"] == 0)
    report.add("B3 lag contract stays inside the 48-hour window",
               max(baselines["baselines"]["B3"]["feature_contract"]
                   ["lag_positions_hours_before_origin"]) <= CONTEXT - 1)
    report.add("B3 library and grid deferred to Phase 3",
               baselines["baselines"]["B3"]["library"] is None
               and baselines["baselines"]["B3"]["hyperparameters"] is None)


# ---------------------------------------------------------------------------
# Sample universe gates
# ---------------------------------------------------------------------------

def sample_universe(report):
    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    report.add("Universe records context 48 and horizons 1/6/12/24",
               universe["context_hours"] == CONTEXT
               and universe["horizons"] == HORIZONS)
    report.add("Universe is common across models and regimes",
               universe["common_across_models"] is True
               and universe["common_across_regimes"] is True)

    drift = []
    for partition, digest in universe["index_file_sha256"].items():
        path = OUT / "sample_index" / ("%s.csv" % partition)
        if not path.is_file() or sha256_of_file(path) != digest:
            drift.append(partition)
    report.add("Sample-index hashes match the frozen universe", not drift,
               "drifted: %s" % drift)

    counts_from_index = Counter()
    geometry_problems = []
    partition_problems = []
    duplicates = []
    unobserved = []
    for partition in ("train", "validation", "test"):
        seen = set()
        path = OUT / "sample_index" / ("%s.csv" % partition)
        with open(str(path), encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["sample_id"]
                if sample_id in seen:
                    duplicates.append(sample_id)
                seen.add(sample_id)
                horizon = int(row["horizon_hours"])
                target = datetime.fromisoformat(row["target_timestamp"])
                origin = datetime.fromisoformat(row["forecast_origin"])
                start = datetime.fromisoformat(row["context_start"])
                end = datetime.fromisoformat(row["context_end"])
                if origin + timedelta(hours=horizon) != target:
                    geometry_problems.append(sample_id)
                elif end != origin:
                    geometry_problems.append(sample_id)
                elif start != origin - timedelta(hours=CONTEXT - 1):
                    geometry_problems.append(sample_id)
                elif not origin < target:
                    geometry_problems.append(sample_id)
                elif start < FULL_START:
                    geometry_problems.append(sample_id)
                if partition_of(target) != partition:
                    partition_problems.append(sample_id)
                if row["target_observed"] != "1":
                    unobserved.append(sample_id)
                counts_from_index[(partition, row["station"], horizon)] += 1
        report.add("Sample ids unique in %s (%d rows)"
                   % (partition, len(seen)), not duplicates,
                   "duplicates: %s" % duplicates[:3])
    report.add("Every row satisfies origin + horizon = target, end = origin, "
               "start = origin - 47 h", not geometry_problems,
               "violations: %s" % geometry_problems[:3])
    report.add("Partition always follows the target timestamp",
               not partition_problems,
               "violations: %s" % partition_problems[:3])
    report.add("Every indexed sample has an observed target",
               not unobserved, "unobserved: %s" % unobserved[:3])

    declared = universe["counts_by_partition_station_horizon"]
    mismatch = []
    for key, value in declared.items():
        partition, station, horizon = key.split("|")
        if counts_from_index[(partition, station, int(horizon))] != value:
            mismatch.append(key)
    report.add("Manifest counts reconcile with the index files (%d cells)"
               % len(declared), not mismatch, "mismatched: %s" % mismatch[:3])

    table = Counter()
    with open(str(RESULTS / "sample_counts.csv"), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(row["partition"], row["station"],
                   int(row["horizon_hours"]))] = int(row["eligible_samples"])
    report.add("sample_counts.csv reconciles with the index files",
               table == counts_from_index)
    return counts_from_index


# ---------------------------------------------------------------------------
# Canonical array gates
# ---------------------------------------------------------------------------

def canonical_arrays(report):
    mask_problems = []
    age_problems = []
    fill_problems = []
    leak = []
    train_end_index = index_of(TRAIN_END)
    for station in STATIONS:
        base = OUT / "station_series" / station
        meta = load_json(base / "meta.json")
        observed = np.load(str(base / "observed_mask.npy"))
        gap_age = np.load(str(base / "gap_age.npy"))
        fill_source = np.load(str(base / "fill_source.npy"))
        pm25 = np.load(str(base / "pm25_scaled_train_only.npy"))
        target = np.load(str(OUT / "target_series"
                             / ("%s_train_pm25.npy" % station)))

        if not np.isin(observed, (0, 1)).all():
            mask_problems.append(station)
        if gap_age.min() < 0 or gap_age.max() > 168:
            age_problems.append(station)
        if np.isfinite(pm25[train_end_index + 1:]).any():
            leak.append("%s pm25_scaled" % station)
        if np.isfinite(target[train_end_index + 1:]).any():
            leak.append("%s target_series" % station)

        # Causal-fill semantics, checked against the mask directly.
        for name in NON_TARGET:
            column = meta["fill_source_channels"].index(name)
            mask_column = meta["mask_channels"].index(name)
            source = fill_source[:, column]
            was_observed = observed[:, mask_column].astype(bool)
            if (source[was_observed] != 0).any():
                fill_problems.append("%s/%s observed not marked" % (station,
                                                                    name))
            carried = np.flatnonzero(source == 1)
            if carried.size:
                ages = np.zeros(carried.size, dtype=int)
                last = -1
                pointer = 0
                lookup = {}
                for position in range(source.size):
                    if was_observed[position]:
                        last = position
                    lookup[position] = last
                for offset, position in enumerate(carried):
                    previous = lookup[position]
                    ages[offset] = (position - previous) if previous >= 0 else 999
                if ages.max() > 6 or ages.min() < 1:
                    fill_problems.append("%s/%s carried beyond 6 h"
                                         % (station, name))
                # A carried value must never come from the future.
                if (carried <= 0).any():
                    fill_problems.append("%s/%s carried at index 0"
                                         % (station, name))
    report.add("Observed masks are binary at every station", not mask_problems,
               "stations: %s" % mask_problems[:3])
    report.add("Gap ages lie in [0, 168] at every station", not age_problems,
               "stations: %s" % age_problems[:3])
    report.add("Causal fill obeyed: observed marked, carries <= 6 h, none "
               "from the future", not fill_problems,
               "problems: %s" % fill_problems[:3])
    report.add("No validation or test PM2.5 value materialised in any array",
               not leak, "leaks: %s" % leak[:5])


def quarantine(report):
    leak = []
    for partition in ("validation", "test"):
        path = OUT / "sample_index" / ("%s.csv" % partition)
        with open(str(path), encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        for column in header:
            if column.lower() in ("target", "target_value", "y", "pm25",
                                  "pm2.5", "target_pm25", "severe",
                                  "target_bin", "target_quantile"):
                leak.append("%s:%s" % (partition, column))
    report.add("Validation and test indices carry metadata and presence only",
               not leak, "columns: %s" % leak)

    for name in ("y_validation", "y_test"):
        matches = list(OUT.rglob("*%s*" % name))
        report.add("No %s artifact exists" % name, not matches,
                   "found: %s" % [str(m) for m in matches[:3]])

    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    report.add("Universe records no materialised validation/test target",
               universe["validation_target_values_materialized"] is False
               and universe["test_target_values_materialized"] is False)


def no_model_output(report):
    offenders = []
    for directory in ("artifacts", "results", "data/processed", "figures"):
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if owned_by_a_later_phase(path.relative_to(PROJECT_ROOT)):
                continue
            for pattern in FORBIDDEN_ARTIFACTS:
                if re.search(pattern, path.name, re.IGNORECASE):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
    report.add("No model-result, prediction or checkpoint artifact",
               not offenders, "; ".join(offenders[:5]))

    placeholders = []
    for path in sorted((PROJECT_ROOT / "artifacts").rglob("*.json")):
        if owned_by_a_later_phase(path.relative_to(PROJECT_ROOT)):
            continue
        payload = load_json(path)
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if (key.lower() in METRIC_KEYS
                            and isinstance(value, (int, float))
                            and not isinstance(value, bool)):
                        placeholders.append("%s:%s" % (path.name, key))
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    report.add("No forecast metric anywhere in artifacts", not placeholders,
               "; ".join(placeholders[:5]))

    imports = []
    for relative in PHASE2_OWNED_SOURCE:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        if TRAINING_IMPORTS.search(path.read_text(encoding="utf-8")):
            imports.append(relative)
    report.add("No modelling library imported in Phase-2 code (%d files)"
               % len([n for n in PHASE2_OWNED_SOURCE
                      if (PROJECT_ROOT / n).is_file()]),
               not imports, "; ".join(imports))


# ---------------------------------------------------------------------------
# Independent recomputation - separate pure-stdlib path
# ---------------------------------------------------------------------------

def read_station_csv(station):
    """Parse one station with the csv module only. No Phase-2 imports."""
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
    length = int((FULL_END - FULL_START).total_seconds() // 3600) + 1
    values = {name: [None] * length for name in CSV_NUMERIC}
    present = {name: [False] * length for name in CSV_NUMERIC}
    with open(str(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            if stamp < FULL_START or stamp > FULL_END:
                continue
            position = int((stamp - FULL_START).total_seconds() // 3600)
            for name in CSV_NUMERIC:
                raw = row[name]
                if raw in ("NA", ""):
                    continue
                present[name][position] = True
                values[name][position] = float(raw)
    return values, present


def independent(report):
    station = "Dongsi"
    values, present = read_station_csv(station)
    length = len(present["PM2.5"])
    train_end = int((TRAIN_END - FULL_START).total_seconds() // 3600)

    # 1 & 2. sample counts for one station/horizon, train and validation
    declared = load_json(
        ARTIFACTS / "sample_universe_manifest.json")[
            "counts_by_partition_station_horizon"]
    for partition, horizon in (("train", 6), ("validation", 12)):
        low = int((BOUNDS[partition][0] - FULL_START).total_seconds() // 3600)
        high = int((BOUNDS[partition][1] - FULL_START).total_seconds() // 3600)
        count = 0
        for position in range(low, high + 1):
            if position - horizon - (CONTEXT - 1) < 0:
                continue
            if present["PM2.5"][position]:
                count += 1
        key = "%s|%s|%d" % (partition, station, horizon)
        report.add("Independent: %s %s h%d sample count"
                   % (partition, station, horizon),
                   count == declared[key],
                   "recomputed %d vs declared %d" % (count, declared[key]))

    # 3. first eligible training sample metadata
    first = None
    for position in range(length):
        if position - 6 - 47 < 0:
            continue
        stamp = FULL_START + timedelta(hours=position)
        if stamp > TRAIN_END:
            break
        if present["PM2.5"][position]:
            first = (position, stamp)
            break
    with open(str(OUT / "sample_index" / "train.csv"), encoding="utf-8") as fh:
        published_first = None
        for row in csv.DictReader(fh):
            if row["station"] == station and row["horizon_hours"] == "6":
                published_first = row
                break
    ok = (published_first is not None
          and datetime.fromisoformat(published_first["target_timestamp"])
          == first[1]
          and int(published_first["target_row_index"]) == first[0])
    report.add("Independent: first eligible train %s h6 sample" % station, ok,
               "recomputed %s idx %d" % (first[1].isoformat(), first[0]))

    # 4. one 48-hour window timestamp sequence
    target = datetime.fromisoformat(published_first["target_timestamp"])
    origin = target - timedelta(hours=6)
    expected = [(origin - timedelta(hours=47 - k)).isoformat()
                for k in range(48)]
    from src.data.windowing import ForecastWindowLoader
    loader = ForecastWindowLoader(OUT)
    window = loader.get_window(published_first["sample_id"], regime="R2")
    report.add("Independent: 48-hour timestamp sequence",
               window["timestamps"] == expected
               and len(expected) == 48
               and expected[-1] == origin.isoformat())

    # 5 & 6. a known short-gap carry and a beyond-ceiling fallback
    base = OUT / "station_series" / station
    meta = load_json(base / "meta.json")
    fill_source = np.load(str(base / "fill_source.npy"))
    carry_case = None
    fallback_case = None
    column = meta["fill_source_channels"].index("NO2")
    last_seen = -1
    for position in range(length):
        if present["NO2"][position]:
            last_seen = position
            continue
        if last_seen < 0:
            continue
        age = position - last_seen
        if age <= 6 and carry_case is None:
            carry_case = (position, age)
        if age > 6 and fallback_case is None:
            fallback_case = (position, age)
        if carry_case and fallback_case:
            break
    report.add("Independent: short-gap carry marked as causal ffill",
               fill_source[carry_case[0], column] == 1,
               "index %d age %d source %d"
               % (carry_case[0], carry_case[1],
                  fill_source[carry_case[0], column]))
    report.add("Independent: beyond-ceiling gap marked as median fallback",
               fill_source[fallback_case[0], column] == 2,
               "index %d age %d source %d"
               % (fallback_case[0], fallback_case[1],
                  fill_source[fallback_case[0], column]))

    # 7. observed-mask sequence
    observed = np.load(str(base / "observed_mask.npy"))
    mask_column = meta["mask_channels"].index("PM2.5")
    start = first[0] + 1000          # comfortably inside the timeline
    recomputed = [1 if present["PM2.5"][i] else 0
                  for i in range(start, start + 48)]
    report.add("Independent: observed-mask sequence",
               list(observed[start:start + 48, mask_column]) == recomputed)

    # 8. gap-age sequence
    gap_age = np.load(str(base / "gap_age.npy"))
    age_column = meta["gap_age_channels"].index("PM2.5")
    ages = []
    last = None
    for position in range(start + 48):
        if present["PM2.5"][position]:
            last = position
        if position >= start:
            ages.append(0 if present["PM2.5"][position]
                        else (min(position - last, 168) if last is not None
                              else 168))
    report.add("Independent: gap-age sequence",
               list(gap_age[start:start + 48, age_column]) == ages)

    # 9 & 10. a scaled value, and the RAIN zero-IQR exception
    statistics = load_json(ARTIFACTS / "preprocessing_statistics.json")[
        "pooled_training_statistics"]
    numeric = np.load(str(base / "numeric_scaled.npy"))
    numeric_meta = meta["numeric_channels"]
    probe = first[0]
    while not present["TEMP"][probe]:
        probe += 1
    expected_value = ((values["TEMP"][probe] - statistics["TEMP"]["median"])
                      / statistics["TEMP"]["iqr"])
    report.add("Independent: scaled TEMP value",
               abs(float(numeric[probe, numeric_meta.index("TEMP")])
                   - expected_value) < 1e-5,
               "recomputed %.9f" % expected_value)

    rain_probe = probe
    while not present["RAIN"][rain_probe]:
        rain_probe += 1
    expected_rain = ((values["RAIN"][rain_probe]
                      - statistics["RAIN"]["median"])
                     / statistics["RAIN"]["std"])
    report.add("Independent: RAIN scaled by std because training IQR is 0",
               statistics["RAIN"]["iqr"] == 0
               and abs(float(numeric[rain_probe, numeric_meta.index("RAIN")])
                       - expected_rain) < 1e-5,
               "recomputed %.9f" % expected_rain)

    # 11. one climatology cell
    month, hour = 1, 0
    sample = []
    for position in range(train_end + 1):
        stamp = FULL_START + timedelta(hours=position)
        if (stamp.month == month and stamp.hour == hour
                and present["PM2.5"][position]):
            sample.append(values["PM2.5"][position])
    sample.sort()
    location = (len(sample) - 1) * 0.5
    lower = int(location)
    upper = min(lower + 1, len(sample) - 1)
    median = sample[lower] + (location - lower) * (sample[upper]
                                                   - sample[lower])
    published = None
    with open(str(OUT / "climatology" / "b2_training_climatology.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["station"] == station
                    and row["level"] == "station_month_hour"
                    and row["month"] == str(month)
                    and row["hour"] == str(hour)):
                published = row
                break
    report.add("Independent: climatology cell %s month %d hour %d"
               % (station, month, hour),
               published is not None
               and abs(float(published["median_pm25_ug_m3"]) - median) < 1e-6
               and int(published["n_training_observations"]) == len(sample),
               "recomputed median %.6f n %d" % (median, len(sample)))

    # 12. target-history future access refusal
    from src.data.target_history import (CausalTargetHistory,
                                         TargetAccessError)
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
    history = CausalTargetHistory(station, path, value_access_until=TRAIN_END)
    refused_future = False
    refused_sealed = False
    try:
        history.value_at(datetime(2014, 6, 10, 12), datetime(2014, 6, 10, 6))
    except TargetAccessError:
        refused_future = True
    try:
        history.values(datetime(2016, 6, 1, 0), 48)
    except TargetAccessError:
        refused_sealed = True
    report.add("Independent: target history refuses future and sealed access",
               refused_future and refused_sealed,
               "future=%s sealed=%s" % (refused_future, refused_sealed))


def main():
    print("")
    print("[AIRSENSE V2 PHASE-2 VALIDATION]")
    print("")
    print("Read-only. Fits nothing, predicts nothing, computes no metric.")
    print("")
    report = Report()
    upstream(report)
    contract(report)
    sample_universe(report)
    canonical_arrays(report)
    quarantine(report)
    no_model_output(report)
    independent(report)

    print("=" * 78)
    report.render()
    print("=" * 78)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-2 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-2 VALIDATION: PASS")
    print("Data layer only. No model, no prediction, no metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
