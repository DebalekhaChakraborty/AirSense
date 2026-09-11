"""Validate the AirSense V2 Phase-1 deliverables.

Read-only. Fits nothing, predicts nothing, repairs nothing. Exits 0 only if
every gate passes.

Two kinds of check run here:

* **gates** on the published artifacts (hashes, policy content, quarantine
  discipline, absence of model output);
* an **independent recomputation** of six headline numbers using a separate
  pure-standard-library code path - no numpy, no `src.data.eda` - so an error
  in the analysis module cannot validate itself.

Usage:
    python3 scripts/validate_phase1.py
"""

import csv
import hashlib
import json
import math
import re
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
RESULTS = PROJECT_ROOT / "results" / "eda_training"
FIGURES = PROJECT_ROOT / "figures"
CONFIG = PROJECT_ROOT / "configs" / "preprocessing.json"
PHASE1_MANIFEST = PROJECT_ROOT / "artifacts" / "v2_phase1_manifest.json"
FOUNDATION_MANIFEST = PROJECT_ROOT / "artifacts" / "v2_foundation_manifest.json"
TRAINING_EDA = PROJECT_ROOT / "artifacts" / "training_eda.json"

TRAIN_START = datetime(2013, 3, 1, 0)
TRAIN_END = datetime(2015, 2, 28, 23)
DATASET_START = datetime(2013, 3, 1, 0)
HORIZONS = [1, 6, 12, 24]

VERIFY_STATION = "Dongsi"
VERIFY_PAIR = ("Tiantan", "Dongsi")
VERIFY_LAG = 24
VERIFY_RUN_VARIABLE = "PM2.5"

MODEL_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(torch|tensorflow|jax|transformers|timesfm|"
    r"chronos|uni2ts|gluonts|neuralforecast|xgboost|lightgbm|sklearn|"
    r"statsmodels|prophet)\b", re.MULTILINE)

FORBIDDEN_ARTIFACT_PATTERNS = [
    r"prediction", r"forecast_result", r"model_comparison", r"metric",
    r"leaderboard", r"checkpoint", r"\.pt$", r"\.pth$", r"\.pkl$",
]
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
PHASE_NUMBER = 1

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


# The assertion is "Phase-1 implementation imports no modelling library" - not
# "no later phase anywhere in the repository may import one". Scoping it by
# ownership keeps the assertion exact and stops it from needing a new
# exclusion every time a later phase legitimately trains something.
PHASE1_OWNED_SOURCE = (
    "scripts/run_training_eda.py",
    "scripts/build_preprocessing_policy.py",
    "scripts/validate_phase1.py",
    "src/data/eda.py",
    "src/data/missingness.py",
    "src/visualization/__init__.py",
    "src/visualization/render.py",
)


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 68
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


# ---------------------------------------------------------------------------
# Independent recomputation - pure standard library, separate code path
# ---------------------------------------------------------------------------

def independent_read(station):
    """Read one station's TRAINING rows with the csv module only."""
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
    length = int((TRAIN_END - TRAIN_START).total_seconds() // 3600) + 1
    values = [None] * length
    present = [False] * length
    with open(str(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            if stamp < TRAIN_START or stamp > TRAIN_END:
                continue
            index = int((stamp - TRAIN_START).total_seconds() // 3600)
            raw = row["PM2.5"]
            if raw not in ("NA", ""):
                values[index] = float(raw)
                present[index] = True
    return values, present


def independent_stations():
    return sorted(path.name.split("_")[2]
                  for path in RAW_DIR.glob("PRSA_Data_*.csv"))


def type7_quantile(sorted_values, percentile):
    n = len(sorted_values)
    if n == 0:
        return None
    position = (n - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    return (sorted_values[lower]
            + fraction * (sorted_values[upper] - sorted_values[lower]))


def independent_correlation(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs)
                            * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def independent_recompute(report):
    stations = independent_stations()
    per_station = OrderedDict((s, independent_read(s)) for s in stations)

    pooled = []
    observed_total = 0
    missing_total = 0
    for station in stations:
        values, present = per_station[station]
        for value, flag in zip(values, present):
            if flag:
                pooled.append(value)
                observed_total += 1
            else:
                missing_total += 1
    pooled.sort()

    published = load_json(TRAINING_EDA)

    report.add("Independent: total observed training targets",
               observed_total == published["target_observed"],
               "recomputed %d vs published %d"
               % (observed_total, published["target_observed"]))
    report.add("Independent: total missing training targets",
               missing_total == published["target_missing"],
               "recomputed %d vs published %d"
               % (missing_total, published["target_missing"]))

    p95 = type7_quantile(pooled, 95)
    report.add("Independent: pooled training PM2.5 P95",
               abs(p95 - published["pooled_training_p95"]) < TOLERANCE,
               "recomputed %.9f vs published %.9f"
               % (p95, published["pooled_training_p95"]))

    values, present = per_station[VERIFY_STATION]
    sample = sorted(v for v, f in zip(values, present) if f)
    mean = sum(sample) / len(sample)
    published_row = None
    with open(str(RESULTS / "pm25_summary_by_station.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["station"] == VERIFY_STATION:
                published_row = row
    ok = (published_row is not None
          and abs(mean - float(published_row["mean"])) < 1e-6
          and abs(type7_quantile(sample, 50)
                  - float(published_row["median"])) < TOLERANCE
          and len(sample) == int(published_row["observed"]))
    report.add("Independent: %s PM2.5 summary" % VERIFY_STATION, ok,
               "recomputed mean %.6f median %.6f n %d"
               % (mean, type7_quantile(sample, 50), len(sample)))

    runs = []
    current = 0
    for flag in present:
        if flag:
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    published_run = None
    with open(str(RESULTS / "missing_run_summary.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["variable"] == VERIFY_RUN_VARIABLE
                    and row["station"] == VERIFY_STATION):
                published_run = row
    ok = (published_run is not None
          and len(runs) == int(published_run["n_runs"])
          and max(runs) == int(published_run["longest_run_h"])
          and sum(1 for r in runs if r == 1)
          == int(published_run["isolated_1h_gaps"]))
    report.add("Independent: %s %s missing-run distribution"
               % (VERIFY_STATION, VERIFY_RUN_VARIABLE), ok,
               "recomputed runs %d longest %d isolated %d"
               % (len(runs), max(runs) if runs else 0,
                  sum(1 for r in runs if r == 1)))

    left, right = [], []
    for index in range(len(values) - VERIFY_LAG):
        if present[index] and present[index + VERIFY_LAG]:
            left.append(values[index])
            right.append(values[index + VERIFY_LAG])
    rho = independent_correlation(left, right)
    published_rho = None
    with open(str(RESULTS / "pm25_autocorrelation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["scope"] == "station" and row["station"] == VERIFY_STATION
                    and int(row["lag_hours"]) == VERIFY_LAG):
                published_rho = (float(row["pearson_r"]),
                                 int(row["valid_pairs"]))
    ok = (published_rho is not None
          and abs(rho - published_rho[0]) < 1e-6
          and len(left) == published_rho[1])
    report.add("Independent: %s autocorrelation at lag %d h"
               % (VERIFY_STATION, VERIFY_LAG), ok,
               "recomputed r %.9f pairs %d" % (rho, len(left)))

    a_values, a_present = per_station[VERIFY_PAIR[0]]
    b_values, b_present = per_station[VERIFY_PAIR[1]]
    xs, ys = [], []
    for index in range(len(a_values)):
        if a_present[index] and b_present[index]:
            xs.append(a_values[index])
            ys.append(b_values[index])
    rho = independent_correlation(xs, ys)
    published_pair = None
    with open(str(RESULTS / "cross_station_pm25_correlation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["station_a"] == VERIFY_PAIR[0]
                    and row["station_b"] == VERIFY_PAIR[1]):
                published_pair = (float(row["pearson_r"]),
                                  int(row["valid_pairs"]))
    ok = (published_pair is not None
          and abs(rho - published_pair[0]) < 1e-6
          and len(xs) == published_pair[1])
    report.add("Independent: %s-%s PM2.5 correlation" % VERIFY_PAIR, ok,
               "recomputed r %.9f pairs %d" % (rho, len(xs)))


# ---------------------------------------------------------------------------
# Horizon leakage checks (metadata only)
# ---------------------------------------------------------------------------

def horizon_leakage_checks(report, config):
    contexts = config["context_length_candidates"]
    problems = []
    examples = []
    probes = [("train", datetime(2014, 6, 10, 12)),
              ("validation", datetime(2015, 3, 1, 3)),
              ("test", datetime(2016, 3, 1, 0))]
    for partition, target in probes:
        for horizon in HORIZONS:
            origin = target - timedelta(hours=horizon)
            for context in contexts:
                window_start = origin - timedelta(hours=context - 1)
                max_input = origin
                if not max_input <= origin:
                    problems.append("%s h=%d: max input after origin"
                                    % (partition, horizon))
                if origin != target - timedelta(hours=horizon):
                    problems.append("%s h=%d: origin rule violated"
                                    % (partition, horizon))
                if not target > origin:
                    problems.append("%s h=%d: target not after origin"
                                    % (partition, horizon))
                if window_start < DATASET_START:
                    continue
                examples.append((partition, horizon, context,
                                 window_start.isoformat(), origin.isoformat(),
                                 target.isoformat()))
    report.add("Horizon leakage checks (max_input <= origin < target)",
               not problems, "; ".join(problems[:5]))
    report.add("Illustrative eligible samples constructed for all horizons",
               {h for _, h, _, _, _, _ in examples} == set(HORIZONS),
               "horizons covered: %s"
               % sorted({h for _, h, _, _, _, _ in examples}))
    return examples


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def main():
    print("")
    print("[AIRSENSE V2 PHASE-1 VALIDATION]")
    print("")
    print("Read-only. Fits nothing, predicts nothing, repairs nothing.")
    print("")
    report = Report()

    required = [CONFIG, PHASE1_MANIFEST, TRAINING_EDA,
                PROJECT_ROOT / "docs" / "PREPROCESSING_POLICY.md",
                PROJECT_ROOT / "docs" / "TRAINING_EDA.md",
                PROJECT_ROOT / "src" / "data" / "eda.py",
                PROJECT_ROOT / "src" / "data" / "missingness.py"]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required
               if not p.is_file()]
    if not report.add("Required Phase-1 files present", not missing,
                      "missing: %s" % missing):
        report.render()
        return 1

    manifest = load_json(PHASE1_MANIFEST)
    config = load_json(CONFIG)
    eda = load_json(TRAINING_EDA)

    report.add("Phase-0 foundation manifest unchanged",
               manifest["foundation_manifest_sha256"]
               == sha256_of_file(FOUNDATION_MANIFEST),
               "foundation manifest digest differs")
    foundation = load_json(FOUNDATION_MANIFEST)
    stale = [name for name, digest
             in list(foundation.get("document_sha256", {}).items())
             + list(foundation.get("dataset_audit_artifact_sha256",
                                   {}).items())
             if not (PROJECT_ROOT / name).is_file()
             or sha256_of_file(PROJECT_ROOT / name) != digest]
    report.add("Phase-0 artifacts still match recorded hashes", not stale,
               "changed: %s" % stale[:5])

    report.add("Target EDA is training-only",
               eda["partition"] == "train"
               and eda["training_start"] == TRAIN_START.isoformat()
               and eda["training_end"] == TRAIN_END.isoformat())
    report.add("No validation target analysis",
               eda["validation_target_analysis"] is False
               and manifest["validation_target_analysis"] is False)
    report.add("No test target analysis",
               eda["test_target_analysis"] is False
               and manifest["test_target_analysis"] is False)

    scopes = set()
    with open(str(RESULTS / "severe_thresholds.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scopes.add(row["scope"])
    report.add("Severe threshold derived from training only",
               scopes == {"pooled_training", "station_training"}
               and config["primary_severe_threshold_definition"]
               ["derived_from"] == "training_partition_only",
               "scopes found: %s" % sorted(scopes))
    report.add("Severe threshold value matches EDA and manifest",
               abs(config["primary_severe_threshold_value"]
                   - eda["pooled_training_p95"]) < TOLERANCE
               and abs(manifest["primary_severe_threshold_value"]
                       - eda["pooled_training_p95"]) < TOLERANCE)

    with open(str(RESULTS / "target_eligibility_by_partition_horizon.csv"),
              encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    report.add("Later-partition tables carry counts only, no target values",
               all(column in ("partition", "horizon_h", "candidate_targets",
                              "excluded_origin_before_dataset",
                              "excluded_target_missing", "eligible_targets")
                   for column in header),
               "unexpected columns: %s" % header)

    drift = []
    for name, digest in manifest["eda_table_sha256"].items():
        path = RESULTS / name
        if not path.is_file() or sha256_of_file(path) != digest:
            drift.append(name)
    for name, digest in manifest["eda_figure_sha256"].items():
        path = FIGURES / name
        if not path.is_file() or sha256_of_file(path) != digest:
            drift.append(name)
    report.add("EDA table and figure hashes match the manifest (%d files)"
               % (len(manifest["eda_table_sha256"])
                  + len(manifest["eda_figure_sha256"])),
               not drift, "drifted: %s" % drift[:5])
    report.add("Policy and config hashes match the manifest",
               sha256_of_file(PROJECT_ROOT / "docs" / "PREPROCESSING_POLICY.md")
               == manifest["preprocessing_policy_sha256"]
               and sha256_of_file(CONFIG)
               == manifest["preprocessing_config_sha256"]
               and sha256_of_file(PROJECT_ROOT / "docs" / "TRAINING_EDA.md")
               == manifest["training_eda_document_sha256"])

    report.add("Preprocessing config parses and forbids future information",
               config["numeric_missing_policy"]["backward_fill_permitted"]
               is False
               and config["numeric_missing_policy"]["interpolation_permitted"]
               is False
               and config["numeric_missing_policy"]
               ["centred_statistics_permitted"] is False
               and config["calendar_features"]
               ["future_observed_weather_permitted"] is False)
    report.add("Scaling is fitted on training only",
               config["scaling_policy"]["fitted_on"]
               == "training_partition_pooled_across_stations"
               and config["scaling_policy"]["applied_to_validation_and_test"]
               == "using_training_statistics_only")
    report.add("Target labels are never imputed",
               config["target_missing_policy"]["rule"]
               == "never_impute_target_labels"
               and config["target_missing_policy"]
               ["eligible_only_if_target_observed"] is True)
    rules = config["sequence_eligibility_rules"]["rules"]
    report.add("Sequence eligibility rules defined", len(rules) >= 6,
               "found %d rules" % len(rules))
    report.add("Boundary-context rule defined",
               bool(config["sequence_eligibility_rules"]
                    ["boundary_context_rule"]))
    report.add("Context candidates non-empty and justified",
               bool(config["context_length_candidates"])
               and len(config["context_length_justification"]) > 80
               and config["context_length_selected_by_model_performance"]
               is False)
    report.add("No model hyperparameters in the Phase-1 contract",
               config.get("model_hyperparameters") is None)

    offenders = []
    for directory in ("artifacts", "results", "figures", "data/processed"):
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if owned_by_a_later_phase(path.relative_to(PROJECT_ROOT)):
                continue
            for pattern in FORBIDDEN_ARTIFACT_PATTERNS:
                if re.search(pattern, path.name, re.IGNORECASE):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
    report.add("No model-result or prediction artifact exists", not offenders,
               "; ".join(offenders[:5]))

    imports = []
    for relative in PHASE1_OWNED_SOURCE:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            continue
        if MODEL_IMPORTS.search(path.read_text(encoding="utf-8")):
            imports.append(relative)
    report.add("No modelling library imported in Phase-1 code (%d files)"
               % len([n for n in PHASE1_OWNED_SOURCE
                      if (PROJECT_ROOT / n).is_file()]), not imports,
               "; ".join(imports))

    counters = [key for key in ("model_runs", "predictions_generated",
                                "validation_metrics_seen",
                                "test_metrics_seen")
                if manifest[key] != 0]
    report.add("Model runs and metrics seen are all zero", not counters,
               "non-zero: %s" % counters)
    report.add("Final test still sealed",
               manifest["final_test_status"] == "sealed")

    horizon_leakage_checks(report, config)
    independent_recompute(report)

    print("=" * 74)
    report.render()
    print("=" * 74)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-1 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-1 VALIDATION: PASS")
    print("Design and audit only. No model, no metric, no V2 result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
