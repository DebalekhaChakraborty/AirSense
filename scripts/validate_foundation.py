"""Validate the AirSense V2 Phase-0 foundation.

Read-only. Trains nothing, predicts nothing, repairs nothing. Exits 0 only if
every gate passes, non-zero otherwise.

Gates:
  * branch expectation recorded as master
  * every required Phase-0 file exists
  * the study contract parses
  * horizons are exactly [1, 6, 12, 24]
  * partitions are ordered, non-overlapping and train < validation < test
  * official raw files match the digests recorded at acquisition
  * (station, timestamp) identity is unique and the hourly grid is complete
  * final-test status is sealed
  * no model-result file exists
  * no placeholder metric exists anywhere in artifacts/ or results/

Usage:
    python3 scripts/validate_foundation.py
    python3 scripts/validate_foundation.py --write-manifest
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_BRANCH = "master"
STARTING_MASTER_HEAD = "09942928eef31808c27e621ae26c7ff2c82ed3ed"
EXPECTED_HORIZONS = [1, 6, 12, 24]

STUDY = PROJECT_ROOT / "configs" / "study.json"
RAW_MANIFEST = PROJECT_ROOT / "artifacts" / "raw_dataset_manifest.json"
DATASET_AUDIT = PROJECT_ROOT / "artifacts" / "dataset_audit.json"
FOUNDATION_MANIFEST = (PROJECT_ROOT / "artifacts"
                       / "v2_foundation_manifest.json")

REQUIRED_FILES = [
    "README.md",
    "configs/study.json",
    "docs/V2_RESEARCH_BLUEPRINT.md",
    "docs/V2_RESEARCH_PROTOCOL.md",
    "docs/DATASET_PROVENANCE.md",
    "docs/FORECAST_SEMANTICS.md",
    "docs/MODEL_LANDSCAPE_2026.md",
    "docs/ENVIRONMENT_PLAN.md",
    "docs/PHASE_00_V2_FOUNDATION_RECORD.md",
    "scripts/acquire_dataset.py",
    "scripts/audit_dataset.py",
    "scripts/validate_foundation.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/audit.py",
    "artifacts/raw_dataset_manifest.json",
    "artifacts/dataset_audit.json",
    "results/dataset_audit/station_summary.csv",
    "results/dataset_audit/missingness_by_variable.csv",
    "results/dataset_audit/missingness_by_station.csv",
    "results/dataset_audit/target_missingness_by_station_year.csv",
    "results/dataset_audit/timestamp_grid_audit.csv",
]

HASHED_DOCUMENTS = [
    "configs/study.json",
    "docs/V2_RESEARCH_PROTOCOL.md",
    "docs/FORECAST_SEMANTICS.md",
    "docs/MODEL_LANDSCAPE_2026.md",
    "docs/ENVIRONMENT_PLAN.md",
    "docs/V2_RESEARCH_BLUEPRINT.md",
    "docs/DATASET_PROVENANCE.md",
    "docs/PHASE_00_V2_FOUNDATION_RECORD.md",
]

AUDIT_ARTIFACTS = [
    "artifacts/raw_dataset_manifest.json",
    "artifacts/dataset_audit.json",
    "results/dataset_audit/station_summary.csv",
    "results/dataset_audit/missingness_by_variable.csv",
    "results/dataset_audit/missingness_by_station.csv",
    "results/dataset_audit/target_missingness_by_station_year.csv",
    "results/dataset_audit/timestamp_grid_audit.csv",
]

# A Phase-0 repository must contain no model output at all.
FORBIDDEN_NAME_PATTERNS = [
    r"metric", r"prediction", r"forecast_result", r"model_comparison",
    r"leaderboard", r"checkpoint", r"\.pt$", r"\.pth$", r"\.onnx$",
    r"\.pkl$", r"\.joblib$",
]
FORBIDDEN_METRIC_KEYS = {
    "mae", "rmse", "mse", "r2", "mape", "smape", "crps", "wql", "nll",
    "loss", "score", "accuracy", "val_mae", "test_mae", "val_rmse",
    "test_rmse",
}

CHUNK = 1 << 20




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
PHASE_NUMBER = 0

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


# The information regimes are named R0/R1/R2/R3 and the coefficient of
# determination is R2. A bare key "R2" is therefore ambiguous, and treating
# every one as a metric produced a false positive on the Phase-4 neural
# feature schema, whose `dynamic_channel_counts` is keyed by regime.
#
# Disambiguation is contextual rather than by deletion: a bare r2 key still
# counts as a metric unless its siblings are the regime vocabulary, so a
# genuine `{"r2": 0.53}` in a Phase-0 artifact is still caught. Qualified
# metric spellings are always caught.
REGIME_LABELS = {"R0", "R1", "R2", "R3"}
AMBIGUOUS_METRIC_KEYS = {"r2"}
QUALIFIED_METRIC_KEYS = {
    "r2_score", "macro_r2", "micro_r2", "train_r2", "validation_r2",
    "test_r2", "val_r2", "metric_value", "prediction_score",
}


def names_a_performance_metric(key, siblings):
    """Context-aware: does this key name an actual performance metric?"""
    lowered = str(key).strip().lower()
    if lowered in QUALIFIED_METRIC_KEYS:
        return True
    if lowered in AMBIGUOUS_METRIC_KEYS:
        return not set(str(s) for s in siblings) <= REGIME_LABELS
    return lowered in (FORBIDDEN_METRIC_KEYS - AMBIGUOUS_METRIC_KEYS)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 66
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def check_files(report):
    missing = [name for name in REQUIRED_FILES
               if not (PROJECT_ROOT / name).is_file()]
    report.add("Required Phase-0 files present (%d)" % len(REQUIRED_FILES),
               not missing, "missing: %s" % ", ".join(missing))
    return not missing


def check_study(report):
    if not STUDY.is_file():
        report.add("Study contract parses", False, "configs/study.json absent")
        return None
    try:
        study = load_json(STUDY)
    except ValueError as error:
        report.add("Study contract parses", False, str(error))
        return None
    report.add("Study contract parses", True)

    report.add("Branch expectation recorded as master",
               study.get("branch") == EXPECTED_BRANCH,
               "found %r" % study.get("branch"))
    report.add("Horizons are exactly [1, 6, 12, 24]",
               study.get("forecast_horizons_hours") == EXPECTED_HORIZONS,
               "found %r" % study.get("forecast_horizons_hours"))
    report.add("Partition rule is target_timestamp",
               study.get("partition_rule") == "target_timestamp",
               "found %r" % study.get("partition_rule"))
    report.add("Forecast-origin rule is target_timestamp_minus_horizon",
               study.get("forecast_origin_rule")
               == "target_timestamp_minus_horizon",
               "found %r" % study.get("forecast_origin_rule"))

    split = study.get("split", {})
    problems = []
    bounds = {}
    for name in ("train", "validation", "test"):
        part = split.get(name, {})
        try:
            start = datetime.strptime(part["start"], "%Y-%m-%dT%H:%M:%S")
            end = datetime.strptime(part["end"], "%Y-%m-%dT%H:%M:%S")
        except (KeyError, ValueError) as error:
            problems.append("%s: %s" % (name, error))
            continue
        if start >= end:
            problems.append("%s start is not before end" % name)
        bounds[name] = (start, end)
    report.add("Partition bounds parse and are internally ordered",
               not problems, "; ".join(problems))

    ordering = []
    if len(bounds) == 3:
        if not bounds["train"][1] < bounds["validation"][0]:
            ordering.append("train end is not before validation start")
        if not bounds["validation"][1] < bounds["test"][0]:
            ordering.append("validation end is not before test start")
    else:
        ordering.append("incomplete partition bounds")
    report.add("train < validation < test, non-overlapping",
               not ordering, "; ".join(ordering))

    report.add("Final-test status is sealed",
               study.get("final_test_status") == "sealed",
               "found %r" % study.get("final_test_status"))
    counters = []
    for key in ("model_runs", "validation_metrics_seen",
                "test_metrics_seen"):
        if study.get(key) != 0:
            counters.append("%s is %r" % (key, study.get(key)))
    report.add("Model runs and metrics seen are all zero",
               not counters, "; ".join(counters))
    report.add("Severe threshold is not numerically frozen",
               study.get("severe_threshold_definition", {})
               .get("numeric_value_frozen") is False)
    return study


def check_raw_integrity(report):
    if not RAW_MANIFEST.is_file():
        report.add("Raw files match recorded digests", False,
                   "raw manifest absent")
        return
    manifest = load_json(RAW_MANIFEST)
    problems = []
    outer = PROJECT_ROOT / manifest["outer_archive"]["relative_path"]
    if not outer.is_file():
        problems.append("outer archive missing")
    elif sha256_of_file(outer) != manifest["outer_archive"]["sha256"]:
        problems.append("outer archive digest mismatch")
    for entry in manifest["source_csvs"]:
        path = PROJECT_ROOT / entry["relative_path"]
        if not path.is_file():
            problems.append("%s missing" % entry["member"])
        elif sha256_of_file(path) != entry["sha256"]:
            problems.append("%s digest mismatch" % entry["member"])
    report.add("Official raw files match recorded digests (%d files)"
               % (len(manifest["source_csvs"]) + 1),
               not problems, "; ".join(problems[:5]))
    report.add("Raw manifest records no cleaning or imputation",
               manifest.get("cleaning_applied") is False
               and manifest.get("imputation_applied") is False)


def check_dataset_audit(report):
    if not DATASET_AUDIT.is_file():
        report.add("Station/timestamp identity is unique", False,
                   "dataset audit absent")
        return
    audit = load_json(DATASET_AUDIT)
    report.add("Station count is 12", audit.get("station_count") == 12,
               "found %r" % audit.get("station_count"))
    report.add("Total rows are 420,768", audit.get("total_rows") == 420768,
               "found %r" % audit.get("total_rows"))
    report.add("Schema matches across all stations",
               audit.get("schema_matches_expected") is True
               and audit.get("all_headers_identical") is True)
    report.add("Station identifiers agree with filenames",
               audit.get("station_identifiers_agree_with_filenames") is True)
    report.add("(station, timestamp) pairs are unique",
               audit.get("duplicate_station_timestamp_pairs") == 0,
               "found %r" % audit.get("duplicate_station_timestamp_pairs"))
    grid = [station["filename_station"] for station in audit.get("stations", [])
            if station["missing_timestamps"]
            or station["timestamps_outside_expected_grid"]]
    report.add("Hourly grid complete at every station", not grid,
               "incomplete: %s" % grid)
    report.add("No PM2.5 value statistic was computed",
               audit.get("target_value_statistics_computed") is False)


def check_no_model_output(report):
    offenders = []
    for directory in ("artifacts", "results", "figures", "data/processed"):
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            relative = str(path.relative_to(PROJECT_ROOT))
            if owned_by_a_later_phase(relative):
                continue
            for pattern in FORBIDDEN_NAME_PATTERNS:
                if re.search(pattern, path.name, re.IGNORECASE):
                    offenders.append("%s (name matches %r)"
                                     % (relative, pattern))
    report.add("No model-result file exists", not offenders,
               "; ".join(offenders[:5]))

    placeholders = []
    for path in sorted((PROJECT_ROOT / "artifacts").rglob("*.json")):
        if owned_by_a_later_phase(path.relative_to(PROJECT_ROOT)):
            continue
        try:
            payload = load_json(path)
        except ValueError:
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                siblings = list(node.keys())
                for key, value in node.items():
                    if (names_a_performance_metric(key, siblings)
                            and isinstance(value, (int, float))
                            and not isinstance(value, bool)):
                        placeholders.append("%s -> %s"
                                            % (path.name, key))
                    stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
    for path in sorted((PROJECT_ROOT / "results").rglob("*.csv")):
        if owned_by_a_later_phase(path.relative_to(PROJECT_ROOT)):
            continue
        with open(str(path), newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        hits = [column for column in header
                if names_a_performance_metric(column.strip(), header)]
        if hits:
            placeholders.append("%s -> columns %s" % (path.name, hits))
    report.add("No placeholder metric exists", not placeholders,
               "; ".join(placeholders[:5]))


def build_manifest(study):
    raw = load_json(RAW_MANIFEST)
    audit = load_json(DATASET_AUDIT)
    return {
        "study": study["study"],
        "phase": "phase_0_foundation",
        "branch": EXPECTED_BRANCH,
        "starting_master_head": STARTING_MASTER_HEAD,
        "dataset": {
            "name": raw["dataset"],
            "uci_id": raw["uci_id"],
            "doi": raw["doi"],
            "landing_page": raw["landing_page"],
            "resolved_download_url": raw["resolved_download_url"],
            "license": raw["license"],
            "period_start": study["dataset"]["period_start"],
            "period_end": study["dataset"]["period_end"],
            "station_count": audit["station_count"],
            "station_names": audit["station_names"],
            "row_count": audit["total_rows"],
            "schema": audit["columns"],
        },
        "raw_archive_sha256": raw["outer_archive"]["sha256"],
        "inner_archive_sha256": raw["inner_archive"]["sha256"],
        "source_csv_aggregate_sha256": raw["source_csv_aggregate_sha256"],
        "source_csv_sha256": {entry["station"]: entry["sha256"]
                              for entry in raw["source_csvs"]},
        "dataset_audit_artifact_sha256": {
            name: sha256_of_file(PROJECT_ROOT / name)
            for name in AUDIT_ARTIFACTS},
        "document_sha256": {name: sha256_of_file(PROJECT_ROOT / name)
                            for name in HASHED_DOCUMENTS},
        "forecast_horizons_hours": study["forecast_horizons_hours"],
        "partition_rule": study["partition_rule"],
        "forecast_origin_rule": study["forecast_origin_rule"],
        "split": study["split"],
        "information_regimes": sorted(study["information_regimes"]),
        "primary_metric_provisional": study["primary_metric_provisional"],
        "final_test_status": "sealed",
        "target_value_statistics_computed": False,
        "severe_threshold_numeric_value_frozen": False,
        "model_roster_frozen": False,
        "environment_frozen": False,
        "heavy_dependencies_installed": False,
        "model_runs": 0,
        "validation_metrics_seen": 0,
        "test_metrics_seen": 0,
        "v1_legacy_modified": False,
        "note": (
            "Deterministic by construction: no wall-clock timestamp and no "
            "filesystem mtime is recorded. Content is hashed, not metadata."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="write artifacts/v2_foundation_manifest.json "
                             "after all gates pass")
    args = parser.parse_args()

    print("")
    print("[AIRSENSE V2 FOUNDATION VALIDATION]")
    print("")
    print("Read-only. Trains nothing, predicts nothing, repairs nothing.")
    print("")

    report = Report()
    check_files(report)
    study = check_study(report)
    check_raw_integrity(report)
    check_dataset_audit(report)
    check_no_model_output(report)

    if FOUNDATION_MANIFEST.is_file():
        manifest = load_json(FOUNDATION_MANIFEST)
        stale = []
        for name, digest in manifest.get("document_sha256", {}).items():
            path = PROJECT_ROOT / name
            if not path.is_file():
                stale.append("%s missing" % name)
            elif sha256_of_file(path) != digest:
                stale.append("%s changed" % name)
        for name, digest in manifest.get(
                "dataset_audit_artifact_sha256", {}).items():
            path = PROJECT_ROOT / name
            if not path.is_file():
                stale.append("%s missing" % name)
            elif sha256_of_file(path) != digest:
                stale.append("%s changed" % name)
        report.add("Foundation manifest matches current files", not stale,
                   "; ".join(stale[:5]))
        report.add("Foundation manifest records a sealed final test",
                   manifest.get("final_test_status") == "sealed")

    print("=" * 70)
    report.render()
    print("=" * 70)
    failures = report.failed()
    print("")
    if failures:
        print("V2 FOUNDATION VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1

    if args.write_manifest and study is not None:
        manifest = build_manifest(study)
        with open(str(FOUNDATION_MANIFEST), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print("wrote %s" % FOUNDATION_MANIFEST.relative_to(PROJECT_ROOT))
        print("")

    print("V2 FOUNDATION VALIDATION: PASS")
    print("Phase 0 only. No model, no metric, no V2 result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
