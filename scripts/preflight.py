"""AirSense V1 preflight check.

Read-only validation of the project foundation. Reports PASS / WARN / FAIL
for the runtime, the pinned dependency set, the project structure and the
raw dataset.

This script uses the Python standard library only, so that it runs before
any scientific package is installed and on both the reference historical
interpreter (CPython 3.6.7) and the modern host interpreter.

It NEVER modifies the dataset, and never writes anything at all. Every
filesystem operation performed here is a read.

A host newer than the frozen 2019 reference environment is reported as an
explicit HOST-COMPATIBILITY WARNING. It is never reported as a historical
environment PASS.

Usage:
    python scripts/preflight.py

Exit codes:
    0  READY
    1  HOST-COMPATIBILITY WARNING
    2  NOT READY
"""

import io
import json
import os
import platform
import sys
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import (  # noqa: E402
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    audit_csv,
    schema_differences,
)

# ----------------------------------------------------------------------
# Frozen historical reference. Do not relax these to make a check pass.
# ----------------------------------------------------------------------

HISTORICAL_CUTOFF = "2019-04-26"
REFERENCE_PYTHON = (3, 6, 7)
REFERENCE_OS = "Ubuntu 18.04 LTS"

# Import name for each distribution named in requirements-v1-2019.txt.
IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jupyter": "jupyter",
}

# Metapackages that expose no meaningful __version__ attribute.
NO_VERSION_ATTRIBUTE = ("jupyter",)

REQUIRED_DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "scripts",
    "artifacts",
    "results",
    "figures",
    "docs",
    "tests",
]

REQUIRED_DOCS = [
    "README.md",
    "requirements-v1-2019.txt",
    "data/README.md",
    "docs/HISTORICAL_COMPATIBILITY.md",
    "docs/DATASET_AUDIT.md",
    "docs/RESEARCH_QUESTION.md",
    "docs/FEATURE_POLICY.md",
    "docs/V1_RESEARCH_PROTOCOL.md",
]

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
AUDIT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "data_audit.json")
REQUIREMENTS_PATH = os.path.join(PROJECT_ROOT, "requirements-v1-2019.txt")

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

LINE_WIDTH = 62


class Report(object):
    """Accumulates check results and renders the summary block."""

    def __init__(self):
        self.checks = []      # list of (label, status)
        self.details = []     # list of strings, printed under the summary

    def add(self, label, status, detail=None):
        self.checks.append((label, status))
        if detail:
            self.details.append("  - [%s] %s: %s" % (status, label, detail))
        return status

    def statuses(self):
        return [status for _, status in self.checks]

    def render_summary(self):
        lines = []
        for label, status in self.checks:
            dots = "." * max(3, LINE_WIDTH - len(label) - len(status) - 2)
            lines.append("%s %s %s" % (label, dots, status))
        return "\n".join(lines)

    def overall(self):
        statuses = self.statuses()
        if FAIL in statuses:
            return "NOT READY", 2
        if WARN in statuses:
            return "HOST-COMPATIBILITY WARNING", 1
        return "READY", 0


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------

def check_runtime(report):
    """Check 1 - Python version against the frozen reference interpreter."""
    actual = sys.version_info[:3]
    actual_text = "%d.%d.%d" % actual
    reference_text = "%d.%d.%d" % REFERENCE_PYTHON

    print("  Reference interpreter : CPython %s (%s)"
          % (reference_text, REFERENCE_OS))
    print("  Host interpreter      : %s %s"
          % (platform.python_implementation(), actual_text))
    print("  Host platform         : %s" % platform.platform())

    if actual == REFERENCE_PYTHON:
        return report.add("Runtime", PASS)

    if actual > REFERENCE_PYTHON:
        return report.add(
            "Runtime", WARN,
            "host CPython %s is NEWER than the 2019 reference %s; this is a "
            "non-conforming execution environment, not a historical one"
            % (actual_text, reference_text),
        )

    return report.add(
        "Runtime", WARN,
        "host CPython %s is OLDER than the reference %s"
        % (actual_text, reference_text),
    )


def read_pinned_requirements():
    """Parse requirements-v1-2019.txt into an ordered name -> version map."""
    pins = OrderedDict()
    if not os.path.isfile(REQUIREMENTS_PATH):
        return pins
    with io.open(REQUIREMENTS_PATH, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "==" not in line:
                continue
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def check_dependencies(report):
    """Check 2 - installed versions against the frozen historical pins."""
    pins = read_pinned_requirements()
    if not pins:
        return report.add(
            "Dependencies", FAIL,
            "no pins parsed from requirements-v1-2019.txt",
        )

    matched, mismatched, missing = [], [], []

    for name, expected in pins.items():
        import_name = IMPORT_NAMES.get(name, name)
        try:
            module = __import__(import_name)
        except ImportError:
            missing.append("%s==%s (not installed)" % (name, expected))
            print("  %-14s pinned %-8s -> NOT INSTALLED" % (name, expected))
            continue

        if name in NO_VERSION_ATTRIBUTE:
            matched.append("%s (metapackage, importable)" % name)
            print("  %-14s pinned %-8s -> importable (metapackage)"
                  % (name, expected))
            continue

        installed = getattr(module, "__version__", None)
        if installed is None:
            mismatched.append("%s (version attribute unavailable)" % name)
            print("  %-14s pinned %-8s -> version unknown" % (name, expected))
        elif installed == expected:
            matched.append("%s==%s" % (name, installed))
            print("  %-14s pinned %-8s -> %s  MATCH"
                  % (name, expected, installed))
        else:
            mismatched.append("%s: pinned %s, installed %s"
                              % (name, expected, installed))
            print("  %-14s pinned %-8s -> %s  MISMATCH"
                  % (name, expected, installed))

    if not mismatched and not missing:
        return report.add("Dependencies", PASS)

    parts = []
    if mismatched:
        parts.append("version mismatch: " + "; ".join(mismatched))
    if missing:
        parts.append("not installed: " + "; ".join(missing))
    parts.append(
        "the pinned 2019 stack is NOT reconstructed here; see "
        "docs/HISTORICAL_COMPATIBILITY.md section 4"
    )
    return report.add("Dependencies", WARN, " | ".join(parts))


def check_structure(report):
    """Check 3 - required directories and foundation documents."""
    missing_dirs = [
        d for d in REQUIRED_DIRECTORIES
        if not os.path.isdir(os.path.join(PROJECT_ROOT, d))
    ]
    missing_docs = [
        f for f in REQUIRED_DOCS
        if not os.path.isfile(os.path.join(PROJECT_ROOT, f))
    ]

    print("  Directories : %d/%d present"
          % (len(REQUIRED_DIRECTORIES) - len(missing_dirs),
             len(REQUIRED_DIRECTORIES)))
    print("  Documents   : %d/%d present"
          % (len(REQUIRED_DOCS) - len(missing_docs), len(REQUIRED_DOCS)))

    if missing_dirs or missing_docs:
        detail = []
        if missing_dirs:
            detail.append("missing directories: " + ", ".join(missing_dirs))
        if missing_docs:
            detail.append("missing documents: " + ", ".join(missing_docs))
        return report.add("Project structure", FAIL, " | ".join(detail))

    return report.add("Project structure", PASS)


def find_raw_csv_files():
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(
        name for name in os.listdir(RAW_DIR) if name.lower().endswith(".csv")
    )


def load_recorded_audit():
    """Load artifacts/data_audit.json, returning filename -> record."""
    if not os.path.isfile(AUDIT_PATH):
        return None
    try:
        with io.open(AUDIT_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except ValueError:
        return None
    records = {}
    for record in payload.get("files", []):
        records[record.get("filename")] = record
    return records


def check_dataset(report):
    """Checks 4-10 - dataset presence, schema, integrity and contents."""
    csv_names = find_raw_csv_files()

    if not csv_names:
        report.add(
            "Raw dataset", FAIL,
            "no CSV found in data/raw/; see data/README.md for acquisition "
            "instructions",
        )
        for label in ("Dataset schema", "Dataset integrity",
                      "Dataset dimensions", "Missing-value summary",
                      "Duplicate rows", "Target column"):
            report.add(label, SKIP)
        print("  No CSV file present in data/raw/.")
        return

    name = csv_names[0]
    path = os.path.join(RAW_DIR, name)
    if len(csv_names) > 1:
        print("  Note: %d CSV files present; auditing %s"
              % (len(csv_names), name))

    print("  File       : %s" % name)
    print("  Byte size  : %d" % os.path.getsize(path))
    mode = oct(os.stat(path).st_mode & 0o777)[-3:]
    writable = os.access(path, os.W_OK)
    print("  File mode  : %s (%s)"
          % (mode, "writable" if writable else "read-only"))
    report.add("Raw dataset", PASS)

    # Read-only audit of the file as it currently exists on disk.
    record = audit_csv(path)

    # --- Check 5: schema -------------------------------------------------
    header = record["column_names"]
    print("  Columns    : %s" % ", ".join(header))
    if record["schema_matches_expected"]:
        report.add("Dataset schema", PASS)
    else:
        missing_expected, unexpected = schema_differences(header)
        detail = []
        if missing_expected:
            detail.append("missing: " + ", ".join(missing_expected))
        if unexpected:
            detail.append("unexpected: " + ", ".join(unexpected))
        if not detail:
            detail.append("column order differs from expected %s"
                          % ", ".join(EXPECTED_COLUMNS))
        report.add("Dataset schema", FAIL, " | ".join(detail))

    # --- Check 6: SHA-256 integrity --------------------------------------
    digest = record["sha256"]
    print("  SHA-256    : %s" % digest)
    recorded = load_recorded_audit()
    if recorded is None:
        report.add(
            "Dataset integrity", WARN,
            "artifacts/data_audit.json absent or unreadable, so the digest "
            "cannot be verified against a recorded value; run "
            "scripts/audit_dataset.py",
        )
    elif name not in recorded:
        report.add(
            "Dataset integrity", WARN,
            "no recorded audit entry for %s" % name,
        )
    else:
        expected_digest = recorded[name].get("sha256")
        if expected_digest == digest:
            print("  Integrity  : digest matches artifacts/data_audit.json")
            report.add("Dataset integrity", PASS)
        else:
            report.add(
                "Dataset integrity", FAIL,
                "digest mismatch: recorded %s, found %s -- the raw file is "
                "NOT the one this project was audited against"
                % (expected_digest, digest),
            )

    # --- Check 7: dimensions ---------------------------------------------
    rows = record["row_count"]
    columns = record["column_count"]
    print("  Rows       : %d" % rows)
    print("  Columns    : %d" % columns)
    if rows > 0 and columns == len(EXPECTED_COLUMNS):
        if record["ragged_row_count"]:
            report.add(
                "Dataset dimensions", FAIL,
                "%d rows have a field count other than %d"
                % (record["ragged_row_count"], columns),
            )
        else:
            report.add("Dataset dimensions", PASS)
    else:
        report.add(
            "Dataset dimensions", FAIL,
            "expected %d columns and at least one row, found %d columns and "
            "%d rows" % (len(EXPECTED_COLUMNS), columns, rows),
        )

    # --- Check 8: missing values -----------------------------------------
    missing_counts = record["missing_values_per_column"]
    with_missing = [(c, n) for c, n in missing_counts.items() if n]
    if with_missing:
        for column, count in with_missing:
            share = (100.0 * count / rows) if rows else 0.0
            print("  Missing    : %-6s %6d  (%.4f%%)" % (column, count, share))
    else:
        print("  Missing    : none in any column")
    # Missing values are recorded, never removed. Their presence is expected
    # for this dataset and is not itself a failure.
    report.add("Missing-value summary", PASS)

    # --- Check 9: duplicates ---------------------------------------------
    duplicates_full = record["duplicate_rows_including_index_column"]
    duplicates_trimmed = record["duplicate_rows_excluding_index_column"]
    print("  Duplicates : %d (all columns), %d (excluding index column)"
          % (duplicates_full, duplicates_trimmed))
    if duplicates_trimmed == 0:
        report.add("Duplicate rows", PASS)
    else:
        report.add(
            "Duplicate rows", WARN,
            "%d duplicate observations found when the index column is "
            "excluded" % duplicates_trimmed,
        )

    # --- Check 10: target column -----------------------------------------
    if record["target_column_present"]:
        target_missing = missing_counts.get(TARGET_COLUMN, 0)
        usable = rows - target_missing
        share = (100.0 * usable / rows) if rows else 0.0
        print("  Target     : '%s' present, %d usable of %d rows (%.4f%%)"
              % (TARGET_COLUMN, usable, rows, share))
        if usable > 0:
            report.add("Target column", PASS)
        else:
            report.add(
                "Target column", FAIL,
                "target '%s' present but has no usable values"
                % TARGET_COLUMN,
            )
    else:
        report.add(
            "Target column", FAIL,
            "target column '%s' not found in header" % TARGET_COLUMN,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    report = Report()

    print("")
    print("[AIRSENSE V1 PREFLIGHT]")
    print("")
    print("Historical cutoff: %s" % HISTORICAL_CUTOFF)
    print("Reference Python:  %d.%d.%d" % REFERENCE_PYTHON)
    print("Project root:      %s" % PROJECT_ROOT)
    print("")
    print("This check is read-only. No dataset or artifact is modified.")

    print("")
    print("-- Runtime " + "-" * (LINE_WIDTH - 11))
    check_runtime(report)

    print("")
    print("-- Dependencies " + "-" * (LINE_WIDTH - 16))
    check_dependencies(report)

    print("")
    print("-- Project structure " + "-" * (LINE_WIDTH - 21))
    check_structure(report)

    print("")
    print("-- Dataset " + "-" * (LINE_WIDTH - 11))
    check_dataset(report)

    print("")
    print("=" * LINE_WIDTH)
    print(report.render_summary())
    print("=" * LINE_WIDTH)

    overall, exit_code = report.overall()
    print("")
    print("Overall: %s" % overall)

    if report.details:
        print("")
        print("Notes:")
        for detail in report.details:
            print(detail)

    if overall == "HOST-COMPATIBILITY WARNING":
        print("")
        print("The project foundation is valid, but this host does not")
        print("reproduce the frozen 2019 reference environment. V1 modelling")
        print("must not be executed here and reported as period-authentic.")
        print("See docs/HISTORICAL_COMPATIBILITY.md section 4.")

    print("")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
