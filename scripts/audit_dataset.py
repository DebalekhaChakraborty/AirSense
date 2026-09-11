"""Audit the raw AirSense V2 dataset and write deterministic artifacts.

AirSense V2, Protocol Phase 0.

Reads only ``data/raw/``. Repairs nothing, imputes nothing, fits nothing.
Computes no PM2.5 value statistic in any partition: the final-test target is
sealed, so the target is reported by presence/absence counts only.

Outputs (all deterministic - sorted, no wall-clock time, no mtimes):

    artifacts/dataset_audit.json
    results/dataset_audit/station_summary.csv
    results/dataset_audit/missingness_by_variable.csv
    results/dataset_audit/missingness_by_station.csv
    results/dataset_audit/target_missingness_by_station_year.csv
    results/dataset_audit/timestamp_grid_audit.csv

Usage:
    python3 scripts/audit_dataset.py
"""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import (                                  # noqa: E402
    EXPECTED_COLUMNS, TARGET, audit_dataset)

RAW_STATION_DIR = (PROJECT_ROOT / "data" / "raw"
                   / "PRSA_Data_20130301-20170228")
AUDIT_JSON = PROJECT_ROOT / "artifacts" / "dataset_audit.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_audit"


def write_csv(path, header, rows):
    """Write LF-terminated CSV deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def rate(count, total):
    return round(count / float(total), 8) if total else 0.0


def main():
    if not RAW_STATION_DIR.is_dir():
        print("raw station directory missing: %s" % RAW_STATION_DIR,
              file=sys.stderr)
        print("run scripts/acquire_dataset.py first", file=sys.stderr)
        return 2
    paths = sorted(RAW_STATION_DIR.glob("PRSA_Data_*.csv"))
    if not paths:
        print("no station CSVs under %s" % RAW_STATION_DIR, file=sys.stderr)
        return 2

    audit = audit_dataset(paths)
    write_json(AUDIT_JSON, audit)

    stations = sorted(audit["stations"], key=lambda s: s["filename_station"])

    write_csv(
        RESULTS_DIR / "station_summary.csv",
        ["station", "file", "sha256", "size_bytes", "rows",
         "timestamp_min", "timestamp_max", "distinct_timestamps",
         "missing_timestamps", "duplicate_timestamps", "malformed_rows",
         "station_matches_filename", "pm25_missing", "pm25_missing_rate"],
        [[s["filename_station"], s["file"], s["sha256"], s["size_bytes"],
          s["row_count"], s["timestamp_min"], s["timestamp_max"],
          s["observed_distinct_timestamps"], s["missing_timestamps"],
          s["duplicate_timestamps"], s["malformed_rows"],
          s["station_identifier_agrees_with_filename"],
          s["missing_by_variable"][TARGET],
          rate(s["missing_by_variable"][TARGET], s["row_count"])]
         for s in stations])

    write_csv(
        RESULTS_DIR / "missingness_by_variable.csv",
        ["variable", "missing", "present", "missing_rate"],
        [[column, audit["missing_by_variable"][column]["missing"],
          audit["missing_by_variable"][column]["present"],
          audit["missing_by_variable"][column]["missing_rate"]]
         for column in EXPECTED_COLUMNS])

    write_csv(
        RESULTS_DIR / "missingness_by_station.csv",
        ["station", "variable", "missing", "present", "missing_rate"],
        [[s["filename_station"], column, s["missing_by_variable"][column],
          s["row_count"] - s["missing_by_variable"][column],
          rate(s["missing_by_variable"][column], s["row_count"])]
         for s in stations for column in EXPECTED_COLUMNS])

    target_rows = []
    for station in stations:
        years = sorted(station["missing_by_year"], key=int)
        for year in years:
            count = station["missing_by_year"][year].get(TARGET, 0)
            target_rows.append([station["filename_station"], year, count])
    write_csv(
        RESULTS_DIR / "target_missingness_by_station_year.csv",
        ["station", "year", "pm25_missing"], target_rows)

    write_csv(
        RESULTS_DIR / "timestamp_grid_audit.csv",
        ["station", "expected_grid_hours", "observed_distinct_timestamps",
         "missing_timestamps", "timestamps_outside_grid",
         "duplicate_timestamps", "timestamp_min", "timestamp_max",
         "grid_complete"],
        [[s["filename_station"], s["expected_grid_hours"],
          s["observed_distinct_timestamps"], s["missing_timestamps"],
          s["timestamps_outside_expected_grid"], s["duplicate_timestamps"],
          s["timestamp_min"], s["timestamp_max"],
          s["missing_timestamps"] == 0
          and s["timestamps_outside_expected_grid"] == 0
          and s["duplicate_timestamps"] == 0]
         for s in stations])

    print("")
    print("stations              : %d" % audit["station_count"])
    print("total rows            : %d" % audit["total_rows"])
    print("schema matches        : %s" % audit["schema_matches_expected"])
    print("headers identical     : %s" % audit["all_headers_identical"])
    print("station ids agree     : %s"
          % audit["station_identifiers_agree_with_filenames"])
    print("duplicate st/ts pairs : %d"
          % audit["duplicate_station_timestamp_pairs"])
    print("fully duplicated rows : %d" % audit["fully_duplicated_rows"])
    print("PM2.5 missing         : %d (%.4f%%)"
          % (audit["missing_by_variable"][TARGET]["missing"],
             100.0 * audit["missing_by_variable"][TARGET]["missing_rate"]))
    print("target value stats    : %s"
          % audit["target_value_statistics_computed"])
    for path in [AUDIT_JSON] + sorted(RESULTS_DIR.glob("*.csv")):
        digest = hashlib.sha256(open(str(path), "rb").read()).hexdigest()
        print("  %-58s %s" % (path.relative_to(PROJECT_ROOT), digest[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
