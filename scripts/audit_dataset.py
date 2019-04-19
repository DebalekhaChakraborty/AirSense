"""Write the AirSense V1 raw-dataset audit record.

Reads data/raw/, audits every CSV found there, and writes the combined
result to artifacts/data_audit.json.

The raw files are opened read-only. This script never cleans, imputes or
modifies the data; it only measures and records it so that the raw inputs
can be verified as unchanged at any later phase of the project.

Usage:
    python scripts/audit_dataset.py
"""

import io
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import audit_csv  # noqa: E402

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "data_audit.json")


def main():
    if not os.path.isdir(RAW_DIR):
        sys.stderr.write("Raw data directory not found: %s\n" % RAW_DIR)
        return 2

    csv_names = sorted(
        name for name in os.listdir(RAW_DIR) if name.lower().endswith(".csv")
    )
    if not csv_names:
        sys.stderr.write(
            "No CSV files found in %s. See data/README.md for acquisition "
            "instructions.\n" % RAW_DIR
        )
        return 2

    files = []
    for name in csv_names:
        path = os.path.join(RAW_DIR, name)
        sys.stdout.write("Auditing %s ...\n" % name)
        record = audit_csv(path)
        record["relative_path"] = os.path.join("data", "raw", name)
        files.append(record)

    payload = OrderedDict()
    payload["project"] = "AirSense"
    payload["phase"] = "V1 / Phase 2 - dataset provenance and audit"
    payload["audit_generated_utc"] = datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload["audit_generated_note"] = (
        "Timestamp of this reconstruction run, not a historical date."
    )
    payload["historical_cutoff"] = "2019-04-26"
    payload["audit_tool"] = "src/data/audit.py (Python standard library only)"
    payload["cleaning_performed"] = False
    payload["rows_removed"] = 0
    payload["files"] = files

    with io.open(ARTIFACT_PATH, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.write("\n")

    sys.stdout.write("Wrote %s\n" % ARTIFACT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
