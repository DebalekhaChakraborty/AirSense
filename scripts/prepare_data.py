"""Run the AirSense V1 Phase 4 cleaning and chronological split freeze.

Thin entry point. All logic lives in ``src/data/cleaning.py``.

What this does:

* verifies the raw dataset digest and hourly temporal integrity;
* assigns train / validation / test membership from CALENDAR TIME on the
  full 43,824-row timeline, before any target-based exclusion;
* excludes rows whose ``pm2.5`` target is missing from the supervised
  derived datasets only, recording every one in a permanent manifest;
* writes the supervised CSVs, the exclusion manifest, the split manifest
  and the preprocessing audit tables;
* verifies every retained row against the raw source and re-checks the raw
  digest.

What this does NOT do: impute any target, transform PM2.5, remove zero or
high-concentration observations, scale, encode, engineer features, or fit
any model.

Usage:
    venv/bin/python scripts/prepare_data.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.cleaning import IntegrityError, run_preparation  # noqa: E402


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 4]")
    print("Data cleaning and chronological evaluation split freeze")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        run_preparation(PROJECT_ROOT)
    except IntegrityError as exc:
        sys.stderr.write("\nPREPARATION ABORTED: %s\n" % exc)
        return 1
    print("")
    print("Phase 4 complete. No feature was engineered and no model was "
          "trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
