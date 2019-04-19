"""Run the AirSense V1 exploratory data analysis.

Thin entry point. All scientific logic lives in ``src/analysis/eda.py`` so
that the script, and the notebook, present the same calculations rather
than each carrying their own copy.

The raw dataset is read-only throughout. This script performs no cleaning,
imputation, encoding, splitting, feature engineering or modelling.

Usage:
    venv/bin/python scripts/run_eda.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.eda import run_analysis  # noqa: E402


def main():
    print("[AIRSENSE V1 - PHASE 3 EXPLORATORY DATA ANALYSIS]")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        run_analysis(PROJECT_ROOT)
    except RuntimeError as exc:
        sys.stderr.write("\nEDA ABORTED: %s\n" % exc)
        return 1
    print("")
    print("EDA complete. No model was trained and no dataset was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
