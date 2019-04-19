"""Run the AirSense V1 Protocol Phase 6 M0 naive baseline.

Thin entry point. All logic lives in ``src/models/baseline.py``.

M0 is a training-median constant predictor: the constant is the median of
the 2010-2012 development-training target, and it is evaluated on the 2013
validation year only.

The 2014 partition is locked until Protocol Phase 10. This script reads no
test target, no test feature matrix and produces no test prediction or
metric.

Usage:
    venv/bin/python scripts/run_m0_baseline.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline import BaselineError, run_m0_baseline  # noqa: E402


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 6]")
    print("M0 naive baseline: training-median constant predictor")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        manifest = run_m0_baseline(PROJECT_ROOT)
    except BaselineError as exc:
        sys.stderr.write("\nM0 BASELINE ABORTED: %s\n" % exc)
        return 1
    metrics = manifest["validation_metrics"]
    print("")
    print("M0 baseline constant : %r" % manifest["baseline_constant"])
    print("2013 DEVELOPMENT VALIDATION (not final test performance):")
    print("    MAE  : %.6f" % metrics["mae"])
    print("    RMSE : %.6f" % metrics["rmse"])
    print("    R2   : %.6f" % metrics["r2"])
    print("")
    print("2014 final test: not evaluated (quarantined until Phase 10).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
