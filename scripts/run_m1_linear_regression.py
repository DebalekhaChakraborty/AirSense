"""Run the AirSense V1 Protocol Phase 7 M1 linear regression.

Thin entry point. All logic lives in ``src/models/linear_regression.py``.

M1 is ordinary least squares (``sklearn.linear_model.LinearRegression``,
``fit_intercept=True``) on the frozen 43-feature Phase 5 matrix, fitted on
2010-2012 and evaluated on the 2013 validation year only.

The 2014 partition is locked until Protocol Phase 10. This script reads no
test features, no test target, and produces no test prediction or metric.

Usage:
    venv/bin/python scripts/run_m1_linear_regression.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.linear_regression import (  # noqa: E402
    LinearModelError, run_m1_linear_regression)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 7]")
    print("M1 ordinary least-squares linear regression")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        manifest = run_m1_linear_regression(PROJECT_ROOT)
    except LinearModelError as exc:
        sys.stderr.write("\nM1 ABORTED: %s\n" % exc)
        return 1
    metrics = manifest["validation_metrics"]
    comparison = manifest["m0_comparison"]
    print("")
    print("2013 DEVELOPMENT VALIDATION (not final test performance):")
    print("    MAE  : %.6f   (M0 %.6f, improvement %+.6f = %+.4f%%)"
          % (metrics["mae"], comparison["m0_mae"],
             comparison["mae_improvement_absolute"],
             comparison["mae_improvement_percent"]))
    print("    RMSE : %.6f   (M0 %.6f, improvement %+.6f = %+.4f%%)"
          % (metrics["rmse"], comparison["m0_rmse"],
             comparison["rmse_improvement_absolute"],
             comparison["rmse_improvement_percent"]))
    print("    R2   : %.6f   (M0 %.6f, delta %+.6f)"
          % (metrics["r2"], comparison["m0_r2"], comparison["r2_delta"]))
    print("")
    print("2014 final test: not evaluated (quarantined until Phase 10).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
