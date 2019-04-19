"""Run the AirSense V1 Protocol Phase 9 M3 random forest.

Thin entry point. All logic lives in ``src/models/random_forest.py``.

M3 evaluates 24 predeclared RandomForestRegressor configurations - 100 trees
each - on the frozen 43-feature Phase 5 matrix, fitting on 2010-2012 and
selecting one by lowest 2013 validation MAE under a tie-break policy frozen
before fitting.

The 2014 partition is locked until Protocol Phase 10. This script reads no
test features, no test target, and produces no test prediction or metric.
No 2014 information influences hyperparameter selection.

Usage:
    venv/bin/python scripts/run_m3_random_forest.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.random_forest import (  # noqa: E402
    RandomForestError, run_m3_random_forest)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 9]")
    print("M3 random forest regression")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        manifest = run_m3_random_forest(PROJECT_ROOT)
    except RandomForestError as exc:
        sys.stderr.write("\nM3 ABORTED: %s\n" % exc)
        return 1

    config = manifest["selected_configuration"]
    metrics = manifest["validation_metrics"]
    comparison = manifest["model_comparison"]
    print("")
    print("Selected: candidate %d  max_depth=%s  min_samples_leaf=%s  "
          "max_features=%s  (n_estimators=%d)"
          % (config["candidate_id"], config["max_depth"],
             config["min_samples_leaf"], config["max_features"],
             config["n_estimators"]))
    print("")
    print("2013 DEVELOPMENT VALIDATION (not final test performance):")
    print("    MAE  : %.6f" % metrics["mae"])
    print("    RMSE : %.6f" % metrics["rmse"])
    print("    R2   : %.6f" % metrics["r2"])
    for key, name in (("m0", "M0"), ("m1", "M1"), ("m2", "M2")):
        block = comparison[key]
        print("    vs %s: MAE %+.6f (%+.4f%%)  RMSE %+.6f  R2 %+.6f"
              % (name, block["mae_improvement_absolute"],
                 block["mae_improvement_percent"],
                 block["rmse_improvement_absolute"], block["r2_delta"]))
    print("")
    print("2014 final test: not evaluated (quarantined until Phase 10).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
