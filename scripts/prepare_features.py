"""Run the AirSense V1 Protocol Phase 5 feature preparation.

Thin entry point. All logic lives in ``src/features/preparation.py``.

Builds one shared 43-feature representation for M1, M2 and M3 from the
frozen Phase 4 partitions. Categorical vocabularies come from the 2010-2012
development-training predictors only; validation and test are applied
against that frozen schema.

The 2014 target is quarantined: the test file is read with an explicit
column list that omits ``pm2.5``, and no ``y_test`` artifact is produced.

This script fits no model, produces no prediction, and computes no metric.

Usage:
    venv/bin/python scripts/prepare_features.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.preparation import (  # noqa: E402
    FeatureError, run_feature_preparation)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 5]")
    print("Feature preparation")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    try:
        schema = run_feature_preparation(PROJECT_ROOT)
    except FeatureError as exc:
        sys.stderr.write("\nFEATURE PREPARATION ABORTED: %s\n" % exc)
        return 1
    print("")
    print("Phase 5 complete: %d features shared by M1, M2 and M3."
          % schema["feature_count"])
    print("No model was trained and no metric was calculated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
