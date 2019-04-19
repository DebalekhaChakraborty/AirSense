"""Protocol Phase 10, Stage A: blind final refit and 2014 predictions.

Thin entry point. All logic lives in ``src/evaluation/final_refit.py``.

Refits all four frozen models on the full permitted 2010-2013 development
population and generates 2014 predictions from **predictors only**. The
2014 target is sealed throughout: this script never loads
``data/processed/airsense_test_2014.csv``, computes no test metric, and
writes no target value.

Run this TWICE and confirm byte-identical artifacts before Stage B is
allowed to open the target.

Usage:
    venv/bin/python scripts/prepare_final_predictions.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.final_refit import (  # noqa: E402
    FinalRefitError, run_stage_a)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 10, STAGE A]")
    print("Blind final refit on 2010-2013 and 2014 prediction generation")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("Project root: %s" % PROJECT_ROOT)
    print("")
    print("The 2014 TARGET IS SEALED in this stage.")
    print("")
    try:
        result = run_stage_a(PROJECT_ROOT)
    except FinalRefitError as exc:
        sys.stderr.write("\nSTAGE A ABORTED: %s\n" % exc)
        return 1
    print("")
    print("Blind predictions SHA-256: %s" % result["blind_sha256"])
    print("M0 final constant        : %r" % result["m0_constant"])
    print("")
    print("test_target_status : sealed")
    print("test_metrics_status: not_computed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
