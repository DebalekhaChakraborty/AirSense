"""Protocol Phase 11: post-evaluation error analysis.

Thin entry point. All logic lives in ``src/evaluation/error_analysis.py``.

Reads the frozen Phase-10 predictions and target snapshot and characterises
where the four models succeed and fail. It fits nothing, predicts nothing,
changes nothing, and never reopens the original 2014 test source.

Usage:
    venv/bin/python scripts/run_error_analysis.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.error_analysis import (  # noqa: E402
    ErrorAnalysisError, run_error_analysis)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 11]")
    print("Post-evaluation error analysis (descriptive only)")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("")
    print("The final test is already exhausted. No model is fitted here.")
    print("")
    try:
        manifest = run_error_analysis(PROJECT_ROOT)
    except ErrorAnalysisError as exc:
        sys.stderr.write("\nERROR ANALYSIS ABORTED: %s\n" % exc)
        return 1
    print("")
    print("models_fitted             : %d" % manifest["models_fitted"])
    print("predictions_generated     : %d"
          % manifest["predictions_generated"])
    print("model_modification_status : %s"
          % manifest["model_modification_status"])
    print("final_test_status         : %s" % manifest["final_test_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
