"""Write the Phase-3 pre-validation freeze and the validation opening receipt.

Protocol Phase 3, Stage A.

The freeze must exist **before** the first numerical validation-target access.
After it exists, the B3 grid and feature schema may not change because of
validation results.

Usage:
    python3 scripts/build_phase3_freeze.py --freeze     # Stage A
    python3 scripts/build_phase3_freeze.py --receipt    # opens Stage B
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS               # noqa: E402
from src.models import b3_grid                                    # noqa: E402
from src.models.b3_features import (                              # noqa: E402
    CONTEXT_HOURS, LAGS, MASK_FRACTION_WINDOWS, REGIME_NUMERIC,
    REGIME_POLLUTANTS, SUMMARY_STATS, SUMMARY_WINDOWS, feature_names)

ARTIFACTS = PROJECT_ROOT / "artifacts"
FREEZE = ARTIFACTS / "phase3_prevalidation_freeze.json"
RECEIPT = ARTIFACTS / "validation_development_opening_receipt.json"

N_JOBS = min(16, os.cpu_count() or 1)
SEVERE_THRESHOLD = 244.0


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_json(path, payload):
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_freeze():
    import lightgbm
    import numpy
    import scipy

    grid = b3_grid.candidates(N_JOBS)
    grid_payload = [OrderedDict([
        ("candidate_id", c["candidate_id"]),
        ("learning_rate", c["learning_rate"]),
        ("num_leaves", c["num_leaves"]),
        ("min_child_samples", c["min_child_samples"]),
        ("parameters", c["parameters"]),
    ]) for c in grid]
    grid_hash = hashlib.sha256(
        json.dumps(grid_payload, sort_keys=True).encode("utf-8")).hexdigest()

    schema = OrderedDict()
    for regime in b3_grid.REGIMES:
        schema[regime] = OrderedDict([
            ("numeric_variables", REGIME_NUMERIC[regime]),
            ("gap_age_variables", REGIME_POLLUTANTS[regime]),
            ("feature_count", len(feature_names(regime))),
            ("feature_names_sha256", hashlib.sha256(
                "\n".join(feature_names(regime)).encode("utf-8")).hexdigest()),
        ])

    return OrderedDict([
        ("study", "AirSense V2"),
        ("phase", "phase_3_prevalidation_freeze"),
        ("stage", "A_before_any_numerical_validation_target_access"),
        ("runtime", OrderedDict([
            ("python", platform.python_version()),
            ("implementation", platform.python_implementation()),
            ("environment_path", ".venv-v2"),
            ("numpy", numpy.__version__),
            ("scipy", scipy.__version__),
            ("lightgbm", lightgbm.__version__),
            ("lightgbm_version_source",
             "https://pypi.org/pypi/lightgbm/json (info.version)"),
            ("cpu_logical_cores", os.cpu_count()),
            ("n_jobs_fixed_across_candidates", N_JOBS),
            ("thread_policy",
             "fixed n_jobs for every candidate; threads are never changed "
             "between candidates"),
        ])),
        ("b3_feature_schema", OrderedDict([
            ("context_hours", CONTEXT_HOURS),
            ("lags", LAGS),
            ("lag_48_excluded_reason",
             "lag 48 lies outside the canonical [t-47, t] window; admitting "
             "it would require a hidden 49th hour"),
            ("summary_windows_hours", SUMMARY_WINDOWS),
            ("summary_statistics", SUMMARY_STATS),
            ("summary_windows_end_at", "forecast_origin"),
            ("centred_windows", False),
            ("exponentially_weighted_features", False),
            ("mask_fraction_windows_hours", MASK_FRACTION_WINDOWS),
            ("wind_encoding",
             "deterministic one-hot over 16 frozen training categories "
             "plus MISSING"),
            ("station_encoding",
             "deterministic one-hot over the frozen 12-station vocabulary"),
            ("target_encoding_used", False),
            ("calendar",
             "frozen canonical origin and target cyclic channels; "
             "day_of_week and weekend excluded"),
            ("horizon_as_column", False),
            ("horizon_as_column_reason",
             "models are trained per horizon, so horizon is constant within "
             "a matrix"),
            ("per_regime", schema),
        ])),
        ("b3_candidate_grid", OrderedDict([
            ("count", len(grid_payload)),
            ("generation_order",
             "nested loop: learning_rate [0.03, 0.07] outer, num_leaves "
             "[31, 63], min_child_samples [50, 200] inner"),
            ("candidate_ids", [c["candidate_id"] for c in grid_payload]),
            ("n_estimators_num_boost_round", b3_grid.N_ESTIMATORS),
            ("parameter_spelling", "native LightGBM Booster API"),
            ("parameter_spelling_reason",
             "the lightgbm 4.7.0 sklearn wrapper requires scikit-learn, "
             "which is not part of the frozen V2 runtime; the scientific "
             "grid is identical and only the spelling is native"),
            ("sklearn_equivalent_names", b3_grid.SKLEARN_EQUIVALENT),
            ("early_stopping", False),
            ("early_stopping_reason",
             "an early-stopped iteration count would be one more quantity "
             "selected by looking at validation"),
            ("grid_sha256", grid_hash),
            ("candidates", grid_payload),
        ])),
        ("selection", OrderedDict([
            ("criterion", b3_grid.SELECTION_CRITERION),
            ("criterion_definition",
             "MAE computed independently for each of the 12 stations at that "
             "horizon, then averaged with equal weight"),
            ("selected_independently_for", "regime x horizon"),
            ("tie_break_order", b3_grid.TIE_BREAK_ORDER),
            ("comparison_tolerance", b3_grid.COMPARISON_TOLERANCE),
            ("micro_mae_used_for_selection", False),
        ])),
        ("regimes", b3_grid.REGIMES),
        ("r3_implemented", False),
        ("horizons", b3_grid.HORIZONS),
        ("determinism", OrderedDict([
            ("random_state", 42),
            ("deterministic", True),
            ("force_col_wise", True),
            ("subsample", 1.0),
            ("colsample_bytree", 1.0),
            ("bagging_randomness", False),
            ("feature_subsampling_randomness", False),
        ])),
        ("upstream", OrderedDict([
            ("foundation_manifest_sha256",
             sha256_of_file(ARTIFACTS / "v2_foundation_manifest.json")),
            ("phase1_manifest_sha256",
             sha256_of_file(ARTIFACTS / "v2_phase1_manifest.json")),
            ("phase2_manifest_sha256",
             sha256_of_file(ARTIFACTS / "v2_phase2_manifest.json")),
            ("sample_universe_manifest_sha256",
             sha256_of_file(ARTIFACTS / "sample_universe_manifest.json")),
            ("preprocessing_statistics_sha256",
             sha256_of_file(ARTIFACTS / "preprocessing_statistics.json")),
            ("windowing_config_sha256",
             sha256_of_file(PROJECT_ROOT / "configs" / "windowing.json")),
            ("baselines_config_sha256",
             sha256_of_file(PROJECT_ROOT / "configs" / "baselines.json")),
            ("preprocessing_config_sha256",
             sha256_of_file(PROJECT_ROOT / "configs" / "preprocessing.json")),
        ])),
        ("severe_threshold", SEVERE_THRESHOLD),
        ("severe_threshold_source", "training_pooled_p95"),
        ("residual_convention", "residual = actual - prediction"),
        ("validation_period", OrderedDict([
            ("start", PARTITION_BOUNDS["validation"][0].isoformat()),
            ("end", PARTITION_BOUNDS["validation"][1].isoformat()),
        ])),
        ("final_test_status", "sealed"),
        ("validation_metrics_seen", 0),
        ("test_metrics_seen", 0),
        ("test_predictions_generated", 0),
    ])


def build_receipt():
    if not FREEZE.is_file():
        raise SystemExit("refusing to open validation: no prevalidation freeze")
    return OrderedDict([
        ("study", "AirSense V2"),
        ("event", "development_validation_opening"),
        ("phase3_prevalidation_freeze", str(FREEZE.relative_to(PROJECT_ROOT))),
        ("phase3_prevalidation_freeze_sha256", sha256_of_file(FREEZE)),
        ("validation_interval", OrderedDict([
            ("start", PARTITION_BOUNDS["validation"][0].isoformat()),
            ("end", PARTITION_BOUNDS["validation"][1].isoformat()),
        ])),
        ("allowed_purposes", [
            "rolling historical PM2.5 context strictly before the forecast "
            "origin",
            "development scoring after a prediction has been produced",
        ]),
        ("forbidden_purposes", [
            "changing the Phase-2 sample universe",
            "changing preprocessing",
            "changing the severe threshold",
            "inventing features after seeing performance",
            "changing the primary metric",
            "any access to the locked final test",
        ]),
        ("b3_candidate_grid_already_frozen", True),
        ("b3_feature_set_already_frozen", True),
        ("primary_metric_already_frozen", True),
        ("severe_threshold_already_frozen", True),
        ("severe_threshold", SEVERE_THRESHOLD),
        ("final_test_status", "sealed"),
        ("note", "This receipt records the opening of development "
                 "validation. It deliberately contains no performance value."),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--receipt", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        write_json(FREEZE, build_freeze())
        print("wrote %s" % FREEZE.relative_to(PROJECT_ROOT))
        print("sha256 %s" % sha256_of_file(FREEZE))
    if args.receipt:
        write_json(RECEIPT, build_receipt())
        print("wrote %s" % RECEIPT.relative_to(PROJECT_ROOT))
        print("sha256 %s" % sha256_of_file(RECEIPT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
