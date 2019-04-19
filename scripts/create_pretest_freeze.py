"""Write the AirSense V1 pre-test development freeze.

Run at the end of Protocol Phase 9, **after** M3 has been selected,
validated and shown reproducible, and **before** Protocol Phase 10 ever
opens the 2014 target.

The freeze seals the complete pre-test development state in one artifact:
the raw dataset digest, the split and feature-schema digests, and for each
of M0-M3 its specification, selected hyperparameters, development manifest
digest and Phase-10 refit rule.

Phase 10 can begin by validating this single file and know that nothing in
the development state has changed.

**Immutability.** Once this freeze exists, Phase 10 may not change the
feature schema, the M0 strategy, the M1 specification, the M2 or M3
selected hyperparameters, the split boundaries, the primary metric, or the
test policy. If a later defect requires changing any of those, the planned
final-test evaluation must be invalidated and reset, and the deviation
documented, **before** the test is reopened. Nothing may be silently
repaired after seeing 2014.

This script reads no test features and no test target, and records no 2014
target statistic or metric. It writes no execution timestamp, so repeated
runs are byte-identical.

Usage:
    venv/bin/python scripts/create_pretest_freeze.py
"""

import io
import json
import os
import sys
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import sha256_of_file  # noqa: E402

RAW_RELATIVE = os.path.join("data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

FREEZE_RELATIVE = os.path.join("artifacts", "pretest_development_freeze.json")

MODEL_MANIFESTS = OrderedDict([
    ("M0", "m0_baseline_manifest.json"),
    ("M1", "m1_linear_regression_manifest.json"),
    ("M2", "m2_decision_tree_manifest.json"),
    ("M3", "m3_random_forest_manifest.json"),
])

REFIT_RULE = (
    "At Protocol Phase 10: combine the permitted 2010-2013 development "
    "data, apply the already-frozen Phase 5 feature policy unchanged, "
    "refit once with exactly the specification recorded here, generate "
    "2014 predictions once, open the locked 2014 target once, and "
    "calculate MAE/RMSE/R2 once. Do not retune after seeing 2014.")


class FreezeError(RuntimeError):
    """Raised when the freeze cannot be written or verified."""


def _digest(relative):
    path = os.path.join(PROJECT_ROOT, relative)
    if not os.path.isfile(path):
        raise FreezeError("%s not found" % relative)
    return sha256_of_file(path)


def build_freeze():
    freeze = OrderedDict()
    freeze["project"] = "AirSense"
    freeze["version"] = "V1"
    freeze["freeze_type"] = "pre_final_test"
    freeze["created_at_protocol_phase"] = (
        "Protocol Phase 9 - after M3 selection, validation and "
        "reproducibility verification")
    freeze["historical_cutoff"] = "2019-04-26"
    freeze["note"] = (
        "Seals the complete development state before Protocol Phase 10 "
        "opens the 2014 target. Contains no 2014 target statistic and no "
        "2014 metric. No execution timestamp is recorded so that repeated "
        "runs are byte-identical.")

    # -- test seal -------------------------------------------------------
    freeze["final_test_period"] = "2014"
    freeze["final_test_status"] = "sealed"
    freeze["test_opening_policy"] = "Protocol Phase 10 only"
    freeze["test_target_read_during_development"] = False
    freeze["test_features_used_during_development"] = False
    freeze["test_information_used_in_any_selection"] = False

    # -- periods ---------------------------------------------------------
    periods = OrderedDict()
    periods["development_train"] = "2010-01-01 00:00:00 to 2012-12-31 23:00:00"
    periods["development_validation"] = (
        "2013-01-01 00:00:00 to 2013-12-31 23:00:00")
    periods["final_refit_period"] = (
        "2010-01-01 00:00:00 to 2013-12-31 23:00:00")
    periods["locked_test_period"] = (
        "2014-01-01 00:00:00 to 2014-12-31 23:00:00")
    freeze["periods"] = periods

    # -- metrics ---------------------------------------------------------
    freeze["primary_final_metric"] = "MAE"
    freeze["reported_secondary_metrics"] = ["RMSE", "R2"]
    freeze["metric_definitions"] = OrderedDict([
        ("MAE", "mean(abs(y_true - y_pred))"),
        ("RMSE", "sqrt(mean((y_true - y_pred) ** 2))"),
        ("R2", "1 - SS_res / SS_tot"),
    ])

    # -- data and feature provenance -------------------------------------
    provenance = OrderedDict()
    provenance[RAW_RELATIVE] = _digest(RAW_RELATIVE)
    if provenance[RAW_RELATIVE] != EXPECTED_RAW_SHA256:
        raise FreezeError("raw dataset SHA-256 does not match the declared "
                          "value")
    for relative in (os.path.join("artifacts", "split_manifest.json"),
                     os.path.join("artifacts", "feature_schema.json"),
                     os.path.join("artifacts",
                                  "feature_row_manifest.csv")):
        provenance[relative] = _digest(relative)
    for filename in ("X_train.csv", "X_validation.csv", "y_train.csv",
                     "y_validation.csv"):
        relative = os.path.join("data", "processed", "features", filename)
        provenance[relative] = _digest(relative)
    freeze["provenance_sha256"] = provenance

    schema = json.load(io.open(
        os.path.join(PROJECT_ROOT, "artifacts", "feature_schema.json"),
        encoding="utf-8"))
    freeze["feature_count"] = int(schema["feature_count"])
    freeze["feature_names"] = list(schema["feature_names"])
    freeze["feature_schema_shared_by"] = ["M1", "M2", "M3"]

    # -- models ----------------------------------------------------------
    models = OrderedDict()
    manifest_digests = OrderedDict()
    for model_id, filename in MODEL_MANIFESTS.items():
        relative = os.path.join("artifacts", filename)
        digest = _digest(relative)
        manifest_digests[relative] = digest
        manifest = json.load(io.open(os.path.join(PROJECT_ROOT, relative),
                                     encoding="utf-8"))
        entry = OrderedDict()
        entry["model_id"] = model_id
        entry["development_manifest"] = relative
        entry["development_manifest_sha256"] = digest
        entry["fit_partition_during_development"] = "2010-2012"
        entry["evaluation_partition_during_development"] = "2013"
        if model_id == "M0":
            entry["strategy"] = manifest["strategy"]
            entry["model_type"] = manifest["model_type"]
            entry["development_baseline_constant"] = manifest[
                "baseline_constant"]
            entry["hyperparameters"] = None
            entry["predictors_used"] = []
            entry["refit_rule"] = (
                "Recompute the constant as the median of the combined "
                "2010-2013 development target. " + REFIT_RULE)
        elif model_id == "M1":
            entry["estimator"] = manifest["estimator"]
            entry["specification"] = OrderedDict([
                ("fit_intercept", manifest["fit_intercept"]),
                ("scaling", "none"),
                ("regularization", "none"),
            ])
            entry["feature_count"] = int(manifest["feature_count"])
            entry["hyperparameters"] = None
            entry["refit_rule"] = REFIT_RULE
        else:
            entry["estimator"] = manifest["estimator"]
            entry["selected_candidate_id"] = manifest[
                "selected_configuration"]["candidate_id"]
            entry["selected_hyperparameters"] = manifest[
                "selected_configuration"]
            entry["candidate_count"] = manifest["search_space"][
                "candidate_count"]
            entry["selection_metric"] = manifest["selection_metric"]
            entry["refit_rule"] = (
                "Keep the selected hyperparameters above frozen, including "
                "random_state. " + REFIT_RULE)
        entry["development_validation_metrics"] = OrderedDict([
            ("evaluation_period_label", "2013_validation"),
            ("mae", manifest["validation_metrics"]["mae"]),
            ("rmse", manifest["validation_metrics"]["rmse"]),
            ("r2", manifest["validation_metrics"]["r2"]),
            ("status", "development validation only - not final test"),
        ])
        entry["test_evaluation_status"] = "not_evaluated"
        entry["final_test_status"] = "not_evaluated"
        models[model_id] = entry
    freeze["models"] = models
    freeze["model_manifest_sha256"] = manifest_digests

    # -- immutability statement -------------------------------------------
    freeze["immutability"] = OrderedDict([
        ("statement", "Once this freeze exists, Protocol Phase 10 may not "
                      "change any item in frozen_items below."),
        ("frozen_items", [
            "feature schema (the ordered 43 columns)",
            "M0 strategy",
            "M1 specification",
            "M2 selected hyperparameters",
            "M3 selected hyperparameters",
            "split boundaries",
            "primary metric (MAE)",
            "test policy",
        ]),
        ("deviation_procedure",
         "If a later defect requires changing any frozen item, the planned "
         "final-test evaluation must be invalidated and reset, and the "
         "deviation explicitly documented, BEFORE the test is reopened. "
         "Nothing may be silently repaired after seeing 2014."),
    ])
    return freeze


def verify_freeze(freeze):
    """Re-verify every hash the freeze references, from disk."""
    problems = []
    checked = 0
    for relative, recorded in freeze["provenance_sha256"].items():
        observed = _digest(relative)
        checked += 1
        if observed != recorded:
            problems.append("%s: recorded %s, found %s"
                            % (relative, recorded, observed))
    for relative, recorded in freeze["model_manifest_sha256"].items():
        observed = _digest(relative)
        checked += 1
        if observed != recorded:
            problems.append("%s: recorded %s, found %s"
                            % (relative, recorded, observed))
    for model_id, entry in freeze["models"].items():
        observed = _digest(entry["development_manifest"])
        checked += 1
        if observed != entry["development_manifest_sha256"]:
            problems.append("%s manifest digest mismatch" % model_id)
    if problems:
        raise FreezeError("freeze is not self-consistent: %s"
                          % "; ".join(problems))
    return checked


def main():
    print("[AIRSENSE V1 - PRE-TEST DEVELOPMENT FREEZE]")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("")
    try:
        freeze = build_freeze()
        path = os.path.join(PROJECT_ROOT, FREEZE_RELATIVE)
        with io.open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(freeze, indent=2))
            handle.write("\n")
        print("Wrote %s" % FREEZE_RELATIVE)
        checked = verify_freeze(freeze)
        print("Verified %d referenced hashes from disk - all consistent"
              % checked)
    except FreezeError as exc:
        sys.stderr.write("\nFREEZE ABORTED: %s\n" % exc)
        return 1
    print("")
    print("final_test_period : %s" % freeze["final_test_period"])
    print("final_test_status : %s" % freeze["final_test_status"])
    print("models sealed     : %s" % ", ".join(freeze["models"].keys()))
    print("primary metric    : %s" % freeze["primary_final_metric"])
    print("")
    print("The 2014 final test remains sealed. Protocol Phase 10 has not "
          "started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
