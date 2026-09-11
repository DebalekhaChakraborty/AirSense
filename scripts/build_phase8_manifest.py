"""Phase-8 manifest for the AirSense V2 locked final test.

Protocol Phase 8, section 46. Run last, after the primary-results lock, the
tables, the figures and independent verification.

The manifest carries no volatile timestamp: every field is either a digest, a
count, or a frozen protocol constant, so repeated runs over unchanged evidence
produce byte-identical output.

Usage:
    .venv-v2/bin/python scripts/build_phase8_manifest.py
"""

import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
TABLES = ARTIFACTS / "phase8_final_test_tables"
PRED_DIR = ARTIFACTS / "phase8_predictions"
FIGURES = PROJECT_ROOT / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"

PRETEST_GIT_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
MANIFESTS = ["v2_foundation_manifest.json", "v2_phase1_manifest.json",
             "v2_phase2_manifest.json", "v2_phase3_manifest.json",
             "v2_phase4_manifest.json", "v2_phase5_manifest.json",
             "v2_phase6_manifest.json"]
FREEZES = ["phase7_candidate_freeze.json", "phase7_pretest_freeze.json",
           "phase7_pretest_freeze_addendum.json"]
MODELS = ["B3_R2", "GRU_R1", "B0"]


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def git(*args):
    result = subprocess.run(["git"] + list(args), cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def main():
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    preopening = load_json(ARTIFACTS
                           / "phase8_preopening_integrity_receipt.json")

    figure_hashes = OrderedDict()
    figure_manifest = ARTIFACTS / "phase8_figure_manifest.json"
    if figure_manifest.is_file():
        figure_hashes = load_json(figure_manifest)["figure_sha256"]

    manifest = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "v2_phase8_manifest"),
        ("phase", "phase_8_locked_final_test"),
        ("branch", "master"),
        ("pretest_git_snapshot", PRETEST_GIT_SHA),
        ("head_at_evaluation", git("rev-parse", "HEAD")),
        ("legacy_unchanged", git("rev-parse", "legacy")
         == "16c9030cf74508b620cc6d28f90346aa379f29cd"),
        ("protocol_amendment", OrderedDict([
            ("artifact", "artifacts/phase8_preopening_protocol_amendment.json"),
            ("sha256", sha256_of_file(
                ARTIFACTS / "phase8_preopening_protocol_amendment.json")),
            ("scientific_protocol_changed", False),
            ("storage_namespace_corrected", True),
        ])),
        ("upstream_manifest_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in MANIFESTS)),
        ("phase7_freeze_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in FREEZES)),
        ("verification_tooling_registry_sha256", sha256_of_file(
            ARTIFACTS / "verification_tooling_registry.json")),
        ("phase7_registry_mutated_by_phase8", False),
        ("phase8_verification_tooling_registry_sha256", sha256_of_file(
            ARTIFACTS / "phase8_verification_tooling_registry.json")),
        ("phase8_validator_sha256", sha256_of_file(
            PROJECT_ROOT / "scripts" / "validate_phase8.py")),
        ("phase8_code_sha256", OrderedDict([
            ("src/evaluation/final_test.py", sha256_of_file(
                PROJECT_ROOT / "src/evaluation/final_test.py")),
            ("scripts/run_phase8_final_test.py", sha256_of_file(
                PROJECT_ROOT / "scripts/run_phase8_final_test.py")),
            ("scripts/build_phase8_receipts.py", sha256_of_file(
                PROJECT_ROOT / "scripts/build_phase8_receipts.py")),
            ("scripts/verify_phase8_independently.py", sha256_of_file(
                PROJECT_ROOT / "scripts/verify_phase8_independently.py")),
            ("scripts/build_phase8_figures.py", sha256_of_file(
                PROJECT_ROOT / "scripts/build_phase8_figures.py")),
            ("tests/test_phase8_final_test.py", sha256_of_file(
                PROJECT_ROOT / "tests/test_phase8_final_test.py")),
        ])),
        ("preopening_integrity_receipt_sha256", sha256_of_file(
            ARTIFACTS / "phase8_preopening_integrity_receipt.json")),
        ("opening_receipt_sha256", sha256_of_file(
            ARTIFACTS / "final_test_opening_receipt.json")),
        ("test_index_sha256", sha256_of_file(
            PROCESSED / "sample_index" / "test.csv")),
        ("model_artifact_sha256", lock["model_artifact_sha256"]),
        ("prediction_sha256", OrderedDict(
            (name, lock["models"][name]["sha256"]) for name in MODELS)),
        ("prediction_file", OrderedDict(
            (name, lock["models"][name]["prediction_file"])
            for name in MODELS)),
        ("prediction_lock_sha256", sha256_of_file(
            ARTIFACTS / "phase8_prediction_lock.json")),
        ("metric_table_sha256", results["metric_table_sha256"]),
        ("confirmatory_endpoints_sha256", sha256_of_file(
            TABLES / "confirmatory_endpoints.json")),
        ("primary_results_lock_sha256", sha256_of_file(
            ARTIFACTS / "phase8_primary_results_lock.json")),
        ("final_test_metrics_sha256", sha256_of_file(
            ARTIFACTS / "phase8_final_test_metrics.json")),
        ("figure_sha256", figure_hashes),
        ("test_interval", OrderedDict([
            ("start", "2016-03-01T00:00:00"),
            ("end", "2017-02-28T23:00:00"),
            ("partition_by", "target timestamp"),
        ])),
        ("test_sample_count", lock["test_sample_count"]),
        ("severe_sample_count", results["secondary_severe_n"]),
        ("context_hours", 48),
        ("horizons", [1, 6, 12, 24]),
        ("station_horizon_cells", 48),
        ("severe_threshold", 244.0),
        ("severe_rule", "actual > 244.0"),
        ("residual_convention", "actual - prediction"),
        ("primary_model", "B3_R2"),
        ("primary_endpoint", "macro_station_horizon_MAE"),
        ("secondary_model", "GRU_R1"),
        ("secondary_model_seed", 42),
        ("secondary_endpoint", "severe_MAE_gt_244.0"),
        ("reference_model", "B0"),
        ("models_evaluated", 3),
        ("model_retraining", 0),
        ("model_refitting", 0),
        ("train_validation_refit_performed", False),
        ("model_selection_after_test", False),
        ("predictions_clipped", False),
        ("future_target_access_violations", 0),
        ("test_predictions_generated",
         sum(lock["models"][name]["length"] for name in MODELS)),
        ("test_metrics_seen", True),
        ("post_test_exploration_performed", False),
        ("environment", preopening["environment"]),
        ("final_test_status", "evaluated"),
        ("next_phase",
         "post-test error analysis, clearly labelled as post-test and held "
         "for human review before it begins"),
    ])
    path = ARTIFACTS / "v2_phase8_manifest.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("v2_phase8_manifest.json  %s" % sha256_of_file(path))
    print("total test predictions   %d"
          % manifest["test_predictions_generated"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
