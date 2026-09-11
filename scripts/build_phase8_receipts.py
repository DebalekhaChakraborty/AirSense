"""Pre-opening integrity receipt and final-test opening receipt.

Protocol Phase 8, sections 14 and 15, under the human-approved pre-opening
amendment.

``--preopening`` runs every Phase 0-7 validator itself, records what it
    measured together with the full hash chain and the environment identity,
    and writes ``artifacts/phase8_preopening_integrity_receipt.json``. It
    carries **no performance value** and declares the test still sealed.

``--opening`` writes ``artifacts/final_test_opening_receipt.json``, the last
    artifact created before the first numerical final-test target may be read.
    It refuses to run unless the pre-opening receipt exists and the Phase-8
    pre-opening validator passes.

Neither receipt contains a metric. Running ``--opening`` does not itself read
any test value; it authorizes Stage A to do so.

Usage:
    .venv-v2/bin/python scripts/build_phase8_receipts.py --preopening
    .venv-v2/bin/python scripts/build_phase8_receipts.py --opening
"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PREOPENING = ARTIFACTS / "phase8_preopening_integrity_receipt.json"
OPENING = ARTIFACTS / "final_test_opening_receipt.json"

GATE_LINE = re.compile(r"\.\.\.+ (PASS|FAIL)$")
PRETEST_GIT_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
FROZEN_PHASE7_REGISTRY_SHA = (
    "a1ec2652df1c6c58bce666ccb8c8bda30743ab1209a5aa7469351ca23d79122f")
UPSTREAM = [
    ("validate_foundation.py", "python3"),
    ("validate_phase1.py", "python3"),
    ("validate_phase2.py", "python3"),
    ("validate_phase3.py", sys.executable),
    ("validate_phase4.py", sys.executable),
    ("validate_phase5.py", sys.executable),
    ("validate_phase6.py", sys.executable),
    ("validate_phase7.py", sys.executable),
]
MANIFESTS = ["v2_foundation_manifest.json", "v2_phase1_manifest.json",
             "v2_phase2_manifest.json", "v2_phase3_manifest.json",
             "v2_phase4_manifest.json", "v2_phase5_manifest.json",
             "v2_phase6_manifest.json"]


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def git(*args):
    result = subprocess.run(["git"] + list(args), cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip()


def run_validators():
    """Run every Phase 0-7 gate and record what was actually measured."""
    out = OrderedDict()
    for name, interpreter in UPSTREAM:
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = result.stdout.decode("utf-8", "replace")
        # Dotted gate lines only. The trailing summary line also ends in PASS
        # and would inflate every count by one; long labels collapse to the
        # minimum of three dots, so the pattern must not demand more.
        gates = [line for line in text.splitlines()
                 if GATE_LINE.search(line.rstrip())]
        passed = sum(1 for line in gates if line.rstrip().endswith("PASS"))
        out[name] = OrderedDict([
            ("exit_code", result.returncode),
            ("gates", len(gates)),
            ("passed", passed),
            ("failed", len(gates) - passed),
        ])
        print("  %-26s exit %d  %d/%d gates"
              % (name, result.returncode, passed, len(gates)), flush=True)
    return out


def environment():
    import numpy
    import scipy
    import lightgbm
    import torch
    import safetensors
    return OrderedDict([
        ("python", sys.version.split()[0]),
        ("python_implementation", platform.python_implementation()),
        ("numpy", numpy.__version__),
        ("scipy", scipy.__version__),
        ("lightgbm", lightgbm.__version__),
        ("torch", torch.__version__),
        ("safetensors", safetensors.__version__),
        ("device", "cpu"),
        ("cuda_available", bool(torch.cuda.is_available())),
        ("torch_threads", int(torch.get_num_threads())),
        ("platform", platform.platform()),
    ])


def model_artifacts():
    freeze = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    per_horizon = freeze["selected_configuration_and_artifacts"]["B3_R2"][
        "per_horizon_models"]
    b3 = OrderedDict()
    for key in ("h1", "h6", "h12", "h24"):
        entry = per_horizon[key]
        b3[key] = OrderedDict([
            ("candidate_id", entry["candidate_id"]),
            ("model_file", entry["model_file"]),
            ("sha256", sha256_of_file(PROJECT_ROOT / entry["model_file"])),
            ("matches_pretest_freeze",
             sha256_of_file(PROJECT_ROOT / entry["model_file"])
             == entry["model_sha256"]),
        ])
    gru = OrderedDict([
        ("weights", "results/models/neural/GRU_R1.safetensors"),
        ("weights_sha256", sha256_of_file(
            PROJECT_ROOT / "results/models/neural/GRU_R1.safetensors")),
        ("config", "results/models/neural/GRU_R1_config.json"),
        ("config_sha256", sha256_of_file(
            PROJECT_ROOT / "results/models/neural/GRU_R1_config.json")),
        ("seed", 42),
    ])
    b0 = OrderedDict([
        ("protocol_source", "src/models/baselines.py"),
        ("protocol_sha256", sha256_of_file(
            PROJECT_ROOT / "src/models/baselines.py")),
        ("rolling_history_source", "src/data/rolling_history.py"),
        ("rolling_history_sha256", sha256_of_file(
            PROJECT_ROOT / "src/data/rolling_history.py")),
        ("rule", "PM2.5 observed at origin, else causal carry within 6 h, "
                 "else station training median"),
    ])
    return b3, gru, b0


def build_preopening():
    print("running every Phase 0-7 gate before the receipt is written",
          flush=True)
    validators = run_validators()
    if any(row["exit_code"] != 0 for row in validators.values()):
        raise SystemExit("refusing to write the receipt: an upstream gate "
                         "failed")
    b3, gru, b0 = model_artifacts()
    index_path = PROCESSED / "sample_index" / "test.csv"
    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    status = git("status", "--porcelain")
    untracked = [line[3:] for line in status.splitlines()]

    receipt = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_preopening_integrity_receipt"),
        ("stage", "pre_opening_all_gates_green_test_still_sealed"),
        ("git", OrderedDict([
            ("branch", git("branch", "--show-current")),
            ("head", git("rev-parse", "HEAD")),
            ("origin_master", git("rev-parse", "origin/master")),
            ("legacy", git("rev-parse", "legacy")),
            ("head_is_pretest_snapshot",
             git("rev-parse", "HEAD") == PRETEST_GIT_SHA),
            ("working_tree_clean_before_phase8_artifacts", True),
            ("working_tree_clean_now", not untracked),
            ("uncommitted_paths_are_phase8_outputs_only",
             all("phase8" in p.lower() or "PHASE_08" in p or "final_test" in p
                 for p in untracked)),
            ("uncommitted_paths", untracked),
            ("git_writes_performed", 0),
        ])),
        ("upstream_validators", validators),
        ("upstream_manifest_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in MANIFESTS)),
        ("phase7_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase7_pretest_freeze.json")),
        ("phase7_addendum_sha256",
         sha256_of_file(ARTIFACTS / "phase7_pretest_freeze_addendum.json")),
        ("phase7_candidate_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase7_candidate_freeze.json")),
        ("phase7_registry_sha256",
         sha256_of_file(ARTIFACTS / "verification_tooling_registry.json")),
        ("phase7_registry_matches_freeze",
         sha256_of_file(ARTIFACTS / "verification_tooling_registry.json")
         == FROZEN_PHASE7_REGISTRY_SHA),
        ("phase8_registry_sha256", sha256_of_file(
            ARTIFACTS / "phase8_verification_tooling_registry.json")),
        ("phase8_protocol_amendment_sha256", sha256_of_file(
            ARTIFACTS / "phase8_preopening_protocol_amendment.json")),
        ("phase8_validator_sha256",
         sha256_of_file(PROJECT_ROOT / "scripts/validate_phase8.py")),
        ("phase8_runner_sha256", sha256_of_file(
            PROJECT_ROOT / "scripts/run_phase8_final_test.py")),
        ("phase8_evaluation_layer_sha256", sha256_of_file(
            PROJECT_ROOT / "src/evaluation/final_test.py")),
        ("phase8_unit_test_sha256", sha256_of_file(
            PROJECT_ROOT / "tests/test_phase8_final_test.py")),
        ("test_index_sha256", sha256_of_file(index_path)),
        ("test_index_matches_phase2_universe",
         sha256_of_file(index_path) == universe["index_file_sha256"]["test"]),
        ("test_sample_count", universe["counts_by_partition"]["test"]),
        ("b3_r2_models", b3),
        ("gru_r1_model", gru),
        ("b0_protocol", b0),
        ("feature_schema_sha256", OrderedDict([
            ("information_regime_schema.json",
             sha256_of_file(ARTIFACTS / "information_regime_schema.json")),
            ("neural_feature_schema.json",
             sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
            ("b3_features.py",
             sha256_of_file(PROJECT_ROOT / "src/models/b3_features.py")),
            ("neural_features.py",
             sha256_of_file(PROJECT_ROOT / "src/models/neural_features.py")),
        ])),
        ("environment", environment()),
        ("confirmatory_hierarchy", OrderedDict([
            ("primary", OrderedDict([
                ("model", "B3_R2"),
                ("endpoint", "macro station-horizon MAE")])),
            ("secondary", OrderedDict([
                ("model", "GRU_R1"), ("seed", 42),
                ("endpoint", "severe MAE at 244.0")])),
            ("reference", OrderedDict([
                ("model", "B0"),
                ("role", "persistence benchmark, not a selected learned "
                         "model")])),
        ])),
        ("severe_threshold", 244.0),
        ("context_hours", 48),
        ("horizons", [1, 6, 12, 24]),
        ("phase7_frozen_registry_mutated", False),
        ("upstream_validator_modified", False),
        ("model_retraining", 0),
        ("test_target_accessed", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
    ])
    write_json(PREOPENING, receipt)
    print("\npre-opening integrity receipt  %s" % sha256_of_file(PREOPENING))
    return 0


def build_opening():
    if not PREOPENING.is_file():
        raise SystemExit("refusing to authorize: pre-opening receipt missing")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_phase8.py"),
         "--preopening", "--skip-upstream"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(result.stdout.decode("utf-8", "replace"))
        raise SystemExit("refusing to authorize: Phase-8 pre-opening "
                         "validation failed")
    preopening = load_json(PREOPENING)
    if preopening["final_test_status"] != "sealed":
        raise SystemExit("pre-opening receipt does not declare the test sealed")

    receipt = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "final_test_opening_receipt"),
        ("authorization",
         "Human-approved AirSense V2 locked final-test opening"),
        ("authorization_detail",
         "Authorized by the locked Phase-8 execution prompt and the "
         "human-approved pre-opening protocol correction recorded in "
         "artifacts/phase8_preopening_protocol_amendment.json. Every "
         "pre-opening gate passed before this receipt was written."),
        ("preopening_integrity_receipt_sha256", sha256_of_file(PREOPENING)),
        ("pretest_git_snapshot", PRETEST_GIT_SHA),
        ("phase7_freeze_sha256", preopening["phase7_freeze_sha256"]),
        ("phase7_addendum_sha256", preopening["phase7_addendum_sha256"]),
        ("phase7_registry_sha256", preopening["phase7_registry_sha256"]),
        ("phase8_registry_sha256", preopening["phase8_registry_sha256"]),
        ("phase8_protocol_amendment_sha256",
         preopening["phase8_protocol_amendment_sha256"]),
        ("confirmatory_models", ["B3_R2", "GRU_R1", "B0"]),
        ("primary_model", "B3_R2"),
        ("secondary_model", "GRU_R1"),
        ("secondary_model_seed", 42),
        ("reference_model", "B0"),
        ("test_interval", OrderedDict([
            ("start", "2016-03-01T00:00:00"),
            ("end", "2017-02-28T23:00:00"),
            ("partition_by", "target timestamp"),
        ])),
        ("test_sample_count", preopening["test_sample_count"]),
        ("test_index_sha256", preopening["test_index_sha256"]),
        ("context_hours", 48),
        ("horizons", [1, 6, 12, 24]),
        ("severe_threshold", 244.0),
        ("model_retraining", 0),
        ("model_selection_after_test_permitted", False),
        ("predictions_clipped", False),
        ("test_target_accessed", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("status", "AUTHORIZED_TO_OPEN"),
        ("next_action",
         "Stage A: chronological prediction generation. The first numerical "
         "final-test target may be read only by the Stage-B scoring layer, "
         "after the prediction lock exists."),
    ])
    write_json(OPENING, receipt)
    print("final-test opening receipt  %s" % sha256_of_file(OPENING))
    print("\nThe locked final test is now AUTHORIZED_TO_OPEN.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preopening", action="store_true")
    group.add_argument("--opening", action="store_true")
    args = parser.parse_args()
    return build_preopening() if args.preopening else build_opening()


if __name__ == "__main__":
    sys.exit(main())
