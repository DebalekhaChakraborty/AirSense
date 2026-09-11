"""Independent verification of the AirSense V2 locked final test.

Protocol Phase 8. **This source was finalized before the sealed test was
numerically opened** and is pinned in
``artifacts/phase8_verification_tooling_registry.json``.

Two modes:

``--preopening``
    Everything that must hold *before* the first final-test target is read.
    Run as the last gate before the opening receipt is written.

(default)
    The full gate set, including the post-opening evidence chain: prediction
    lock before scoring, alignment, the reconstructed confirmatory endpoints,
    and the absence of any excluded model or post-test exploration.

Exits non-zero on any failure.

Usage:
    .venv-v2/bin/python scripts/validate_phase8.py --preopening
    .venv-v2/bin/python scripts/validate_phase8.py
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED_DIR = ARTIFACTS / "phase8_predictions"
TABLES = ARTIFACTS / "phase8_final_test_tables"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"

PRETEST_GIT_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LEGACY_SHA = "16c9030cf74508b620cc6d28f90346aa379f29cd"
FROZEN_PHASE7_REGISTRY_SHA = (
    "a1ec2652df1c6c58bce666ccb8c8bda30743ab1209a5aa7469351ca23d79122f")
PRETEST_FREEZE_SHA = (
    "78a0e36080e52941694581eef3280b42f654f00c8865dc306deaac369ccade60")
ADDENDUM_SHA = (
    "6a1342b7a70eb6b238df95ae5c62950db258e23f70d7feb6c3245de7013708e2")
CANDIDATE_FREEZE_SHA = (
    "0b6b0518182503425aff92a83a8f5d0f9f3b4721e8e667e052c87785339520c8")

EXPECTED_TEST_SAMPLES = 411012
SEVERE_THRESHOLD = 244.0
CONFIRMATORY = ["B3_R2", "GRU_R1", "B0"]
EXCLUDED = ["TCN_R1", "SA_R3", "iTransformer_R1", "iTransformer_R2", "SA_R2",
            "B1", "B2", "B3_R0", "B3_R1", "GRU_R0", "GRU_R2", "TCN_R0",
            "TCN_R2", "Chronos", "TimesFM", "Moirai"]
UPSTREAM_VALIDATORS = [
    ("validate_foundation.py", "python3"),
    ("validate_phase1.py", "python3"),
    ("validate_phase2.py", "python3"),
    ("validate_phase3.py", sys.executable),
    ("validate_phase4.py", sys.executable),
    ("validate_phase5.py", sys.executable),
    ("validate_phase6.py", sys.executable),
    ("validate_phase7.py", sys.executable),
]
AUTHORIZED_PREFIXES = (
    "artifacts/phase8_", "artifacts/v2_phase8_manifest.json",
    "artifacts/final_test_opening_receipt.json", "figures/phase8_",
    "docs/PHASE_08_", "scripts/validate_phase8.py",
    "scripts/run_phase8_final_test.py", "src/evaluation/final_test.py",
    "tests/test_phase8_final_test.py",
)


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 66
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


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


# ---------------------------------------------------------------------------
# A-E: the frozen Phase-7 state and the append-only Phase-8 registry
# ---------------------------------------------------------------------------

def frozen_state_checks(report):
    registry_path = ARTIFACTS / "verification_tooling_registry.json"
    live = sha256_of_file(registry_path)
    report.add("Phase-7 tooling registry is byte-identical to its freeze",
               live == FROZEN_PHASE7_REGISTRY_SHA,
               "live %s" % live)

    freeze = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    report.add("Phase-7 freeze still hashes that exact historical registry",
               freeze["evidence_sha256"].get(
                   "artifacts/verification_tooling_registry.json")
               == FROZEN_PHASE7_REGISTRY_SHA)
    report.add("Phase-7 pre-test freeze unchanged",
               sha256_of_file(ARTIFACTS / "phase7_pretest_freeze.json")
               == PRETEST_FREEZE_SHA)
    report.add("Phase-7 addendum unchanged",
               sha256_of_file(ARTIFACTS / "phase7_pretest_freeze_addendum"
                              ".json") == ADDENDUM_SHA)
    report.add("Phase-7 candidate freeze unchanged",
               sha256_of_file(ARTIFACTS / "phase7_candidate_freeze.json")
               == CANDIDATE_FREEZE_SHA)

    phase8 = load_json(ARTIFACTS / "phase8_verification_tooling_registry.json")
    report.add("Phase-8 registry names the frozen Phase-7 registry as parent",
               phase8["parent_registry"]
               == "artifacts/verification_tooling_registry.json"
               and phase8["parent_registry_sha256"]
               == FROZEN_PHASE7_REGISTRY_SHA)
    report.add("Phase-8 registry declares no upstream mutation",
               phase8["upstream_validator_sources_changed"] is False
               and phase8["phase7_registry_changed"] is False
               and phase8["phase7_freeze_changed"] is False
               and phase8["scientific_assertions_changed"] is False
               and phase8["gates_removed"] == 0
               and phase8["gates_disabled"] == 0)
    pinned = phase8["validators"]
    report.add("Phase-8 registry pins validate_phase7.py",
               "scripts/validate_phase7.py" in pinned
               and pinned["scripts/validate_phase7.py"]["sha256"]
               == sha256_of_file(PROJECT_ROOT / "scripts/validate_phase7.py"))
    report.add("Phase-8 registry pins validate_phase8.py",
               "scripts/validate_phase8.py" in pinned
               and pinned["scripts/validate_phase8.py"]["sha256"]
               == sha256_of_file(PROJECT_ROOT / "scripts/validate_phase8.py"))
    drift = [name for name, entry in pinned.items()
             if sha256_of_file(PROJECT_ROOT / name) != entry["sha256"]]
    report.add("Every validator matches the Phase-8 registry", not drift,
               "changed: %s" % drift)

    # E: upstream sources byte-identical to their pre-Phase-8 state, judged
    # against the frozen Phase-7 registry rather than anything Phase 8 wrote.
    parent = load_json(ARTIFACTS / "verification_tooling_registry.json")
    changed = [name for name, entry in parent["validators"].items()
               if sha256_of_file(PROJECT_ROOT / name) != entry["sha256"]]
    report.add("Phase 0-6 validator sources unchanged by Phase 8", not changed,
               "changed: %s" % changed)
    report.add("validate_phase7.py unchanged since it was registered",
               pinned["scripts/validate_phase7.py"]["sha256"]
               == sha256_of_file(PROJECT_ROOT / "scripts/validate_phase7.py"))

    amendment = load_json(ARTIFACTS
                          / "phase8_preopening_protocol_amendment.json")
    report.add("Protocol amendment changes no scientific assertion",
               amendment["scientific_protocol_changed"] is False
               and amendment["confirmatory_hierarchy_changed"] is False
               and amendment["primary_endpoint_changed"] is False
               and amendment["secondary_endpoint_changed"] is False
               and amendment["sample_universe_changed"] is False
               and amendment["model_changed"] is False
               and amendment["test_opened_before_amendment"] is False)


def git_checks(report):
    report.add("Branch is master", git("branch", "--show-current") == "master")
    report.add("HEAD is the pre-test evidence snapshot",
               git("rev-parse", "HEAD") == PRETEST_GIT_SHA,
               git("rev-parse", "HEAD"))
    report.add("origin/master matches the pre-test snapshot",
               git("rev-parse", "origin/master") == PRETEST_GIT_SHA)
    report.add("V1 legacy is untouched",
               git("rev-parse", "legacy") == LEGACY_SHA)


def upstream_checks(report):
    for name, interpreter in UPSTREAM_VALIDATORS:
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)


def hierarchy_checks(report):
    addendum = load_json(ARTIFACTS / "phase7_pretest_freeze_addendum.json")
    hierarchy = addendum["final_test_hierarchy"]
    report.add("Confirmatory hierarchy is the frozen three, in order",
               [row["model"] for row in hierarchy] == CONFIRMATORY)
    report.add("Primary endpoint unchanged",
               hierarchy[0]["prespecified_endpoint"]
               == "macro station-horizon MAE")
    report.add("Secondary endpoint unchanged at threshold 244.0",
               hierarchy[1]["severe_threshold"] == SEVERE_THRESHOLD)
    report.add("Promotion after test access remains forbidden",
               addendum["excluded_from_confirmatory_set"][
                   "promotion_after_test_access_permitted"] is False
               and addendum["multiple_comparison_discipline"][
                   "model_selection_after_observing_test_results_permitted"]
               is False)


def universe_checks(report):
    index_path = PROCESSED / "sample_index" / "test.csv"
    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    report.add("Canonical test index unchanged since Phase 2",
               sha256_of_file(index_path)
               == universe["index_file_sha256"]["test"])
    report.add("Test sample count is the frozen universe",
               universe["counts_by_partition"]["test"]
               == EXPECTED_TEST_SAMPLES)
    with open(str(index_path), newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    forbidden = [c for c in header
                 if c.strip().lower() in ("pm2.5", "pm25", "target",
                                          "target_value", "value", "y")]
    report.add("Test index carries no target value column", not forbidden,
               "columns: %s" % forbidden)


def model_artifact_checks(report):
    freeze = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    per_horizon = freeze["selected_configuration_and_artifacts"]["B3_R2"][
        "per_horizon_models"]
    drift = [key for key, entry in per_horizon.items()
             if sha256_of_file(PROJECT_ROOT / entry["model_file"])
             != entry["model_sha256"]]
    report.add("B3_R2 uses the four exact frozen boosters", not drift,
               "drifted: %s" % drift)
    report.add("B3_R2 booster set is exactly four horizons",
               sorted(per_horizon) == ["h1", "h12", "h24", "h6"])

    config = load_json(PROJECT_ROOT
                       / "results/models/neural/GRU_R1_config.json")
    report.add("GRU_R1 is the exact seed-42 R1 artifact",
               config["model"] == "GRU_R1" and config["regime"] == "R1"
               and int(config["training"]["seed"]) == 42
               and config["context_hours"] == 48
               and config["output_constrained"] is False)
    phase4 = load_json(ARTIFACTS / "v2_phase4_manifest.json")
    weights = phase4["model_config_and_weight_sha256"]
    report.add("GRU_R1 weights match the Phase-4 manifest",
               sha256_of_file(PROJECT_ROOT
                              / "results/models/neural/GRU_R1.safetensors")
               == weights["GRU_R1.safetensors"]
               and sha256_of_file(PROJECT_ROOT
                                  / "results/models/neural/GRU_R1_config.json")
               == weights["GRU_R1_config.json"])
    report.add("No seed other than 42 has a GRU_R1 weight artifact",
               not list((PROJECT_ROOT / "results/models/robustness")
                        .glob("*GRU_R1*seed*.safetensors")))
    report.add("B0 rule source is the frozen Phase-3 baseline module",
               (PROJECT_ROOT / "src/models/baselines.py").is_file()
               and "b0_causal_persistence" in (
                   PROJECT_ROOT / "src/models/baselines.py").read_text())


def namespace_checks(report):
    report.add("No results/final_test directory exists",
               not (PROJECT_ROOT / "results" / "final_test").exists())
    stray = []
    for base in ("results", "artifacts", "figures", "docs"):
        root = PROJECT_ROOT / base
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(PROJECT_ROOT))
            if "final_test" not in relative and "phase8" not in relative.lower():
                continue
            if not relative.startswith(AUTHORIZED_PREFIXES):
                stray.append(relative)
    report.add("Final-test outputs live only in authorized Phase-8 paths",
               not stray, "stray: %s" % stray[:5])


def excluded_model_checks(report):
    offenders = []
    for base in (PRED_DIR, TABLES):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            for name in EXCLUDED:
                if path.name.startswith(name + "."):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
    report.add("No excluded model has a final-test prediction artifact",
               not offenders, "offenders: %s" % offenders[:5])


def preopening_only(report):
    report.add("No final-test prediction array exists yet",
               not PRED_DIR.exists() or not list(PRED_DIR.glob("*.npy")))
    report.add("No prediction lock exists yet",
               not (ARTIFACTS / "phase8_prediction_lock.json").is_file())
    report.add("No primary-results lock exists yet",
               not (ARTIFACTS / "phase8_primary_results_lock.json").is_file())
    report.add("No final-test metric table exists yet",
               not TABLES.exists() or not list(TABLES.glob("*")))
    receipt_path = ARTIFACTS / "phase8_preopening_integrity_receipt.json"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        report.add("Pre-opening receipt still declares the test sealed",
                   receipt["test_target_accessed"] is False
                   and receipt["test_predictions_generated"] == 0
                   and receipt["test_metrics_seen"] == 0
                   and receipt["final_test_status"] == "sealed")


# ---------------------------------------------------------------------------
# Post-opening evidence chain
# ---------------------------------------------------------------------------

def opening_checks(report):
    opening = load_json(ARTIFACTS / "final_test_opening_receipt.json")
    preopening = load_json(ARTIFACTS
                           / "phase8_preopening_integrity_receipt.json")
    report.add("Opening receipt carries the pre-opening integrity hash",
               opening["preopening_integrity_receipt_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "phase8_preopening_integrity_receipt.json"))
    report.add("Opening receipt pins the pre-test git snapshot",
               opening["pretest_git_snapshot"] == PRETEST_GIT_SHA)
    report.add("Opening receipt names exactly the frozen three",
               opening["confirmatory_models"] == CONFIRMATORY)
    report.add("Opening receipt recorded zero results at authorization",
               opening["test_target_accessed"] is False
               and opening["test_predictions_generated"] == 0
               and opening["test_metrics_seen"] == 0
               and opening["status"] == "AUTHORIZED_TO_OPEN")
    report.add("Pre-opening receipt recorded the test as still sealed",
               preopening["final_test_status"] == "sealed"
               and preopening["phase7_frozen_registry_mutated"] is False
               and preopening["upstream_validator_modified"] is False)


def prediction_checks(report):
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    report.add("Prediction lock certifies predictions preceded scoring",
               lock["prediction_generation_complete"] is True
               and lock["predictions_frozen_before_scoring"] is True
               and lock["test_metrics_seen"] == 0)
    report.add("Prediction lock carries the opening receipt hash",
               lock["opening_receipt_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "final_test_opening_receipt.json"))
    report.add("Future-target access violations are zero",
               lock["future_target_access_violations"] == 0
               and lock["reveal_ledger"][
                   "future_target_access_violations"] == 0)
    report.add("No model was retrained", lock["model_retraining"] == 0)
    report.add("Predictions were not clipped",
               lock["predictions_clipped"] is False)

    present = sorted(p.stem for p in PRED_DIR.glob("*.npy")
                     if not p.stem.endswith("_source"))
    report.add("Exactly the three confirmatory prediction arrays exist",
               present == sorted(CONFIRMATORY), "found: %s" % present)
    for name in CONFIRMATORY:
        entry = lock["models"][name]
        path = PROJECT_ROOT / entry["prediction_file"]
        live = sha256_of_file(path)
        array = np.load(str(path))
        report.add("%s prediction array unchanged since the lock" % name,
                   live == entry["sha256"])
        report.add("%s predicted every canonical sample" % name,
                   array.size == EXPECTED_TEST_SAMPLES
                   and np.isfinite(array).all(),
                   "length %d" % array.size)


def alignment_and_metric_checks(report):
    from src.evaluation import metrics as M
    from src.evaluation.final_test import (
        FinalTestTargetOracle, load_test_index)
    from src.data.preprocessing import STATIONS

    index = load_test_index(PROCESSED)
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    report.add("Prediction lock pins the canonical test index",
               lock["test_index_sha256"]
               == sha256_of_file(PROCESSED / "sample_index" / "test.csv")
               and lock["test_sample_count"] == EXPECTED_TEST_SAMPLES)

    raw_dir = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
    stored = OrderedDict()
    for name in CONFIRMATORY:
        stored[name] = np.load(str(
            PROJECT_ROOT / lock["models"][name]["prediction_file"])
        ).astype(np.float64)

    actual = np.empty(index["target_index"].size, dtype=np.float64)
    for station in STATIONS:
        rows = np.flatnonzero(index["station"] == station)
        oracle = FinalTestTargetOracle(
            station, next(raw_dir.glob("PRSA_Data_%s_*.csv" % station)))
        actual[rows] = oracle.actuals(index["target_index"][rows],
                                      stored["B3_R2"][rows])

    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    cells = M.cell_metrics(actual, stored["B3_R2"], index["station"],
                           index["horizon"])
    report.add("Primary metric reconciles from exactly 48 cells",
               len(cells) == 48)
    recomputed = M.macro_from_cells(cells, "mae")
    report.add("B3_R2 primary endpoint reproduces independently",
               abs(recomputed - results["primary_value"]) < 1e-9,
               "%.9f vs %.9f" % (recomputed, results["primary_value"]))

    severe = actual > SEVERE_THRESHOLD
    report.add("Severe rule is strictly greater than 244.0",
               results["severe_threshold"] == SEVERE_THRESHOLD)
    gru_severe = float(np.abs(actual[severe]
                              - stored["GRU_R1"][severe]).mean())
    report.add("GRU_R1 secondary endpoint reproduces independently",
               abs(gru_severe - results["secondary_value"]) < 1e-9,
               "%.9f vs %.9f" % (gru_severe, results["secondary_value"]))
    report.add("Severe sample count reconciles",
               int(severe.sum()) == results["secondary_severe_n"],
               "%d vs %d" % (int(severe.sum()),
                             results["secondary_severe_n"]))
    b0_macro = M.macro_from_cells(
        M.cell_metrics(actual, stored["B0"], index["station"],
                       index["horizon"]), "mae")
    report.add("B0 reference endpoints reproduce independently",
               abs(b0_macro - results["reference_primary_value"]) < 1e-9
               and abs(float(np.abs(actual[severe] - stored["B0"][severe])
                             .mean())
                       - results["reference_severe_value"]) < 1e-9)
    report.add("Residual convention is actual minus prediction",
               float(M.residuals([10.0], [4.0])[0]) == 6.0)

    report.add("Results lock records no post-test selection",
               results["model_selection_after_test"] is False
               and results["prediction_changes_after_scoring"] is False
               and results["post_test_exploration_performed"] is False)
    report.add("Final-test status is evaluated",
               results["final_test_status"] == "evaluated")
    drift = [name for name, digest in results["metric_table_sha256"].items()
             if sha256_of_file(TABLES / name) != digest]
    report.add("Metric tables unchanged since the results lock", not drift,
               "drifted: %s" % drift)


def post_test_exploration_checks(report):
    banned = ("residual_acf", "episode", "peak_onset", "error_case",
              "attention_diagnostic", "clustering", "weather_conditioned",
              "season_conditioned")
    offenders = []
    for base in ("artifacts", "figures", "results"):
        root = PROJECT_ROOT / base
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "phase8" in path.name.lower():
                for token in banned:
                    if token in path.name.lower():
                        offenders.append(str(path.relative_to(PROJECT_ROOT)))
    report.add("No post-test exploratory artifact exists", not offenders,
               "offenders: %s" % offenders[:5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preopening", action="store_true")
    parser.add_argument("--skip-upstream", action="store_true",
                        help="for reruns only; upstream gates still required")
    args = parser.parse_args()

    report = Report()
    git_checks(report)
    frozen_state_checks(report)
    hierarchy_checks(report)
    universe_checks(report)
    model_artifact_checks(report)
    namespace_checks(report)
    excluded_model_checks(report)
    if not args.skip_upstream:
        upstream_checks(report)

    if args.preopening:
        preopening_only(report)
    else:
        opening_checks(report)
        prediction_checks(report)
        alignment_and_metric_checks(report)
        post_test_exploration_checks(report)

    report.render()
    stage = "PRE-OPENING" if args.preopening else "FULL"
    if report.failed():
        print("\nPHASE 8 %s VALIDATION FAILED: %d gate(s)"
              % (stage, len(report.failed())))
        return 1
    print("\nPHASE 8 %s VALIDATION PASSED" % stage)
    if args.preopening:
        print("Final test: SEALED; test predictions: 0; test metrics: 0")
    else:
        print("Final test: EVALUATED under the frozen confirmatory hierarchy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
