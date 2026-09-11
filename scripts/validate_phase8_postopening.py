"""Post-opening final-state validation for AirSense V2.

Protocol Phase 8, human-approved post-opening validation correction.

Phase 3 carries a historical seal gate - "No test prediction or test metric
artifact exists" - which asserted that the locked final test had never been
opened. That assertion was true through Phase 7 and is **intentionally false**
after the authorized Phase-8 one-way opening: the opening receipt is itself one
of the artifacts that trips it.

The historical validator is therefore preserved untouched, and this validator
checks the state that actually exists now. It is deliberately strict about the
distinction:

* Phases 0-2 must still pass outright;
* Phase 3 must fail on **exactly** the seal gate and on nothing else, triggered
  by **exactly** the two authorized Phase-8 artifacts;
* Phases 4-7 must show no independently owned failure - their only tolerated
  failures are the nested upstream cascade from Phase 3;
* every non-upstream Phase-8 gate from the pre-opening-frozen validator must
  still pass;
* the confirmatory values must reconcile independently against the frozen
  result artifacts, which are the numerical authority.

Nothing here trains, predicts, rewrites or recomputes through model code. It
reads frozen evidence and re-derives declared assertions.

Usage:
    .venv-v2/bin/python scripts/validate_phase8_postopening.py
"""

import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
TABLES = ARTIFACTS / "phase8_final_test_tables"
PRED_DIR = ARTIFACTS / "phase8_predictions"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"

GATE_LINE = re.compile(r"^(.*?)\s\.{3,}\s(PASS|FAIL)$")
UPSTREAM_CASCADE = re.compile(r"^Upstream validate_phase[34567]\.py passes$")
SEAL_GATE = "No test prediction or test metric artifact exists"
AUTHORIZED_SEAL_TRIGGERS = sorted([
    "artifacts/final_test_opening_receipt.json",
    "artifacts/phase8_final_test_metrics.json",
])

PRETEST_GIT_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
EXPECTED_TEST_SAMPLES = 411012
EXPECTED_TOTAL_PREDICTIONS = 1233036
EXPECTED_SEVERE_N = 20336
SEVERE_THRESHOLD = 244.0
CONFIRMATORY = ["B3_R2", "GRU_R1", "B0"]
CLEAN_UPSTREAM = [("validate_foundation.py", "python3", 24),
                  ("validate_phase1.py", "python3", 31),
                  ("validate_phase2.py", "python3", 58)]
CASCADE_UPSTREAM = ["validate_phase4.py", "validate_phase5.py",
                    "validate_phase6.py", "validate_phase7.py"]


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


def run_validator(name, interpreter):
    """Run a validator and parse its dotted gate lines and failure details."""
    result = subprocess.run(
        [interpreter, str(PROJECT_ROOT / "scripts" / name)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = result.stdout.decode("utf-8", "replace").splitlines()
    gates, failures, details = [], [], {}
    for position, line in enumerate(lines):
        match = GATE_LINE.match(line.rstrip())
        if not match:
            continue
        label, status = match.group(1).strip(), match.group(2)
        gates.append((label, status))
        if status == "FAIL":
            failures.append(label)
            if position + 1 < len(lines) and lines[position + 1].startswith(
                    "      "):
                details[label] = lines[position + 1].strip()
    return OrderedDict([
        ("exit_code", result.returncode),
        ("gates", len(gates)),
        ("passed", sum(1 for _, s in gates if s == "PASS")),
        ("failures", failures),
        ("details", details),
    ])


# ---------------------------------------------------------------------------
# Sections 5-7: the upstream state, told apart precisely
# ---------------------------------------------------------------------------

def phase_0_to_2(report, record):
    for name, interpreter, expected in CLEAN_UPSTREAM:
        outcome = run_validator(name, interpreter)
        record[name] = OrderedDict([("status", "PASS"),
                                    ("exit_code", outcome["exit_code"]),
                                    ("gates", outcome["gates"]),
                                    ("failures", outcome["failures"])])
        ok = (outcome["exit_code"] == 0 and not outcome["failures"]
              and outcome["gates"] == expected)
        if not ok:
            record[name]["status"] = "FAIL"
        report.add("%s passes with its existing %d gates" % (name, expected),
                   ok, "exit %d, %d gates, failures %s"
                   % (outcome["exit_code"], outcome["gates"],
                      outcome["failures"]))


def phase_3_seal_transition(report, record):
    outcome = run_validator("validate_phase3.py", sys.executable)
    failures = outcome["failures"]
    report.add("Phase 3 returns non-zero, as expected after the opening",
               outcome["exit_code"] != 0,
               "exit %d" % outcome["exit_code"])
    only_seal = failures == [SEAL_GATE]
    report.add("Phase 3 fails on exactly one gate, the historical seal gate",
               only_seal, "failures: %s" % failures)
    report.add("Every other Phase-3 scientific and hash gate passes",
               outcome["passed"] == outcome["gates"] - 1 and only_seal,
               "%d/%d passed" % (outcome["passed"], outcome["gates"]))

    triggers = []
    detail = outcome["details"].get(SEAL_GATE, "")
    if detail.startswith("found:"):
        try:
            triggers = sorted(ast.literal_eval(detail[len("found:"):].strip()))
        except (ValueError, SyntaxError):
            triggers = []
    report.add("Seal gate is triggered only by the authorized Phase-8 "
               "artifacts", triggers == AUTHORIZED_SEAL_TRIGGERS,
               "triggers: %s" % triggers)
    record["validate_phase3.py"] = OrderedDict([
        ("status", "EXPECTED_HISTORICAL_SEAL_TRANSITION"),
        ("phase3_historical_seal_gate", "EXPECTED_TRANSITION"),
        ("exit_code", outcome["exit_code"]),
        ("gates", outcome["gates"]),
        ("passed", outcome["passed"]),
        ("failures", failures),
        ("seal_gate_triggers", triggers),
        ("scientific_gates", "PASS"),
    ])
    return only_seal and triggers == AUTHORIZED_SEAL_TRIGGERS


def phase_4_to_7_cascade(report, record):
    for name in CASCADE_UPSTREAM:
        outcome = run_validator(name, sys.executable)
        independent = [label for label in outcome["failures"]
                       if not UPSTREAM_CASCADE.match(label)]
        report.add("%s has no independently owned gate failure" % name,
                   not independent, "independent failures: %s" % independent)
        report.add("%s failures are only the nested Phase-3 seal cascade"
                   % name,
                   bool(outcome["failures"]) and not independent,
                   "failures: %s" % outcome["failures"])
        record[name] = OrderedDict([
            ("status",
             "SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE"),
            ("scientific_gates", "PASS"),
            ("historical_seal_dependency", "EXPECTED_TRANSITION"),
            ("exit_code", outcome["exit_code"]),
            ("gates", outcome["gates"]),
            ("passed", outcome["passed"]),
            ("cascade_failures", outcome["failures"]),
            ("independent_failures", independent),
        ])


# ---------------------------------------------------------------------------
# Section 8: the frozen Phase-8 validator, upstream aside
# ---------------------------------------------------------------------------

def frozen_phase8_validator(report, record):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_phase8.py"),
         "--skip-upstream"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = result.stdout.decode("utf-8", "replace").splitlines()
    gates, failures = [], []
    for line in lines:
        match = GATE_LINE.match(line.rstrip())
        if match:
            gates.append(match.group(2))
            if match.group(2) == "FAIL":
                failures.append(match.group(1).strip())
    report.add("Every non-upstream gate of the frozen Phase-8 validator "
               "passes", result.returncode == 0 and not failures,
               "exit %d, failures %s" % (result.returncode, failures))
    record["validate_phase8.py"] = OrderedDict([
        ("status", "non_upstream_gates_PASS"),
        ("source_unchanged", True),
        ("sha256", sha256_of_file(PROJECT_ROOT / "scripts"
                                  / "validate_phase8.py")),
        ("gates_checked", len(gates)),
        ("failures", failures),
        ("note", "Pre-opening-frozen Phase-8 validator preserved as "
                 "historical evidence; its full-mode run also reports the "
                 "expected upstream seal cascade."),
    ])


# ---------------------------------------------------------------------------
# Sections 8-9: the frozen result artifacts are the numerical authority
# ---------------------------------------------------------------------------

def read_actuals():
    from src.data.preprocessing import STATIONS, TARGET, index_of
    from src.evaluation.final_test import TEST_END, TEST_START
    low, high = index_of(TEST_START), index_of(TEST_END)
    values = {}
    for station in STATIONS:
        series = np.full(high + 1, np.nan, dtype=np.float64)
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        with open(str(path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = index_of(stamp)
                if position < low or position > high:
                    continue
                raw = row[TARGET]
                if raw in ("NA", ""):
                    continue
                series[position] = float(raw)
        values[station] = series
    return values


def scientific_results(report, record):
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    manifest = load_json(ARTIFACTS / "v2_phase8_manifest.json")

    report.add("Pre-test git snapshot unchanged",
               manifest["pretest_git_snapshot"] == PRETEST_GIT_SHA)
    report.add("Total prediction count is 1,233,036",
               manifest["test_predictions_generated"]
               == EXPECTED_TOTAL_PREDICTIONS,
               str(manifest["test_predictions_generated"]))
    report.add("No retraining and no train+validation refit",
               manifest["model_retraining"] == 0
               and manifest["model_refitting"] == 0
               and manifest["train_validation_refit_performed"] is False)
    report.add("Confirmatory hierarchy unchanged",
               manifest["primary_model"] == "B3_R2"
               and manifest["secondary_model"] == "GRU_R1"
               and manifest["secondary_model_seed"] == 42
               and manifest["reference_model"] == "B0"
               and manifest["models_evaluated"] == 3)
    report.add("Model selection after test remains false",
               manifest["model_selection_after_test"] is False
               and results["model_selection_after_test"] is False)
    report.add("Post-test exploration still not started",
               manifest["post_test_exploration_performed"] is False
               and results["post_test_exploration_performed"] is False
               and not list(ARTIFACTS.glob("phase8_post_test*"))
               and not list(ARTIFACTS.glob("phase9*")))
    report.add("Severe threshold and rule unchanged",
               manifest["severe_threshold"] == SEVERE_THRESHOLD
               and manifest["severe_rule"] == "actual > 244.0")
    report.add("Predictions were not clipped",
               manifest["predictions_clipped"] is False
               and lock["predictions_clipped"] is False)
    report.add("Future-target access violations remain zero",
               manifest["future_target_access_violations"] == 0
               and lock["future_target_access_violations"] == 0)
    report.add("Final-test status is evaluated",
               manifest["final_test_status"] == "evaluated"
               and results["final_test_status"] == "evaluated")

    stored = OrderedDict()
    for name in CONFIRMATORY:
        entry = lock["models"][name]
        path = PROJECT_ROOT / entry["prediction_file"]
        report.add("%s prediction array matches the prediction lock" % name,
                   sha256_of_file(path) == entry["sha256"])
        stored[name] = np.load(str(path)).astype(np.float64)
        report.add("%s holds exactly %d predictions"
                   % (name, EXPECTED_TEST_SAMPLES),
                   stored[name].size == EXPECTED_TEST_SAMPLES)
    present = sorted(p.stem for p in PRED_DIR.glob("*.npy")
                     if not p.stem.endswith("_source"))
    report.add("Only B3_R2, GRU_R1 and B0 were evaluated",
               present == sorted(CONFIRMATORY), "found: %s" % present)

    from src.evaluation.final_test import load_test_index
    index = load_test_index(PROCESSED)
    actual_by_station = read_actuals()
    actual = np.empty(index["target_index"].size, dtype=np.float64)
    for station in sorted(set(index["station"].tolist())):
        rows = np.flatnonzero(index["station"] == station)
        actual[rows] = actual_by_station[station][index["target_index"][rows]]
    report.add("Every canonical target is observed",
               bool(np.isfinite(actual).all()))

    buckets = OrderedDict()
    for position in range(actual.size):
        key = (index["station"][position], int(index["horizon"][position]))
        buckets.setdefault(key, []).append(position)
    report.add("Primary metric reconciles across exactly 48 cells",
               len(buckets) == 48, "%d cells" % len(buckets))

    def macro(name):
        cells = [float(np.abs(actual[rows] - stored[name][rows]).mean())
                 for rows in buckets.values()]
        return float(np.mean(cells))

    severe = actual > SEVERE_THRESHOLD
    report.add("Severe sample count reconciles at %d" % EXPECTED_SEVERE_N,
               int(severe.sum()) == EXPECTED_SEVERE_N
               == results["secondary_severe_n"],
               "%d" % int(severe.sum()))
    # The rule is strict: a target sitting exactly on the threshold is not
    # severe. Checking that boundary samples exist and are all excluded is
    # what distinguishes ">" from ">=" - 232 samples turn on it.
    boundary = int((actual == SEVERE_THRESHOLD).sum())
    report.add("Severe rule is strictly greater than 244.0",
               bool((actual[severe] > SEVERE_THRESHOLD).all())
               and int(((actual == SEVERE_THRESHOLD) & severe).sum()) == 0
               and int((actual >= SEVERE_THRESHOLD).sum())
               == int(severe.sum()) + boundary,
               "%d samples sit exactly at the threshold; all are excluded"
               % boundary)
    record["severe_boundary_samples_excluded"] = boundary

    reconciled = OrderedDict()
    for label, value, stored_value in (
            ("B3_R2 primary macro station-horizon MAE", macro("B3_R2"),
             results["primary_value"]),
            ("B0 reference macro station-horizon MAE", macro("B0"),
             results["reference_primary_value"]),
            ("GRU_R1 secondary severe MAE",
             float(np.abs(actual[severe] - stored["GRU_R1"][severe]).mean()),
             results["secondary_value"]),
            ("B0 reference severe MAE",
             float(np.abs(actual[severe] - stored["B0"][severe]).mean()),
             results["reference_severe_value"])):
        report.add("%s reconciles independently" % label,
                   abs(value - stored_value) < 1e-9,
                   "%.12f vs frozen %.12f" % (value, stored_value))
        reconciled[label] = OrderedDict([
            ("recomputed", value), ("frozen", stored_value)])

    drift = [name for name, digest in results["metric_table_sha256"].items()
             if sha256_of_file(TABLES / name) != digest]
    report.add("Metric tables unchanged since the primary-results lock",
               not drift, "drifted: %s" % drift)
    record["confirmatory_reconciliation"] = reconciled
    return results


# ---------------------------------------------------------------------------
# Section 10: the state transition must read consistently
# ---------------------------------------------------------------------------

def state_transition(report):
    preopening = load_json(ARTIFACTS
                           / "phase8_preopening_integrity_receipt.json")
    opening = load_json(ARTIFACTS / "final_test_opening_receipt.json")
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    manifest = load_json(ARTIFACTS / "v2_phase8_manifest.json")

    report.add("Historical Phase 0-7 state still reads sealed",
               preopening["final_test_status"] == "sealed"
               and load_json(ARTIFACTS / "phase7_pretest_freeze.json")[
                   "final_test_status"] == "sealed")
    report.add("Opening receipt reads AUTHORIZED_TO_OPEN",
               opening["status"] == "AUTHORIZED_TO_OPEN"
               and opening["test_predictions_generated"] == 0
               and opening["test_metrics_seen"] == 0)
    report.add("Opening receipt carries the pre-opening integrity hash",
               opening["preopening_integrity_receipt_sha256"]
               == sha256_of_file(
                   ARTIFACTS / "phase8_preopening_integrity_receipt.json"))
    report.add("Stage-A lock reads opened, with metrics still unseen",
               lock["final_test_status"] == "opened_for_locked_evaluation"
               and lock["test_metrics_seen"] == 0
               and lock["predictions_frozen_before_scoring"] is True)
    report.add("Prediction lock carries the opening receipt hash",
               lock["opening_receipt_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "final_test_opening_receipt.json"))
    report.add("Results lock reads evaluated, with metrics seen",
               results["final_test_status"] == "evaluated"
               and results["test_metrics_seen"] is True
               and results["prediction_lock_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "phase8_prediction_lock.json"))
    current = {"phase8_prediction_lock.json": lock,
               "phase8_primary_results_lock.json": results,
               "v2_phase8_manifest.json": manifest}
    offenders = [name for name, payload in current.items()
                 if payload.get("final_test_status") == "sealed"]
    report.add("No current-state Phase-8 artifact still claims sealed",
               not offenders, "offenders: %s" % offenders)


def write_receipt(path, report, record):
    """Record what this run measured. Written only when every gate passed."""
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase8_postopening_validation_receipt"),
        ("stage", "post_opening_final_state_verified"),
        ("phase8_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase8_manifest.json")),
        ("primary_results_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_primary_results_lock.json")),
        ("prediction_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_prediction_lock.json")),
        ("opening_receipt_sha256",
         sha256_of_file(ARTIFACTS / "final_test_opening_receipt.json")),
        ("preopening_integrity_receipt_sha256", sha256_of_file(
            ARTIFACTS / "phase8_preopening_integrity_receipt.json")),
        ("postopening_correction_record_sha256", sha256_of_file(
            ARTIFACTS / "phase8_postopening_validator_correction.json")),
        ("postopening_tooling_registry_sha256", sha256_of_file(
            ARTIFACTS / "phase8_postopening_tooling_registry.json")),
        ("postopening_validator_sha256", sha256_of_file(
            PROJECT_ROOT / "scripts" / "validate_phase8_postopening.py")),
        ("validators", OrderedDict([
            ("validate_foundation.py", "PASS"),
            ("validate_phase1.py", "PASS"),
            ("validate_phase2.py", "PASS"),
            ("validate_phase3.py", "EXPECTED_HISTORICAL_SEAL_TRANSITION"),
            ("validate_phase4.py",
             "SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE"),
            ("validate_phase5.py",
             "SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE"),
            ("validate_phase6.py",
             "SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE"),
            ("validate_phase7.py",
             "SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE"),
            ("validate_phase8.py",
             "historical result recorded, source unchanged"),
            ("validate_phase8_postopening.py", "PASS"),
        ])),
        ("validator_detail", record),
        ("postopening_validator_gates", len(report.rows)),
        ("postopening_validator_failures", len(report.failed())),
        ("phase3_historical_seal_gate", "EXPECTED_TRANSITION"),
        ("phase3_seal_gate_triggers", AUTHORIZED_SEAL_TRIGGERS),
        ("historical_gate_suppressed", False),
        ("historical_gate_reinterpreted", False),
        ("upstream_validator_modified", False),
        ("existing_phase8_validator_modified", False),
        ("scientific_drift", 0),
        ("prediction_drift", 0),
        ("metric_drift", 0),
        ("models_changed", False),
        ("predictions_changed", False),
        ("metrics_changed", False),
        ("post_test_exploration", False),
        ("final_test_status", "evaluated"),
    ])
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return sha256_of_file(path)


def main():
    receipt_path = None
    if "--receipt" in sys.argv:
        receipt_path = ARTIFACTS / "phase8_postopening_validation_receipt.json"

    report = Report()
    record = OrderedDict()

    phase_0_to_2(report, record)
    phase_3_seal_transition(report, record)
    phase_4_to_7_cascade(report, record)
    frozen_phase8_validator(report, record)
    scientific_results(report, record)
    state_transition(report)

    report.render()
    print("")
    if report.failed():
        print("PHASE 8 POST-OPENING VALIDATION FAILED: %d gate(s)"
              % len(report.failed()))
        return 1
    print("PHASE 8 POST-OPENING VALIDATION PASSED")
    print("Final test: EVALUATED. Phase 3 seal gate: EXPECTED_TRANSITION.")
    print("Scientific drift: 0. Prediction drift: 0. Metric drift: 0.")
    if receipt_path is not None:
        print("post-opening validation receipt  %s"
              % write_receipt(receipt_path, report, record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
