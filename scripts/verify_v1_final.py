"""Verify the frozen AirSense V1 evidence package.

Protocol Phase 12 verifier. **It trains nothing, predicts nothing,
recomputes no scientific experiment, and repairs nothing.** Its only job is
to confirm that the frozen V1 package is intact and self-consistent.

Exits 0 only if every required gate passes; non-zero otherwise.

Usage:
    venv/bin/python scripts/verify_v1_final.py
"""

import io
import json
import os
import platform
import sys
from collections import OrderedDict

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import audit_csv, sha256_of_file  # noqa: E402

REQUIRED_PYTHON = (3, 6, 7)
REQUIRED_PINS = OrderedDict([
    ("numpy", "1.15.4"), ("pandas", "0.23.4"), ("scipy", "1.1.0"),
    ("sklearn", "0.20.0"), ("matplotlib", "3.0.2"), ("seaborn", "0.9.0"),
])
IMPORT_TO_DIST = {"sklearn": "scikit-learn"}

RAW = "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"
EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")
EXPECTED_RAW_ROWS = 43824
EXPECTED_RAW_COLUMNS = 13
EXPECTED_TEST_ROWS = 8661
EXPECTED_RANKING = ["M3", "M2", "M1", "M0"]
MODELS = ["M0", "M1", "M2", "M3"]
TOLERANCE = 1e-12

REGISTRY = "artifacts/v1_evidence_registry.csv"
FINAL_MANIFEST = "artifacts/v1_final_evidence_manifest.json"
FREEZE_RECEIPT = "artifacts/v1_final_freeze_receipt.json"
EVALUATION_MANIFEST = "artifacts/final_evaluation_manifest.json"
ERROR_MANIFEST = "artifacts/error_analysis_manifest.json"
FINAL_METRICS = "results/final_test/final_test_metrics.csv"
SUMMARY_TABLE = "results/v1_summary/final_model_comparison.csv"

REQUIRED_PRESENT = [
    "results/final_test/final_test_target_snapshot.csv",
    "artifacts/pretest_development_freeze.json",
    "artifacts/final_blind_prediction_freeze.json",
    "artifacts/final_test_opening_receipt.json",
    "artifacts/v1_claims_ledger.csv",
    "docs/V1_FINAL_REPORT.md",
    "docs/V1_REPRODUCIBILITY.md",
]


def path(relative):
    return os.path.join(PROJECT_ROOT, relative)


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 62
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [r for r in self.rows if not r[1]]


def check_runtime(report):
    actual = sys.version_info[:3]
    report.add("Python is exactly 3.6.7",
               actual == REQUIRED_PYTHON
               and platform.python_implementation() == "CPython",
               "found %s %s" % (platform.python_implementation(),
                                ".".join(str(p) for p in actual)))
    problems = []
    for module, expected in REQUIRED_PINS.items():
        try:
            found = getattr(__import__(module), "__version__", None)
        except ImportError:
            found = None
        if found != expected:
            problems.append("%s: expected %s, found %r"
                            % (IMPORT_TO_DIST.get(module, module), expected,
                               found))
    report.add("Frozen direct package pins", not problems,
               "; ".join(problems))


def check_raw(report):
    target = path(RAW)
    if not os.path.isfile(target):
        report.add("Raw dataset present", False, "%s missing" % RAW)
        report.add("Raw dataset SHA-256", False)
        report.add("Raw dataset dimensions", False)
        return
    report.add("Raw dataset present", True)
    observed = sha256_of_file(target)
    report.add("Raw dataset SHA-256", observed == EXPECTED_RAW_SHA256,
               "found %s" % observed)
    record = audit_csv(target)
    report.add("Raw dataset dimensions",
               record["row_count"] == EXPECTED_RAW_ROWS
               and record["column_count"] == EXPECTED_RAW_COLUMNS,
               "found %d x %d" % (record["row_count"],
                                  record["column_count"]))


def check_present(report):
    missing = [p for p in REQUIRED_PRESENT if not os.path.isfile(path(p))]
    report.add("Required freeze artifacts present", not missing,
               "missing: %s" % ", ".join(missing))


def check_manifest_integrity(report, label, relative, keys):
    target = path(relative)
    if not os.path.isfile(target):
        report.add(label, False, "%s missing" % relative)
        return None
    manifest = json.load(io.open(target, encoding="utf-8"))
    problems = []
    for key in keys:
        for item, recorded in manifest.get(key, {}).items():
            digest = recorded["sha256"] if isinstance(recorded, dict) \
                else recorded
            candidate = path(item)
            if not os.path.isfile(candidate):
                problems.append("%s missing" % item)
            elif sha256_of_file(candidate) != digest:
                problems.append("%s hash mismatch" % item)
    report.add(label, not problems, "; ".join(problems[:5]))
    return manifest


def check_registry(report):
    target = path(REGISTRY)
    if not os.path.isfile(target):
        report.add("Evidence registry present", False)
        report.add("Evidence registry integrity", False)
        return None
    report.add("Evidence registry present", True)
    registry = pd.read_csv(target)
    problems = []
    for _, row in registry.iterrows():
        candidate = path(row["relative_path"])
        if not os.path.isfile(candidate):
            problems.append("%s missing" % row["relative_path"])
        elif sha256_of_file(candidate) != row["sha256"]:
            problems.append("%s hash mismatch" % row["relative_path"])
    report.add("Evidence registry integrity (%d files)" % len(registry),
               not problems, "; ".join(problems[:5]))
    return registry


def check_final_results(report):
    for relative in (FINAL_METRICS, SUMMARY_TABLE):
        if not os.path.isfile(path(relative)):
            report.add("Final results present", False,
                       "%s missing" % relative)
            return
    report.add("Final results present", True)
    metrics = pd.read_csv(path(FINAL_METRICS)).set_index("model")
    summary = pd.read_csv(path(SUMMARY_TABLE)).set_index("model")

    problems = []
    for model_id in MODELS:
        for column in ("mae", "rmse", "r2"):
            a = float(metrics.loc[model_id, column])
            b = float(summary.loc[model_id, column])
            if abs(a - b) > TOLERANCE:
                problems.append("%s %s differs" % (model_id, column))
    report.add("Summary table matches frozen final metrics", not problems,
               "; ".join(problems))

    ranking = sorted(MODELS, key=lambda m: float(metrics.loc[m, "mae"]))
    report.add("Final ranking is M3 < M2 < M1 < M0 by lowest MAE",
               ranking == EXPECTED_RANKING, "found %s" % " < ".join(ranking))

    counts = set(int(v) for v in metrics["n_test"])
    report.add("Final test row count is 8,661",
               counts == {EXPECTED_TEST_ROWS}, "found %s" % counts)


def check_freeze_receipt(report):
    target = path(FREEZE_RECEIPT)
    if not os.path.isfile(target):
        report.add("Final freeze receipt present", False)
        return
    report.add("Final freeze receipt present", True)
    receipt = json.load(io.open(target, encoding="utf-8"))
    problems = []
    for key, relative in (("evidence_registry_sha256", REGISTRY),
                          ("final_evidence_manifest_sha256",
                           FINAL_MANIFEST)):
        candidate = path(relative)
        if not os.path.isfile(candidate):
            problems.append("%s missing" % relative)
        elif sha256_of_file(candidate) != receipt.get(key):
            problems.append("%s hash mismatch" % relative)
    if receipt.get("raw_dataset_sha256") != EXPECTED_RAW_SHA256:
        problems.append("raw dataset digest mismatch")
    if receipt.get("v1_status") != "complete_and_frozen":
        problems.append("v1_status is %r" % receipt.get("v1_status"))
    if receipt.get("final_test_status") != "exhausted":
        problems.append("final_test_status is %r"
                        % receipt.get("final_test_status"))
    report.add("Final freeze receipt consistent", not problems,
               "; ".join(problems))


def check_final_manifest(report):
    target = path(FINAL_MANIFEST)
    if not os.path.isfile(target):
        report.add("Final evidence manifest present", False)
        return
    report.add("Final evidence manifest present", True)
    manifest = json.load(io.open(target, encoding="utf-8"))
    problems = []
    closing = manifest.get("closing_artifacts_sha256", {})
    for key, relative in (
            ("v1_evidence_registry_sha256", REGISTRY),
            ("claims_ledger_sha256", "artifacts/v1_claims_ledger.csv"),
            ("final_model_comparison_sha256", SUMMARY_TABLE),
            ("key_findings_sha256", "results/v1_summary/key_findings.csv"),
            ("v1_final_report_sha256", "docs/V1_FINAL_REPORT.md"),
            ("v1_reproducibility_sha256", "docs/V1_REPRODUCIBILITY.md"),
            ("phase_10_final_v1_record_sha256",
             "docs/PHASE_10_FINAL_V1_RECORD.md"),
            ("readme_sha256", "README.md"),
            ("v1_research_protocol_sha256",
             "docs/V1_RESEARCH_PROTOCOL.md")):
        candidate = path(relative)
        if not os.path.isfile(candidate):
            problems.append("%s missing" % relative)
        elif sha256_of_file(candidate) != closing.get(key):
            problems.append("%s hash mismatch" % relative)
    if manifest.get("final_test_status") != "exhausted":
        problems.append("final_test_status is %r"
                        % manifest.get("final_test_status"))
    report.add("Final evidence manifest consistent", not problems,
               "; ".join(problems))


def check_claims_ledger(report):
    target = path("artifacts/v1_claims_ledger.csv")
    if not os.path.isfile(target):
        report.add("Claims ledger verified", False)
        return
    ledger = pd.read_csv(target)
    unverified = ledger[~ledger["verified"].astype(bool)]
    report.add("Claims ledger verified (%d claims)" % len(ledger),
               len(unverified) == 0,
               "unverified: %s"
               % ", ".join(unverified["claim_id"].astype(str).tolist()))


def main():
    print("")
    print("[AIRSENSE V1 FINAL VERIFICATION]")
    print("")
    print("Read-only. This verifier trains nothing, predicts nothing and")
    print("repairs nothing.")
    print("")
    report = Report()
    check_runtime(report)
    check_raw(report)
    check_present(report)
    check_manifest_integrity(report, "Final evaluation manifest integrity",
                             EVALUATION_MANIFEST, ["outputs_sha256"])
    check_manifest_integrity(report, "Error-analysis manifest integrity",
                             ERROR_MANIFEST,
                             ["provenance_sha256", "outputs", "figures"])
    check_registry(report)
    check_final_results(report)
    check_claims_ledger(report)
    check_final_manifest(report)
    check_freeze_receipt(report)

    print("=" * 62)
    report.render()
    print("=" * 62)
    failures = report.failed()
    print("")
    if failures:
        print("V1 FINAL VERIFICATION: FAIL (%d gate(s))" % len(failures))
        print("V1 must NOT be marked complete.")
        return 1
    print("V1 FINAL VERIFICATION: PASS")
    print("AirSense V1 is complete and frozen. The 2014 test is exhausted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
