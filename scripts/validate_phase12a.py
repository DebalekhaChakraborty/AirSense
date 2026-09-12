#!/usr/bin/env python3
"""AirSense V2 - Phase 12A validator.

Publication / reproducibility hardening. NON-SCIENTIFIC RELEASE WORK.

Phase 12A adds tests, retrospective provenance metadata and release audits. It
performs no science. This validator exists to prove that, and in particular to
prove that nothing frozen in Phases 0-11 moved while it happened.

Read-only. Writes no manifest and no artifact of any kind. Exits non-zero on
any failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
TESTS = PROJECT_ROOT / "tests"

PRE_TEST_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LEGACY_SHA = "16c9030cf74508b620cc6d28f90346aa379f29cd"

# The frozen Phase 0-11 evidence Phase 12 must never mutate (task section 1).
FROZEN = [
    "artifacts/phase7_pretest_freeze.json",
    "artifacts/phase7_pretest_freeze_addendum.json",
    "artifacts/phase8_prediction_lock.json",
    "artifacts/phase8_primary_results_lock.json",
    "artifacts/phase8_final_test_metrics.json",
    "artifacts/v2_phase8_manifest.json",
    "artifacts/phase9_analysis_freeze.json",
    "artifacts/v2_phase9_manifest.json",
    "artifacts/v2_phase10_manifest.json",
    "artifacts/phase10_master_evidence_index.json",
    "artifacts/phase11_manuscript_freeze.json",
    "artifacts/v2_phase11_manifest.json",
]

# Every Phase 0-11 manifest. None may be rewritten by Phase 12.
PHASE_MANIFESTS = [
    "artifacts/v2_foundation_manifest.json", "artifacts/v2_phase1_manifest.json",
    "artifacts/v2_phase2_manifest.json", "artifacts/v2_phase3_manifest.json",
    "artifacts/v2_phase4_manifest.json", "artifacts/v2_phase5_manifest.json",
    "artifacts/v2_phase6_manifest.json", "artifacts/v2_phase8_manifest.json",
    "artifacts/v2_phase9_manifest.json", "artifacts/v2_phase10_manifest.json",
    "artifacts/v2_phase11_manifest.json",
]

PHASE12_ARTIFACTS = [
    "artifacts/phase12_phase7_source_provenance.json",
    "artifacts/phase12_readme_update_plan.json",
    "artifacts/phase12_public_release_audit.json",
]

LOCKED = {
    "primary_value": 31.98722559774534,
    "secondary_value": 109.3328897571946,
    "reference_primary_value": 36.138513538776856,
    "reference_severe_value": 93.22447875688434,
    "secondary_severe_n": 20336,
    "test_target_count": 411012,
    "severe_threshold": 244.0,
    "final_test_status": "evaluated",
}

# Keys that would indicate a newly computed scientific metric.
METRIC_KEY = re.compile(
    r"(macro_station_horizon_(mae|rmse|r2)|severe_mae|severe_rmse|micro_(mae|rmse)"
    r"|_mae$|_rmse$|_r2$|residual_acf|detection_recall|underprediction_pct)", re.I)


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def add(self, label: str, ok: bool, detail: str = "") -> None:
        pad = max(3, 74 - len(label))
        print(f"{label} {'.' * pad} {'PASS' if ok else 'FAIL'}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"{label}: {detail}" if detail else label)
            if detail:
                print(f"    -> {detail}")

    def section(self, name: str) -> None:
        print(f"\n--- {name} ---")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args],
                                   text=True).strip()


def historical_sha(path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(PROJECT_ROOT), "show",
                        f"{PRE_TEST_SHA}:{path}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def load(rel: str):
    return json.loads((PROJECT_ROOT / rel).read_text())


def walk_numbers(obj, path=""):
    """Yield (json path, key, value) for every numeric leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, path.rsplit("/", 1)[-1], obj


def main() -> int:
    r = Report()
    print("AirSense V2 - Phase 12A validator")
    print("Release hardening. No new science. Read-only.\n")

    # ---- 1. frozen Phase 0-11 evidence unchanged -------------------------
    r.section("1. Frozen Phase 0-11 evidence")
    for rel in FROZEN:
        p = PROJECT_ROOT / rel
        r.add(f"Present: {rel.split('/')[-1]}", p.is_file())

    changed = git("diff", "--name-only", "HEAD").splitlines()
    staged = git("diff", "--cached", "--name-only").splitlines()
    touched = set(changed) | set(staged)
    hit = sorted(set(FROZEN) & touched)
    r.add("No frozen Phase 0-11 artifact is modified in the working tree",
          not hit, f"modified={hit}")

    # Internal consistency: each manifest still describes the tree it pins.
    m10 = load("artifacts/v2_phase10_manifest.json")
    d10 = [k for k, v in m10["artifact_sha256"].items()
           if not (PROJECT_ROOT / k).exists() or sha256(PROJECT_ROOT / k) != v]
    r.add("Phase-10 manifest reconciles with the tree", not d10, f"drift={d10[:4]}")

    m11 = load("artifacts/v2_phase11_manifest.json")
    d11 = [k for k, v in m11["artifact_sha256"].items()
           if not (PROJECT_ROOT / k).exists() or sha256(PROJECT_ROOT / k) != v]
    r.add("Phase-11 manifest reconciles with the tree", not d11, f"drift={d11[:4]}")

    # ---- 2. Phase-8 locked values ---------------------------------------
    r.section("2. Phase-8 locked confirmatory values")
    lock = load("artifacts/phase8_primary_results_lock.json")
    for key, expected in LOCKED.items():
        r.add(f"Locked value unchanged: {key}", lock.get(key) == expected,
              f"got={lock.get(key)!r} expected={expected!r}")
    metrics = load("artifacts/phase8_final_test_metrics.json")
    r.add("Final-test metrics still report 411,012 samples",
          metrics["test_sample_count"] == 411012)
    r.add("Final-test metrics still report 20,336 severe samples",
          metrics["severe_sample_count"] == 20336)
    r.add("Severe rule is still strictly greater than 244.0",
          metrics["severe_rule"] == "actual > 244.0")

    # ---- 3. no new science in Phase-12 artifacts -------------------------
    r.section("3. No new prediction or scientific metric in Phase-12 output")
    frozen_blob = "".join(
        (PROJECT_ROOT / f).read_text()
        for f in FROZEN + ["artifacts/phase10_canonical_conclusion.json",
                           "artifacts/phase10_dataset_ledger.json",
                           "artifacts/phase5_prevalidation_freeze.json"]
        if (PROJECT_ROOT / f).is_file())

    for rel in PHASE12_ARTIFACTS:
        p = PROJECT_ROOT / rel
        if not p.is_file():
            r.add(f"Phase-12 artifact exists: {rel.split('/')[-1]}", False)
            continue
        doc = json.loads(p.read_text())
        for field in ("models_trained", "predictions_generated"):
            if field in doc:
                r.add(f"{rel.split('/')[-1]} declares {field} == 0",
                      doc[field] == 0, f"got={doc.get(field)}")
        # Any metric-shaped key must carry a value already present in frozen
        # evidence; a genuinely new metric would not appear there.
        novel = []
        for jpath, key, value in walk_numbers(doc):
            if METRIC_KEY.search(key) and repr(value) not in frozen_blob \
               and str(value) not in frozen_blob:
                novel.append(f"{jpath}={value}")
        r.add(f"{rel.split('/')[-1]} introduces no new scientific metric",
              not novel, f"novel={novel[:4]}")
        r.add(f"{rel.split('/')[-1]} contains no prediction array",
              not re.search(r'"prediction[s]?"\s*:\s*\[', p.read_text()))

    # ---- 4. Phase-7 provenance recomputes from history -------------------
    r.section("4. Phase-7 source provenance recomputation")
    prov_path = ARTIFACTS / "phase12_phase7_source_provenance.json"
    r.add("Provenance artifact exists", prov_path.is_file())
    if prov_path.is_file():
        prov = json.loads(prov_path.read_text())
        r.add("Provenance names the pre-test snapshot as its anchor",
              prov["historical_anchor"]["pre_test_snapshot_sha"] == PRE_TEST_SHA)
        r.add("Anchor is an ancestor of HEAD",
              subprocess.run(["git", "-C", str(PROJECT_ROOT), "merge-base",
                              "--is-ancestor", PRE_TEST_SHA, "HEAD"],
                             capture_output=True).returncode == 0)
        entries = prov["required_phase7_sources"] + \
            prov.get("other_phase7_files_at_snapshot", [])
        r.add("Provenance covers the six required Phase-7 sources",
              len(prov["required_phase7_sources"]) == 6)
        mism = []
        for e in entries:
            h = historical_sha(e["path"])
            if h != e["historical_sha256"]:
                mism.append(e["path"])
            cur = (PROJECT_ROOT / e["path"])
            c = sha256(cur) if cur.is_file() else None
            if c != e["current_sha256"]:
                mism.append(e["path"] + " (current)")
        r.add(f"Every recorded hash recomputes from git ({len(entries)} entries)",
              not mism, f"mismatch={mism[:4]}")
        claimed = prov["summary"]["byte_identical_to_pretest_snapshot"]
        actual = sum(1 for e in prov["required_phase7_sources"]
                     if e["byte_identical_to_pretest_snapshot"])
        r.add("Summary byte-identical count agrees with the entries",
              claimed == actual, f"{claimed} vs {actual}")

    # ---- 5/6. new tests exist -------------------------------------------
    r.section("5-6. Phase-12A test files")
    for name, subject in (("test_scoring.py", "ValidationTargetOracle"),
                          ("test_itransformer.py", "ITransformerForecaster")):
        p = TESTS / name
        r.add(f"Test file exists: {name}", p.is_file())
        if p.is_file():
            body = p.read_text()
            r.add(f"{name} exercises {subject}", subject in body)
            r.add(f"{name} defines test cases",
                  len(re.findall(r"def test_", body)) >= 8,
                  f"n={len(re.findall(r'def test_', body))}")
    scoring_src = (PROJECT_ROOT / "src/evaluation/scoring.py")
    r.add("scoring.py is unmodified in the working tree",
          "src/evaluation/scoring.py" not in touched)
    r.add("itransformer_forecaster.py is unmodified in the working tree",
          "src/models/itransformer_forecaster.py" not in touched)

    # ---- 7. Phase-12 artifacts self-identify ----------------------------
    r.section("7. Phase-12 artifacts declare themselves retrospective")
    for rel in PHASE12_ARTIFACTS:
        p = PROJECT_ROOT / rel
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        cls = str(doc.get("classification", "")).upper()
        r.add(f"{rel.split('/')[-1]} declares RETROSPECTIVE/release metadata",
              "RETROSPECTIVE" in cls)
        r.add(f"{rel.split('/')[-1]} is not named like a phase manifest",
              not re.match(r"v2_phase\d+_manifest", Path(rel).name))
    prov = ARTIFACTS / "phase12_phase7_source_provenance.json"
    if prov.is_file():
        cls = json.loads(prov.read_text())["classification"].upper()
        r.add("Provenance denies being a pre-test preregistration artifact",
              "NOT A PRE-TEST PREREGISTRATION" in cls)
        r.add("Provenance denies superseding the Phase-7 freeze",
              "DOES NOT MODIFY OR SUPERSEDE" in cls)
    r.add("No file named v2_phase7_manifest.json was created",
          not (ARTIFACTS / "v2_phase7_manifest.json").exists())

    # ---- 8. legacy untouched --------------------------------------------
    r.section("8. legacy branch")
    try:
        r.add("legacy tip is unchanged", git("rev-parse", "legacy") == LEGACY_SHA)
    except subprocess.CalledProcessError:
        r.add("legacy tip is unchanged", False, "legacy ref missing")
    r.add("legacy is not an ancestor of HEAD (separate V1 lineage)",
          subprocess.run(["git", "-C", str(PROJECT_ROOT), "merge-base",
                          "--is-ancestor", LEGACY_SHA, "HEAD"],
                         capture_output=True).returncode != 0)

    # ---- 9. no Phase 0-11 manifest rewritten -----------------------------
    r.section("9. Phase 0-11 manifests")
    rewritten = sorted(set(PHASE_MANIFESTS) & touched)
    r.add("No Phase 0-11 manifest is modified by this task",
          not rewritten, f"modified={rewritten}")
    for rel in PHASE_MANIFESTS:
        r.add(f"Manifest present: {rel.split('/')[-1]}",
              (PROJECT_ROOT / rel).is_file())

    r.section("Scientific source untouched")
    sci = [f for f in touched if f.startswith(("src/", "results/", "figures/",
                                               "data/"))]
    r.add("No src/, results/, figures/ or data/ file modified", not sci,
          f"modified={sci[:5]}")

    total = r.passed + r.failed
    print(f"\n{'=' * 70}")
    print(f"Phase 12A validation: {r.passed}/{total} gates passed, {r.failed} failed")
    if r.failed:
        print("\nFAILURES:")
        for f in r.failures:
            print(f"  - {f}")
    print("=" * 70)
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
