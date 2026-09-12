#!/usr/bin/env python3
"""AirSense V2 - Phase 11 validator.

Manuscript construction, publication package and final research freeze.

Phase 11 performs NO new science. This validator exists to prove that:
  * the Phase-10 evidence base is bit-identical to what Phase 10 froze;
  * the manuscript was drafted after a freeze that declared zero new science;
  * every quantitative statement in the manuscript traces to frozen evidence;
  * evidence classes were never upgraded, and no C3 finding is dressed as C1;
  * the locked confirmatory hierarchy and the hypothesis statuses are unchanged;
  * the negative severe-tail result is still visible in the manuscript;
  * no reference was fabricated and unverified citations stay visibly marked.

It deliberately does NOT encode manuscript prose as expected strings. It checks
structure, provenance, numbers and prohibited language only.

Read-only. Exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DOCS = PROJECT_ROOT / "docs"
MANUSCRIPT = PROJECT_ROOT / "manuscript"
FIGURES = PROJECT_ROOT / "figures"

PRE_TEST_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LOCKED_EVAL_SHA = "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"
POST_EVAL_SHA = "7ccfc7e54f7d9008421cfcf764fcfaacc353e1db"
PHASE9_CLOSURE_SHA = "8cc3a7ab8a069b9a187904be68f14b9040928ab4"

MANUSCRIPT_MD = MANUSCRIPT / "AirSense_V2_Manuscript.md"
MANUSCRIPT_TXT = MANUSCRIPT / "AirSense_V2_Manuscript_plaintext.txt"
TRACEABILITY = MANUSCRIPT / "MANUSCRIPT_CLAIM_TRACEABILITY.csv"

# Frozen locked values. Full precision, from the Phase-8 primary results lock.
LOCKED = {
    "primary": 31.98722559774534,
    "secondary_severe": 109.3328897571946,
    "reference_macro": 36.138513538776856,
    "reference_severe": 93.22447875688434,
    "b3r2_severe": 131.90112073760537,
    "gru_macro": 33.08209845348378,
    "improvement_pct": 11.487157424383705,
    "severe_n": 20336,
    "test_n": 411012,
    "threshold": 244.0,
}

# Rounded manuscript spellings that must be present.
REQUIRED_ROUNDED = ["31.99", "36.14", "11.49", "109.33", "93.22", "131.90", "20,336", "411,012"]

# Hard-prohibited language. Word boundaries throughout: "unresolved" contains
# "solve" and is the opposite of the banned claim.
BANNED_PATTERNS = [
    (r"\bstate[- ]of[- ]the[- ]art\b", "state-of-the-art"),
    (r"\bSOTA\b", "SOTA"),
    (r"\bbreakthrough\b", "breakthrough"),
    (r"\bnovel(ty|ties)?\b", "novel/novelty"),
    (r"\bprove[sd]?\b", "prove/proves/proved"),
    (r"\bproven\b", "proven"),
    (r"\bsolve[sd]?\b", "solve/solves/solved"),
    (r"\beliminate[sd]?\b", "eliminate"),
    (r"\bspatial reasoning\b", "spatial reasoning"),
    (r"\bpollution propagation\b", "pollution propagation"),
    (r"\boperationally ready\b", "operationally ready"),
    (r"\bearly[- ]warning[- ]ready\b", "early-warning ready"),
    (r"\ball hypotheses (were )?confirmed\b", "all hypotheses confirmed"),
    (r"\bfoundation models failed\b", "foundation models failed"),
    (r"\bunseen[- ]station\b", "unseen-station"),
    (r"\bp[- ]value[s]?\b", "p-value"),
    (r"\bconfidence interval[s]?\b", "confidence interval"),
    (r"\bsignifican(t|ce|tly)\b", "significant/significance"),
    (r"\bwinner\b", "winner"),
    (r"\bbeat persistence in the severe\b", "beats persistence in the severe tail"),
    (r"\bbias[-– ]variance decomposition\b", "bias-variance decomposition"),
]

# Terms naming an inferential device. The study predeclared none, so the
# manuscript must never assert one - but it is REQUIRED to state that none was
# predeclared. These fail only in a sentence that lacks a negation.
NEGATION_REQUIRED = {
    "p-value",
    "confidence interval",
    "significant/significance",
    "bias-variance decomposition",
    # The role ledger requires the manuscript to state that no overall winner
    # label was assigned. Banned as an assertion, required as a disclaimer.
    "winner",
}

NEGATION_MARKERS = (
    r"\b(no|not|never|none|without|neither|nor|absent|prohibited|"
    r"predeclar\w*|declined|refrain\w*|is why|cannot|could not)\b"
)

REQUIRED_PHRASES = [
    "locked final test",
    "post-test exploratory",
    "training pooled P95",
    "held-out target-station supervision transfer",
    "reference benchmark",
    "primary confirmatory model",
]

PHASE10_ARTIFACT_LIST = [
    "artifacts/phase10_synthesis_freeze.json",
    "artifacts/phase10_model_role_ledger.json",
    "artifacts/phase10_hypothesis_ledger.json",
    "artifacts/phase10_claim_evidence_matrix.csv",
    "artifacts/phase10_table_locked_test.csv",
    "artifacts/phase10_table_development_models.csv",
    "artifacts/phase10_table_multiseed_robustness.csv",
    "artifacts/phase10_table_posttest_diagnostics.csv",
    "artifacts/phase10_dataset_ledger.json",
    "artifacts/phase10_information_regime_ledger.json",
    "artifacts/phase10_v1_v2_structural_ledger.json",
    "artifacts/phase10_foundation_model_governance.json",
    "artifacts/phase10_limitations_ledger.json",
    "artifacts/phase10_future_work_ledger.json",
    "artifacts/phase10_reproducibility_ledger.json",
    "artifacts/phase10_canonical_conclusion.json",
    "artifacts/phase10_publication_table_index.json",
    "artifacts/phase10_publication_figure_index.json",
    "artifacts/phase10_master_evidence_index.json",
    "artifacts/phase10_final_synthesis_summary.json",
    "docs/PUBLICATION_CLAIM_GUIDE.md",
    "docs/PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md",
]

PACKAGE_FILES = [
    "manuscript/AirSense_V2_Manuscript.md",
    "manuscript/AirSense_V2_Manuscript_plaintext.txt",
    "manuscript/TITLE_CANDIDATES.md",
    "manuscript/REFERENCES_VERIFIED.md",
    "manuscript/RELATED_WORK_CITATION_NEEDS.md",
    "manuscript/FIGURE_PLAN.md",
    "manuscript/SUPPLEMENT_PLAN.md",
    "manuscript/SUPPLEMENTARY_METHODS.md",
    "manuscript/MANUSCRIPT_CLAIM_TRACEABILITY.csv",
    "manuscript/LANGUAGE_AUDIT.md",
    "manuscript/MANUSCRIPT_READINESS_REVIEW.md",
    "manuscript/tables/table1_dataset_split.md",
    "manuscript/tables/table2_model_regimes.md",
    "manuscript/tables/table3_development_results.md",
    "manuscript/tables/table4_multiseed_robustness.md",
    "manuscript/tables/table5_locked_final_test.md",
    "manuscript/tables/table6_posttest_diagnostics.md",
]


# ---------------------------------------------------------------------------
# Phase-8 / Phase-9 digest verification.
#
# Each manifest uses its own key schema. The resolvers below map a manifest
# entry to the file it pins. Floors are the number of entries that MUST be
# verified: they turn "verified nothing" from a silent pass into a failure.
# ---------------------------------------------------------------------------

PHASE8_DIGEST_FLOOR = 35
PHASE9_DIGEST_FLOOR = 31

PHASE8_SCALAR_DIGESTS = {
    "prediction_lock_sha256": "artifacts/phase8_prediction_lock.json",
    "primary_results_lock_sha256": "artifacts/phase8_primary_results_lock.json",
    "final_test_metrics_sha256": "artifacts/phase8_final_test_metrics.json",
    "opening_receipt_sha256": "artifacts/final_test_opening_receipt.json",
    "confirmatory_endpoints_sha256":
        "artifacts/phase8_final_test_tables/confirmatory_endpoints.json",
    "verification_tooling_registry_sha256":
        "artifacts/verification_tooling_registry.json",
    "phase8_verification_tooling_registry_sha256":
        "artifacts/phase8_verification_tooling_registry.json",
    "phase8_validator_sha256": "scripts/validate_phase8.py",
    "preopening_integrity_receipt_sha256":
        "artifacts/phase8_preopening_integrity_receipt.json",
    "test_index_sha256": "data/processed/phase2/sample_index/test.csv",
}

PHASE9_SCALAR_DIGESTS = {
    "phase9_analysis_freeze_sha256": "artifacts/phase9_analysis_freeze.json",
    "summary_sha256": "artifacts/phase9_post_test_analysis_summary.json",
    "phase9_validator_sha256": "scripts/validate_phase9.py",
    "phase9_tooling_registry_sha256":
        "artifacts/phase9_verification_tooling_registry.json",
    "record_sha256": "docs/PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md",
    "protocol_sha256": "docs/PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md",
    "v1_v2_comparison_sha256": "docs/PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md",
    "phase8_manifest_sha256": "artifacts/v2_phase8_manifest.json",
    "phase8_primary_results_lock_sha256": "artifacts/phase8_primary_results_lock.json",
    "phase8_prediction_lock_sha256": "artifacts/phase8_prediction_lock.json",
    "phase8_final_validation_receipt_sha256":
        "artifacts/phase8_postcommit_validation_receipt.json",
}

# Derived and gitignored: absent from a fresh clone until Phase-2 is rebuilt.
# Skipping these is legitimate; skipping them silently is not.
OPTIONAL_DIGEST_TARGETS = {"data/processed/phase2/sample_index/test.csv"}


def _resolve_digest_entry(phase: str, manifest: dict, group: str, key: str) -> str | None:
    """Map one manifest digest entry to the repository path it pins."""
    if phase == "Phase-8":
        if group == "prediction_sha256":
            return manifest.get("prediction_file", {}).get(key)
        if group == "model_artifact_sha256":
            if key.startswith("B3_R2_h"):
                horizon = key.split("_h", 1)[1]
                hits = sorted((PROJECT_ROOT / "results" / "models" / "b3")
                              .glob(f"b3_R2_h{horizon}_*.txt"))
                if len(hits) != 1:
                    return None
                return str(hits[0].relative_to(PROJECT_ROOT))
            return f"results/models/neural/{key}"
        if group == "metric_table_sha256":
            return f"artifacts/phase8_final_test_tables/{key}"
        if group == "figure_sha256":
            return f"figures/{key}"
        if group in ("upstream_manifest_sha256", "phase7_freeze_sha256"):
            return f"artifacts/{key}"
        if group == "phase8_code_sha256":
            return key
        return None
    if group == "table_sha256":
        return f"artifacts/{key}"
    if group == "figure_sha256":
        return f"figures/{key}"
    if group == "phase9_code_sha256":
        return key
    return None


def verify_manifest_digests(phase: str, manifest: dict):
    """Return (entries_checked, drifted, unresolved) for one phase manifest."""
    checked, drift, unresolved = 0, [], []
    for group, value in manifest.items():
        if not (group.endswith("sha256") and isinstance(value, dict)):
            continue
        for key, digest in value.items():
            if not (isinstance(digest, str) and len(digest) == 64):
                continue
            rel = _resolve_digest_entry(phase, manifest, group, key)
            target = (PROJECT_ROOT / rel) if rel else None
            if target is None or not target.is_file():
                unresolved.append(f"{group}:{key}")
                continue
            checked += 1
            if sha256(target) != digest:
                drift.append(f"{group}:{key} -> {rel}")
    return checked, drift, unresolved


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def add(self, label: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        pad = max(3, 78 - len(label))
        print(f"{label} {'.' * pad} {status}")
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
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True).strip()


def is_ancestor(sha: str, ref: str = "HEAD") -> bool:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", sha, ref],
        capture_output=True,
    ).returncode == 0


def load_json(path: Path):
    return json.loads(path.read_text())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_sections(md: str) -> dict[str, str]:
    """Split a markdown document on level-2 headings."""
    sections: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []
    for line in md.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf)
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf)
    return sections


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", flat)


NUM_TOKEN = re.compile(
    r"\d{4}-\d{2}-\d{2}"          # ISO date
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"   # grouped: 20,336 / 1,233,036
    r"|\d+(?:\.\d+)?%?"            # plain: 411012 / 31.987
)

# Frozen artifacts are CSV and JSON, where a comma is a DELIMITER, not a
# thousands separator. Scanning them with the grouping-aware pattern above
# silently welds adjacent fields together ("...h24,229.790698" reads as the
# single number 24229.790698). Source scanning therefore uses a
# grouping-free pattern; rendered prose and tables use NUM_TOKEN.
SRC_NUM_TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?")

# Identifiers and labels that contain digits but assert no quantity. Stripped
# before the numeric-traceability scan. This is a scoping rule, not an
# exemption: it keeps the gate pointed at scientific numbers. Directly
# analogous to the Phase-3 scoping correction, where repository-wide gates were
# narrowed to the outputs each phase actually owns.
LABEL_PATTERNS = [
    r"\b(?:Table|Figure|Fig\.|Section|Stage|Appendix)\s+\d+",   # cross-references
    r"\b[A-Za-z]+_R\d\b",                                       # B3_R2, GRU_R1, SA_R3
    r"\bB[0-3]\b", r"\bR[0-3]\b", r"\bC[1-5]\b", r"\bH[1-5]\b",
    r"\bPM2\.5\b", r"\bPM10\b", r"\bSO2\b", r"\bNO2\b", r"\bO3\b",
    r"\bCC BY 4\.0\b", r"\bChronos-2\b", r"\bTimesFM-3\.0\b",
    r"\bMoirai-2\.0-R-small\b", r"\bAirSense V[12]\b", r"\bV[12]\b",
    r"\bIPv\d\b", r"\bUCI ID \d+\b",
    r"\bSHA-256\b", r"\bx86-64\b", r"\bUTF-8\b",   # identifiers, not quantities
    r"10\.\d{4,9}/[^\s)\],;]+",                                  # DOIs
    r"arXiv:\d+\.\d+",                                          # arXiv ids
]

# Section headings whose numbers are bibliographic rather than scientific.
NON_CLAIM_SECTIONS = ("References",)


def numeric_tokens(text: str) -> list[str]:
    """Numeric tokens from manuscript body, excluding structural markdown.

    Excluded: fenced blocks, headings, markdown table rules, the References
    section, citation-placeholder text, and digit-bearing identifiers that name
    a thing rather than measure one (model names, table/figure cross-references,
    chemical species, DOIs).
    """
    out: list[str] = []
    in_fence = False
    in_refs = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            in_refs = any(heading.endswith(s) or heading == s for s in NON_CLAIM_SECTIONS)
            continue
        if stripped.startswith("#"):
            continue
        if in_refs:
            continue
        if re.fullmatch(r"\|[\s:|-]*\|", stripped):
            continue
        # Ordered-list markers number items; they measure nothing.
        scrubbed = re.sub(r"^\s*\d+\.\s+", " ", line)
        for pat in LABEL_PATTERNS:
            scrubbed = re.sub(pat, " ", scrubbed)
        # Citation placeholders describe what is needed; they assert nothing.
        scrubbed = re.sub(r"\[REF-NEEDED:[^\]]*\]", " ", scrubbed)
        out.extend(NUM_TOKEN.findall(scrubbed))
    return out


def normalise_num(tok: str) -> str:
    tok = tok.strip().rstrip("%").replace(",", "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
        return tok
    try:
        f = float(tok)
    except ValueError:
        return tok
    if f == int(f) and abs(f) < 1e15 and "." not in tok:
        return str(int(f))
    return tok.rstrip("0").rstrip(".") if "." in tok else tok


def main() -> int:
    r = Report()
    print("AirSense V2 - Phase 11 validator")
    print("Manuscript construction. No new science. Read-only.\n")

    # ---------------------------------------------------------------- Git
    r.section("Git state and historical ancestry")
    branch = git("branch", "--show-current")
    r.add("Branch is master", branch == "master", f"branch={branch}")

    head = git("rev-parse", "HEAD")
    r.add("HEAD resolves", bool(re.fullmatch(r"[0-9a-f]{40}", head)), head)

    r.add("Pre-test freeze is an ancestor of HEAD", is_ancestor(PRE_TEST_SHA))
    r.add("Locked-evaluation freeze is an ancestor of HEAD", is_ancestor(LOCKED_EVAL_SHA))
    r.add("Post-evaluation semantics commit is an ancestor of HEAD", is_ancestor(POST_EVAL_SHA))
    r.add("Phase-9 closure is an ancestor of HEAD", is_ancestor(PHASE9_CLOSURE_SHA))

    r.add("Ancestry order: pre-test precedes locked evaluation",
          is_ancestor(PRE_TEST_SHA, LOCKED_EVAL_SHA))
    r.add("Ancestry order: locked evaluation precedes post-evaluation semantics",
          is_ancestor(LOCKED_EVAL_SHA, POST_EVAL_SHA))
    r.add("Ancestry order: post-evaluation semantics precedes Phase-9 closure",
          is_ancestor(POST_EVAL_SHA, PHASE9_CLOSURE_SHA))

    legacy_sha = "16c9030cf74508b620cc6d28f90346aa379f29cd"
    try:
        legacy_tip = git("rev-parse", "legacy")
        r.add("legacy branch is unchanged", legacy_tip == legacy_sha, f"tip={legacy_tip}")
    except subprocess.CalledProcessError:
        r.add("legacy branch is unchanged", False, "legacy ref not found")

    r.add("legacy is not an ancestor of master (separate V1 lineage)",
          not is_ancestor(legacy_sha, "HEAD"))

    # --------------------------------------------------- Phase-10 integrity
    r.section("Phase-10 zero-drift integrity gate")
    manifest10 = load_json(ARTIFACTS / "v2_phase10_manifest.json")
    drift = []
    for rel, expected in manifest10["artifact_sha256"].items():
        p = PROJECT_ROOT / rel
        if not p.exists() or sha256(p) != expected:
            drift.append(rel)
    r.add("Every Phase-10 manifest artifact is bit-identical", not drift, f"drift={drift[:5]}")
    r.add("Phase-10 manifest covers the full evidence base",
          all(a in manifest10["artifact_sha256"] for a in PHASE10_ARTIFACT_LIST),
          "missing=" + str([a for a in PHASE10_ARTIFACT_LIST
                            if a not in manifest10["artifact_sha256"]][:5]))
    r.add("Phase-10 status is SYNTHESIS_FROZEN",
          manifest10["phase10_status"] == "SYNTHESIS_FROZEN")
    r.add("Phase-10 recorded zero models trained", manifest10["models_trained"] == 0)
    r.add("Phase-10 recorded zero predictions generated", manifest10["predictions_generated"] == 0)
    r.add("Phase-10 recorded zero Phase-8 drift", manifest10["phase8_scientific_drift"] == 0)
    r.add("Phase-10 recorded zero Phase-9 drift", manifest10["phase9_scientific_drift"] == 0)
    r.add("Phase-10 confirmatory hierarchy unchanged",
          manifest10["confirmatory_hierarchy_changed"] is False)

    # ------------------------------------------------ Phase-8 locked values
    r.section("Locked confirmatory evidence")
    lock = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    r.add("Primary value matches the frozen lock", lock["primary_value"] == LOCKED["primary"])
    r.add("Secondary value matches the frozen lock", lock["secondary_value"] == LOCKED["secondary_severe"])
    r.add("Reference macro value matches the frozen lock",
          lock["reference_primary_value"] == LOCKED["reference_macro"])
    r.add("Reference severe value matches the frozen lock",
          lock["reference_severe_value"] == LOCKED["reference_severe"])
    r.add("Severe n matches the frozen lock", lock["secondary_severe_n"] == LOCKED["severe_n"])
    r.add("Test sample count matches the frozen lock", lock["test_target_count"] == LOCKED["test_n"])
    r.add("Severe threshold is 244.0", lock["severe_threshold"] == LOCKED["threshold"])
    r.add("final_test_status is evaluated", lock["final_test_status"] == "evaluated")
    r.add("No model selection occurred after the test", lock["model_selection_after_test"] is False)
    r.add("No prediction changed after scoring", lock["prediction_changes_after_scoring"] is False)

    roles = load_json(ARTIFACTS / "phase10_model_role_ledger.json")
    r.add("B3_R2 is the primary confirmatory model",
          roles["roles"]["B3_R2"] == "PRIMARY CONFIRMATORY MODEL")
    r.add("GRU_R1 is the secondary confirmatory severe-tail model",
          roles["roles"]["GRU_R1"] == "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL")
    r.add("B0 is the reference benchmark", roles["roles"]["B0"] == "REFERENCE BENCHMARK")
    r.add("B0 is not a selected learned model", roles["B0_is_a_selected_learned_model"] is False)
    r.add("No winner label was assigned", roles["winner_label_assigned"] is False)
    r.add("Post-test promotion is not permitted",
          roles["promotion_after_test_access_permitted"] is False)
    r.add("Thirteen development-only models are recorded",
          len(roles["development_only_models"]) == 13)
    r.add("Zero foundation models were executed", roles["foundation_models_executed"] == 0)

    # ---------------------------------------------------- Manuscript freeze
    r.section("Phase-11 manuscript freeze")
    freeze_path = ARTIFACTS / "phase11_manuscript_freeze.json"
    r.add("Manuscript freeze exists", freeze_path.exists())
    freeze = load_json(freeze_path) if freeze_path.exists() else {}
    r.add("Freeze declares MANUSCRIPT_CONSTRUCTION",
          freeze.get("classification") == "MANUSCRIPT_CONSTRUCTION")
    r.add("Freeze was written before prose", freeze.get("written_before_manuscript_prose") is True)
    r.add("Freeze declares science complete", freeze.get("science_complete") is True)
    for field in ("new_scientific_experiments", "new_test_analysis", "new_metrics",
                  "models_trained", "predictions_generated"):
        r.add(f"Freeze declares {field} == 0", freeze.get(field) == 0)
    for field in ("phase8_mutable", "phase9_mutable", "phase10_mutable",
                  "hypothesis_status_mutable", "confirmatory_hierarchy_mutable"):
        r.add(f"Freeze declares {field} is false", freeze.get(field) is False)
    r.add("Freeze pins the master evidence index",
          freeze.get("publication_evidence_source_sha256")
          == sha256(ARTIFACTS / "phase10_master_evidence_index.json"))
    r.add("Freeze pins the publication claim guide",
          freeze.get("claim_language_authority_sha256")
          == sha256(DOCS / "PUBLICATION_CLAIM_GUIDE.md"))
    # Ancestry, not equality - the same correction applied to the Git-discipline
    # gate below. The freeze records which commit WAS the Phase-10 closure; that
    # is a historical fact and stays true forever. Comparing it to current HEAD
    # asserted that Phase 11 had never been committed, which stops being true the
    # moment it is.
    _p10 = freeze.get("git_anchors", {}).get("phase10_closure_sha")
    r.add("Freeze records a Phase-10 closure SHA that is an ancestor of HEAD",
          bool(_p10) and bool(re.fullmatch(r"[0-9a-f]{40}", _p10)) and is_ancestor(_p10),
          f"recorded={_p10}")
    r.add("Freeze declares zero new figures",
          freeze.get("planned_figures", {}).get("new_figures_created") == 0)
    r.add("Freeze prohibits evidence-class upgrades",
          freeze.get("claim_language_policy", {}).get("class_upgrade_permitted") is False)
    r.add("Freeze prohibits presenting C3 as C1",
          freeze.get("claim_language_policy", {}).get("c3_as_c1_permitted") is False)
    r.add("Freeze prohibits reference fabrication",
          freeze.get("citation_policy", {}).get("fabrication_prohibited") is True)
    fl = freeze.get("frozen_locked_values", {})
    r.add("Freeze carries the full-precision primary value",
          fl.get("primary_B3_R2_macro_station_horizon_MAE") == LOCKED["primary"])
    r.add("Freeze carries the full-precision secondary value",
          fl.get("secondary_GRU_R1_severe_MAE_gt_244") == LOCKED["secondary_severe"])
    r.add("Freeze carries the full-precision reference values",
          fl.get("reference_B0_macro_station_horizon_MAE") == LOCKED["reference_macro"]
          and fl.get("reference_B0_severe_MAE_gt_244") == LOCKED["reference_severe"])

    # ------------------------------------------------------- No new science
    r.section("No new science was performed in Phase 11")
    phase11_new = sorted(p.name for p in ARTIFACTS.glob("phase11_*"))
    allowed_phase11 = {
        "phase11_manuscript_freeze.json",
        "phase11_verification_tooling_registry.json",
        "phase11_manuscript_summary.json",
        "phase11_publication_package_index.json",
    }
    r.add("Phase-11 artifacts are governance only, no results files",
          set(phase11_new) <= allowed_phase11,
          f"unexpected={sorted(set(phase11_new) - allowed_phase11)}")

    banned_new = []
    for pat in ("phase11_*metric*", "phase11_*prediction*", "phase11_*result*",
                "phase11_*model*", "phase11_*eval*"):
        banned_new.extend(p.name for p in ARTIFACTS.glob(pat))
    r.add("No Phase-11 metric, prediction, result or model artifact exists", not banned_new,
          f"found={banned_new}")

    r.add("No Phase-11 figure was created",
          not list(FIGURES.glob("phase11_*")),
          f"found={[p.name for p in FIGURES.glob('phase11_*')][:5]}")

    # Phase 8 and 9 evidence must be untouched.
    #
    # These manifests do NOT use an "artifact_sha256" map - that key belongs to
    # the Phase-10 and Phase-11 manifests only. Phase 8 records digests under
    # metric_table_sha256 / prediction_sha256 / model_artifact_sha256 /
    # figure_sha256 / phase8_code_sha256 / upstream_manifest_sha256 /
    # phase7_freeze_sha256, and Phase 9 under table_sha256 / figure_sha256 /
    # phase9_code_sha256. An earlier revision of this validator read
    # .get("artifact_sha256", {}) on both, iterated an empty dict, and PASSED
    # WITHOUT VERIFYING ANYTHING. The floors and the unresolved-count gates
    # below exist so that failure mode can never recur silently: a check that
    # can verify nothing must fail, not pass.
    for phase, manifest_name, floor in (("Phase-8", "v2_phase8_manifest.json", PHASE8_DIGEST_FLOOR),
                                        ("Phase-9", "v2_phase9_manifest.json", PHASE9_DIGEST_FLOOR)):
        manifest = load_json(ARTIFACTS / manifest_name)
        checked, drift, unresolved = verify_manifest_digests(phase, manifest)
        r.add(f"{phase} manifest digests all resolve to real files",
              not unresolved, f"unresolved={unresolved[:6]}")
        r.add(f"{phase} artifacts are bit-identical", not drift, f"drift={drift[:5]}")
        r.add(f"{phase} digest check is non-vacuous (>= {floor} entries)",
              checked >= floor, f"checked={checked}")

        scalars = PHASE8_SCALAR_DIGESTS if phase == "Phase-8" else PHASE9_SCALAR_DIGESTS
        s_checked, s_drift, s_absent = 0, [], []
        for field, rel in scalars.items():
            expected = manifest.get(field)
            if not isinstance(expected, str):
                s_absent.append(f"{field}:no-such-field")
                continue
            target = PROJECT_ROOT / rel
            if not target.is_file():
                # data/processed is gitignored and absent from a fresh clone
                # until it is rebuilt. Skipping is legitimate; skipping
                # silently is not, so the skip is reported.
                s_absent.append(f"{field}:{rel}")
                continue
            s_checked += 1
            if sha256(target) != expected:
                s_drift.append(f"{field}:{rel}")
        r.add(f"{phase} scalar digests are bit-identical", not s_drift, f"drift={s_drift}")
        r.add(f"{phase} scalar digest check is non-vacuous",
              s_checked >= len(scalars) - len(OPTIONAL_DIGEST_TARGETS),
              f"checked={s_checked} skipped={s_absent}")
        unexpected_absent = [a for a in s_absent
                             if a.split(":", 1)[1] not in OPTIONAL_DIGEST_TARGETS]
        r.add(f"{phase} scalar digest targets are all present or declared optional",
              not unexpected_absent, f"absent={unexpected_absent}")

    # ------------------------------------------------------ Package presence
    r.section("Publication package completeness")
    for rel in PACKAGE_FILES:
        r.add(f"Package file exists: {rel.split('/')[-1]}", (PROJECT_ROOT / rel).exists())

    if not MANUSCRIPT_MD.exists():
        print("\nManuscript absent; remaining gates cannot run.")
        return 1

    md = read_text(MANUSCRIPT_MD)
    sections = split_sections(md)
    lower = md.lower()

    # ------------------------------------------------------ Locked values
    r.section("Locked values as reported in the manuscript")
    for spelling in REQUIRED_ROUNDED:
        r.add(f"Manuscript reports {spelling}", spelling in md)
    r.add("Manuscript reports the full-precision primary value",
          repr(LOCKED["primary"]) in md or "31.98722559774534" in md)
    r.add("Manuscript reports the full-precision secondary value",
          "109.3328897571946" in md)
    r.add("Manuscript reports the full-precision reference macro value",
          "36.138513538776856" in md)
    r.add("Manuscript reports the full-precision reference severe value",
          "93.22447875688434" in md)
    r.add("Manuscript states the severe rule as strictly greater than 244.0",
          re.search(r">\s*244\.0", md) is not None)
    r.add("Manuscript never states the severe rule as >= 244.0",
          re.search(r">=\s*244(\.0)?|≥\s*244(\.0)?", md) is None)
    r.add("Manuscript states the residual convention",
          re.search(r"actual\s*[-−]\s*prediction", md, re.I) is not None)
    r.add("Manuscript states positive residual means under-prediction",
          re.search(r"positive[^.]{0,80}under[- ]prediction", md, re.I) is not None)

    # ------------------------------------------- Headline: both halves
    r.section("Headline discipline")
    abstract = sections.get("Abstract", "")
    r.add("Abstract exists", bool(abstract.strip()))
    r.add("Abstract carries the improvement over persistence",
          "11.49" in abstract or "11.487" in abstract)
    r.add("Abstract carries the primary value", "31.99" in abstract)
    r.add("Abstract carries the reference macro value", "36.14" in abstract)
    r.add("Abstract carries the secondary severe value", "109.33" in abstract)
    r.add("Abstract carries the reference severe value", "93.22" in abstract)
    r.add("Abstract states the severe threshold provenance",
          "training pooled p95" in abstract.lower())
    r.add("Abstract does not imply H1-H4 were confirmed on the locked test",
          not re.search(r"confirm\w*[^.]{0,60}(H1|H2|H3|H4|hypothes)", abstract, re.I))
    r.add("Abstract records that persistence held the severe tail",
          re.search(r"persistence[^.]{0,120}(strongest|lowest|best)", abstract, re.I) is not None)

    conclusion = sections.get("14. Conclusion", "")
    r.add("Conclusion section exists", bool(conclusion.strip()))
    r.add("Conclusion keeps the severe tail unresolved",
          re.search(r"(unresolved|open problem|remains? (an )?open)", conclusion, re.I) is not None)
    r.add("Conclusion retains persistence as strongest severe benchmark",
          re.search(r"persistence", conclusion, re.I) is not None)

    # -------------------------------------------------- Prohibited language
    r.section("Prohibited language")
    for pattern, label in BANNED_PATTERNS:
        if label in NEGATION_REQUIRED:
            # The guide prohibits MAKING an inferential claim, not disclaiming
            # one. "No confidence interval was predeclared" is a required
            # limitation; "the gain is significant" is a prohibited claim.
            offending = [s for s in sentences(md)
                         if re.search(pattern, s, re.I)
                         and not re.search(NEGATION_MARKERS, s, re.I)]
            r.add(f"Manuscript asserts no '{label}'", not offending,
                  f"hits={[s[:110] for s in offending[:2]]}")
        else:
            hits = [m.group(0) for m in re.finditer(pattern, md, re.I)]
            r.add(f"Manuscript avoids '{label}'", not hits, f"hits={hits[:4]}")

    r.section("Required terminology")
    for phrase in REQUIRED_PHRASES:
        r.add(f"Manuscript uses '{phrase}'", phrase.lower() in lower)

    # -------------------------------- Evidence-class and hypothesis fidelity
    r.section("Evidence classes and hypothesis status")
    for cls in ("C1", "C2", "C3", "C4", "C5"):
        r.add(f"Manuscript labels {cls} evidence", re.search(rf"\b{cls}\b", md) is not None)

    hyp = load_json(ARTIFACTS / "phase10_hypothesis_ledger.json")["hypotheses"]
    r.add("Hypothesis ledger still carries exactly H1-H5", sorted(hyp) == ["H1","H2","H3","H4","H5"])
    for hid, body in hyp.items():
        r.add(f"{hid} was not retested on the locked test",
              body["locked_test_retested"] is False)
    r.add("H1 status unchanged", hyp["H1"]["status"] == "SUPPORTED ON DEVELOPMENT VALIDATION")
    r.add("H2 status unchanged", hyp["H2"]["status"] == "SUPPORTED ON DEVELOPMENT VALIDATION")
    r.add("H3 status unchanged", hyp["H3"]["status"] == "WEAK / MODEL-DEPENDENT DEVELOPMENT SUPPORT")
    r.add("H4 status unchanged", hyp["H4"]["status"].startswith("ROBUSTLY SUPPORTED"))
    r.add("H5 status unchanged", hyp["H5"]["status"].startswith("PARTIALLY SUPPORTED"))

    hyp_section = sections.get("1. Introduction", "") + sections.get("12. Limitations", "")
    r.add("Manuscript states H1-H4 were not re-tested on the locked test",
          re.search(r"not\s+(all\s+)?(independently\s+)?(re-?tested|confirmed)",
                    re.sub(r"\s+", " ", md), re.I) is not None)

    # C3 must never be sold as confirmatory.
    posttest = sections.get("10. Post-test exploratory error analysis", "")
    r.add("Post-test section exists", bool(posttest.strip()))
    r.add("Post-test section is labelled exploratory",
          "exploratory" in posttest.lower())
    r.add("Post-test section declares it followed the locked evaluation",
          re.search(r"after[^.]{0,80}(locked|test was (opened|scored))", posttest, re.I) is not None)
    r.add("Post-test section never claims confirmatory status",
          not re.search(r"\bconfirmatory\b(?![^.]{0,40}(not|never|rather than))", posttest, re.I)
          or "not confirmatory" in posttest.lower())

    locked_section = sections.get("9. Locked final-test results", "")
    r.add("Locked final-test section exists", bool(locked_section.strip()))
    for dev_model in ("SA_R2", "SA_R3", "TCN_R1", "iTransformer"):
        r.add(f"Locked-test section does not report {dev_model}",
              dev_model not in locked_section)
    r.add("Locked-test section names the three frozen roles",
          all(m in locked_section for m in ("B3_R2", "GRU_R1", "B0")))

    # ------------------------------------------------- V1/V2 and governance
    r.section("Cross-study and governance discipline")
    v1_sentences = [s for s in sentences(md) if re.search(r"\bV1\b", s)]
    bad_v1 = [s for s in v1_sentences if re.search(r"\d+(\.\d+)?\s*%", s)]
    r.add("No V1/V2 percentage comparison", not bad_v1, f"hits={bad_v1[:2]}")
    bad_v1_metric = [s for s in v1_sentences
                     if re.search(r"\bMAE\b[^.]{0,40}\d", s) and not re.search(r"not|never|invalid", s, re.I)]
    r.add("No V1/V2 metric benchmark", not bad_v1_metric, f"hits={bad_v1_metric[:2]}")

    fm_sentences = [s for s in sentences(md)
                    if re.search(r"Chronos|TimesFM|Moirai|foundation model", s, re.I)]
    bad_fm = [s for s in fm_sentences
              if re.search(r"\b(worse|poorly|underperform\w*|lost|beaten|outperformed by)\b", s, re.I)]
    r.add("No foundation-model performance claim", not bad_fm, f"hits={bad_fm[:2]}")
    r.add("Manuscript records zero foundation models executed",
          re.search(r"(no|zero|none)[^.]{0,60}foundation model[^.]{0,40}(executed|evaluated|run)",
                    md, re.I) is not None
          or re.search(r"foundation model[^.]{0,60}(was|were) (not|never) (executed|run|evaluated)",
                       md, re.I) is not None)
    r.add("Chronos-2 exclusion is stated as overlap risk, not demonstrated contamination",
          not re.search(r"byte[- ]?(level|identical)[^.]{0,60}contaminat", md, re.I))

    sa_sentences = [s for s in sentences(md) if re.search(r"SA_R3|cross-station attention", s)]
    bad_sa = [s for s in sa_sentences
              if re.search(r"\b(propagation|transport|spatial reasoning|geographic)\b", s, re.I)
              and not re.search(r"\bnot\b|\bno\b|\bnever\b|must not", s, re.I)]
    r.add("Cross-station attention is not framed as propagation", not bad_sa, f"hits={bad_sa[:2]}")
    r.add("Manuscript records the attention-distance diagnostic",
          "0.136" in md)

    loso_sentences = [s for s in sentences(md) if re.search(r"LOSO|leave-one-station", s, re.I)]
    r.add("LOSO is described with the required qualifier",
          any("held-out target-station supervision transfer" in s for s in loso_sentences)
          or "held-out target-station supervision transfer" in lower)

    # ------------------------------------------------------- Limitations
    r.section("Limitations and negative results")
    limits = load_json(ARTIFACTS / "phase10_limitations_ledger.json")
    lim_section = sections.get("12. Limitations", "")
    r.add("Limitations section exists", bool(lim_section.strip()))
    lim_items = [ln for ln in lim_section.splitlines()
                 if re.match(r"^\s*(?:\d+\.|[-*])\s+\S", ln)]
    r.add("Limitations section is not a token paragraph", len(lim_items) >= 12,
          f"items={len(lim_items)}")
    r.add("Limitations ledger still records 17 items", limits["limitation_count"] == 17)

    for topic, pattern in [
        ("single metropolitan network", r"single (metropolitan|city)|one city"),
        ("no external-city validation", r"external[- ]city|other cities|another city"),
        ("historical not forecast meteorology", r"forecast\w* meteorolog|observed meteorolog"),
        ("GRU seed-42 artifact", r"seed 42"),
        ("hypotheses not all retested", r"not (all )?(independently )?re-?tested"),
        ("training P95 threshold", r"training pooled P95"),
        ("post-test exploratory status", r"post-test exploratory"),
        ("no predeclared intervals", r"no[^.]{0,60}(interval|uncertainty)"),
        ("LOSO scope", r"held-out target-station supervision transfer"),
        ("coordinates analysis only", r"coordinate\w*[^.]{0,60}(analysis|descriptive|never)"),
        ("attention not geographic", r"attention[^.]{0,80}(not|no)[^.]{0,40}(proximity|distance|geograph)"),
        ("foundation-model exclusion", r"foundation model"),
        ("negatives unclipped", r"unclipped|not clipped|never clipped"),
        ("residual dependence", r"residual[^.]{0,40}(dependence|autocorrelation)"),
        ("severe forecasting unresolved", r"severe[^.]{0,60}(unresolved|open)"),
    ]:
        r.add(f"Limitations cover: {topic}",
              re.search(pattern, re.sub(r"\s+", " ", lim_section), re.I) is not None)

    r.add("Negative severe-tail result retained in the manuscript",
          re.search(r"93\.22", md) is not None
          and re.search(r"persistence[^.]{0,160}(strongest|lowest)", md, re.I) is not None)
    r.add("Severe under-prediction at long horizon retained", "99.9" in md or "99.90" in md)
    r.add("Severe detection recall collapse retained", "0.0069" in md or "0.69" in md)
    r.add("Negative predictions retained", "1,358" in md or "1358" in md)

    # ----------------------------------------------------------- Future work
    r.section("Future work")
    fw_section = sections.get("13. Future work", "")
    r.add("Future-work section exists", bool(fw_section.strip()))
    r.add("Future work is marked as not tested in this study",
          re.search(r"not tested in this study", fw_section, re.I) is not None)
    r.add("Future work claims no improvement",
          not re.search(r"\bwill improve\b|\bwould improve\b", fw_section, re.I))

    # ------------------------------------------------------------- Citations
    r.section("Citations")
    refs_path = MANUSCRIPT / "REFERENCES_VERIFIED.md"
    refs = read_text(refs_path) if refs_path.exists() else ""
    r.add("References file has a VERIFIED section", "## VERIFIED" in refs)
    r.add("References file has a NEEDS VERIFICATION section", "## NEEDS VERIFICATION" in refs)

    placeholders = re.findall(r"\[REF-NEEDED:[^\]]*\]", md)
    r.add("Every citation placeholder is well formed",
          all(p.startswith("[REF-NEEDED:") and len(p) > 14 for p in placeholders),
          f"n={len(placeholders)}")

    # A DOI in the manuscript must also appear in the verified reference list.
    def _dois(t: str) -> set[str]:
        # Trailing sentence punctuation is not part of a DOI.
        return {d.rstrip(".,;:*)") for d in re.findall(r"10\.\d{4,9}/[^\s)\],;]+", t)}
    dois, ref_dois = _dois(md), _dois(refs)
    missing_dois = sorted(d for d in dois if d not in ref_dois)
    r.add("Every DOI cited in the manuscript is in the reference list",
          not missing_dois, f"missing={missing_dois[:3]}")

    verified_block = refs.split("## NEEDS VERIFICATION")[0] if "## NEEDS VERIFICATION" in refs else refs
    for field in ("Authors", "Title", "Venue", "Year", "Verification source"):
        r.add(f"Verified references record {field}", field in verified_block)

    r.add("No unverified reference is presented as verified",
          "[REF-NEEDED" not in verified_block)

    # ------------------------------------------------ Tables map to evidence
    r.section("Tables map to frozen evidence")
    table_sources = {
        "manuscript/tables/table3_development_results.md": "artifacts/phase10_table_development_models.csv",
        "manuscript/tables/table4_multiseed_robustness.md": "artifacts/phase10_table_multiseed_robustness.csv",
        "manuscript/tables/table5_locked_final_test.md": "artifacts/phase10_table_locked_test.csv",
        "manuscript/tables/table6_posttest_diagnostics.md": "artifacts/phase10_table_posttest_diagnostics.csv",
    }
    for tbl, src in table_sources.items():
        tp, sp = PROJECT_ROOT / tbl, PROJECT_ROOT / src
        if not tp.exists():
            r.add(f"{Path(tbl).name} maps to frozen evidence", False, "table file missing")
            continue
        body = read_text(tp)
        r.add(f"{Path(tbl).name} records its source artifact", src in body)
        r.add(f"{Path(tbl).name} records the source SHA-256", sha256(sp) in body)
        r.add(f"{Path(tbl).name} declares an evidence class",
              re.search(r"\*\*Evidence class:\*\*\s*C[1-5]", body) is not None)
        r.add(f"{Path(tbl).name} carries a caption", "**Caption" in body)

        # Every numeric cell in the table must appear in the source artifact.
        # Scope: the rendered data rows only. The provenance header carries a
        # SHA-256 whose hex digits are not quantities, and the footnotes carry
        # prose claims traced through the claim-traceability matrix instead.
        data_rows = [ln for ln in body.splitlines()
                     if ln.strip().startswith("|")
                     and not re.fullmatch(r"\|[\s:|-]*\|", ln.strip())]
        src_nums = {normalise_num(t) for t in SRC_NUM_TOKEN.findall(read_text(sp))}
        tbl_nums = {normalise_num(t) for t in numeric_tokens("\n".join(data_rows))}
        # Rounded values are permitted if they round-trip from a source value.
        unmatched = []
        for n in tbl_nums:
            if n in src_nums:
                continue
            try:
                v = float(n)
            except ValueError:
                unmatched.append(n)
                continue
            if any(abs(v - float(s)) < 5e-3 * max(1.0, abs(v))
                   for s in src_nums if re.fullmatch(r"-?\d+(\.\d+)?", s)):
                continue
            unmatched.append(n)
        r.add(f"{Path(tbl).name} numbers all trace to the source artifact",
              not unmatched, f"unmatched={sorted(unmatched)[:6]}")

    # ------------------------------------------------ Figures are frozen
    r.section("Figures reference frozen assets")
    fig_index = load_json(ARTIFACTS / "phase10_publication_figure_index.json")
    r.add("Figure index declares zero new figures", fig_index["new_figures_created"] == 0)
    r.add("Figure index declares zero modified figures", fig_index["existing_figures_modified"] == 0)
    plan_path = MANUSCRIPT / "FIGURE_PLAN.md"
    plan = read_text(plan_path) if plan_path.exists() else ""
    for fig in fig_index["figures"]:
        src = fig["source_path"]
        r.add(f"{fig['figure_id']} source exists on disk", (PROJECT_ROOT / src).exists())
        r.add(f"{fig['figure_id']} is listed in the figure plan", src in plan)
    r.add("Figure plan records evidence classes", plan.count("Evidence class") >= 10)
    r.add("Figure plan records what each figure must not claim",
          plan.lower().count("must not") >= 10)

    # ------------------------------------------------ Claim traceability
    r.section("Claim traceability")
    if not TRACEABILITY.exists():
        r.add("Traceability CSV exists", False)
    else:
        r.add("Traceability CSV exists", True)
        with TRACEABILITY.open() as fh:
            rows = list(csv.DictReader(fh))
        required_cols = {"manuscript_section", "claim_excerpt", "claim_id", "evidence_class",
                         "source_artifact", "source_field", "claim_value",
                         "source_sha256", "verification_status"}
        r.add("Traceability CSV has the required columns",
              rows and required_cols <= set(rows[0].keys()),
              f"missing={sorted(required_cols - set(rows[0].keys())) if rows else 'no rows'}")
        r.add("Traceability CSV is not empty", len(rows) > 0, f"rows={len(rows)}")
        bad_status = [x["claim_id"] for x in rows if x["verification_status"] != "VERIFIED"]
        r.add("Every traceability row is VERIFIED", not bad_status, f"unverified={bad_status[:5]}")
        bad_class = [x["claim_id"] for x in rows
                     if x["evidence_class"] not in {"C1", "C2", "C3", "C4", "C5"}]
        r.add("Every traceability row carries a valid evidence class",
              not bad_class, f"bad={bad_class[:5]}")

        # Source artifacts must exist and hash as recorded.
        bad_sha = []
        for x in rows:
            art = x["source_artifact"].strip()
            if not art or art == "n/a":
                continue
            p = PROJECT_ROOT / art
            if not p.exists():
                bad_sha.append(f"{x['claim_id']}:missing")
            elif x["source_sha256"].strip() and sha256(p) != x["source_sha256"].strip():
                bad_sha.append(f"{x['claim_id']}:sha")
        r.add("Every traceability source artifact exists and hashes as recorded",
              not bad_sha, f"bad={bad_sha[:5]}")

        # No C3 row may be described as confirmatory.
        c3_conf = [x["claim_id"] for x in rows
                   if x["evidence_class"] == "C3" and "confirmatory" in x["claim_excerpt"].lower()]
        r.add("No C3 claim is described as confirmatory", not c3_conf, f"bad={c3_conf[:5]}")

        # Every number in the manuscript must be a declared claim value.
        declared = set()
        for x in rows:
            for tok in NUM_TOKEN.findall(x["claim_value"]):
                declared.add(normalise_num(tok))
        body_nums = [normalise_num(t) for t in numeric_tokens(md)]
        untraceable = sorted({n for n in body_nums if n not in declared})
        r.add("Every numeric token in the manuscript is a declared claim value",
              not untraceable, f"untraceable={untraceable[:12]}")

    # ------------------------------------------------- Plaintext portability
    r.section("Plaintext portability copy")
    if MANUSCRIPT_TXT.exists():
        txt = read_text(MANUSCRIPT_TXT)
        r.add("Plaintext copy is non-trivial", len(txt) > 5000, f"chars={len(txt)}")
        for spelling in ("31.99", "36.14", "109.33", "93.22"):
            r.add(f"Plaintext copy carries {spelling}", spelling in txt)
        r.add("Plaintext copy carries no markdown table pipes",
              "|---" not in txt and "| ---" not in txt)
    else:
        r.add("Plaintext copy exists", False)

    # ------------------------------------------------------- Tooling registry
    r.section("Append-only verification tooling registry")
    reg_path = ARTIFACTS / "phase11_verification_tooling_registry.json"
    r.add("Phase-11 tooling registry exists", reg_path.exists())
    if reg_path.exists():
        reg = load_json(reg_path)
        r.add("Registry parent is the Phase-10 registry",
              reg.get("parent_registry") == "artifacts/phase10_verification_tooling_registry.json")
        r.add("Registry pins the Phase-10 registry hash",
              reg.get("parent_registry_sha256")
              == sha256(ARTIFACTS / "phase10_verification_tooling_registry.json"))
        pinned = reg.get("tools", {})
        r.add("Registry pins validate_phase11.py", "scripts/validate_phase11.py" in pinned)
        for ref in ("scripts/validate_phase8.py", "scripts/validate_phase8_postopening.py",
                    "scripts/validate_phase9.py", "scripts/validate_phase10.py"):
            r.add(f"Registry references {Path(ref).name}", ref in pinned)
        bad_pins = [k for k, v in pinned.items()
                    if (PROJECT_ROOT / k).exists()
                    and v.get("sha256") and k != "scripts/validate_phase11.py"
                    and sha256(PROJECT_ROOT / k) != v["sha256"]]
        r.add("Every referenced validator hashes as recorded", not bad_pins, f"bad={bad_pins}")
        r.add("Registry declares append-only semantics",
              reg.get("append_only") is True)

    # Prior registries must be untouched.
    prior = {
        "artifacts/verification_tooling_registry.json": "a1ec2652",
        "artifacts/phase8_verification_tooling_registry.json": "c59abf27",
        "artifacts/phase8_postopening_tooling_registry.json": "d61b7ee7",
        "artifacts/phase8_postcommit_tooling_registry.json": "19797553",
        "artifacts/phase9_verification_tooling_registry.json": "187ac165",
        "artifacts/phase10_verification_tooling_registry.json": "0b1d8c11",
    }
    bad_prior = [k for k, pre in prior.items()
                 if not (PROJECT_ROOT / k).exists() or not sha256(PROJECT_ROOT / k).startswith(pre)]
    r.add("No earlier tooling registry was rewritten", not bad_prior, f"changed={bad_prior}")

    # --------------------------------------------------------- Git is clean
    r.section("Git write discipline")
    staged = git("diff", "--cached", "--name-only")
    r.add("Nothing is staged", staged == "", f"staged={staged.splitlines()[:5]}")
    # Ancestry, not equality. An earlier revision asserted
    # head == phase10_closure_sha, which is true only while Phase 11 is still
    # uncommitted: committing the manuscript package would have made this
    # validator fail on its own commit, creating a third permanently-failing
    # validator alongside validate_phase3 and validate_phase8. This project has
    # already learned that lesson once - validate_phase8_postopening replaced
    # exact-HEAD gates with freeze-point ancestry for exactly this reason.
    p10_closure = freeze.get("git_anchors", {}).get("phase10_closure_sha")
    r.add("Phase-10 closure is an ancestor of HEAD",
          bool(p10_closure) and is_ancestor(p10_closure),
          f"phase10_closure={p10_closure}")

    # ------------------------------------------------------------- Summary
    total = r.passed + r.failed
    print(f"\n{'=' * 72}")
    print(f"Phase 11 validation: {r.passed}/{total} gates passed, {r.failed} failed")
    if r.failed:
        print("\nFAILURES:")
        for f in r.failures:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
