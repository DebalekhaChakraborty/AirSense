"""Independent verification of the AirSense V2 final scientific synthesis.

Protocol Phase 10. **This source was finalized before any Phase-10 synthesis
output was produced** and is pinned in
``artifacts/phase10_verification_tooling_registry.json``.

Phase 10 freezes evidence; it runs no experiment. The gates below are mostly
negative - no model, no prediction, no new metric, no reclassification of
exploratory findings as confirmatory - plus positive traceability: every
publication claim must name an artifact, and every table value must reconstruct
from frozen evidence.

Prose is not gated verbatim. What is gated is classification, traceability,
arithmetic and the absence of prohibited claims.

Exits non-zero on any failure.

Usage:
    .venv-v2/bin/python scripts/validate_phase10.py
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "figures"

PRE_TEST_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LOCKED_EVALUATION_SHA = "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"
POST_EVAL_SHA = "7ccfc7e54f7d9008421cfcf764fcfaacc353e1db"
PHASE9_CLOSURE_SHA = "8cc3a7ab8a069b9a187904be68f14b9040928ab4"
LEGACY_SHA = "16c9030cf74508b620cc6d28f90346aa379f29cd"

CONFIRMATORY = ["B3_R2", "GRU_R1", "B0"]
DEVELOPMENT_ONLY = ["B1", "B2", "B3_R0", "B3_R1", "GRU_R0", "GRU_R2", "TCN_R0",
                    "TCN_R1", "TCN_R2", "iTransformer_R1", "iTransformer_R2",
                    "SA_R2", "SA_R3"]
CLASSES = ["C1", "C2", "C3", "C4", "C5"]
LOCKED = OrderedDict([
    ("primary_B3_R2_macro_station_horizon_MAE", 31.98722559774534),
    ("secondary_GRU_R1_severe_MAE_gt_244", 109.3328897571946),
    ("severe_n", 20336),
    ("reference_B0_macro_station_horizon_MAE", 36.138513538776856),
    ("reference_B0_severe_MAE_gt_244", 93.22447875688434),
])
MATRIX_COLUMNS = ["claim_id", "claim_text", "evidence_class", "phase_source",
                  "artifact_source", "metric_or_fact", "scope", "limitations",
                  "permitted_wording", "prohibited_wording"]
REQUIRED_CLAIM_TOPICS = ["dataset_audit", "chronological_split",
                         "test_sealing", "causal_features", "b3r2_locked",
                         "b0_comparison", "gru_severe", "persistence_severe",
                         "h1", "h2", "h3", "h4", "h5", "seed_instability",
                         "loso", "spatial_attention", "foundation_models",
                         "residual_acf", "severe_events", "seasonality",
                         "complementarity", "v1_v2"]

# Claims the synthesis must never make, matched on whole words.
PROHIBITED_PATTERNS = [
    (r"\bsolves?\s+severe\b", "claims severe forecasting is solved"),
    (r"\bGRU\w*\s+beats?\s+persistence\b", "claims GRU beats persistence"),
    (r"pollution\s+propagation", "claims learned pollution propagation"),
    (r"\bspatial\s+reasoning\b", "claims spatial reasoning"),
    (r"foundation\s+models?\s+failed", "claims foundation models failed"),
    (r"all\s+hypotheses\s+(were\s+)?confirmed", "claims all H confirmed"),
    (r"\boperationally\s+ready\b", "claims operational readiness"),
    (r"bias[-\s]variance\s+decomposition\s+(was\s+)?(performed|showed)",
     "claims a bias-variance decomposition was performed"),
    (r"improves?\s+V1[^.]{0,40}\d+(\.\d+)?\s*%",
     "cross-study numeric improvement claim"),
]


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


def read_csv(path):
    with open(str(path), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def git(*args):
    result = subprocess.run(["git"] + list(args), cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stdout.decode("utf-8").strip()


def is_ancestor(a, b):
    return git("merge-base", "--is-ancestor", a, b)[0] == 0


def anchor_checks(report):
    _, head = git("rev-parse", "HEAD")
    report.add("Phase-9 closure commit exists",
               git("cat-file", "-e", "%s^{commit}" % PHASE9_CLOSURE_SHA)[0]
               == 0)
    report.add("Ancestry: pre-test -> locked evaluation",
               is_ancestor(PRE_TEST_SHA, LOCKED_EVALUATION_SHA))
    report.add("Ancestry: locked evaluation -> post-evaluation validation",
               is_ancestor(LOCKED_EVALUATION_SHA, POST_EVAL_SHA))
    report.add("Ancestry: post-evaluation validation -> Phase-9 closure",
               is_ancestor(POST_EVAL_SHA, PHASE9_CLOSURE_SHA))
    report.add("Ancestry: Phase-9 closure -> current HEAD",
               is_ancestor(PHASE9_CLOSURE_SHA, head), head)
    report.add("V1 legacy unchanged",
               git("rev-parse", "legacy")[1] == LEGACY_SHA)
    freeze = load_json(ARTIFACTS / "phase10_synthesis_freeze.json")
    report.add("Synthesis freeze records the Phase-9 closure checkpoint",
               freeze["git_anchors"]["phase9_closure_sha"]
               == PHASE9_CLOSURE_SHA)
    report.add("Working tree was clean at Phase-10 start",
               all(not line.strip().startswith(("M ", " M", "D ", " D"))
                   for line in git("status", "--porcelain")[1].splitlines()),
               "modified or deleted tracked files present")


def drift_checks(report):
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    phase8 = load_json(ARTIFACTS / "v2_phase8_manifest.json")
    drift = []
    for name, entry in lock["models"].items():
        if sha256_of_file(PROJECT_ROOT / entry["prediction_file"]) \
                != entry["sha256"]:
            drift.append("prediction:" + name)
    for name, digest in results["metric_table_sha256"].items():
        if sha256_of_file(ARTIFACTS / "phase8_final_test_tables" / name) \
                != digest:
            drift.append("table:" + name)
    for group in ("upstream_manifest_sha256", "phase7_freeze_sha256"):
        for name, digest in phase8[group].items():
            if sha256_of_file(ARTIFACTS / name) != digest:
                drift.append(name)
    report.add("Phase-8 scientific drift is zero", not drift,
               "drifted: %s" % drift[:5])

    phase9 = load_json(ARTIFACTS / "v2_phase9_manifest.json")
    drift9 = []
    for name, digest in phase9["table_sha256"].items():
        if sha256_of_file(ARTIFACTS / name) != digest:
            drift9.append(name)
    for name, digest in phase9["figure_sha256"].items():
        if sha256_of_file(FIGURES / name) != digest:
            drift9.append(name)
    for key, path in (("phase9_analysis_freeze_sha256",
                       ARTIFACTS / "phase9_analysis_freeze.json"),
                      ("phase9_validator_sha256",
                       PROJECT_ROOT / "scripts" / "validate_phase9.py"),
                      ("phase9_tooling_registry_sha256",
                       ARTIFACTS / "phase9_verification_tooling_registry.json"),
                      ("summary_sha256",
                       ARTIFACTS / "phase9_post_test_analysis_summary.json"),
                      ("record_sha256",
                       DOCS / "PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md"),
                      ("v1_v2_comparison_sha256",
                       DOCS / "PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md")):
        if phase9.get(key) and sha256_of_file(path) != phase9[key]:
            drift9.append(str(path.name))
    report.add("Phase-9 scientific drift is zero", not drift9,
               "drifted: %s" % drift9[:5])


def no_experiment_checks(report, freeze):
    report.add("Synthesis freeze declares no new experimentation",
               freeze["new_scientific_experiment"] is False
               and freeze["new_test_analysis"] is False
               and freeze["models_trained"] == 0
               and freeze["predictions_generated"] == 0
               and freeze["new_metrics_computed"] == 0)
    report.add("Synthesis freeze forbids mutating Phase 8, Phase 9 and the "
               "hierarchy",
               freeze["phase8_results_mutable"] is False
               and freeze["phase9_results_mutable"] is False
               and freeze["confirmatory_hierarchy_mutable"] is False)
    report.add("Manuscript drafting is not permitted in Phase 10",
               freeze["manuscript_drafting_permitted"] is False)
    arrays = sorted(p.stem for p in (ARTIFACTS / "phase8_predictions").glob(
        "*.npy") if not p.stem.endswith("_source"))
    report.add("No prediction artifact was created in Phase 10",
               arrays == sorted(CONFIRMATORY)
               and not list(ARTIFACTS.glob("phase10*pred*"))
               and not list(ARTIFACTS.glob("phase10*.npy")))
    report.add("No model artifact was created in Phase 10",
               not list(PROJECT_ROOT.glob("**/phase10*.safetensors"))
               and not list(PROJECT_ROOT.glob("**/phase10*.txt")))
    report.add("No Phase-10 table introduces a new scientific metric",
               not list(ARTIFACTS.glob("phase10_*metrics*.csv")))


def ledger_checks(report):
    roles = load_json(ARTIFACTS / "phase10_model_role_ledger.json")
    report.add("Model role ledger preserves the frozen three roles",
               roles["roles"]["B3_R2"] == "PRIMARY CONFIRMATORY MODEL"
               and roles["roles"]["GRU_R1"]
               == "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL"
               and roles["roles"]["B0"] == "REFERENCE BENCHMARK")
    report.add("Model role ledger lists no other confirmatory model",
               sorted(roles["roles"]) == sorted(CONFIRMATORY))
    report.add("Development-only models are recorded as such",
               sorted(roles["development_only_models"])
               == sorted(DEVELOPMENT_ONLY))
    report.add("No winner label is assigned",
               roles["winner_label_assigned"] is False
               and roles["foundation_models_executed"] == 0)
    report.add("GRU secondary artifact is the seed-42 model",
               roles["secondary_model_seed"] == 42)

    hypotheses = load_json(ARTIFACTS / "phase10_hypothesis_ledger.json")
    report.add("Hypothesis ledger covers H1-H5",
               sorted(hypotheses["hypotheses"]) == ["H1", "H2", "H3", "H4",
                                                    "H5"])
    entries = hypotheses["hypotheses"]
    report.add("No hypothesis is marked as retested on the locked test",
               all(entry["locked_test_retested"] is False
                   for entry in entries.values()),
               "retested: %s" % [k for k, v in entries.items()
                                 if v["locked_test_retested"]])
    report.add("H3 is not overclaimed as general co-pollutant benefit",
               "WEAK" in entries["H3"]["status"].upper()
               and "MODEL-DEPENDENT" in entries["H3"]["status"].upper())
    report.add("H4 is not called locked-test confirmed",
               "NOT INDEPENDENTLY CONFIRMED"
               in entries["H4"]["status"].upper())
    report.add("H5 is not called fully supported",
               "NOT SUPPORTED AS SUPERIORITY OVER PERSISTENCE"
               in entries["H5"]["status"].upper())
    report.add("Every hypothesis carries an evidence class of C2 or C3",
               all(entry["evidence_class"] in ("C2", "C3", "C2/C3")
                   for entry in entries.values()))


def matrix_checks(report):
    header, rows = read_csv(ARTIFACTS / "phase10_claim_evidence_matrix.csv")
    report.add("Claim-evidence matrix matches the frozen schema",
               header == MATRIX_COLUMNS, "header: %s" % header)
    report.add("Every claim carries a valid evidence class",
               all(row["evidence_class"] in CLASSES for row in rows),
               "bad: %s" % [r["claim_id"] for r in rows
                            if r["evidence_class"] not in CLASSES][:5])
    report.add("Every claim names an artifact source",
               all(row["artifact_source"].strip() for row in rows))
    untraceable = [row["claim_id"] for row in rows
                   if not any((PROJECT_ROOT / part.strip()).exists()
                              for part in row["artifact_source"].split(";"))]
    report.add("Every claim's artifact source exists on disk", not untraceable,
               "untraceable: %s" % untraceable[:5])
    topics = set(row["claim_id"] for row in rows)
    missing = [t for t in REQUIRED_CLAIM_TOPICS if t not in topics]
    report.add("Matrix covers every required claim topic (%d)"
               % len(REQUIRED_CLAIM_TOPICS), not missing,
               "missing: %s" % missing)
    c1 = [row for row in rows if row["evidence_class"] == "C1"]
    report.add("C1 claims cite only Phase-8 locked evidence",
               all("phase8" in row["artifact_source"]
                   or "final_test" in row["artifact_source"] for row in c1),
               "bad: %s" % [r["claim_id"] for r in c1
                            if "phase8" not in r["artifact_source"]
                            and "final_test" not in r["artifact_source"]])
    c3 = [row for row in rows if row["evidence_class"] == "C3"]
    report.add("C3 claims cite only Phase-9 exploratory evidence",
               all("phase9" in row["artifact_source"].lower()
                   or "PHASE_09" in row["artifact_source"] for row in c3),
               "bad: %s" % [r["claim_id"] for r in c3][:5])
    report.add("No Phase-9 finding is classified as confirmatory",
               not [row for row in rows
                    if row["evidence_class"] == "C1"
                    and "phase9" in row["artifact_source"].lower()])


def table_checks(report):
    _, locked = read_csv(ARTIFACTS / "phase10_table_locked_test.csv")
    report.add("Locked-test table holds exactly the three frozen models",
               sorted(r["model"] for r in locked) == sorted(CONFIRMATORY))
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    primary = next(r for r in locked if r["model"] == "B3_R2")
    secondary = next(r for r in locked if r["model"] == "GRU_R1")
    reference = next(r for r in locked if r["model"] == "B0")
    report.add("Locked-test table reconstructs the primary endpoint",
               abs(float(primary["macro_station_horizon_MAE"])
                   - results["primary_value"]) < 1e-9)
    report.add("Locked-test table reconstructs the secondary endpoint",
               abs(float(secondary["severe_MAE_gt_244"])
                   - results["secondary_value"]) < 1e-9)
    report.add("Locked-test table reconstructs the reference values",
               abs(float(reference["macro_station_horizon_MAE"])
                   - results["reference_primary_value"]) < 1e-9
               and abs(float(reference["severe_MAE_gt_244"])
                       - results["reference_severe_value"]) < 1e-9)
    report.add("Locked-test table labels confirmatory roles explicitly",
               all(r["role"] in ("PRIMARY CONFIRMATORY MODEL",
                                 "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL",
                                 "REFERENCE BENCHMARK") for r in locked))
    report.add("Locked-test table carries no Phase-9-only column",
               not any(key and ("acf" in key.lower() or "event" in key.lower()
                                or "detection" in key.lower())
                       for key in locked[0]))

    _, development = read_csv(ARTIFACTS
                              / "phase10_table_development_models.csv")
    report.add("Development table labels seed provenance explicitly",
               all(r.get("seed_provenance", "").strip() for r in development))
    _, multiseed = read_csv(ARTIFACTS
                            / "phase10_table_multiseed_robustness.csv")
    report.add("Multi-seed table covers the four Phase-7 models",
               sorted(set(r["model"] for r in multiseed))
               == sorted(["GRU_R1", "SA_R2", "SA_R3", "TCN_R1"]))
    _, posttest = read_csv(ARTIFACTS
                           / "phase10_table_posttest_diagnostics.csv")
    report.add("Every post-test diagnostic row is marked exploratory",
               all(r["evidence_class"] == "C3" for r in posttest))


def arithmetic_checks(report):
    conclusion = load_json(ARTIFACTS / "phase10_canonical_conclusion.json")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    expected = 100.0 * (results["reference_primary_value"]
                        - results["primary_value"]) \
        / results["reference_primary_value"]
    report.add("B3_R2 relative improvement over B0 reconstructs exactly",
               abs(conclusion["b3r2_relative_improvement_vs_B0_pct"]
                   - expected) < 1e-9,
               "%.12f vs %.12f"
               % (conclusion["b3r2_relative_improvement_vs_B0_pct"], expected))
    for key, value in LOCKED.items():
        recorded = conclusion["canonical_locked_results"][key]
        report.add("Canonical conclusion carries exact %s" % key,
                   recorded == value, "%r vs %r" % (recorded, value))
    report.add("Canonical conclusion is evidence-bounded and cites artifacts",
               bool(conclusion["supporting_artifacts"])
               and all((PROJECT_ROOT / p).exists()
                       for p in conclusion["supporting_artifacts"].values()))


def governance_checks(report):
    governance = load_json(ARTIFACTS
                           / "phase10_foundation_model_governance.json")
    report.add("Foundation-model governance records zero execution",
               governance["foundation_models_executed"] == 0
               and governance["foundation_model_predictions_generated"] == 0)
    report.add("Chronos-2 exclusion is worded as overlap risk, not proven "
               "contamination",
               "material" in governance["models"]["Chronos-2"]["reason"]
               .lower()
               and "byte" not in governance["models"]["Chronos-2"]["reason"]
               .lower())
    report.add("Governance is framed as reproducibility, not model failure",
               governance["framing"].lower().startswith("reproducibility")
               or "governance" in governance["framing"].lower())

    limitations = load_json(ARTIFACTS / "phase10_limitations_ledger.json")
    report.add("Limitations ledger holds at least 15 entries",
               len(limitations["limitations"]) >= 15,
               "%d" % len(limitations["limitations"]))
    joined = json.dumps(limitations).lower()
    report.add("Limitations record the LOSO target-transfer caveat",
               "target" in joined and "unseen" in joined)
    report.add("Limitations record the seed-42 secondary caveat",
               "seed 42" in joined or "seed-42" in joined)

    future = load_json(ARTIFACTS / "phase10_future_work_ledger.json")
    report.add("Future work is marked untested in this study",
               all(entry["tested_in_this_study"] is False
                   for entry in future["future_work"]))
    report.add("Future work claims no expected improvement",
               future["improvement_claimed"] is False)


def narrative_checks(report):
    texts = {}
    for name in ("PUBLICATION_CLAIM_GUIDE.md",
                 "PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md"):
        texts[name] = (DOCS / name).read_text(encoding="utf-8")
    guide = texts["PUBLICATION_CLAIM_GUIDE.md"]
    for section in ("Claims we can make strongly", "requiring qualifiers",
                    "Post-test exploratory", "must NOT make",
                    "terminology", "classification"):
        report.add("Claim guide has a section on: %s" % section[:38],
                   section.lower() in guide.lower())

    # The guide legitimately quotes prohibited claims in order to ban them, so
    # prohibited patterns are checked in the synthesis document only.
    synthesis = texts["PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md"]
    hits = [why for pattern, why in PROHIBITED_PATTERNS
            if re.search(pattern, synthesis, re.I)]
    report.add("Synthesis document makes no prohibited claim", not hits,
               "found: %s" % hits)
    report.add("Synthesis separates development, locked and post-test "
               "evidence",
               all(token in synthesis for token in
                   ("C1", "C2", "C3", "LOCKED CONFIRMATORY",
                    "POST-TEST EXPLORATORY")))
    report.add("Synthesis states persistence remained the stronger severe "
               "benchmark",
               re.search(r"persistence[^.]{0,120}sever", synthesis, re.I)
               is not None)


def index_checks(report):
    tables = load_json(ARTIFACTS / "phase10_publication_table_index.json")
    report.add("Publication table index proposes at least six tables",
               len(tables["tables"]) >= 6, "%d" % len(tables["tables"]))
    report.add("Every proposed table names an evidence class and a source",
               all(entry["evidence_class"] in CLASSES
                   and (PROJECT_ROOT / entry["artifact_source"]).exists()
                   for entry in tables["tables"]))

    figures = load_json(ARTIFACTS / "phase10_publication_figure_index.json")
    report.add("Figure index selects only existing frozen figures",
               all((PROJECT_ROOT / entry["source_path"]).exists()
                   for entry in figures["figures"]))
    report.add("Figure index creates no new figure",
               figures["new_figures_created"] == 0
               and not list(FIGURES.glob("phase10_*")))
    report.add("Every indexed figure carries an evidence class",
               all(entry["evidence_class"] in CLASSES
                   for entry in figures["figures"]))

    index = load_json(ARTIFACTS / "phase10_master_evidence_index.json")
    missing = [key for key, entry in index["evidence"].items()
               if not (PROJECT_ROOT / entry["artifact"]).exists()
               or sha256_of_file(PROJECT_ROOT / entry["artifact"])
               != entry["sha256"]]
    report.add("Master evidence index reconciles every artifact hash",
               not missing, "unreconciled: %s" % missing[:5])
    report.add("Master evidence index covers the required backbone",
               all(key in index["evidence"] for key in
                   ("locked_primary_result", "locked_secondary_result",
                    "reference_benchmark", "hypothesis_ledger",
                    "claim_evidence_matrix", "limitations", "future_work")))


def manifest_checks(report):
    summary = load_json(ARTIFACTS / "phase10_final_synthesis_summary.json")
    report.add("Phase-10 summary is classified as synthesis",
               summary["classification"] == "FINAL_SCIENTIFIC_SYNTHESIS"
               and summary["phase"] == 10)
    report.add("Phase-10 summary records zero experimentation",
               summary["new_scientific_experiments"] == 0
               and summary["new_test_analyses"] == 0
               and summary["models_trained"] == 0
               and summary["predictions_generated"] == 0
               and summary["new_test_models"] == 0)
    report.add("Phase-10 summary records nothing reclassified or changed",
               summary["confirmatory_hierarchy_changed"] is False
               and summary["locked_test_result_changed"] is False
               and summary[
                   "post_test_findings_reclassified_as_confirmatory"] is False
               and summary["manuscript_drafting_started"] is False)

    manifest = load_json(ARTIFACTS / "v2_phase10_manifest.json")
    drift = [name for name, digest in manifest["artifact_sha256"].items()
             if not (PROJECT_ROOT / name).exists()
             or sha256_of_file(PROJECT_ROOT / name) != digest]
    report.add("Phase-10 manifest hashes reconcile", not drift,
               "drifted: %s" % drift[:5])
    report.add("Phase-10 manifest records the synthesis status",
               manifest["phase10_status"] == "SYNTHESIS_FROZEN"
               and manifest["phase8_scientific_drift"] == 0
               and manifest["phase9_scientific_drift"] == 0
               and manifest["confirmatory_hierarchy_changed"] is False)

    registry = load_json(ARTIFACTS
                         / "phase10_verification_tooling_registry.json")
    report.add("Phase-10 registry pins this validator",
               registry["validators"]["scripts/validate_phase10.py"]["sha256"]
               == sha256_of_file(PROJECT_ROOT / "scripts"
                                 / "validate_phase10.py"))
    report.add("Phase-10 registry parents the Phase-9 registry",
               registry["parent_registry"]
               == "artifacts/phase9_verification_tooling_registry.json")
    pinned = registry["validators"]
    report.add("Phase-10 registry preserves historical validator semantics",
               all(name in pinned for name in
                   ("scripts/validate_phase8.py",
                    "scripts/validate_phase8_postopening.py",
                    "scripts/validate_phase9.py"))
               and all(sha256_of_file(PROJECT_ROOT / name)
                       == entry["sha256"] for name, entry in pinned.items()))


def main():
    report = Report()
    freeze = load_json(ARTIFACTS / "phase10_synthesis_freeze.json")
    anchor_checks(report)
    drift_checks(report)
    no_experiment_checks(report, freeze)
    ledger_checks(report)
    matrix_checks(report)
    table_checks(report)
    arithmetic_checks(report)
    governance_checks(report)
    narrative_checks(report)
    index_checks(report)
    manifest_checks(report)

    report.render()
    print("")
    if report.failed():
        print("PHASE 10 VALIDATION FAILED: %d gate(s)" % len(report.failed()))
        return 1
    print("PHASE 10 VALIDATION PASSED")
    print("Classification: FINAL_SCIENTIFIC_SYNTHESIS. Status: "
          "SYNTHESIS_FROZEN.")
    print("Models trained: 0. Predictions generated: 0. New metrics: 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
