"""Build the AirSense V1 evidence registry, final manifest and freeze receipt.

Protocol Phase 12, closing step. **Audit and synthesis only** - this script
fits nothing, predicts nothing, and recomputes no scientific result. It
enumerates the canonical research files, hashes them, and writes the three
artifacts that close V1.

Ordering is deliberately acyclic:

    registry  ->  final evidence manifest  ->  freeze receipt

Each references only artifacts created before it. The registry excludes all
three closing files, because they are written after it and cannot
self-reference; the manifest does not contain its own digest; the receipt
does not contain its own.

Registry scope is the intended research record. Version-control internals,
the untracked runtime prefix, byte-code caches and editor/OS metadata are
excluded. Anything encountered that is neither clearly canonical nor
clearly excluded aborts the run rather than being silently swept in.

No execution timestamp is written, so repeated runs are byte-identical.

Usage:
    venv/bin/python scripts/freeze_v1_evidence.py
"""

import io
import json
import os
import sys
from collections import OrderedDict

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import sha256_of_file  # noqa: E402

REGISTRY = os.path.join("artifacts", "v1_evidence_registry.csv")
FINAL_MANIFEST = os.path.join("artifacts",
                              "v1_final_evidence_manifest.json")
FREEZE_RECEIPT = os.path.join("artifacts", "v1_final_freeze_receipt.json")

# Created after the registry; cannot be inside it.
SELF_EXCLUDED = {REGISTRY, FINAL_MANIFEST, FREEZE_RECEIPT}

# Directory trees never registered.
EXCLUDED_DIRS = {".git", "venv", "__pycache__", ".ipynb_checkpoints",
                 ".idea", ".vscode", "bin"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".orig", ".rej",
                     ".tmp", ".bak")
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep"}

# Roots that make up the intended V1 research record.
REGISTERED_ROOTS = ["docs", "src", "scripts", "data", "artifacts",
                    "results", "figures", "notebooks"]
REGISTERED_FILES = ["README.md", ".gitignore",
                    "requirements-v1-2019.txt",
                    "requirements-v1-2019-lock.txt"]

CATEGORIES = [
    ("src/", "source"),
    ("scripts/", "script"),
    ("docs/", "documentation"),
    ("data/raw/", "raw_data"),
    ("data/processed/", "processed_data"),
    ("artifacts/", "artifact"),
    ("results/", "result"),
    ("figures/", "figure"),
    ("notebooks/", "notebook"),
]

REQUIRED_COMPONENTS = OrderedDict([
    ("raw dataset", "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"),
    ("EDA summary", "artifacts/eda_summary.json"),
    ("split manifest", "artifacts/split_manifest.json"),
    ("feature schema", "artifacts/feature_schema.json"),
    ("M0 manifest", "artifacts/m0_baseline_manifest.json"),
    ("M1 manifest", "artifacts/m1_linear_regression_manifest.json"),
    ("M2 manifest", "artifacts/m2_decision_tree_manifest.json"),
    ("M3 manifest", "artifacts/m3_random_forest_manifest.json"),
    ("pre-test freeze", "artifacts/pretest_development_freeze.json"),
    ("blind prediction freeze",
     "artifacts/final_blind_prediction_freeze.json"),
    ("test opening receipt", "artifacts/final_test_opening_receipt.json"),
    ("final evaluation manifest",
     "artifacts/final_evaluation_manifest.json"),
    ("error analysis manifest", "artifacts/error_analysis_manifest.json"),
    ("final report", "docs/V1_FINAL_REPORT.md"),
    ("reproducibility document", "docs/V1_REPRODUCIBILITY.md"),
    ("claims ledger", "artifacts/v1_claims_ledger.csv"),
    ("final model comparison",
     "results/v1_summary/final_model_comparison.csv"),
    ("key findings", "results/v1_summary/key_findings.csv"),
    ("research protocol", "docs/V1_RESEARCH_PROTOCOL.md"),
    ("repository summary", "README.md"),
    ("dependency spec", "requirements-v1-2019.txt"),
    ("dependency lock", "requirements-v1-2019-lock.txt"),
])

MODELS = ["M0", "M1", "M2", "M3"]


class FreezeError(RuntimeError):
    """Raised when the evidence package cannot be frozen."""


def path(relative):
    return os.path.join(PROJECT_ROOT, relative)


def categorise(relative):
    if relative in ("requirements-v1-2019.txt",
                    "requirements-v1-2019-lock.txt"):
        return "dependency_spec"
    if relative in ("README.md", ".gitignore"):
        return "repository_summary"
    # data/README.md is dataset provenance prose, not data.
    if relative == "data/README.md":
        return "documentation"
    for prefix, category in CATEGORIES:
        if relative.startswith(prefix):
            return category
    raise FreezeError("uncategorised file: %s" % relative)


def collect():
    """Enumerate canonical files in deterministic lexicographic order."""
    found = []
    for name in REGISTERED_FILES:
        if os.path.isfile(path(name)):
            found.append(name)
    for root_name in REGISTERED_ROOTS:
        root = path(root_name)
        if not os.path.isdir(root):
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
            for filename in sorted(files):
                if filename in EXCLUDED_NAMES:
                    continue
                if filename.endswith(EXCLUDED_SUFFIXES):
                    continue
                absolute = os.path.join(current, filename)
                relative = os.path.relpath(absolute, PROJECT_ROOT)
                relative = relative.replace(os.sep, "/")
                if relative in SELF_EXCLUDED:
                    continue
                found.append(relative)
    return sorted(set(found))


def build_registry():
    rows = []
    for relative in collect():
        absolute = path(relative)
        rows.append(OrderedDict([
            ("relative_path", relative),
            ("category", categorise(relative)),
            ("size_bytes", int(os.path.getsize(absolute))),
            ("sha256", sha256_of_file(absolute)),
        ]))
    frame = pd.DataFrame(rows)

    registered = set(frame["relative_path"])
    missing = [name for name, rel in REQUIRED_COMPONENTS.items()
               if rel not in registered]
    if missing:
        raise FreezeError(
            "required evidence components missing from the registry: %s"
            % ", ".join(missing))
    return frame


def build_final_manifest(registry_sha):
    metrics = pd.read_csv(
        path("results/final_test/final_test_metrics.csv")).set_index("model")
    evaluation = json.load(io.open(
        path("artifacts/final_evaluation_manifest.json"), encoding="utf-8"))
    error = json.load(io.open(
        path("artifacts/error_analysis_manifest.json"), encoding="utf-8"))
    pretest = json.load(io.open(
        path("artifacts/pretest_development_freeze.json"), encoding="utf-8"))
    refit = json.load(io.open(
        path("artifacts/final_refit_manifest.json"), encoding="utf-8"))
    schema = json.load(io.open(
        path("artifacts/feature_schema.json"), encoding="utf-8"))
    split = json.load(io.open(
        path("artifacts/split_manifest.json"), encoding="utf-8"))
    severe = pd.read_csv(
        path("results/error_analysis/severe_hour_summary.csv")).set_index(
        "model")
    episodes = pd.read_csv(
        path("results/error_analysis/severe_episode_summary.csv")).iloc[0]
    auto = pd.read_csv(
        path("results/error_analysis/residual_autocorrelation.csv"))
    auto3 = auto[auto["model"] == "M3"].set_index("lag_hours")

    manifest = OrderedDict()
    manifest["project"] = "AirSense"
    manifest["version"] = "V1"
    manifest["title"] = ("Air Quality Prediction Using Statistical Analysis "
                         "and Machine Learning")
    manifest["artifact_type"] = "final_evidence_manifest"
    manifest["historical_cutoff"] = "2019-04-26"
    manifest["task_definition"] = (
        "concurrent-hour PM2.5 estimation from meteorological and calendar "
        "covariates under chronological generalisation; NOT future-horizon "
        "forecasting - no lagged target is used")
    manifest["protocol_completion_status"] = "Phase 12 complete"

    manifest["raw_dataset_sha256"] = evaluation["raw_dataset_sha256"]
    manifest["runtime"] = OrderedDict([
        ("interpreter", "CPython 3.6.7"),
        ("direct_dependencies", OrderedDict([
            ("numpy", "1.15.4"), ("pandas", "0.23.4"), ("scipy", "1.1.0"),
            ("scikit-learn", "0.20.0"), ("matplotlib", "3.0.2"),
            ("seaborn", "0.9.0"), ("jupyter", "1.0.0")])),
        ("note", "period-exact Python environment on a modern substrate"),
    ])

    reconciliation = split["membership_reconciliation"]
    manifest["population"] = OrderedDict([
        ("raw_hourly_records", reconciliation["raw_total_rows"]),
        ("excluded_missing_target",
         reconciliation["excluded_target_missing"]),
        ("supervised_observations",
         reconciliation["supervised_total_rows"]),
    ])
    manifest["chronological_split"] = OrderedDict(
        (name, OrderedDict([
            ("period", "%s to %s" % (entry["declared_start"],
                                     entry["declared_end"])),
            ("supervised_rows", entry["supervised_rows"])]))
        for name, entry in split["partitions"].items())
    manifest["final_feature_count"] = schema["feature_count"]

    specs = OrderedDict()
    specs["M0"] = OrderedDict([
        ("strategy", pretest["models"]["M0"]["strategy"]),
        ("final_constant",
         refit["models"]["M0"]["final_development_median"])])
    specs["M1"] = OrderedDict([
        ("estimator", pretest["models"]["M1"]["estimator"]),
        ("fit_intercept", True), ("scaling", "none"),
        ("regularization", "none"),
        ("final_intercept", refit["models"]["M1"]["intercept"])])
    specs["M2"] = OrderedDict([
        ("estimator", pretest["models"]["M2"]["estimator"]),
        ("frozen_hyperparameters",
         pretest["models"]["M2"]["selected_hyperparameters"])])
    specs["M3"] = OrderedDict([
        ("estimator", pretest["models"]["M3"]["estimator"]),
        ("frozen_hyperparameters",
         pretest["models"]["M3"]["selected_hyperparameters"])])
    manifest["model_specifications"] = specs

    manifest["final_held_out_metrics"] = OrderedDict(
        (m, OrderedDict([("mae", float(metrics.loc[m, "mae"])),
                         ("rmse", float(metrics.loc[m, "rmse"])),
                         ("r2", float(metrics.loc[m, "r2"])),
                         ("mae_rank", int(metrics.loc[m, "mae_rank"]))]))
        for m in MODELS)
    manifest["primary_metric"] = "MAE"
    manifest["primary_ranking_by_mae"] = sorted(
        MODELS, key=lambda m: int(metrics.loc[m, "mae_rank"]))
    manifest["final_winner"] = evaluation["best_final_model_by_mae"]
    manifest["metric_ranking_note"] = (
        "M2 outranks M1 on MAE while M1 outranks M2 on RMSE and R2; MAE was "
        "pre-registered and governs. M3 is best on all three.")

    manifest["severe_tail_headline"] = OrderedDict([
        ("threshold_source", "development P95 (2010-2013), study-specific, "
                             "not regulatory"),
        ("threshold_value", float(error["thresholds"]["P95"])),
        ("severe_hours", int(episodes["number_of_severe_hours"])),
        ("severe_episodes", int(episodes["number_of_episodes"])),
        ("m3_severe_mae", float(severe.loc["M3", "mae"])),
        ("m3_severe_underprediction_percent",
         float(severe.loc["M3", "underprediction_percent"])),
    ])
    manifest["residual_autocorrelation_headline"] = OrderedDict(
        ("m3_lag_%dh" % lag, float(auto3.loc[lag, "pearson_r"]))
        for lag in (1, 6, 12, 24, 48, 168))
    manifest["residual_autocorrelation_note"] = (
        "computed by exact timestamp matching, not row shifting; errors "
        "remain strongly temporally structured at short lags")

    manifest["final_test_status"] = "exhausted"
    manifest["test_exhaustion_statement"] = (
        "The 2014 target has been opened and scored once. It is no longer "
        "an unseen test set. No model changed after this point may claim a "
        "fresh evaluation on the same split.")
    manifest["model_modification_after_final_test"] = "none"

    closing = OrderedDict()
    closing["v1_evidence_registry_sha256"] = registry_sha
    for key, relative in (
            ("claims_ledger_sha256", "artifacts/v1_claims_ledger.csv"),
            ("final_model_comparison_sha256",
             "results/v1_summary/final_model_comparison.csv"),
            ("key_findings_sha256",
             "results/v1_summary/key_findings.csv"),
            ("v1_final_report_sha256", "docs/V1_FINAL_REPORT.md"),
            ("v1_reproducibility_sha256", "docs/V1_REPRODUCIBILITY.md"),
            ("phase_10_final_v1_record_sha256",
             "docs/PHASE_10_FINAL_V1_RECORD.md"),
            ("readme_sha256", "README.md"),
            ("v1_research_protocol_sha256",
             "docs/V1_RESEARCH_PROTOCOL.md")):
        target = path(relative)
        if not os.path.isfile(target):
            raise FreezeError("%s not found; it must exist before the "
                              "final manifest" % relative)
        closing[key] = sha256_of_file(target)
    manifest["closing_artifacts_sha256"] = closing
    manifest["note"] = (
        "This manifest does not contain its own digest. No execution "
        "timestamp is recorded so that repeated runs are byte-identical.")
    return manifest


def build_receipt(registry_sha, manifest_sha, raw_sha):
    return OrderedDict([
        ("freeze_name", "AirSense_V1_Final"),
        ("v1_status", "complete_and_frozen"),
        ("historical_cutoff", "2019-04-26"),
        ("evidence_registry", REGISTRY),
        ("evidence_registry_sha256", registry_sha),
        ("final_evidence_manifest", FINAL_MANIFEST),
        ("final_evidence_manifest_sha256", manifest_sha),
        ("raw_dataset_sha256", raw_sha),
        ("final_test_status", "exhausted"),
        ("model_development_status", "closed"),
        ("model_modification_after_final_test", "none"),
        ("future_work_policy", "V2_must_be_separate_from_frozen_V1"),
        ("post_freeze_policy",
         "AirSense V1 is immutable scientific evidence. Future work must "
         "not silently edit V1 to improve results, methodology, features, "
         "models or conclusions. A genuine factual or documentation "
         "correction must be recorded transparently as a post-freeze "
         "correction leaving the original evidence intact."),
        ("note", "This receipt does not contain its own digest."),
    ])


def main():
    print("[AIRSENSE V1 - EVIDENCE FREEZE]")
    print("Audit and synthesis only. No model, no prediction, no new "
          "analysis.")
    print("")
    try:
        registry = build_registry()
        registry.to_csv(path(REGISTRY), index=False)
        registry_sha = sha256_of_file(path(REGISTRY))
        total_bytes = int(registry["size_bytes"].sum())
        print("Wrote %s" % REGISTRY)
        print("  files registered : %d" % len(registry))
        print("  total bytes      : %d" % total_bytes)
        print("  by category      :")
        for category, count in sorted(
                registry["category"].value_counts().items()):
            print("      %-18s %4d" % (category, count))
        print("  registry SHA-256 : %s" % registry_sha)

        manifest = build_final_manifest(registry_sha)
        with io.open(path(FINAL_MANIFEST), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2))
            handle.write("\n")
        manifest_sha = sha256_of_file(path(FINAL_MANIFEST))
        print("")
        print("Wrote %s" % FINAL_MANIFEST)
        print("  SHA-256 : %s" % manifest_sha)

        raw_sha = manifest["raw_dataset_sha256"]
        receipt = build_receipt(registry_sha, manifest_sha, raw_sha)
        with io.open(path(FREEZE_RECEIPT), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2))
            handle.write("\n")
        print("")
        print("Wrote %s" % FREEZE_RECEIPT)
        print("  SHA-256 : %s" % sha256_of_file(path(FREEZE_RECEIPT)))
        print("  v1_status: %s" % receipt["v1_status"])
    except FreezeError as exc:
        sys.stderr.write("\nEVIDENCE FREEZE ABORTED: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
