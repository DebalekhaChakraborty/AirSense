"""Independent verification of the AirSense V2 post-test exploratory analysis.

Protocol Phase 9. **This source was finalized before any Phase-9 numerical
output was computed** and is pinned in
``artifacts/phase9_verification_tooling_registry.json``.

Phase 9 is exploratory. Its job is to explain the locked final test, never to
change it, so the gates here are mostly about what must *not* have happened:
no model loaded, no prediction generated, no fourth model analysed, no
definition altered after the freeze, no confirmatory promotion, and no Phase-8
artifact touched.

The four locked Phase-8 values are recomputed from the frozen prediction arrays
and the raw station truth. Disagreement is a stop condition for human review,
never grounds to adjust Phase 8.

No expected Phase-9 finding is encoded here. The validator checks structure,
definitions and reconciliation - not conclusions.

Exits non-zero on any failure.

Usage:
    .venv-v2/bin/python scripts/validate_phase9.py
"""

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
FIGURES = PROJECT_ROOT / "figures"
DOCS = PROJECT_ROOT / "docs"
PRED_DIR = ARTIFACTS / "phase8_predictions"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"

PRE_TEST_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LOCKED_EVALUATION_SHA = "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"
LEGACY_SHA = "16c9030cf74508b620cc6d28f90346aa379f29cd"

MODELS = ["B3_R2", "GRU_R1", "B0"]
EXPECTED_TEST_SAMPLES = 411012
SEVERE_THRESHOLD = 244.0
TRAINING_MEDIAN = 61.0
ACF_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}
REGIMES = ["LOW", "ELEVATED", "SEVERE"]
HORIZONS = [1, 6, 12, 24]
FIGURE_LABEL = "POST-TEST EXPLORATORY"

SCHEMAS = {
    "generalization_gap": ["model", "metric", "development_value",
                           "locked_test_value", "absolute_change",
                           "relative_change_pct", "note"],
    "horizon_error_analysis": ["model", "horizon_hours", "n", "MAE", "RMSE",
                               "R2", "mean_residual",
                               "underprediction_rate_pct", "mae_diff_vs_B0",
                               "mae_rel_improvement_vs_B0_pct"],
    "station_error_analysis": ["model", "station", "n", "MAE", "RMSE", "R2",
                               "mean_residual", "underprediction_rate_pct",
                               "severe_n", "severe_MAE"],
    "station_horizon_error_analysis": ["model", "station", "horizon_hours",
                                       "n", "MAE", "RMSE", "R2",
                                       "mean_residual",
                                       "underprediction_rate_pct"],
    "residual_acf": ["model", "station", "horizon_hours", "lag_hours", "acf",
                     "n_pairs"],
    "residual_acf_summary": ["model", "horizon_hours", "lag_hours", "mean",
                             "median", "min", "max", "q25", "q75", "iqr",
                             "n_stations"],
    "test_target_acf": ["station", "lag_hours", "acf", "n_pairs"],
    "concentration_regime_analysis": ["model", "regime", "stratum_label", "n",
                                      "MAE", "RMSE", "mean_residual",
                                      "underprediction_rate_pct",
                                      "median_abs_error", "p90_abs_error"],
    "severe_tail_analysis": ["model", "horizon_hours", "severe_n", "MAE",
                             "RMSE", "mean_residual", "median_residual",
                             "underprediction_pct", "median_abs_error",
                             "p90_abs_error", "p95_abs_error",
                             "mean_abs_error_diff_vs_B0",
                             "mean_abs_error_diff_vs_B3_R2"],
    "severe_events": ["station", "event_id", "start_timestamp",
                      "end_timestamp", "duration_hours", "actual_mean",
                      "actual_peak", "peak_timestamp"],
    "severe_event_model_analysis": ["station", "event_id", "model",
                                    "horizon_hours", "severe_hours",
                                    "event_MAE", "event_mean_residual",
                                    "prediction_at_peak", "residual_at_peak",
                                    "abs_error_at_peak", "underpredicted_peak",
                                    "predicted_severe_at_first_severe_hour",
                                    "predicted_severe_hours"],
    "severe_detection_analysis": ["model", "horizon_hours", "TP", "FP", "TN",
                                  "FN", "precision", "recall", "F1",
                                  "specificity"],
    "seasonal_error_analysis": ["model", "season", "horizon_hours", "n", "MAE",
                                "RMSE", "mean_residual",
                                "underprediction_rate_pct", "severe_n",
                                "severe_MAE"],
    "hour_of_day_error_analysis": ["model", "horizon_hours", "target_hour",
                                   "n", "MAE", "mean_residual",
                                   "underprediction_rate_pct"],
    "error_complementarity": ["stratum", "model_a", "model_b", "n",
                              "pearson_abs_error", "spearman_abs_error",
                              "mean_paired_abs_error_diff",
                              "median_paired_abs_error_diff",
                              "fraction_a_strictly_better",
                              "fraction_b_strictly_better", "fraction_tied"],
    "negative_prediction_analysis": ["model", "scope", "key", "n",
                                     "n_negative", "pct_negative",
                                     "min_prediction"],
}

BANNED_PHRASES = ["proved", "solved", "eliminated", "spatial reasoning",
                  "learned pollution propagation"]


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


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def anchor_checks(report, freeze):
    _, head = git("rev-parse", "HEAD")
    report.add("Pre-test snapshot is an ancestor of locked evaluation",
               is_ancestor(PRE_TEST_SHA, LOCKED_EVALUATION_SHA))
    report.add("Locked evaluation is an ancestor of current HEAD",
               is_ancestor(LOCKED_EVALUATION_SHA, head), head)
    report.add("V1 legacy is unchanged",
               git("rev-parse", "legacy")[1] == LEGACY_SHA)

    drift = [name for name, digest in freeze["phase8_anchor_sha256"].items()
             if sha256_of_file(
                 PRED_DIR / name if name.endswith(".npy")
                 else ARTIFACTS / name) != digest]
    report.add("No Phase-8 anchor artifact changed", not drift,
               "drifted: %s" % drift)

    receipt = load_json(ARTIFACTS / "phase8_postcommit_validation_receipt.json")
    report.add("Phase-8 final validation receipt records a clean final state",
               receipt["postopening_validator_failures"] == 0
               and receipt["scientific_drift"] == 0
               and receipt["prediction_drift"] == 0
               and receipt["metric_drift"] == 0
               and receipt["final_test_status"] == "evaluated")
    report.add("Phase-8 receipt is tracked at HEAD",
               git("ls-tree", "-r", "HEAD", "--name-only",
                   "artifacts/phase8_postcommit_validation_receipt.json")[1]
               != "")


def freeze_checks(report, freeze):
    report.add("Phase-9 freeze declares exploratory, non-confirmatory status",
               freeze["phase9_status"] == "POST_TEST_EXPLORATORY"
               and freeze["confirmatory_status"] == "NONE")
    report.add("Phase-9 freeze forbids training, inference and selection",
               freeze["model_training_permitted"] is False
               and freeze["model_inference_permitted"] is False
               and freeze["model_selection_permitted"] is False
               and freeze["new_test_model_evaluation_permitted"] is False
               and freeze["ensemble_construction_permitted"] is False
               and freeze["phase8_results_mutable"] is False)
    report.add("Phase-9 freeze analyses exactly the frozen three models",
               freeze["models_analysed"] == MODELS and freeze["gru_seed"] == 42)
    report.add("Protocol document unchanged since the freeze",
               sha256_of_file(DOCS / "PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md")
               == freeze["protocol_document_sha256"])
    report.add("Confirmatory hierarchy unchanged in the freeze",
               freeze["confirmatory_hierarchy_changed"] is False
               and freeze["primary_model"] == "B3_R2"
               and freeze["secondary_model"] == "GRU_R1"
               and freeze["reference_model"] == "B0")
    report.add("Freeze records zero training and zero generated predictions",
               freeze["models_trained"] == 0
               and freeze["predictions_generated"] == 0)

    definitions = freeze["definitions"]
    report.add("Severe boundary is exactly 244.0, strictly greater than",
               definitions["severe_threshold"] == SEVERE_THRESHOLD
               and definitions["severe_rule"] == "actual > 244.0")
    report.add("Severe threshold is declared as training pooled P95",
               "P95" in definitions["severe_threshold_source"]
               and "regulatory" in definitions["severe_threshold_source"])
    report.add("Training median boundary is exactly 61.0",
               definitions["training_median"] == TRAINING_MEDIAN)
    report.add("Concentration regimes use training-frozen boundaries only",
               definitions["concentration_regimes"]["test_quantiles_used"]
               is False
               and definitions["concentration_regimes"]["LOW"]
               == "actual <= 61.0"
               and definitions["concentration_regimes"]["SEVERE"]
               == "actual > 244.0")
    report.add("ACF lags are exactly the frozen nine",
               definitions["acf"]["lags"] == ACF_LAGS)
    report.add("ACF forbids gap bridging and imputation",
               definitions["acf"]["gap_bridging"] is False
               and definitions["acf"]["imputation"] is False)
    report.add("Season definitions are exact and unoptimised",
               all(definitions["seasons"][k] == v
                   for k, v in SEASONS.items())
               and definitions["seasons"]["boundaries_optimised"] is False)
    report.add("Hour-of-day uses all 24 exact hours, no adaptive binning",
               definitions["hour_of_day"]["values"] == list(range(24))
               and definitions["hour_of_day"]["adaptive_binning"] is False)
    report.add("Severe-event rule forbids gap bridging and tolerance",
               definitions["severe_event"]["gap_bridging"] is False
               and definitions["severe_event"]["tolerance_window"] is False
               and definitions["severe_event"]["minimum_duration_hours"] == 1
               and "unique hourly truth series"
               in definitions["severe_event"]["series"])
    report.add("Severe detection forbids threshold tuning",
               definitions["severe_detection"]["threshold_tuning"] is False
               and definitions["severe_detection"]["roc_optimisation"] is False
               and definitions["severe_detection"]["alternate_threshold"]
               is False)
    report.add("Complementarity forbids oracle or ensemble construction",
               definitions["complementarity"]["oracle_or_ensemble"] is False
               and "ties" in definitions["complementarity"])
    report.add("Freeze carries no Phase-9 numerical finding",
               not any(key.endswith(("_finding", "_result", "_conclusion"))
                       for key in freeze))


# ---------------------------------------------------------------------------
# No inference, no extra models
# ---------------------------------------------------------------------------

def no_execution_checks(report, freeze):
    arrays = sorted(p.stem for p in PRED_DIR.glob("*.npy")
                    if not p.stem.endswith("_source"))
    report.add("Only the three frozen prediction arrays exist",
               arrays == sorted(MODELS), "found: %s" % arrays)
    report.add("No Phase-9 prediction array was created",
               not list(ARTIFACTS.glob("phase9*pred*.npy"))
               and not list(ARTIFACTS.glob("phase9_predictions")))
    report.add("No new model weight or booster exists under a Phase-9 name",
               not list(PROJECT_ROOT.glob("**/phase9*.safetensors"))
               and not list(PROJECT_ROOT.glob("**/phase9*.txt")))
    report.add("results/final_test was never created",
               not (PROJECT_ROOT / "results" / "final_test").exists())

    forbidden = ["B1", "B2", "B3_R0", "B3_R1", "GRU_R0", "GRU_R2", "TCN_R0",
                 "TCN_R1", "TCN_R2", "iTransformer_R1", "iTransformer_R2",
                 "SA_R2", "SA_R3", "Chronos", "TimesFM", "Moirai"]
    offenders = []
    for path in sorted(ARTIFACTS.glob("phase9_*.csv")):
        _, rows = read_csv(path)
        for row in rows:
            for column in ("model", "model_a", "model_b"):
                value = row.get(column)
                if value and value in forbidden:
                    offenders.append("%s:%s" % (path.name, value))
    report.add("No excluded model appears in any Phase-9 table",
               not offenders, "offenders: %s" % sorted(set(offenders))[:5])


def table_checks(report, freeze):
    missing, bad_schema = [], []
    for name, columns in SCHEMAS.items():
        path = ARTIFACTS / ("phase9_%s.csv" % name)
        if not path.is_file():
            missing.append(name)
            continue
        header, _ = read_csv(path)
        if header != columns:
            bad_schema.append("%s: %s" % (name, header))
    report.add("Every planned Phase-9 table exists (%d)" % len(SCHEMAS),
               not missing, "missing: %s" % missing)
    report.add("Every Phase-9 table matches its frozen schema",
               not bad_schema, "mismatched: %s" % bad_schema[:3])
    planned = set(freeze["planned_tables"])
    produced = set("artifacts/%s" % p.name
                   for p in ARTIFACTS.glob("phase9_*.csv"))
    report.add("No unplanned Phase-9 table was produced",
               produced <= planned,
               "unplanned: %s" % sorted(produced - planned))

    if missing:
        return
    _, acf = read_csv(ARTIFACTS / "phase9_residual_acf.csv")
    lags = sorted(set(int(r["lag_hours"]) for r in acf))
    report.add("Residual ACF uses exactly the frozen lags", lags == ACF_LAGS,
               "found: %s" % lags)
    _, target_acf = read_csv(ARTIFACTS / "phase9_test_target_acf.csv")
    report.add("Target ACF uses exactly the frozen lags",
               sorted(set(int(r["lag_hours"]) for r in target_acf)) == ACF_LAGS)
    _, regimes = read_csv(ARTIFACTS
                          / "phase9_concentration_regime_analysis.csv")
    report.add("Concentration table uses exactly the three frozen strata",
               sorted(set(r["regime"] for r in regimes)) == sorted(REGIMES))
    report.add("Concentration strata are labelled exploratory",
               all(r["stratum_label"] == "POST-TEST EXPLORATORY STRATA"
                   for r in regimes))
    _, seasons = read_csv(ARTIFACTS / "phase9_seasonal_error_analysis.csv")
    report.add("Seasonal table uses exactly DJF/MAM/JJA/SON",
               sorted(set(r["season"] for r in seasons))
               == sorted(SEASONS))
    _, hours = read_csv(ARTIFACTS / "phase9_hour_of_day_error_analysis.csv")
    report.add("Hour-of-day table covers exactly hours 0-23",
               sorted(set(int(r["target_hour"]) for r in hours))
               == list(range(24)))
    _, horizon = read_csv(ARTIFACTS / "phase9_horizon_error_analysis.csv")
    report.add("Horizon table covers exactly the frozen four horizons",
               sorted(set(int(r["horizon_hours"]) for r in horizon))
               == HORIZONS)


# ---------------------------------------------------------------------------
# Reconciliation against the locked Phase-8 values
# ---------------------------------------------------------------------------

def load_truth_and_predictions():
    from src.data.preprocessing import STATIONS, TARGET, index_of
    from src.evaluation.final_test import TEST_END, TEST_START, load_test_index
    index = load_test_index(PROCESSED)
    low, high = index_of(TEST_START), index_of(TEST_END)
    by_station = {}
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
                if row[TARGET] in ("NA", ""):
                    continue
                series[position] = float(row[TARGET])
        by_station[station] = series
    actual = np.empty(index["target_index"].size, dtype=np.float64)
    for station in STATIONS:
        rows = np.flatnonzero(index["station"] == station)
        actual[rows] = by_station[station][index["target_index"][rows]]
    stored = OrderedDict(
        (m, np.load(str(PRED_DIR / ("%s.npy" % m))).astype(np.float64))
        for m in MODELS)
    return index, actual, stored, by_station


def reconciliation_checks(report, freeze, index, actual, stored):
    locked = freeze["immutable_phase8_results"]
    buckets = OrderedDict()
    for position in range(actual.size):
        key = (index["station"][position], int(index["horizon"][position]))
        buckets.setdefault(key, []).append(position)
    report.add("Primary metric reconciles across exactly 48 cells",
               len(buckets) == 48)

    def macro(name):
        return float(np.mean([
            float(np.abs(actual[rows] - stored[name][rows]).mean())
            for rows in buckets.values()]))

    severe = actual > SEVERE_THRESHOLD
    checks = [
        ("B3_R2 locked primary MAE", macro("B3_R2"),
         locked["primary_B3_R2_macro_station_horizon_MAE"]),
        ("B0 locked macro MAE", macro("B0"),
         locked["reference_B0_macro_station_horizon_MAE"]),
        ("GRU_R1 locked severe MAE",
         float(np.abs(actual[severe] - stored["GRU_R1"][severe]).mean()),
         locked["secondary_GRU_R1_severe_MAE_gt_244"]),
        ("B0 locked severe MAE",
         float(np.abs(actual[severe] - stored["B0"][severe]).mean()),
         locked["reference_B0_severe_MAE_gt_244"]),
    ]
    for label, value, expected in checks:
        report.add("%s reconciles" % label, abs(value - expected) < 1e-9,
                   "%.12f vs locked %.12f" % (value, expected))
    report.add("Severe sample count reconciles",
               int(severe.sum()) == locked["severe_n"],
               "%d vs %d" % (int(severe.sum()), locked["severe_n"]))
    report.add("Targets exactly at 244.0 are excluded from severe",
               int(((actual == SEVERE_THRESHOLD) & severe).sum()) == 0,
               "%d targets sit at exactly 244.0"
               % int((actual == SEVERE_THRESHOLD).sum()))
    report.add("Every prediction array covers the canonical universe",
               all(stored[m].size == EXPECTED_TEST_SAMPLES for m in MODELS))


# ---------------------------------------------------------------------------
# Narrative and figure discipline
# ---------------------------------------------------------------------------

def narrative_checks(report):
    figures = load_json(ARTIFACTS / "phase9_figure_manifest.json")
    titles = figures["figure_titles"]
    unlabelled = [name for name, title in titles.items()
                  if FIGURE_LABEL not in title.upper()]
    report.add("Every Phase-9 figure is marked POST-TEST EXPLORATORY",
               not unlabelled, "unlabelled: %s" % unlabelled)
    on_disk = sorted(p.name for p in FIGURES.glob("phase9_*.png"))
    report.add("Figure manifest matches the figures on disk",
               sorted(titles) == on_disk,
               "manifest %s vs disk %s" % (sorted(titles), on_disk))
    drift = [name for name, digest in figures["figure_sha256"].items()
             if sha256_of_file(FIGURES / name) != digest]
    report.add("Figures unchanged since the figure manifest", not drift,
               "drifted: %s" % drift)

    comparison = (DOCS / "PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md").read_text(
        encoding="utf-8")
    report.add("V1/V2 comparison is labelled qualitative and structural",
               "QUALITATIVE" in comparison.upper()
               and "STRUCTURAL" in comparison.upper())
    report.add("V1/V2 comparison carries the direct-comparison warning",
               "not directly comparable" in comparison.lower()
               or "not comparable" in comparison.lower())
    report.add("V1/V2 comparison makes no cross-study numeric improvement "
               "claim",
               not re.search(r"improves?\s+V1[^.]{0,40}\d+(\.\d+)?\s*%",
                             comparison, re.I))

    record = (DOCS / "PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md").read_text(
        encoding="utf-8")
    report.add("Phase-9 record declares itself post-test exploratory",
               "POST-TEST EXPLORATORY" in record.upper())
    report.add("Phase-9 record states no model was changed",
               "no model was" in record.lower()
               or "models_trained" in record)
    # Word boundaries, not substrings: "unresolved" contains "solved" and is
    # the opposite of the claim being guarded against.
    lowered = record.lower() + comparison.lower()
    used = [phrase for phrase in BANNED_PHRASES
            if re.search(r"\b%s\b" % re.escape(phrase), lowered)]
    report.add("Phase-9 narrative avoids overclaiming vocabulary", not used,
               "found: %s" % used)


def summary_checks(report, freeze):
    summary = load_json(ARTIFACTS / "phase9_post_test_analysis_summary.json")
    report.add("Summary is classified POST_TEST_EXPLORATORY",
               summary["classification"] == "POST_TEST_EXPLORATORY")
    report.add("Summary records zero training, inference and new models",
               summary["models_trained"] == 0
               and summary["new_test_models_evaluated"] == 0
               and summary["predictions_generated"] == 0)
    report.add("Summary records Phase-8 untouched",
               summary["phase8_predictions_modified"] is False
               and summary["phase8_metrics_modified"] is False
               and summary["confirmatory_hierarchy_changed"] is False
               and summary["post_test_exploration_performed"] is True)
    report.add("Summary references the locked results by hash, not by copy",
               summary["phase8_locked_results"]["primary_results_lock_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "phase8_primary_results_lock.json"))
    report.add("Summary cites the Phase-9 analysis freeze",
               summary["analysis_freeze_sha256"]
               == sha256_of_file(ARTIFACTS / "phase9_analysis_freeze.json"))

    manifest = load_json(ARTIFACTS / "v2_phase9_manifest.json")
    report.add("Manifest records the immutable git anchors",
               manifest["pre_test_sha"] == PRE_TEST_SHA
               and manifest["locked_evaluation_sha"] == LOCKED_EVALUATION_SHA)
    report.add("Manifest records zero drift and zero training",
               manifest["phase8_prediction_drift"] == 0
               and manifest["phase8_metric_drift"] == 0
               and manifest["models_trained"] == 0
               and manifest["predictions_generated"] == 0
               and manifest["model_weights_changed"] is False)
    report.add("Manifest is classified POST_TEST_EXPLORATORY",
               manifest["analysis_classification"] == "POST_TEST_EXPLORATORY"
               and manifest["confirmatory_hierarchy_changed"] is False)
    drift = [name for name, digest in manifest["table_sha256"].items()
             if sha256_of_file(ARTIFACTS / name) != digest]
    report.add("Manifest table hashes match the tables on disk", not drift,
               "drifted: %s" % drift[:5])

    registry = load_json(ARTIFACTS
                         / "phase9_verification_tooling_registry.json")
    report.add("Phase-9 registry pins this validator",
               registry["validators"]["scripts/validate_phase9.py"]["sha256"]
               == sha256_of_file(PROJECT_ROOT / "scripts"
                                 / "validate_phase9.py"))
    report.add("Phase-9 registry parents the latest Phase-8 registry",
               registry["parent_registry"]
               == "artifacts/phase8_postcommit_tooling_registry.json"
               and registry["parent_registry_sha256"]
               == sha256_of_file(ARTIFACTS
                                 / "phase8_postcommit_tooling_registry.json"))


def main():
    report = Report()
    freeze = load_json(ARTIFACTS / "phase9_analysis_freeze.json")

    anchor_checks(report, freeze)
    freeze_checks(report, freeze)
    no_execution_checks(report, freeze)
    table_checks(report, freeze)
    index, actual, stored, _ = load_truth_and_predictions()
    reconciliation_checks(report, freeze, index, actual, stored)
    narrative_checks(report)
    summary_checks(report, freeze)

    report.render()
    print("")
    if report.failed():
        print("PHASE 9 VALIDATION FAILED: %d gate(s)" % len(report.failed()))
        return 1
    print("PHASE 9 VALIDATION PASSED")
    print("Classification: POST-TEST EXPLORATORY. Confirmatory status: NONE.")
    print("Models trained: 0. Predictions generated: 0. Phase-8 drift: 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
