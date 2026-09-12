"""Phase-9 master summary and manifest for AirSense V2.

Protocol Phase 9, sections 32 and 34. Run last, after every table, figure and
verification. Reads the frozen Phase-9 tables and records what they show; it
computes no new statistic of its own beyond selecting and summarising.

The locked Phase-8 results are carried **by hash and reference**, never
rewritten.

Usage:
    .venv-v2/bin/python scripts/build_phase9_summary_and_manifest.py
"""

import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = PROJECT_ROOT / "figures"
DOCS = PROJECT_ROOT / "docs"

MODELS = ["B3_R2", "GRU_R1", "B0"]
HORIZONS = [1, 6, 12, 24]
PRE_TEST_SHA = "75494266792e09bbcfa51c00aaacec58c5b0fb4a"
LOCKED_EVALUATION_SHA = "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"
TABLES = ["generalization_gap", "horizon_error_analysis",
          "station_error_analysis", "station_horizon_error_analysis",
          "residual_acf", "residual_acf_summary", "test_target_acf",
          "concentration_regime_analysis", "severe_tail_analysis",
          "severe_events", "severe_event_model_analysis",
          "severe_detection_analysis", "seasonal_error_analysis",
          "hour_of_day_error_analysis", "error_complementarity",
          "negative_prediction_analysis"]


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table(name):
    with open(str(ARTIFACTS / ("phase9_%s.csv" % name)), newline="",
              encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick(rows, **where):
    for row in rows:
        if all(str(row[k]) == str(v) for k, v in where.items()):
            return row
    return None


def main():
    gap = table("generalization_gap")
    horizon = table("horizon_error_analysis")
    station = table("station_error_analysis")
    severe = table("severe_tail_analysis")
    detection = table("severe_detection_analysis")
    regime = table("concentration_regime_analysis")
    acf = table("residual_acf_summary")
    target_acf = table("test_target_acf")
    seasonal = table("seasonal_error_analysis")
    hourly = table("hour_of_day_error_analysis")
    comp = table("error_complementarity")
    negative = table("negative_prediction_analysis")
    events = table("severe_events")
    event_models = table("severe_event_model_analysis")

    def f(row, key):
        return None if row is None or row[key] in ("NA", "") else float(
            row[key])

    station_mae = [(r["station"], float(r["MAE"])) for r in station
                   if r["model"] == "B3_R2"]
    station_mae.sort(key=lambda pair: pair[1])

    def peak_stats(model, horizon_value):
        rows = [r for r in event_models if r["model"] == model
                and r["horizon_hours"] == str(horizon_value)
                and r["abs_error_at_peak"] not in ("NA", "")]
        under = sum(1 for r in rows if r["underpredicted_peak"] == "true")
        flagged = sum(1 for r in rows
                      if r["predicted_severe_at_first_severe_hour"] == "true")
        return OrderedDict([
            ("events_with_a_peak_hour_sample", len(rows)),
            ("mean_abs_error_at_peak",
             round(sum(float(r["abs_error_at_peak"]) for r in rows)
                   / len(rows), 6) if rows else None),
            ("underpredicted_peak_pct",
             round(100.0 * under / len(rows), 4) if rows else None),
            ("flagged_first_severe_hour_pct",
             round(100.0 * flagged / len(rows), 4) if rows else None),
        ])

    summary = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase9_post_test_analysis_summary"),
        ("classification", "POST_TEST_EXPLORATORY"),
        ("confirmatory_status", "NONE"),
        ("analysis_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase9_analysis_freeze.json")),
        ("protocol_document_sha256", sha256_of_file(
            DOCS / "PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md")),
        ("phase8_locked_results", OrderedDict([
            ("carried_by", "reference and hash; never rewritten"),
            ("primary_results_lock", "artifacts/phase8_primary_results_lock.json"),
            ("primary_results_lock_sha256", sha256_of_file(
                ARTIFACTS / "phase8_primary_results_lock.json")),
            ("prediction_lock_sha256", sha256_of_file(
                ARTIFACTS / "phase8_prediction_lock.json")),
            ("phase8_manifest_sha256", sha256_of_file(
                ARTIFACTS / "v2_phase8_manifest.json")),
            ("primary_model", "B3_R2"),
            ("secondary_model", "GRU_R1"),
            ("reference_model", "B0"),
        ])),
        ("generalization", OrderedDict([
            ("summary", "All three models lost overall accuracy moving from "
                        "development validation to the locked test and gained "
                        "severe-tail accuracy, the 2016-17 tail being "
                        "slightly easier than 2015-16. Relative ordering is "
                        "preserved on both endpoints."),
            ("macro_MAE_relative_change_pct", OrderedDict(
                (m, f(pick(gap, model=m, metric="macro_station_horizon_MAE"),
                      "relative_change_pct")) for m in MODELS)),
            ("severe_MAE_relative_change_pct", OrderedDict(
                (m, f(pick(gap, model=m, metric="severe_MAE"),
                      "relative_change_pct")) for m in MODELS)),
            ("largest_overall_degradation", "B0 (+9.09%)"),
            ("smallest_overall_degradation", "B3_R2 (+6.23%)"),
        ])),
        ("horizon", OrderedDict([
            ("summary", "B3_R2 beats persistence at every horizon, by 9.47% "
                        "at h=1 rising to 12.30% at h=24. GRU_R1 is markedly "
                        "worse than persistence at h=1 (-25.16%) and better "
                        "at longer leads."),
            ("mae_rel_improvement_vs_B0_pct", OrderedDict(
                (m, OrderedDict((h, f(pick(horizon, model=m,
                                           horizon_hours=h),
                                      "mae_rel_improvement_vs_B0_pct"))
                                for h in HORIZONS)) for m in MODELS)),
            ("B0_micro_R2_at_h24", f(pick(horizon, model="B0",
                                          horizon_hours=24), "R2")),
            ("note", "B0's micro R2 at h=24 is negative: at a full day's lead "
                     "persistence is worse than predicting the test mean, "
                     "even though it wins the severe tail."),
        ])),
        ("station", OrderedDict([
            ("summary", "Northern suburban sites are easiest and central "
                        "urban sites hardest, for every model. The ordering "
                        "is near-identical across models, which is associated "
                        "with site character rather than model behaviour."),
            ("B3_R2_easiest_station", station_mae[0][0]),
            ("B3_R2_easiest_MAE", round(station_mae[0][1], 6)),
            ("B3_R2_hardest_station", station_mae[-1][0]),
            ("B3_R2_hardest_MAE", round(station_mae[-1][1], 6)),
        ])),
        ("residual_acf", OrderedDict([
            ("summary", "Residual autocorrelation at lag 1 remains ~0.95-0.96 "
                        "for every model at h=24: a large share of the "
                        "remaining error is systematic temporal structure, "
                        "not noise. At lag 24 the models separate by sign."),
            ("lag1_mean_h24", OrderedDict(
                (m, f(pick(acf, model=m, horizon_hours=24, lag_hours=1),
                      "mean")) for m in MODELS)),
            ("lag24_mean_h24", OrderedDict(
                (m, f(pick(acf, model=m, horizon_hours=24, lag_hours=24),
                      "mean")) for m in MODELS)),
            ("lag168_mean_h24", OrderedDict(
                (m, f(pick(acf, model=m, horizon_hours=24, lag_hours=168),
                      "mean")) for m in MODELS)),
        ])),
        ("target_acf", OrderedDict([
            ("summary", "Observed PM2.5 is strongly autocorrelated: ~0.97 at "
                        "1 h, ~0.43 at 24 h. This is the context in which "
                        "persistence remains hard to beat in the tail."),
            ("mean_over_stations", OrderedDict(
                (str(lag), round(sum(float(r["acf"]) for r in target_acf
                                     if int(r["lag_hours"]) == lag
                                     and r["acf"] != "NA")
                                 / sum(1 for r in target_acf
                                       if int(r["lag_hours"]) == lag
                                       and r["acf"] != "NA"), 6))
                for lag in (1, 6, 24, 168))),
        ])),
        ("concentration_regimes", OrderedDict([
            ("summary", "Classic regression to the mean: every model "
                        "over-predicts in the LOW stratum (negative mean "
                        "residual) and under-predicts in ELEVATED and SEVERE. "
                        "Error grows steeply with concentration."),
            ("mean_residual", OrderedDict(
                (m, OrderedDict((g, f(pick(regime, model=m, regime=g),
                                      "mean_residual"))
                                for g in ("LOW", "ELEVATED", "SEVERE")))
                for m in MODELS)),
            ("MAE", OrderedDict(
                (m, OrderedDict((g, f(pick(regime, model=m, regime=g), "MAE"))
                                for g in ("LOW", "ELEVATED", "SEVERE")))
                for m in MODELS)),
        ])),
        ("severe_tail", OrderedDict([
            ("summary", "The persistence advantage in the tail is entirely a "
                        "long-horizon phenomenon. At h=1 the three models are "
                        "close; by h=24 B3_R2 degrades far faster than either "
                        "GRU_R1 or B0."),
            ("severe_MAE_by_horizon", OrderedDict(
                (m, OrderedDict((h, f(pick(severe, model=m, horizon_hours=h),
                                      "MAE")) for h in HORIZONS))
                for m in MODELS)),
            ("underprediction_pct_by_horizon", OrderedDict(
                (m, OrderedDict((h, f(pick(severe, model=m, horizon_hours=h),
                                      "underprediction_pct"))
                                for h in HORIZONS)) for m in MODELS)),
        ])),
        ("severe_events", OrderedDict([
            ("summary", "586 strict severe events, median duration 4 h, "
                        "median peak 292 ug/m3. At h=24 no model is usable as "
                        "an early warning: B3_R2 flags the first severe hour "
                        "in 0.3% of events."),
            ("n_events", len(events)),
            ("peak_behaviour", OrderedDict(
                ("%s_h%d" % (m, h), peak_stats(m, h))
                for m in MODELS for h in (1, 24))),
        ])),
        ("severe_detection", OrderedDict([
            ("summary", "Recall collapses with horizon for the learned "
                        "models. At h=24 B3_R2 recalls 0.69% of severe hours "
                        "while B0 recalls 35.6%, so persistence is also the "
                        "best severe detector at long lead."),
            ("recall_by_horizon", OrderedDict(
                (m, OrderedDict((h, f(pick(detection, model=m,
                                           horizon_hours=h), "recall"))
                                for h in HORIZONS)) for m in MODELS)),
        ])),
        ("seasonal", OrderedDict([
            ("summary", "Error concentrates in winter. DJF carries the "
                        "overwhelming majority of severe hours and the "
                        "largest MAE for every model; JJA has almost no "
                        "severe hours."),
            ("severe_n_by_season_h24", OrderedDict(
                (s, int(pick(seasonal, model="B3_R2", season=s,
                             horizon_hours=24)["severe_n"]))
                for s in ("DJF", "MAM", "JJA", "SON"))),
        ])),
        ("hour_of_day", OrderedDict([
            ("summary", "A shallow diurnal pattern: error is lowest in the "
                        "morning around 08:00 and highest just after "
                        "midnight, with a spread of roughly 7-8 ug/m3 at "
                        "h=24. Small next to the horizon and concentration "
                        "effects."),
        ])),
        ("model_complementarity", OrderedDict([
            ("summary", "The models largely fail together - absolute errors "
                        "correlate 0.67-0.87 - but B0 complements the learned "
                        "models in the tail, holding the strictly smallest "
                        "error on 52.88% of severe samples. No ensemble was "
                        "constructed."),
            ("pearson_abs_error", OrderedDict(
                ("%s|%s|%s" % (r["stratum"], r["model_a"], r["model_b"]),
                 f(r, "pearson_abs_error")) for r in comp
                if r["model_b"] != "BEST_OF_THREE")),
            ("strictly_best_of_three_pct", OrderedDict(
                ("%s|%s" % (r["stratum"], r["model_a"]),
                 f(r, "fraction_a_strictly_better")) for r in comp
                if r["model_b"] == "BEST_OF_THREE")),
        ])),
        ("negative_predictions", OrderedDict([
            ("summary", "GRU_R1's unconstrained head produces 26x more "
                        "negative predictions than B3_R2, concentrated at "
                        "h=1. B0 is structurally non-negative. Nothing was "
                        "clipped."),
            ("overall", OrderedDict(
                (m, OrderedDict([
                    ("n_negative", int(pick(negative, model=m, scope="overall",
                                            key="all")["n_negative"])),
                    ("pct", f(pick(negative, model=m, scope="overall",
                                   key="all"), "pct_negative")),
                    ("min_prediction", f(pick(negative, model=m,
                                              scope="overall", key="all"),
                                         "min_prediction")),
                ])) for m in MODELS)),
        ])),
        ("v1_v2_failure_mode_comparison", OrderedDict([
            ("document", "docs/PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md"),
            ("sha256", sha256_of_file(
                DOCS / "PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md")),
            ("classification", "QUALITATIVE / STRUCTURAL COMPARISON"),
            ("direct_numeric_comparison_made", False),
            ("unresolved_in_both", ["systematic severe under-prediction",
                                    "strong residual autocorrelation"]),
            ("weakened", ["overall skill against the available naive "
                          "reference", "blanket under-prediction at all "
                          "concentrations"]),
            ("newly_demonstrated_in_v2",
             "persistence is hard to beat in the severe tail; V1 could not "
             "test this because its task had no persistence baseline"),
        ])),
        ("limitations", [
            "Single test period, one city, one pollutant.",
            "GRU_R1's confirmatory role rested on five-seed evidence while "
            "the evaluated artifact is the single seed-42 model.",
            "The locked test did not evaluate the paired regime contrasts "
            "that H1-H4 would need, so those remain development-supported "
            "and not independently re-tested.",
            "No uncertainty quantification: none was frozen before the test "
            "was opened, so none is reported.",
            "Severe strata are exploratory, not confirmatory endpoints.",
            "Error decomposition into bias and variance is only partially "
            "identifiable from point forecasts.",
        ]),
        ("future_work_candidates", [
            "Quantile or distributional forecasting for the upper tail.",
            "Explicit severe-weighted or asymmetric loss.",
            "Residual models that exploit the remaining lag-1 structure.",
            "A persistence-anchored hybrid, given B0's tail behaviour.",
            "Separately frozen post-test uncertainty quantification.",
            "Event-onset detection framed as classification rather than "
            "point regression.",
        ]),
        ("models_trained", 0),
        ("new_test_models_evaluated", 0),
        ("predictions_generated", 0),
        ("models_analysed", MODELS),
        ("phase8_predictions_modified", False),
        ("phase8_metrics_modified", False),
        ("confirmatory_hierarchy_changed", False),
        ("post_test_exploration_performed", True),
    ])
    path = ARTIFACTS / "phase9_post_test_analysis_summary.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("summary   %s" % sha256_of_file(path))

    figure_manifest = json.load(open(str(ARTIFACTS
                                         / "phase9_figure_manifest.json"),
                                     encoding="utf-8"))
    lock = json.load(open(str(ARTIFACTS / "phase8_prediction_lock.json"),
                          encoding="utf-8"))
    results = json.load(open(str(ARTIFACTS
                                 / "phase8_primary_results_lock.json"),
                             encoding="utf-8"))
    prediction_drift = sum(
        1 for name, entry in lock["models"].items()
        if sha256_of_file(PROJECT_ROOT / entry["prediction_file"])
        != entry["sha256"])
    metric_drift = sum(
        1 for name, digest in results["metric_table_sha256"].items()
        if sha256_of_file(ARTIFACTS / "phase8_final_test_tables" / name)
        != digest)

    manifest = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "v2_phase9_manifest"),
        ("phase", "phase_9_post_test_exploratory_error_analysis"),
        ("analysis_classification", "POST_TEST_EXPLORATORY"),
        ("confirmatory_status", "NONE"),
        ("phase8_closure_head_at_phase9_start",
         "adf19874fb7863579e8b0ca5a1b45d3c7975c097"),
        ("pre_test_sha", PRE_TEST_SHA),
        ("locked_evaluation_sha", LOCKED_EVALUATION_SHA),
        ("phase8_final_validation_receipt_sha256", sha256_of_file(
            ARTIFACTS / "phase8_postcommit_validation_receipt.json")),
        ("phase8_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase8_manifest.json")),
        ("phase8_prediction_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_prediction_lock.json")),
        ("phase8_primary_results_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_primary_results_lock.json")),
        ("phase9_analysis_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase9_analysis_freeze.json")),
        ("phase9_validator_sha256",
         sha256_of_file(PROJECT_ROOT / "scripts" / "validate_phase9.py")),
        ("phase9_tooling_registry_sha256", sha256_of_file(
            ARTIFACTS / "phase9_verification_tooling_registry.json")),
        ("phase9_code_sha256", OrderedDict([
            ("scripts/run_phase9_post_test_analysis.py", sha256_of_file(
                PROJECT_ROOT / "scripts/run_phase9_post_test_analysis.py")),
            ("scripts/verify_phase9_independently.py", sha256_of_file(
                PROJECT_ROOT / "scripts/verify_phase9_independently.py")),
            ("scripts/build_phase9_figures.py", sha256_of_file(
                PROJECT_ROOT / "scripts/build_phase9_figures.py")),
            ("scripts/build_phase9_summary_and_manifest.py", sha256_of_file(
                PROJECT_ROOT
                / "scripts/build_phase9_summary_and_manifest.py")),
        ])),
        ("table_sha256", OrderedDict(
            ("phase9_%s.csv" % name,
             sha256_of_file(ARTIFACTS / ("phase9_%s.csv" % name)))
            for name in TABLES)),
        ("figure_sha256", figure_manifest["figure_sha256"]),
        ("summary_sha256", sha256_of_file(path)),
        ("record_sha256", sha256_of_file(
            DOCS / "PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md")
         if (DOCS / "PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md").is_file()
         else None),
        ("protocol_sha256", sha256_of_file(
            DOCS / "PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md")),
        ("v1_v2_comparison_sha256", sha256_of_file(
            DOCS / "PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md")),
        ("models_trained", 0),
        ("model_weights_changed", False),
        ("predictions_generated", 0),
        ("models_analysed", MODELS),
        ("phase8_prediction_drift", prediction_drift),
        ("phase8_metric_drift", metric_drift),
        ("confirmatory_hierarchy_changed", False),
        ("primary_model", "B3_R2"),
        ("secondary_model", "GRU_R1"),
        ("reference_model", "B0"),
        ("post_test_exploration_performed", True),
        ("final_test_status", "evaluated"),
    ])
    manifest_path = ARTIFACTS / "v2_phase9_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("manifest  %s" % sha256_of_file(manifest_path))
    print("prediction drift %d | metric drift %d"
          % (prediction_drift, metric_drift))
    return 0


if __name__ == "__main__":
    sys.exit(main())
