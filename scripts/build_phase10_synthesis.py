"""Final scientific synthesis ledgers for AirSense V2.

Protocol Phase 10. **No experiment, no model, no prediction, no new metric.**
Every value here is parsed from a frozen artifact or is an arithmetic
restatement of already-frozen values.

Two stages so the manifest can hash the documents written between them:

    --ledgers    the role, hypothesis, claim, table, dataset, regime, V1/V2,
                 governance, limitations, future-work and reproducibility
                 ledgers, plus the publication indices
    --finalize   the master evidence index, the Phase-10 summary and the
                 Phase-10 manifest

Usage:
    .venv-v2/bin/python scripts/build_phase10_synthesis.py --ledgers
    .venv-v2/bin/python scripts/build_phase10_synthesis.py --finalize
"""

import csv
import hashlib
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
DOCS = PROJECT_ROOT / "docs"
FIGURES = PROJECT_ROOT / "figures"
RESULTS = PROJECT_ROOT / "results"

CONFIRMATORY = ["B3_R2", "GRU_R1", "B0"]
DEVELOPMENT_ONLY = ["B1", "B2", "B3_R0", "B3_R1", "GRU_R0", "GRU_R2", "TCN_R0",
                    "TCN_R1", "TCN_R2", "iTransformer_R1", "iTransformer_R2",
                    "SA_R2", "SA_R3"]
PHASE9_CLOSURE_SHA = "8cc3a7ab8a069b9a187904be68f14b9040928ab4"


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
        return list(csv.DictReader(handle))


def write_json(name, payload):
    path = ARTIFACTS / name
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("  %-52s %s" % (name, sha256_of_file(path)[:16]))
    return path


def write_csv(name, header, rows):
    path = ARTIFACTS / name
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    print("  %-52s %6d rows  %s" % (name, len(rows), sha256_of_file(path)[:16]))
    return path


def build_ledgers():
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    freeze7 = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    addendum = load_json(ARTIFACTS / "phase7_pretest_freeze_addendum.json")
    phase9 = load_json(ARTIFACTS / "phase9_post_test_analysis_summary.json")
    foundation = load_json(ARTIFACTS / "v2_foundation_manifest.json")
    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    regimes = load_json(ARTIFACTS / "information_regime_schema.json")
    phase5 = load_json(ARTIFACTS / "v2_phase5_manifest.json")
    dev = {r["model"]: r for r in
           read_csv(RESULTS / "validation" / "phase6_model_comparison.csv")}
    dev_severe = {r["model"]: r for r in
                  read_csv(RESULTS / "validation" / "phase6_severe_metrics.csv")}
    seeds = read_csv(RESULTS / "models" / "robustness" / "multiseed_metrics.csv")
    h4 = {r["scope"]: r for r in
          read_csv(RESULTS / "validation" / "phase6_h4_summary.csv")}

    print("model role ledger, hypothesis ledger, claim matrix, tables:")

    # -- 10: model role ledger ------------------------------------------
    write_json("phase10_model_role_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_model_role_ledger"),
        ("evidence_class", "C1"),
        ("frozen_before", "the locked final test was opened"),
        ("source", "artifacts/phase7_pretest_freeze_addendum.json"),
        ("roles", OrderedDict([
            ("B3_R2", "PRIMARY CONFIRMATORY MODEL"),
            ("GRU_R1", "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL"),
            ("B0", "REFERENCE BENCHMARK"),
        ])),
        ("endpoints", OrderedDict([
            ("B3_R2", "macro station-horizon MAE"),
            ("GRU_R1", "severe MAE at the frozen threshold 244.0"),
            ("B0", "both endpoints, as reference only"),
        ])),
        ("secondary_model_seed", 42),
        ("secondary_model_artifact",
         "results/models/neural/GRU_R1.safetensors"),
        ("B0_is_a_selected_learned_model", False),
        ("development_only_models", DEVELOPMENT_ONLY),
        ("development_only_permitted_use",
         "historical development tables only; no locked-test confirmatory "
         "role"),
        ("promotion_after_test_access_permitted", False),
        ("winner_label_assigned", False),
        ("winner_label_rationale",
         "The study reports roles frozen before the test was opened and the "
         "measured outcome for each. A single overall winner label would "
         "collapse endpoints that behave differently: the primary model leads "
         "overall while the reference benchmark leads in the severe tail."),
        ("foundation_models_executed", 0),
        ("foundation_model_predictions_generated", 0),
    ]))

    # -- 11: hypothesis ledger -------------------------------------------
    write_json("phase10_hypothesis_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_hypothesis_ledger"),
        ("rule", "The locked final test evaluated three models on two "
                 "endpoints. It did not evaluate the paired regime contrasts "
                 "H1-H4 require, so no hypothesis is marked as retested "
                 "there."),
        ("hypotheses", OrderedDict([
            ("H1", OrderedDict([
                ("hypothesis", "Target PM2.5 history improves forecasting."),
                ("original_definition",
                 "Adding target-station PM2.5 history (R0 -> R1) improves "
                 "macro station-horizon MAE."),
                ("evidence_class", "C2"),
                ("supporting_phases", ["Phase 3", "Phase 4"]),
                ("evidence", OrderedDict([
                    ("B3_R0_macro_MAE", float(dev["B3_R0"][
                        "macro_station_horizon_MAE"])),
                    ("B3_R1_macro_MAE", float(dev["B3_R1"][
                        "macro_station_horizon_MAE"])),
                    ("absolute_improvement", round(
                        float(dev["B3_R0"]["macro_station_horizon_MAE"])
                        - float(dev["B3_R1"]["macro_station_horizon_MAE"]),
                        6)),
                ])),
                ("status", "SUPPORTED ON DEVELOPMENT VALIDATION"),
                ("locked_test_retested", False),
                ("why_not_retested",
                 "B3_R0 and B3_R1 were not authorized final-test models."),
                ("limitations",
                 "A single paired contrast on one model family, measured "
                 "before the test was opened."),
                ("permitted_wording",
                 "Target PM2.5 history substantially improved development "
                 "validation accuracy."),
                ("prohibited_wording",
                 "Confirmed on the final test."),
            ])),
            ("H2", OrderedDict([
                ("hypothesis",
                 "The benefit of PM2.5 history is largest at short horizons."),
                ("original_definition",
                 "The R0 -> R1 gain decreases as forecast horizon grows."),
                ("evidence_class", "C2"),
                ("supporting_phases", ["Phase 3", "Phase 4"]),
                ("status", "SUPPORTED ON DEVELOPMENT VALIDATION"),
                ("locked_test_retested", False),
                ("why_not_retested",
                 "No R0/R1 pair was evaluated on the locked test."),
                ("limitations",
                 "Locked-test horizon results are consistent with the "
                 "pattern but are not a paired contrast and do not test it."),
                ("permitted_wording",
                 "On development validation the benefit of target history "
                 "was largest at the shortest horizon and diminished with "
                 "lead time."),
                ("prohibited_wording",
                 "The final test confirmed the horizon-dependent benefit."),
            ])),
            ("H3", OrderedDict([
                ("hypothesis",
                 "Co-pollutant history provides additional predictive "
                 "value."),
                ("original_definition",
                 "Adding co-pollutant history (R1 -> R2) improves macro "
                 "station-horizon MAE."),
                ("evidence_class", "C2"),
                ("supporting_phases", ["Phase 3", "Phase 4", "Phase 5"]),
                ("evidence", OrderedDict([
                    ("B3_R1_to_B3_R2", round(
                        float(dev["B3_R1"]["macro_station_horizon_MAE"])
                        - float(dev["B3_R2"]["macro_station_horizon_MAE"]),
                        6)),
                    ("GRU_R1_to_GRU_R2", round(
                        float(dev["GRU_R1"]["macro_station_horizon_MAE"])
                        - float(dev["GRU_R2"]["macro_station_horizon_MAE"]),
                        6)),
                    ("TCN_R1_to_TCN_R2", round(
                        float(dev["TCN_R1"]["macro_station_horizon_MAE"])
                        - float(dev["TCN_R2"]["macro_station_horizon_MAE"]),
                        6)),
                    ("iTransformer_R1_to_R2", round(
                        float(dev["iTransformer_R1"][
                            "macro_station_horizon_MAE"])
                        - float(dev["iTransformer_R2"][
                            "macro_station_horizon_MAE"]), 6)),
                    ("note", "Positive means the R2 variant improved. Only "
                             "the gradient-boosted family improved; all "
                             "three learned families degraded."),
                ])),
                ("status", "WEAK / MODEL-DEPENDENT DEVELOPMENT SUPPORT"),
                ("locked_test_retested", False),
                ("why_not_retested",
                 "Only B3_R2 was evaluated on the locked test, so no R1/R2 "
                 "contrast exists there."),
                ("limitations",
                 "The direction of the effect depends on model family; it "
                 "was positive only for gradient boosting."),
                ("permitted_wording",
                 "Co-pollutant history helped the gradient-boosted model on "
                 "development validation and degraded the neural and "
                 "transformer variants."),
                ("prohibited_wording",
                 "Co-pollutants generally improve forecasting."),
            ])),
            ("H4", OrderedDict([
                ("hypothesis",
                 "Cross-station information provides additional predictive "
                 "value."),
                ("original_definition",
                 "A cross-station regime (R3) improves on the same "
                 "architecture without it (R2)."),
                ("evidence_class", "C2"),
                ("supporting_phases", ["Phase 6", "Phase 7"]),
                ("evidence", OrderedDict([
                    ("SA_R2_macro_MAE", float(h4["overall"][
                        "SA_R2_macro_MAE"])),
                    ("SA_R3_macro_MAE", float(h4["overall"][
                        "SA_R3_macro_MAE"])),
                    ("absolute_improvement", float(h4["overall"][
                        "absolute_MAE_improvement"])),
                    ("relative_improvement_pct", float(h4["overall"][
                        "relative_MAE_improvement_pct"])),
                    ("paired_seeds_R3_better", "5/5 overall"),
                    ("h24_direction_reliable_across_seeds", False),
                    ("h24_single_seed_value", float(h4["horizon_24"][
                        "relative_MAE_improvement_pct"])),
                    ("loso_mean_relative_penalty_pct",
                     freeze7["leave_one_station_out"][
                         "mean_relative_MAE_penalty_pct"]),
                    ("loso_stations_improved_when_held_out",
                     freeze7["leave_one_station_out"][
                         "stations_improved_when_held_out"]),
                ])),
                ("status", "ROBUSTLY SUPPORTED ON DEVELOPMENT / ROBUSTNESS "
                           "EVIDENCE, BUT NOT INDEPENDENTLY CONFIRMED ON THE "
                           "LOCKED TEST"),
                ("locked_test_retested", False),
                ("why_not_retested",
                 "SA_R2 and SA_R3 were excluded from the confirmatory set."),
                ("limitations",
                 "The h=24 effect was not reliably directional across seeds. "
                 "Phase-7 spatial diagnostics found no proximity effect: "
                 "Spearman(distance, attention) = %s."
                 % freeze7["spatial_analysis"][
                     "spearman_distance_vs_attention"]),
                ("permitted_wording",
                 "Cross-station context improved the same architecture in a "
                 "paired comparison across all five seeds on development "
                 "validation."),
                ("prohibited_wording",
                 "Geographic propagation, spatial reasoning, "
                 "distance-driven attention, or confirmation on the final "
                 "test."),
            ])),
            ("H5", OrderedDict([
                ("hypothesis",
                 "Modern and spatiotemporal methods reduce severe-tail "
                 "error and under-prediction."),
                ("original_definition",
                 "Sequence and cross-station models improve severe MAE "
                 "relative to engineered and naive baselines."),
                ("evidence_class", "C2/C3"),
                ("supporting_phases", ["Phase 4", "Phase 6", "Phase 7",
                                       "Phase 8", "Phase 9"]),
                ("evidence", OrderedDict([
                    ("development_GRU_R1_severe_MAE", float(
                        dev_severe["GRU_R1"]["severe_MAE"])),
                    ("development_B3_R2_severe_MAE", float(
                        dev_severe["B3_R2"]["severe_MAE"])),
                    ("development_SA_R3_severe_MAE", float(
                        dev_severe["SA_R3"]["severe_MAE"])),
                    ("development_B0_severe_MAE", float(
                        dev_severe["B0"]["severe_MAE"])),
                    ("locked_GRU_R1_severe_MAE", results["secondary_value"]),
                    ("locked_B3_R2_severe_MAE", 131.901121),
                    ("locked_B0_severe_MAE",
                     results["reference_severe_value"]),
                    ("locked_severe_n", results["secondary_severe_n"]),
                ])),
                ("status", "PARTIALLY SUPPORTED AGAINST SOME LEARNED AND "
                           "ENGINEERED MODELS, BUT NOT SUPPORTED AS "
                           "SUPERIORITY OVER PERSISTENCE ON THE LOCKED TEST"),
                ("locked_test_retested", False),
                ("locked_test_note",
                 "GRU_R1 and B3_R2 both carried confirmatory roles, so the "
                 "GRU_R1 vs B3_R2 severe comparison is measurable on the "
                 "locked test. The hypothesis as originally framed concerns "
                 "a model class rather than these two artifacts, and the "
                 "class-level contrast was not retested."),
                ("limitations",
                 "The reference persistence benchmark had the lowest severe "
                 "MAE on the locked test. Phase-7 showed severe-tail metrics "
                 "swinging 29-35 ug/m3 across seeds, and the evaluated GRU "
                 "artifact is a single seed."),
                ("permitted_wording",
                 "On the locked test GRU_R1 achieved a lower severe MAE than "
                 "B3_R2, while persistence remained the strongest severe "
                 "benchmark."),
                ("prohibited_wording",
                 "The severe-tail problem is addressed; GRU beats "
                 "persistence in severe pollution; H5 is confirmed."),
            ])),
        ])),
    ]))

    # -- 12: claim-evidence matrix ---------------------------------------
    P8 = "artifacts/phase8_primary_results_lock.json"
    P9S = "artifacts/phase9_post_test_analysis_summary.json"
    claims = [
        ("dataset_audit",
         "The study uses the UCI Beijing Multi-Site Air-Quality dataset "
         "(ID 501), 12 stations, 420,768 hourly rows, 2013-03-01 to "
         "2017-02-28, with every source CSV hash-pinned.",
         "C5", "Phase 0", "artifacts/v2_foundation_manifest.json",
         "420768 rows; 12 stations", "whole dataset",
         "Raw missingness is documented; no imputation at source.",
         "The dataset and its integrity checks are hash-pinned.",
         "Any claim the data were cleaned or corrected."),
        ("chronological_split",
         "Train, validation and locked test are contiguous chronological "
         "blocks partitioned by target timestamp.",
         "C5", "Phase 2", "artifacts/phase10_dataset_ledger.json",
         "train 2013-03 to 2015-02; validation 2015-03 to 2016-02; test "
         "2016-03 to 2017-02", "whole study",
         "Early test targets legitimately have origins inside validation.",
         "Partitioning is by target timestamp, not by origin.",
         "Any claim of random splitting."),
        ("test_sealing",
         "The final test was sealed until Phase 8 and opened once, after an "
         "authorization receipt, with zero future-target access violations.",
         "C5", "Phase 8", "artifacts/final_test_opening_receipt.json;"
         "artifacts/phase8_prediction_lock.json",
         "future_target_access_violations = 0", "locked test",
         "Predictions were frozen and hashed before any metric was seen.",
         "The test was opened once under a pre-registered hierarchy.",
         "Any claim of repeated test evaluation."),
        ("causal_features",
         "Every model input obeys timestamp <= forecast origin; future "
         "observed meteorology and pollutants are prohibited.",
         "C5", "Phase 2-8", "artifacts/phase10_information_regime_ledger.json",
         "1,233,036 index checks, 0 violations", "all models",
         "Deterministic future calendar is permitted by design.",
         "Inputs are strictly causal with respect to the forecast origin.",
         "Any claim that future weather was used."),
        ("b3r2_locked",
         "On the locked final test B3_R2 achieved a macro station-horizon "
         "MAE of 31.98722559774534.",
         "C1", "Phase 8", P8,
         "31.98722559774534", "411,012 test samples, 48 cells",
         "Single test period, one city.",
         "This is the primary confirmatory result.",
         "Any restatement as an average across studies."),
        ("b0_comparison",
         "B3_R2 improved macro station-horizon MAE relative to causal "
         "persistence B0 on the locked test.",
         "C1", "Phase 8", P8,
         "B3_R2 31.98722559774534 vs B0 36.138513538776856",
         "411,012 test samples",
         "No inferential interval was predeclared.",
         "B3_R2 improved on persistence at every horizon.",
         "Any significance or confidence claim."),
        ("gru_severe",
         "On the locked test GRU_R1 seed 42 achieved a severe MAE of "
         "109.3328897571946 over 20,336 severe samples.",
         "C1", "Phase 8", P8,
         "109.3328897571946; n=20336", "actual PM2.5 > 244.0",
         "The evaluated artifact is a single seed; Phase 7 showed large "
         "severe-tail seed spread.",
         "This is the prespecified secondary confirmatory result.",
         "Any claim it beats the reference benchmark."),
        ("persistence_severe",
         "Causal persistence B0 had the lowest severe MAE on the locked "
         "test at 93.22447875688434.",
         "C1", "Phase 8", P8,
         "93.22447875688434", "actual PM2.5 > 244.0",
         "B0 remains the reference benchmark and is not promoted.",
         "Persistence remained the strongest severe benchmark.",
         "Any claim that persistence is the recommended model."),
        ("h1", "Target PM2.5 history improves forecasting.",
         "C2", "Phase 3-4", "artifacts/phase10_hypothesis_ledger.json",
         "B3_R0 40.623282 -> B3_R1 30.472594", "development validation",
         "Not retested on the locked test.",
         "Supported on development validation.",
         "Confirmed on the final test."),
        ("h2", "The benefit of PM2.5 history is largest at short horizons.",
         "C2", "Phase 3-4", "artifacts/phase10_hypothesis_ledger.json",
         "largest R0->R1 gain at h=1", "development validation",
         "Not retested on the locked test.",
         "Supported on development validation.",
         "Confirmed on the final test."),
        ("h3", "Co-pollutant history provides additional predictive value.",
         "C2", "Phase 3-5", "artifacts/phase10_hypothesis_ledger.json",
         "improves B3 only; degrades GRU, TCN and iTransformer",
         "development validation",
         "Direction depends on model family.",
         "Weak, model-dependent development support.",
         "Co-pollutants generally improve forecasting."),
        ("h4", "Cross-station information provides additional predictive "
               "value.",
         "C2", "Phase 6-7", "artifacts/phase10_hypothesis_ledger.json",
         "SA_R3 better than SA_R2 in 5/5 paired seeds; +2.193123% overall",
         "development validation and multi-seed robustness",
         "h=24 direction not reliable across seeds; not retested on the "
         "locked test.",
         "Robustly supported on development and robustness evidence.",
         "Geographic propagation or locked-test confirmation."),
        ("h5", "Modern methods reduce severe-tail error (hypothesis "
               "status).",
         "C2", "Phase 4-9", "artifacts/phase10_hypothesis_ledger.json",
         "development severe MAE: GRU_R1 118.965, SA_R3 121.270, B3_R2 "
         "136.612, B0 101.273",
         "development validation; hypothesis status only",
         "The locked-test outcome is carried separately by the C1 claims "
         "gru_severe and persistence_severe, where persistence remained the "
         "strongest severe benchmark. This row states hypothesis status, "
         "not a confirmatory result.",
         "Partially supported against some learned and engineered models.",
         "The severe-tail problem is addressed, or H5 was confirmed."),
        ("seed_instability",
         "Severe-tail metrics vary substantially across training seeds.",
         "C2", "Phase 7",
         "artifacts/phase10_table_multiseed_robustness.csv",
         "severe-tail spread of roughly 29-35 ug/m3 across five seeds",
         "stochastic models",
         "Five seeds only; descriptive, not inferential.",
         "Single-seed severe claims should be treated as provisional.",
         "Any confidence interval framing."),
        ("loso",
         "Withholding a station's training targets produced a small mean "
         "penalty, with several stations improving when held out.",
         "C2", "Phase 7", "artifacts/phase7_pretest_freeze.json",
         "mean relative penalty +0.134022%; 5/12 stations improved",
         "leave-one-station-out, seed 42",
         "The held-out station's causal historical observations remained "
         "available to the shared encoder.",
         "Held-out target-station supervision transfer with causal "
         "historical observations from the held-out station available.",
         "Completely unseen-station forecasting."),
        ("spatial_attention",
         "Cross-station attention was not proximity-driven.",
         "C2", "Phase 7", "artifacts/phase7_spatial_analysis.json",
         "Spearman(distance, attention) = 0.136289", "12 stations",
         "Coordinates were used for descriptive diagnostics only, never as "
         "a model input.",
         "Attention weights showed no proximity relationship.",
         "Learned pollution propagation or spatial reasoning."),
        ("foundation_models",
         "No foundation model was executed, for pretraining-overlap and "
         "licensing reasons.",
         "C5", "Phase 5",
         "artifacts/phase10_foundation_model_governance.json",
         "foundation_models_executed = 0", "study scope",
         "This is a governance outcome, not a measured model result.",
         "No eligible foundation model could be evaluated under the study's "
         "constraints.",
         "Foundation models failed or performed poorly."),
        ("residual_acf",
         "Substantial short-lag residual dependence remains after "
         "forecasting.",
         "C3", "Phase 9", "artifacts/phase9_residual_acf_summary.csv;" + P9S,
         "lag-1 residual ACF approximately 0.95-0.96 at h=24",
         "post-test exploratory",
         "Association only; no causal mechanism is established.",
         "Substantial unresolved temporal structure remains in the locked "
         "predictions.",
         "Any claim the models fail because autocorrelation is high."),
        ("severe_events",
         "At long horizons the models under-predict severe event peaks and "
         "rarely flag event onset.",
         "C3", "Phase 9", "artifacts/phase9_severe_event_model_analysis.csv;"
         + P9S,
         "586 events; at h=24 B3_R2 under-predicted the peak in 100% and "
         "flagged onset in 0.3%", "post-test exploratory",
         "Operational usability was never formally defined or tested.",
         "The frozen severe-event diagnostics are insufficient to establish "
         "early-warning capability at 24 h under the fixed 244.0 threshold.",
         "The models are or are not operationally ready."),
        ("seasonality",
         "Severe hours and error concentrate in winter.",
         "C3", "Phase 9", "artifacts/phase9_seasonal_error_analysis.csv",
         "DJF holds 3,120 of the h=24 severe hours against 28 in JJA",
         "post-test exploratory",
         "Descriptive stratification, not a predeclared endpoint.",
         "Severe error is associated with the winter season.",
         "Any seasonal forecasting claim."),
        ("complementarity",
         "The three models largely fail on the same samples, but "
         "persistence is complementary in the severe tail.",
         "C3", "Phase 9", "artifacts/phase9_error_complementarity.csv",
         "absolute-error correlation 0.67-0.87; B0 strictly best on 52.88% "
         "of severe samples", "post-test exploratory",
         "No ensemble, oracle or weighting was constructed.",
         "Error complementarity was measured descriptively.",
         "Any ensemble performance claim."),
        ("v1_v2",
         "AirSense V1 and V2 share two failure modes despite being "
         "different tasks.",
         "C4", "Phase 9",
         "docs/PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md",
         "severe under-prediction and residual autocorrelation recur",
         "qualitative / structural only",
         "V1 was concurrent-hour estimation with a different severe "
         "threshold; metric comparison is invalid.",
         "Structural continuity of failure modes.",
         "Any cross-study percentage improvement."),
    ]
    write_csv("phase10_claim_evidence_matrix.csv",
              ["claim_id", "claim_text", "evidence_class", "phase_source",
               "artifact_source", "metric_or_fact", "scope", "limitations",
               "permitted_wording", "prohibited_wording"],
              [list(c) for c in claims])

    # -- 13: locked-test publication table --------------------------------
    # Full precision, from the frozen Phase-8 metrics artifact. The Phase-8
    # comparison CSV rounds to six decimals, which is fine for a display table
    # but not for a confirmatory record that must reconcile exactly.
    metrics8 = load_json(ARTIFACTS / "phase8_final_test_metrics.json")["models"]
    rows = []
    roles = {"B3_R2": "PRIMARY CONFIRMATORY MODEL",
             "GRU_R1": "SECONDARY CONFIRMATORY SEVERE-TAIL MODEL",
             "B0": "REFERENCE BENCHMARK"}
    endpoint = {"B3_R2": "macro station-horizon MAE (PRIMARY ENDPOINT)",
                "GRU_R1": "severe MAE > 244.0 (SECONDARY ENDPOINT)",
                "B0": "both, as REFERENCE METRICS"}
    for model in CONFIRMATORY:
        row = metrics8[model]
        rows.append([model, roles[model], endpoint[model], 411012,
                     repr(row["macro_station_horizon_MAE"]),
                     repr(row["micro_MAE"]),
                     repr(row["macro_station_horizon_RMSE"]),
                     repr(row["micro_RMSE"]),
                     repr(row["macro_station_horizon_R2"]),
                     row["severe_n"], repr(row["severe_mae"]),
                     repr(row["severe_rmse"]),
                     repr(row["severe_mean_residual"]),
                     repr(row["severe_underprediction_pct"]),
                     row["n_negative"], repr(row["min_prediction"])])
    write_csv("phase10_table_locked_test.csv",
              ["model", "role", "confirmatory_endpoint", "n",
               "macro_station_horizon_MAE", "micro_MAE",
               "macro_station_horizon_RMSE", "micro_RMSE",
               "macro_station_horizon_R2", "severe_n", "severe_MAE_gt_244",
               "severe_RMSE", "severe_mean_residual",
               "severe_underprediction_pct", "n_negative", "min_prediction"],
              rows)

    # -- 14: development model table --------------------------------------
    family = {m: dev[m]["family"] for m in dev}
    regime_of = {"B0": "n/a", "B1": "n/a", "B2": "n/a", "B3_R0": "R0",
                 "B3_R1": "R1", "B3_R2": "R2", "GRU_R0": "R0", "GRU_R1": "R1",
                 "GRU_R2": "R2", "TCN_R0": "R0", "TCN_R1": "R1",
                 "TCN_R2": "R2", "iTransformer_R1": "R1",
                 "iTransformer_R2": "R2", "SA_R2": "R2", "SA_R3": "R3"}
    role_of = dict((m, "development only") for m in DEVELOPMENT_ONLY)
    role_of.update(roles)
    order = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2", "GRU_R1", "GRU_R2",
             "TCN_R1", "TCN_R2", "iTransformer_R1", "iTransformer_R2",
             "SA_R2", "SA_R3"]
    rows = []
    for model in order:
        rows.append([
            model, family.get(model, ""), regime_of.get(model, ""),
            dev[model]["macro_station_horizon_MAE"],
            dev_severe[model]["severe_MAE"],
            dev_severe[model]["severe_underprediction_pct"],
            dev[model]["evidence_phase"] if "evidence_phase" in dev[model]
            else dev[model].get("source", ""),
            "single run (deterministic)" if model in
            ("B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2")
            else "single seed 42",
            role_of.get(model, "development only")])
    write_csv("phase10_table_development_models.csv",
              ["model", "family", "regime", "validation_macro_MAE",
               "validation_severe_MAE", "validation_severe_underprediction_pct",
               "evidence_phase", "seed_provenance", "role"], rows)

    # -- 15: multi-seed robustness table -----------------------------------
    rows = []
    for model in ("GRU_R1", "TCN_R1", "SA_R3", "SA_R2"):
        values = [float(r["macro_station_horizon_MAE"]) for r in seeds
                  if r["model"] == model]
        severe_values = [float(r["severe_MAE"]) for r in seeds
                         if r["model"] == model]
        n = len(values)
        mean = sum(values) / n
        sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
        smean = sum(severe_values) / n
        ssd = (sum((v - smean) ** 2 for v in severe_values)
               / (n - 1)) ** 0.5
        rows.append([model, n, round(mean, 6), round(sd, 6),
                     round(min(values), 6), round(max(values), 6),
                     round(max(values) - min(values), 6), round(smean, 6),
                     round(ssd, 6), round(min(severe_values), 6),
                     round(max(severe_values), 6),
                     round(max(severe_values) - min(severe_values), 6),
                     "paired control" if model == "SA_R2" else "candidate"])
    write_csv("phase10_table_multiseed_robustness.csv",
              ["model", "seed_count", "macro_MAE_mean", "macro_MAE_sd",
               "macro_MAE_min", "macro_MAE_max", "macro_MAE_range",
               "severe_MAE_mean", "severe_MAE_sd", "severe_MAE_min",
               "severe_MAE_max", "severe_MAE_range", "role_in_phase7"], rows)

    # -- 16: post-test diagnostics table ------------------------------------
    acf = {(r["model"], r["lag_hours"]): r for r in
           read_csv(ARTIFACTS / "phase9_residual_acf_summary.csv")
           if r["horizon_hours"] == "24"}
    severe9 = {(r["model"], r["horizon_hours"]): r for r in
               read_csv(ARTIFACTS / "phase9_severe_tail_analysis.csv")}
    detect = {(r["model"], r["horizon_hours"]): r for r in
              read_csv(ARTIFACTS / "phase9_severe_detection_analysis.csv")}
    neg = {r["model"]: r for r in
           read_csv(ARTIFACTS / "phase9_negative_prediction_analysis.csv")
           if r["scope"] == "overall"}
    target_acf = read_csv(ARTIFACTS / "phase9_test_target_acf.csv")

    def target_mean(lag):
        values = [float(r["acf"]) for r in target_acf
                  if r["lag_hours"] == str(lag) and r["acf"] != "NA"]
        return round(sum(values) / len(values), 6)

    rows = []
    for model in CONFIRMATORY:
        rows.append([model, "residual_acf_lag1_h24",
                     acf[(model, "1")]["mean"], "C3",
                     "artifacts/phase9_residual_acf_summary.csv"])
        rows.append([model, "residual_acf_lag24_h24",
                     acf[(model, "24")]["mean"], "C3",
                     "artifacts/phase9_residual_acf_summary.csv"])
        rows.append([model, "severe_MAE_h24",
                     severe9[(model, "24")]["MAE"], "C3",
                     "artifacts/phase9_severe_tail_analysis.csv"])
        rows.append([model, "severe_underprediction_pct_h24",
                     severe9[(model, "24")]["underprediction_pct"], "C3",
                     "artifacts/phase9_severe_tail_analysis.csv"])
        rows.append([model, "severe_detection_recall_h24",
                     detect[(model, "24")]["recall"], "C3",
                     "artifacts/phase9_severe_detection_analysis.csv"])
        rows.append([model, "negative_prediction_count",
                     neg[model]["n_negative"], "C3",
                     "artifacts/phase9_negative_prediction_analysis.csv"])
    for lag in (1, 24):
        rows.append(["observed_PM2.5", "target_acf_lag%d" % lag,
                     target_mean(lag), "C3",
                     "artifacts/phase9_test_target_acf.csv"])
    rows.append(["all_models", "severe_event_count", len(read_csv(
        ARTIFACTS / "phase9_severe_events.csv")), "C3",
        "artifacts/phase9_severe_events.csv"])
    rows.append(["B0", "severe_strictly_best_of_three_pct", "52.876573", "C3",
                 "artifacts/phase9_error_complementarity.csv"])
    rows.append(["all_models", "DJF_severe_hours_h24", "3120", "C3",
                 "artifacts/phase9_seasonal_error_analysis.csv"])
    write_csv("phase10_table_posttest_diagnostics.csv",
              ["subject", "diagnostic", "value", "evidence_class",
               "artifact_source"], rows)

    # -- 17: dataset ledger -------------------------------------------------
    counts = universe["counts_by_partition"]
    write_json("phase10_dataset_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_dataset_ledger"),
        ("evidence_class", "C5"),
        ("dataset", OrderedDict([
            ("name", foundation["dataset"]["name"]),
            ("uci_id", foundation["dataset"]["uci_id"]),
            ("doi", foundation["dataset"]["doi"]),
            ("license", foundation["dataset"]["license"]),
            ("station_count", foundation["dataset"]["station_count"]),
            ("station_names", foundation["dataset"]["station_names"]),
            ("row_count", foundation["dataset"]["row_count"]),
            ("period_start", foundation["dataset"]["period_start"]),
            ("period_end", foundation["dataset"]["period_end"]),
            ("raw_archive_sha256", foundation["raw_archive_sha256"]),
        ])),
        ("partitions", OrderedDict([
            ("train", "2013-03-01T00:00:00 to 2015-02-28T23:00:00"),
            ("validation", "2015-03-01T00:00:00 to 2016-02-29T23:00:00"),
            ("locked_test", "2016-03-01T00:00:00 to 2017-02-28T23:00:00"),
            ("partitioned_by", "TARGET TIMESTAMP"),
        ])),
        ("sample_universe", OrderedDict([
            ("train", counts["train"]),
            ("validation", counts["validation"]),
            ("test", counts["test"]),
            ("source", "artifacts/sample_universe_manifest.json"),
        ])),
        ("context_hours", 48),
        ("horizons", [1, 6, 12, 24]),
        ("station_horizon_cells", 48),
        ("severe_threshold", 244.0),
        ("severe_threshold_source",
         "pooled training PM2.5 P95; a study-internal quantile, not a "
         "regulatory or AQI boundary"),
        ("training_median", 61.0),
        ("split_design_recomputed_in_phase10", False),
    ]))

    # -- 18: information regime ledger ---------------------------------------
    write_json("phase10_information_regime_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_information_regime_ledger"),
        ("evidence_class", "C5"),
        ("source", "artifacts/information_regime_schema.json"),
        ("context_hours_primary", regimes["context_hours_primary"]),
        ("horizons", regimes["horizons"]),
        ("common_sample_universe", regimes["common_sample_universe"]),
        ("regimes", OrderedDict([
            ("R0", "meteorology, wind direction, rain occurrence, calendar "
                   "and masks; no pollutant history"),
            ("R1", "R0 plus target-station PM2.5 history, its mask and its "
                   "gap age"),
            ("R2", "R1 plus co-pollutant history (PM10, SO2, NO2, CO, O3) "
                   "with masks and gap ages"),
            ("R3", "R2 plus cross-station history"),
        ])),
        ("causality_rules", OrderedDict([
            ("future_observed_meteorology", "PROHIBITED"),
            ("future_observed_pollutants", "PROHIBITED"),
            ("deterministic_future_calendar", "ALLOWED"),
            ("historical_observations",
             "only timestamps <= forecast origin"),
            ("rolling_pm25_reveal",
             "permitted once a timestamp becomes past relative to the "
             "forecast origin"),
        ])),
        ("enforcement", OrderedDict([
            ("index_checks", 1233036),
            ("future_target_access_violations", 0),
            ("source", "artifacts/phase8_prediction_lock.json"),
        ])),
    ]))

    # -- 22: V1/V2 structural ledger ------------------------------------------
    write_json("phase10_v1_v2_structural_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_v1_v2_structural_ledger"),
        ("evidence_class", "C4"),
        ("comparison_type", "QUALITATIVE / STRUCTURAL ONLY"),
        ("direct_metric_comparison_valid", False),
        ("direct_metric_comparison_reason",
         "V1 performed concurrent-hour PM2.5 estimation on a different "
         "period with a single-station design and a different severe "
         "threshold (282.0 vs 244.0). V2 performs future multi-horizon "
         "forecasting across 12 stations. The tasks and the severe "
         "populations differ."),
        ("v1_task", "concurrent-hour PM2.5 estimation"),
        ("v2_task", "true future multi-horizon forecasting"),
        ("v1_reference", "legacy branch 16c9030cf74508b620cc6d28f90346aa379f29cd"),
        ("structural_continuity_supported", [
            "systematic severe under-prediction remains",
            "strong residual temporal structure remains",
            "carry-forward of recent extreme values remains important"]),
        ("structural_progress", [
            "strict chronological forecasting protocol",
            "multi-station evidence",
            "explicit information regimes",
            "multi-horizon evaluation",
            "sealed locked test opened once",
            "multi-seed robustness analysis",
            "paired cross-station controlled experiment"]),
        ("percentage_improvements_computed", 0),
    ]))

    # -- 23: foundation-model governance ---------------------------------------
    write_json("phase10_foundation_model_governance.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_foundation_model_governance"),
        ("evidence_class", "C5"),
        ("framing", "Reproducibility and governance outcome, not a modelling "
                    "failure and not a measured model result."),
        ("source", "artifacts/v2_phase5_manifest.json;"
                   "docs/CHRONOS2_PRETRAINING_AUDIT.md"),
        ("models", OrderedDict([
            ("Chronos-2", OrderedDict([
                ("executed", False),
                ("checkpoint_downloaded",
                 phase5["chronos2_checkpoint_downloaded"]),
                ("reason", "material pretraining overlap identified with the "
                           "sealed test period through KDD Cup 2018 Beijing "
                           "data"),
                ("gate", phase5["chronos2_gate"]),
                ("audit_document", "docs/CHRONOS2_PRETRAINING_AUDIT.md"),
                ("audit_sha256", phase5["model_audit_sha256"][
                    "CHRONOS2_PRETRAINING_AUDIT.md"]),
                ("evidence_strength",
                 "overlap risk established by provenance audit; byte-level "
                 "contamination was not demonstrated and is not claimed"),
            ])),
            ("TimesFM-3.0", OrderedDict([
                ("executed", False),
                ("reason", phase5["timesfm3_excluded"]),
                ("category", "weight licence incompatible with the study's "
                             "intended use"),
            ])),
            ("Moirai-2.0-R-small", OrderedDict([
                ("executed", False),
                ("reason", phase5["moirai2_excluded"]),
                ("category", "non-commercial weight licence"),
            ])),
        ])),
        ("foundation_models_executed", 0),
        ("foundation_model_predictions_generated", 0),
        ("permitted_wording",
         "No eligible foundation model could be evaluated under the study's "
         "pretraining-overlap and licensing constraints."),
        ("prohibited_wording",
         "Foundation models failed, underperformed, or were beaten."),
    ]))

    # -- 24: limitations ledger -------------------------------------------------
    write_json("phase10_limitations_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_limitations_ledger"),
        ("limitations", [
            "Single metropolitan air-quality network; one city, one "
            "pollutant.",
            "No independent external-city test set.",
            "Only historical observed meteorology was available; no future "
            "weather forecast inputs were used.",
            "The GRU secondary test artifact is seed 42, while its secondary "
            "role was motivated by five-seed robustness evidence.",
            "H1 to H4 were not all independently re-tested on the locked "
            "final test.",
            "The severe threshold is the training pooled P95, not a "
            "regulatory AQI boundary.",
            "Phase-9 event, seasonal, detection and residual analyses are "
            "post-test exploratory.",
            "No formal inferential confidence intervals were predeclared for "
            "the locked test.",
            "LOSO evaluates held-out target supervision while retaining the "
            "held-out station's causal historical observations; it is not "
            "completely unseen-site forecasting.",
            "Coordinates were used for descriptive spatial diagnostics only, "
            "never as model features.",
            "SA_R3 attention was not proximity-driven and must not be "
            "interpreted as physical pollution transport.",
            "No eligible foundation model was evaluated, because of "
            "pretraining-overlap and licensing governance constraints.",
            "Negative neural predictions were intentionally left unclipped.",
            "Substantial short-lag residual autocorrelation remains in the "
            "locked predictions.",
            "Severe-episode forecasting remains an open problem.",
            "Operational early-warning usability was never formally defined "
            "or tested.",
            "GRU predictions carry an order-1e-5 dependence on inference "
            "batch composition; immaterial at reported precision but "
            "relevant to bitwise reproduction.",
        ]),
        ("limitation_count", 17),
    ]))

    # -- 25: future-work ledger --------------------------------------------------
    future = [
        "Tail-aware loss functions.",
        "Probabilistic forecasting.",
        "Quantile forecasting.",
        "Event-aware training objectives.",
        "Persistence/model hybrid gating.",
        "Forecast meteorology as an exogenous input.",
        "Cross-city external validation.",
        "A clean-pretraining foundation-model benchmark.",
        "Graph or geographic inductive biases.",
        "Calibrated severe-event warning probabilities.",
        "Longer contexts where scientifically justified.",
        "Uncertainty quantification with predeclared temporal block methods.",
    ]
    write_json("phase10_future_work_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_future_work_ledger"),
        ("improvement_claimed", False),
        ("note", "These are candidate directions motivated by the observed "
                 "failure modes. None was attempted, and no claim is made "
                 "that any would improve results."),
        ("future_work", [OrderedDict([("direction", item),
                                      ("tested_in_this_study", False)])
                         for item in future]),
        ("future_work_count", len(future)),
    ]))

    # -- 26: reproducibility ledger ------------------------------------------------
    lock = load_json(ARTIFACTS / "phase8_prediction_lock.json")
    preopening = load_json(ARTIFACTS
                           / "phase8_preopening_integrity_receipt.json")
    write_json("phase10_reproducibility_ledger.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_reproducibility_ledger"),
        ("evidence_class", "C5"),
        ("repository", "github.com/DebalekhaChakraborty/AirSense"),
        ("branch", "master"),
        ("git_anchors", OrderedDict([
            ("pre_test_sha", "75494266792e09bbcfa51c00aaacec58c5b0fb4a"),
            ("locked_evaluation_sha",
             "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"),
            ("post_evaluation_validation_sha",
             "7ccfc7e54f7d9008421cfcf764fcfaacc353e1db"),
            ("phase9_closure_sha", PHASE9_CLOSURE_SHA),
            ("legacy_v1_sha", "16c9030cf74508b620cc6d28f90346aa379f29cd"),
        ])),
        ("raw_data", OrderedDict([
            ("raw_archive_sha256", foundation["raw_archive_sha256"]),
            ("source_csv_sha256", foundation["source_csv_sha256"]),
        ])),
        ("processed_data_regeneration",
         ".venv-v2/bin/python scripts/build_forecast_dataset.py "
         "(data/processed is gitignored; every output digest is in "
         "artifacts/v2_phase2_manifest.json)"),
        ("sample_universe_sha256",
         sha256_of_file(ARTIFACTS / "sample_universe_manifest.json")),
        ("environment", preopening["environment"]),
        ("seeds", OrderedDict([
            ("training_seed", 42),
            ("robustness_seeds", [42, 43, 44, 45, 46]),
            ("final_test_artifact_seed", 42),
        ])),
        ("validator_chain", [
            "scripts/validate_foundation.py", "scripts/validate_phase1.py",
            "scripts/validate_phase2.py", "scripts/validate_phase3.py",
            "scripts/validate_phase4.py", "scripts/validate_phase5.py",
            "scripts/validate_phase6.py", "scripts/validate_phase7.py",
            "scripts/validate_phase8.py",
            "scripts/validate_phase8_postopening.py",
            "scripts/validate_phase9.py", "scripts/validate_phase10.py"]),
        ("phase_manifest_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in (
                "v2_foundation_manifest.json", "v2_phase1_manifest.json",
                "v2_phase2_manifest.json", "v2_phase3_manifest.json",
                "v2_phase4_manifest.json", "v2_phase5_manifest.json",
                "v2_phase6_manifest.json", "v2_phase8_manifest.json",
                "v2_phase9_manifest.json"))),
        ("opening_receipt_sha256",
         sha256_of_file(ARTIFACTS / "final_test_opening_receipt.json")),
        ("prediction_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_prediction_lock.json")),
        ("primary_results_lock_sha256",
         sha256_of_file(ARTIFACTS / "phase8_primary_results_lock.json")),
        ("phase8_final_validation_receipt_sha256", sha256_of_file(
            ARTIFACTS / "phase8_postcommit_validation_receipt.json")),
        ("phase9_analysis_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase9_analysis_freeze.json")),
        ("test_access_chronology", [
            "Phases 0-7: test sealed; no target value materialised.",
            "Phase 8 pre-opening: all gates green, receipt written, status "
            "still sealed.",
            "Phase 8 opening receipt: AUTHORIZED_TO_OPEN.",
            "Phase 8 Stage A: chronological prediction generation, metrics "
            "unseen.",
            "Phase 8 prediction lock: arrays hashed before scoring.",
            "Phase 8 Stage B: scoring; status evaluated.",
            "Phase 9: post-test exploratory analysis from frozen arrays.",
            "Phase 10: synthesis only."]),
        ("future_target_access_violations",
         lock["future_target_access_violations"]),
        ("model_retraining_after_test", 0),
        ("new_test_model_selection", 0),
    ]))

    # -- 27: canonical conclusion ----------------------------------------------
    improvement = 100.0 * (results["reference_primary_value"]
                           - results["primary_value"]) \
        / results["reference_primary_value"]
    write_json("phase10_canonical_conclusion.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_canonical_conclusion"),
        ("canonical_locked_results", OrderedDict([
            ("primary_B3_R2_macro_station_horizon_MAE",
             results["primary_value"]),
            ("secondary_GRU_R1_severe_MAE_gt_244",
             results["secondary_value"]),
            ("severe_n", results["secondary_severe_n"]),
            ("reference_B0_macro_station_horizon_MAE",
             results["reference_primary_value"]),
            ("reference_B0_severe_MAE_gt_244",
             results["reference_severe_value"]),
        ])),
        ("b3r2_relative_improvement_vs_B0_pct", improvement),
        ("b3r2_relative_improvement_derivation",
         "100 * (B0 macro - B3_R2 macro) / B0 macro, from the frozen "
         "full-precision values in the primary-results lock. No new metric "
         "and no statistical test."),
        ("conclusion", OrderedDict([
            ("locked_confirmatory_C1",
             "On the sealed 2016-2017 test, opened once under a hierarchy "
             "frozen beforehand, a gradient-boosted model over causal "
             "pollutant, meteorological and temporal history (B3_R2) "
             "achieved a macro station-horizon MAE of %.14f against %.14f "
             "for causal persistence, an improvement of %.6f%%, and it "
             "improved on persistence at every horizon from 1 to 24 hours. "
             "In the severe stratum (actual PM2.5 > 244.0, n = %d) the "
             "prespecified secondary model GRU_R1 seed 42 reached %.13f "
             "against %.13f for B3_R2, while persistence remained the "
             "strongest severe benchmark at %.14f."
             % (results["primary_value"],
                results["reference_primary_value"], improvement,
                results["secondary_severe_n"], results["secondary_value"],
                131.901121, results["reference_severe_value"])),
            ("development_and_robustness_C2",
             "Development and multi-seed experiments showed substantial "
             "value from target PM2.5 history, weak and model-dependent "
             "value from co-pollutant history, and a paired cross-station "
             "benefit favouring R3 over R2 in all five seeds. None of these "
             "contrasts was independently re-tested on the locked final "
             "test, because the confirmatory set contained three models and "
             "no paired regime variant."),
            ("post_test_exploratory_C3",
             "Post-test exploratory analysis is consistent with regression "
             "toward the conditional mean during extreme episodes: severe "
             "errors are strongly one-directional, with the mean residual "
             "accounting for most of the severe MAE, and under-prediction "
             "rising with forecast horizon. Substantial short-lag residual "
             "dependence remains after forecasting, indicating unresolved "
             "temporal structure in the locked predictions."),
            ("principal_open_problem",
             "The principal open problem is reliable forecasting of extreme "
             "pollution episodes rather than average multi-horizon "
             "accuracy."),
        ])),
        ("supporting_artifacts", OrderedDict([
            ("primary_results_lock",
             "artifacts/phase8_primary_results_lock.json"),
            ("prediction_lock", "artifacts/phase8_prediction_lock.json"),
            ("hypothesis_ledger", "artifacts/phase10_hypothesis_ledger.json"),
            ("claim_evidence_matrix",
             "artifacts/phase10_claim_evidence_matrix.csv"),
            ("post_test_summary",
             "artifacts/phase9_post_test_analysis_summary.json"),
            ("limitations", "artifacts/phase10_limitations_ledger.json"),
        ])),
        ("evidence_classes_used", ["C1", "C2", "C3"]),
        ("new_metric_introduced", False),
        ("new_statistical_test_introduced", False),
    ]))

    # -- 30/31: publication indices ----------------------------------------------
    write_json("phase10_publication_table_index.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_publication_table_index"),
        ("latex_formatting_deferred_to", "Phase 11"),
        ("tables", [
            OrderedDict([
                ("table_id", "Table 1"),
                ("title", "Dataset and chronological split"),
                ("artifact_source", "artifacts/phase10_dataset_ledger.json"),
                ("evidence_class", "C5"),
                ("manuscript_section", "Data and protocol"),
                ("allowed_interpretation",
                 "Descriptive; establishes the chronological design and the "
                 "sealed test."),
            ]),
            OrderedDict([
                ("table_id", "Table 2"),
                ("title", "Model families and information regimes"),
                ("artifact_source",
                 "artifacts/phase10_information_regime_ledger.json"),
                ("evidence_class", "C5"),
                ("manuscript_section", "Methods"),
                ("allowed_interpretation",
                 "Definitional; states what each regime may observe."),
            ]),
            OrderedDict([
                ("table_id", "Table 3"),
                ("title", "Development-validation model comparison"),
                ("artifact_source",
                 "artifacts/phase10_table_development_models.csv"),
                ("evidence_class", "C2"),
                ("manuscript_section", "Development results"),
                ("allowed_interpretation",
                 "Development evidence only; must not be read as final-test "
                 "performance."),
            ]),
            OrderedDict([
                ("table_id", "Table 4"),
                ("title", "Multi-seed robustness"),
                ("artifact_source",
                 "artifacts/phase10_table_multiseed_robustness.csv"),
                ("evidence_class", "C2"),
                ("manuscript_section", "Robustness"),
                ("allowed_interpretation",
                 "Descriptive spread across five seeds; not an inferential "
                 "interval."),
            ]),
            OrderedDict([
                ("table_id", "Table 5"),
                ("title", "Locked final-test confirmatory results"),
                ("artifact_source",
                 "artifacts/phase10_table_locked_test.csv"),
                ("evidence_class", "C1"),
                ("manuscript_section", "Locked final-test results"),
                ("allowed_interpretation",
                 "The only confirmatory table. Roles are frozen; no "
                 "post-test promotion."),
            ]),
            OrderedDict([
                ("table_id", "Table 6"),
                ("title", "Post-test severe and error diagnostics"),
                ("artifact_source",
                 "artifacts/phase10_table_posttest_diagnostics.csv"),
                ("evidence_class", "C3"),
                ("manuscript_section", "Post-test error analysis"),
                ("allowed_interpretation",
                 "Exploratory only; must be visibly separated from Table 5."),
            ]),
        ]),
    ]))

    figures = [
        ("Figure 1", "figures/phase9_generalization_gap.png", "Phase 9", "C3",
         "Development validation against the locked test",
         "Caption must state that development and locked-test values come "
         "from different partitions and that GRU_R1 uses its seed-42 row."),
        ("Figure 2", "figures/phase9_mae_by_horizon.png", "Phase 9", "C3",
         "Error growth with forecast horizon",
         "Caption must not imply a confirmatory horizon-wise test."),
        ("Figure 3", "figures/phase9_station_mae_heatmap.png", "Phase 9", "C3",
         "Station-level error structure",
         "Descriptive; ordering is associated with site character."),
        ("Figure 4", "figures/phase9_severe_mae_by_horizon.png", "Phase 9",
         "C3", "Severe-tail error by horizon",
         "Caption must state the severe rule and that 244.0 is the training "
         "pooled P95, not a regulatory threshold."),
        ("Figure 5", "figures/phase9_severe_underprediction.png", "Phase 9",
         "C3", "Severe under-prediction rate by horizon",
         "Caption must state residual = actual - prediction."),
        ("Figure 6", "figures/phase9_residual_acf.png", "Phase 9", "C3",
         "Residual autocorrelation at frozen lags",
         "Association only; no causal claim."),
        ("Figure 7", "figures/phase9_concentration_regimes.png", "Phase 9",
         "C3", "Error by training-frozen concentration stratum",
         "Boundaries are training-derived; no test quantile was computed."),
        ("Figure 8", "figures/phase9_severe_event_peak_error.png", "Phase 9",
         "C3", "Error at severe-event peaks",
         "Must not be framed as an operational early-warning evaluation."),
        ("Figure 9", "figures/phase8_final_test_primary_mae.png", "Phase 8",
         "C1", "Locked final-test primary endpoint",
         "The only confirmatory figure; roles must be labelled."),
        ("Figure 10", "figures/phase8_final_test_severe_mae.png", "Phase 8",
         "C1", "Locked final-test secondary endpoint",
         "Caption must note persistence is a reference benchmark, not a "
         "selected model."),
    ]
    write_json("phase10_publication_figure_index.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_publication_figure_index"),
        ("new_figures_created", 0),
        ("existing_figures_modified", 0),
        ("note", "Selected from existing frozen figures only. Redrawing for "
                 "journal style, if wanted, belongs to a later phase."),
        ("figures", [OrderedDict([
            ("figure_id", fid), ("source_path", path),
            ("source_phase", phase), ("evidence_class", cls),
            ("manuscript_role", role), ("caption_constraints", caption)])
            for fid, path, phase, cls, role, caption in figures]),
    ]))
    return 0


def finalize():
    print("master evidence index, summary, manifest:")
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")

    def entry(artifact, cls, description):
        return OrderedDict([
            ("artifact", artifact),
            ("sha256", sha256_of_file(PROJECT_ROOT / artifact)),
            ("evidence_class", cls),
            ("description", description)])

    evidence = OrderedDict([
        ("research_question", entry(
            "docs/PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md", "C5",
            "Research question, design and synthesis")),
        ("dataset", entry("artifacts/phase10_dataset_ledger.json", "C5",
                          "Dataset and chronological split")),
        ("information_regimes", entry(
            "artifacts/phase10_information_regime_ledger.json", "C5",
            "Information regimes and causality rules")),
        ("locked_primary_result", entry(
            "artifacts/phase8_primary_results_lock.json", "C1",
            "Primary confirmatory endpoint")),
        ("locked_secondary_result", entry(
            "artifacts/phase8_final_test_tables/confirmatory_endpoints.json",
            "C1", "Secondary confirmatory endpoint")),
        ("reference_benchmark", entry(
            "artifacts/phase10_table_locked_test.csv", "C1",
            "Locked-test table including the reference benchmark")),
        ("prediction_lock", entry("artifacts/phase8_prediction_lock.json",
                                  "C1",
                                  "Predictions frozen before scoring")),
        ("model_roles", entry("artifacts/phase10_model_role_ledger.json",
                              "C1", "Frozen model roles")),
        ("hypothesis_ledger", entry(
            "artifacts/phase10_hypothesis_ledger.json", "C2",
            "H1-H5 status and permitted wording")),
        ("development_models", entry(
            "artifacts/phase10_table_development_models.csv", "C2",
            "Development validation comparison")),
        ("robustness_evidence", entry(
            "artifacts/phase10_table_multiseed_robustness.csv", "C2",
            "Multi-seed robustness")),
        ("exploratory_diagnostics", entry(
            "artifacts/phase10_table_posttest_diagnostics.csv", "C3",
            "Post-test exploratory diagnostics")),
        ("post_test_summary", entry(
            "artifacts/phase9_post_test_analysis_summary.json", "C3",
            "Phase-9 findings")),
        ("v1_v2_continuity", entry(
            "artifacts/phase10_v1_v2_structural_ledger.json", "C4",
            "Structural continuity with AirSense V1")),
        ("foundation_model_governance", entry(
            "artifacts/phase10_foundation_model_governance.json", "C5",
            "Why no foundation model was executed")),
        ("claim_evidence_matrix", entry(
            "artifacts/phase10_claim_evidence_matrix.csv", "C5",
            "Every manuscript-level claim and its source")),
        ("limitations", entry("artifacts/phase10_limitations_ledger.json",
                              "C5", "Study limitations")),
        ("future_work", entry("artifacts/phase10_future_work_ledger.json",
                              "C5", "Untested future directions")),
        ("reproducibility", entry(
            "artifacts/phase10_reproducibility_ledger.json", "C5",
            "Reproduction chain")),
        ("canonical_conclusion", entry(
            "artifacts/phase10_canonical_conclusion.json", "C1",
            "Evidence-bounded conclusion")),
        ("publication_claim_guide", entry(
            "docs/PUBLICATION_CLAIM_GUIDE.md", "C5",
            "Permitted and prohibited wording")),
    ])
    write_json("phase10_master_evidence_index.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_master_evidence_index"),
        ("purpose", "Evidence backbone for Phase-11 manuscript drafting. "
                    "Every claim must trace here."),
        ("evidence", evidence),
        ("entry_count", len(evidence)),
    ]))

    write_json("phase10_final_synthesis_summary.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase10_final_synthesis_summary"),
        ("phase", 10),
        ("classification", "FINAL_SCIENTIFIC_SYNTHESIS"),
        ("new_scientific_experiments", 0),
        ("new_test_analyses", 0),
        ("models_trained", 0),
        ("predictions_generated", 0),
        ("new_test_models", 0),
        ("new_metrics_computed", 0),
        ("phase8_drift", 0),
        ("phase9_drift", 0),
        ("confirmatory_hierarchy_changed", False),
        ("locked_test_result_changed", False),
        ("post_test_findings_reclassified_as_confirmatory", False),
        ("manuscript_drafting_started", False),
        ("canonical_locked_results", OrderedDict([
            ("primary_B3_R2_macro_station_horizon_MAE",
             results["primary_value"]),
            ("secondary_GRU_R1_severe_MAE_gt_244", results["secondary_value"]),
            ("severe_n", results["secondary_severe_n"]),
            ("reference_B0_macro_station_horizon_MAE",
             results["reference_primary_value"]),
            ("reference_B0_severe_MAE_gt_244",
             results["reference_severe_value"]),
        ])),
        ("phase10_status", "SYNTHESIS_FROZEN"),
        ("next_phase", "Phase 11 manuscript drafting, after human review"),
    ]))

    tracked = [
        "artifacts/phase10_synthesis_freeze.json",
        "scripts/validate_phase10.py",
        "artifacts/phase10_verification_tooling_registry.json",
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
        "scripts/build_phase10_synthesis.py",
        "scripts/verify_phase10_independently.py",
    ]
    write_json("v2_phase10_manifest.json", OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "v2_phase10_manifest"),
        ("phase", "phase_10_final_scientific_synthesis"),
        ("phase9_closure_sha", PHASE9_CLOSURE_SHA),
        ("pre_test_sha", "75494266792e09bbcfa51c00aaacec58c5b0fb4a"),
        ("locked_evaluation_sha",
         "9a787a6320a3eca2104abfcbf10c8dbd29bfc19b"),
        ("artifact_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / name))
            for name in tracked if (PROJECT_ROOT / name).exists())),
        ("models_trained", 0),
        ("predictions_generated", 0),
        ("new_metrics", 0),
        ("new_scientific_analysis", 0),
        ("phase8_scientific_drift", 0),
        ("phase9_scientific_drift", 0),
        ("confirmatory_hierarchy_changed", False),
        ("primary_model", "B3_R2"),
        ("secondary_model", "GRU_R1"),
        ("reference_model", "B0"),
        ("phase10_status", "SYNTHESIS_FROZEN"),
    ]))
    return 0


def main():
    if "--finalize" in sys.argv:
        return finalize()
    return build_ledgers()


if __name__ == "__main__":
    sys.exit(main())
