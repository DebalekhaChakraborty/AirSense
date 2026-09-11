"""Build the AirSense V2 Phase-7 robustness summary and pre-test freeze.

This is a reporting-only program. It reads development-validation robustness
outputs, records deterministic candidates once, applies the selection rule
frozen before the experiments, and hashes the evidence. It trains no model,
does not import test data, and generates no predictions.

Usage:
    .venv-v2/bin/python scripts/build_phase7_pretest_freeze.py
"""

import csv
import hashlib
import io
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import STATIONS  # noqa
from src.evaluation import metrics as M  # noqa

ARTIFACTS = PROJECT_ROOT / "artifacts"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "models" / "robustness"
NEURAL = PROJECT_ROOT / "results" / "models" / "neural"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"

EXPECTED_CANDIDATE_FREEZE_SHA256 = (
    "0b6b0518182503425aff92a83a8f5d0f9f3b4721e8e667e052c87785339520c8"
)
SEEDS = [42, 43, 44, 45, 46]
STOCHASTIC = ["GRU_R1", "TCN_R1", "SA_R3"]
PAIRED_CONTROL = "SA_R2"
DETERMINISTIC = ["B0", "B3_R2"]
HORIZONS = [1, 6, 12, 24]
T_CRITICAL_95_DF4 = 2.7764451051977987
SCORE_FIELDS = [
    "macro_station_horizon_MAE", "micro_MAE", "macro_RMSE", "micro_RMSE",
    "macro_R2", "severe_MAE", "severe_RMSE", "severe_mean_residual",
    "severe_underprediction_pct", "n_negative", "min_prediction",
]


def sha256_of_file(path):
    with open(str(path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def finite_float(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit("non-finite metric encountered")
    return result


def rounded(value, digits=6):
    return None if value is None else round(float(value), digits)


def metric_summary(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return OrderedDict([
        ("mean", rounded(values.mean())),
        ("sample_std", rounded(values.std(ddof=1))),
        ("min", rounded(values.min())),
        ("max", rounded(values.max())),
    ])


def score(actual, prediction, station, horizon):
    cells = M.cell_metrics(actual, prediction, station, horizon)
    severe = M.severe_metrics(actual, prediction, station, horizon)
    negatives = M.negative_prediction_summary(prediction)
    return OrderedDict([
        ("macro_station_horizon_MAE", M.macro_from_cells(cells)),
        ("micro_MAE", M.mae(actual, prediction)),
        ("macro_RMSE", M.macro_from_cells(cells, "rmse")),
        ("micro_RMSE", M.rmse(actual, prediction)),
        ("macro_R2", M.macro_from_cells(cells, "r2")),
        ("severe_MAE", severe["severe_mae"]),
        ("severe_RMSE", severe["severe_rmse"]),
        ("severe_mean_residual", severe["severe_mean_residual"]),
        ("severe_underprediction_pct",
         severe["severe_underprediction_pct"]),
        ("n_negative", negatives["n_negative"]),
        ("min_prediction", negatives["min_prediction"]),
    ])


def index_arrays():
    rows = read_csv(PROCESSED / "sample_index" / "validation.csv")
    return (
        np.asarray([row["station"] for row in rows]),
        np.asarray([int(row["horizon_hours"]) for row in rows],
                   dtype=np.int32),
    )


def original_timings():
    neural_diagnostics = {
        row["model"]: row for row in
        read_csv(NEURAL / "neural_training_diagnostics.csv")
    }
    neural_models = {
        row["model"]: row for row in
        read_csv(NEURAL / "neural_full_models.csv")
    }
    spatial_diagnostics = {
        row["model"]: row for row in
        read_csv(SPATIAL / "spatiotemporal_training_diagnostics.csv")
    }
    spatial_models = {
        row["model"]: row for row in
        read_csv(SPATIAL / "spatiotemporal_full_models.csv")
    }
    return {
        "GRU_R1": (float(neural_diagnostics["GRU_R1"]["training_seconds"]),
                   float(neural_models["GRU_R1"]["inference_seconds"])),
        "TCN_R1": (float(neural_diagnostics["TCN_R1"]["training_seconds"]),
                   float(neural_models["TCN_R1"]["inference_seconds"])),
        "SA_R2": (float(spatial_diagnostics["SA_R2"]["training_seconds"]),
                  float(spatial_models["SA_R2"]["inference_seconds"])),
        "SA_R3": (float(spatial_diagnostics["SA_R3"]["training_seconds"]),
                  float(spatial_models["SA_R3"]["inference_seconds"])),
    }


def complete_seed_rows():
    path = OUT / "multiseed_metrics.csv"
    horizon_path = OUT / "multiseed_metrics_by_horizon.csv"
    if not path.is_file() or not horizon_path.is_file():
        raise SystemExit("multi-seed run is incomplete")
    rows = read_csv(path)
    expected = {(name, str(seed)) for name in STOCHASTIC + [PAIRED_CONTROL]
                for seed in SEEDS}
    observed = {(row["model"], row["seed"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise SystemExit("multi-seed table does not contain the exact run set")

    timings = original_timings()
    receipt = load_json(ARTIFACTS /
                        "phase7_multiseed_execution_receipt.json")
    candidate = load_json(ARTIFACTS / "phase7_candidate_freeze.json")
    header = list(rows[0]) + ["timing_provenance",
                              "fit_inference_loop_elapsed_seconds"]
    completed = []
    for original in rows:
        row = OrderedDict(original)
        seed = int(row["seed"])
        name = row["model"]
        parameter_count = candidate["candidates"].get(name, {}).get(
            "parameter_count")
        row["parameter_count"] = str(parameter_count or 144001)
        if seed == 42:
            row["training_seconds"] = str(timings[row["model"]][0])
            row["inference_seconds"] = str(timings[row["model"]][1])
            row["timing_provenance"] = "exact_frozen_phase_timing"
            row["fit_inference_loop_elapsed_seconds"] = ""
        else:
            elapsed = receipt["completion_log"][name][str(seed)][
                "fit_inference_loop_elapsed_seconds"]
            row["timing_provenance"] = (
                "completion_log_upper_bound_after_timing_split_lost")
            row["fit_inference_loop_elapsed_seconds"] = str(elapsed)
        completed.append(row)
    write_csv(OUT / "phase7_seed_metrics_complete.csv", header,
              [[row[column] for column in header] for row in completed])
    return completed, read_csv(horizon_path)


def deterministic_rows(actual, station, horizon, freeze):
    header = ["model", "fit_count", "source"] + SCORE_FIELDS
    rows = []
    payload = OrderedDict()
    for name in DETERMINISTIC:
        prediction_path = PRED / (name + "_validation.npy")
        prediction = np.load(str(prediction_path)).astype(np.float64)
        values = score(actual, prediction, station, horizon)
        row = [name, 1, "frozen_phase3"] + [rounded(values[key])
                                             for key in values]
        rows.append(row)
        payload[name] = OrderedDict([
            ("fit_count", 1),
            ("prediction_sha256", sha256_of_file(prediction_path)),
            ("metrics", OrderedDict((key, rounded(value))
                                    for key, value in values.items())),
            ("training_configuration",
             freeze["candidates"][name]["training_configuration"]),
        ])
    write_csv(OUT / "phase7_deterministic_metrics.csv", header, rows)
    return payload


def aggregate_models(seed_rows, freeze):
    fields = [
        "macro_station_horizon_MAE", "micro_MAE", "macro_RMSE",
        "micro_RMSE", "macro_R2", "severe_MAE", "severe_RMSE",
        "severe_mean_residual", "severe_underprediction_pct", "n_negative",
        "min_prediction",
    ]
    summary = OrderedDict()
    for name in STOCHASTIC:
        rows = [row for row in seed_rows if row["model"] == name]
        if [int(row["seed"]) for row in rows] != SEEDS:
            raise SystemExit("unexpected seed order for %s" % name)
        base_configuration = freeze["candidates"][name][
            "training_configuration"]
        configuration_by_seed = OrderedDict()
        for seed in SEEDS:
            configuration = OrderedDict(base_configuration)
            configuration["seed"] = seed
            configuration_by_seed[str(seed)] = configuration
        summary[name] = OrderedDict([
            ("seeds", SEEDS),
            ("fit_count", len(rows)),
            ("configuration_by_seed", configuration_by_seed),
            ("only_configuration_change_across_runs", "seed"),
            ("metrics", OrderedDict((field, metric_summary(rows, field))
                                    for field in fields)),
            ("timing_by_seed", OrderedDict(
                (row["seed"], OrderedDict([
                    ("training_seconds_exact",
                     finite_float(row["training_seconds"])),
                    ("inference_seconds_exact",
                     finite_float(row["inference_seconds"])),
                    ("fit_inference_loop_elapsed_seconds_upper_bound",
                     finite_float(row[
                         "fit_inference_loop_elapsed_seconds"])),
                    ("provenance", row["timing_provenance"]),
                ])) for row in rows)),
            ("prediction_sha256_by_seed", OrderedDict(
                (row["seed"], row["prediction_sha256"]) for row in rows)),
        ])
    return summary


def paired_analysis(seed_rows, horizon_rows):
    by_horizon = {(row["model"], int(row["seed"]), int(row["horizon"])): row
                  for row in horizon_rows}
    h24_rows = []
    horizon_differences = OrderedDict()
    for hz in HORIZONS:
        horizon_differences[hz] = []
        for seed in SEEDS:
            control = float(by_horizon[("SA_R2", seed, hz)]
                            ["macro_station_MAE"])
            relational = float(by_horizon[("SA_R3", seed, hz)]
                               ["macro_station_MAE"])
            difference = control - relational
            horizon_differences[hz].append(difference)
            if hz == 24:
                h24_rows.append([seed, control, relational,
                                 rounded(difference),
                                 rounded(100.0 * difference / control)])
    write_csv(OUT / "phase7_h24_paired_robustness.csv",
              ["seed", "SA_R2_macro_station_MAE",
               "SA_R3_macro_station_MAE", "R3_improvement_absolute",
               "R3_improvement_pct"], h24_rows)

    severe_rows = []
    severe_differences = []
    row_map = {(row["model"], int(row["seed"])): row for row in seed_rows}
    for seed in SEEDS:
        control = float(row_map[("SA_R2", seed)]["severe_MAE"])
        relational = float(row_map[("SA_R3", seed)]["severe_MAE"])
        difference = control - relational
        severe_differences.append(difference)
        severe_rows.append([
            seed, control, relational, rounded(difference),
            rounded(100.0 * difference / control),
            row_map[("SA_R2", seed)]["severe_underprediction_pct"],
            row_map[("SA_R3", seed)]["severe_underprediction_pct"],
            row_map[("SA_R2", seed)]["severe_mean_residual"],
            row_map[("SA_R3", seed)]["severe_mean_residual"],
        ])
    write_csv(OUT / "phase7_severe_paired_robustness.csv",
              ["seed", "SA_R2_severe_MAE", "SA_R3_severe_MAE",
               "R3_improvement_absolute", "R3_improvement_pct",
               "SA_R2_underprediction_pct", "SA_R3_underprediction_pct",
               "SA_R2_mean_residual", "SA_R3_mean_residual"], severe_rows)

    def paired_summary(values):
        values = np.asarray(values, dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        half = T_CRITICAL_95_DF4 * std / math.sqrt(values.size)
        return OrderedDict([
            ("definition", "positive means SA_R3 improves over SA_R2"),
            ("mean_difference", rounded(mean)),
            ("sample_std_difference", rounded(std)),
            ("paired_t_95pct_ci", [rounded(mean - half),
                                    rounded(mean + half)]),
            ("seeds_R3_better", int((values > 0).sum())),
            ("seeds_R3_worse", int((values < 0).sum())),
            ("all_seeds_same_direction", bool(np.all(values > 0)
                                              or np.all(values < 0))),
            ("note", "Small n=5 robustness interval; descriptive, not a "
                     "new selection endpoint."),
        ])

    horizon_summary = OrderedDict(
        ("h%d" % hz, paired_summary(horizon_differences[hz]))
        for hz in HORIZONS)
    write_csv(
        OUT / "phase7_sa_paired_by_horizon_summary.csv",
        ["horizon", "mean_R3_improvement", "sample_std_difference",
         "paired_t_95pct_ci_low", "paired_t_95pct_ci_high",
         "seeds_R3_better", "seeds_R3_worse"],
        [[hz, horizon_summary["h%d" % hz]["mean_difference"],
          horizon_summary["h%d" % hz]["sample_std_difference"],
          horizon_summary["h%d" % hz]["paired_t_95pct_ci"][0],
          horizon_summary["h%d" % hz]["paired_t_95pct_ci"][1],
          horizon_summary["h%d" % hz]["seeds_R3_better"],
          horizon_summary["h%d" % hz]["seeds_R3_worse"]]
         for hz in HORIZONS])

    return OrderedDict([
        ("horizon_SA_R3_vs_SA_R2", horizon_summary),
        ("h24_SA_R3_vs_SA_R2", horizon_summary["h24"]),
        ("severe_tail_SA_R3_vs_SA_R2",
         paired_summary(severe_differences)),
    ])


def loso_analysis():
    path = OUT / "leave_one_station_out.csv"
    horizon_path = OUT / "leave_one_station_out_by_horizon.csv"
    if not path.is_file() or not horizon_path.is_file():
        raise SystemExit("leave-one-station-out run is incomplete")
    rows = read_csv(path)
    if [row["held_out_station"] for row in rows] != list(STATIONS):
        raise SystemExit("leave-one-station-out station set/order is incomplete")
    horizon_rows = read_csv(horizon_path)
    if len(horizon_rows) != len(STATIONS) * len(HORIZONS):
        raise SystemExit("leave-one-station-out horizon table is incomplete")
    loso_mae = np.asarray([float(row["loso_macro_over_horizon_MAE"])
                           for row in rows])
    normal_mae = np.asarray([float(row["in_training_macro_over_horizon_MAE"])
                             for row in rows])
    loso_severe = np.asarray([float(row["loso_severe_MAE"])
                              for row in rows])
    normal_severe = np.asarray([float(row["in_training_severe_MAE"])
                                for row in rows])
    return OrderedDict([
        ("protocol", "withhold every training target for one station; its "
                     "causal history remains an input to the shared encoder"),
        ("seed", 42),
        ("station_count", len(rows)),
        ("mean_station_macro_over_horizon_MAE", rounded(loso_mae.mean())),
        ("normal_mean_station_macro_over_horizon_MAE",
         rounded(normal_mae.mean())),
        ("mean_absolute_MAE_penalty", rounded((loso_mae-normal_mae).mean())),
        ("mean_relative_MAE_penalty_pct",
         rounded(100.0 * ((loso_mae-normal_mae) / normal_mae).mean())),
        ("stations_improved_when_held_out", int((loso_mae < normal_mae).sum())),
        ("mean_station_severe_MAE", rounded(loso_severe.mean())),
        ("normal_mean_station_severe_MAE", rounded(normal_severe.mean())),
        ("mean_severe_MAE_penalty",
         rounded((loso_severe-normal_severe).mean())),
        ("caveat", "Target-transfer test, not a completely unseen sensor: "
                   "held-out station history remains visible."),
        ("prediction_sha256_by_station", OrderedDict(
            (row["held_out_station"], row["prediction_sha256"])
            for row in rows)),
    ])


def selection_table(deterministic, stochastic):
    table = []
    for name in DETERMINISTIC:
        metrics = deterministic[name]["metrics"]
        table.append(OrderedDict([
            ("model", name), ("fit_count", 1),
            ("primary_macro_MAE_mean",
             metrics["macro_station_horizon_MAE"]),
            ("primary_macro_MAE_sample_std", 0.0),
            ("macro_RMSE_mean", metrics["macro_RMSE"]),
            ("macro_R2_mean", metrics["macro_R2"]),
            ("severe_MAE_mean", metrics["severe_MAE"]),
            ("severe_underprediction_pct_mean",
             metrics["severe_underprediction_pct"]),
        ]))
    for name in STOCHASTIC:
        metrics = stochastic[name]["metrics"]
        table.append(OrderedDict([
            ("model", name), ("fit_count", 5),
            ("primary_macro_MAE_mean",
             metrics["macro_station_horizon_MAE"]["mean"]),
            ("primary_macro_MAE_sample_std",
             metrics["macro_station_horizon_MAE"]["sample_std"]),
            ("macro_RMSE_mean", metrics["macro_RMSE"]["mean"]),
            ("macro_R2_mean", metrics["macro_R2"]["mean"]),
            ("severe_MAE_mean", metrics["severe_MAE"]["mean"]),
            ("severe_underprediction_pct_mean",
             metrics["severe_underprediction_pct"]["mean"]),
        ]))
    return sorted(table, key=lambda row: row["primary_macro_MAE_mean"])


def main():
    candidate_path = ARTIFACTS / "phase7_candidate_freeze.json"
    if sha256_of_file(candidate_path) != EXPECTED_CANDIDATE_FREEZE_SHA256:
        raise SystemExit("pre-analysis candidate freeze has drifted")
    freeze = load_json(candidate_path)
    if freeze["final_test_status"] != "sealed":
        raise SystemExit("final test must remain sealed")

    station, horizon = index_arrays()
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    seed_rows, horizon_rows = complete_seed_rows()
    deterministic = deterministic_rows(actual, station, horizon, freeze)
    stochastic = aggregate_models(seed_rows, freeze)
    paired = paired_analysis(seed_rows, horizon_rows)
    loso = loso_analysis()
    spatial = load_json(ARTIFACTS / "phase7_spatial_analysis.json")
    selection = selection_table(deterministic, stochastic)
    selected = selection[0]["model"]

    evidence_paths = [
        "results/models/robustness/multiseed_metrics.csv",
        "results/models/robustness/multiseed_metrics_by_horizon.csv",
        "results/models/robustness/phase7_seed_metrics_complete.csv",
        "results/models/robustness/phase7_deterministic_metrics.csv",
        "results/models/robustness/phase7_h24_paired_robustness.csv",
        "results/models/robustness/phase7_sa_paired_by_horizon_summary.csv",
        "results/models/robustness/phase7_severe_paired_robustness.csv",
        "results/models/robustness/leave_one_station_out.csv",
        "results/models/robustness/leave_one_station_out_by_horizon.csv",
        "results/models/robustness/station_distance_matrix_km.csv",
        "results/models/robustness/spatial_attention_pairs.csv",
        "results/models/robustness/nearest_station_attention.csv",
        "artifacts/phase7_station_coordinates.json",
        "artifacts/phase7_spatial_analysis.json",
        "artifacts/phase7_multiseed_execution_receipt.json",
        "results/models/neural/determinism/GRU_R1_validation.npy",
        "results/models/neural/determinism/TCN_R1_validation.npy",
        "results/models/spatiotemporal/determinism/SA_R3_validation.npy",
        "scripts/validate_phase3.py",
        "artifacts/verification_tooling_registry.json",
    ]
    missing = [path for path in evidence_paths
               if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise SystemExit("missing evidence: %s" % missing)

    selected_entry = freeze["candidates"][selected]
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase7_pretest_freeze"),
        ("stage", "after_robustness_before_final_test"),
        ("candidate_freeze_sha256", sha256_of_file(candidate_path)),
        ("selection_rule", freeze["selection_rule"]),
        ("selection_ranking", selection),
        ("selected_final_candidates", [selected]),
        ("selected_configuration_and_artifacts", OrderedDict([
            (selected, selected_entry),
        ])),
        ("selection_rationale", "The selected model has the lowest "
         "development-validation macro station-horizon MAE under the rule "
         "declared before robustness. Seed means and sample standard "
         "deviations are used for stochastic candidates; deterministic "
         "candidates are recorded once. Severe MAE, macro RMSE and macro R2 "
         "are retained as secondary evidence and do not override the primary "
         "endpoint."),
        ("deterministic_candidates", deterministic),
        ("stochastic_multiseed_summary", stochastic),
        ("paired_cross_station_robustness", paired),
        ("leave_one_station_out", loso),
        ("spatial_analysis", spatial),
        ("determinism_evidence", OrderedDict([
            ("configuration", "torch deterministic algorithms enabled; "
             "Python, NumPy and Torch seeds fixed for each run"),
            ("seed42_GRU_R1_prediction_rerun_byte_identical",
             sha256_of_file(PRED / "GRU_R1_validation.npy")
             == sha256_of_file(PROJECT_ROOT / "results/models/neural/"
                               "determinism/GRU_R1_validation.npy")),
            ("seed42_TCN_R1_prediction_rerun_byte_identical",
             sha256_of_file(PRED / "TCN_R1_validation.npy")
             == sha256_of_file(PROJECT_ROOT / "results/models/neural/"
                               "determinism/TCN_R1_validation.npy")),
            ("seed42_SA_R3_prediction_rerun_byte_identical",
             sha256_of_file(PRED / "SA_R3_validation.npy")
             == sha256_of_file(PROJECT_ROOT / "results/models/spatiotemporal/"
                               "determinism/SA_R3_validation.npy")),
            ("note", "The selected seed-42 artifacts were independently "
             "refit in Phases 4 and 6; Phase 7 reuses and re-scores those "
             "byte-identical frozen predictions."),
        ])),
        ("post_freeze_validator_scope_correction", OrderedDict([
            ("reason", "The required phase7_pretest_freeze.json filename "
             "contains the substring test; the Phase-3 filename quarantine "
             "mistook that sealed-test declaration for a test result."),
            ("candidate_freeze_registry_sha256",
             freeze["frozen_inputs_sha256"]
             ["verification_tooling_registry"]),
            ("current_registry_sha256", sha256_of_file(
                ARTIFACTS / "verification_tooling_registry.json")),
            ("phase3_validator_previous_sha256",
             "ae2a842307eff20718795bc54bfec53952fbe0026318361e1313558a9e802dc5"),
            ("phase3_validator_current_sha256", sha256_of_file(
                PROJECT_ROOT / "scripts" / "validate_phase3.py")),
            ("change", "Exclude only filenames containing pretest from the "
             "broad artifact-name scan; all other test filenames and the "
             "dedicated prediction scan remain enforced."),
            ("scientific_artifacts_changed", False),
            ("models_or_predictions_changed", False),
            ("test_access", "none"),
        ])),
        ("evidence_sha256", OrderedDict(
            (path, sha256_of_file(PROJECT_ROOT / path))
            for path in evidence_paths)),
        ("upstream_manifest_sha256", freeze["upstream_manifest_sha256"]),
        ("severe_threshold", 244.0),
        ("context_hours", 48),
        ("architecture_search_performed", False),
        ("features_changed", False),
        ("context_changed", False),
        ("severe_balancing", False),
        ("target_transformation", False),
        ("predictions_clipped", False),
        ("full_development_refit_performed", False),
        ("test_data_accessed", False),
        ("test_targets_accessed", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("next_action", "STOP before opening the sealed final test or "
                        "refitting on full development data"),
    ])
    output = ARTIFACTS / "phase7_pretest_freeze.json"
    with open(str(output), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % output.relative_to(PROJECT_ROOT))
    print("sha256 %s" % sha256_of_file(output))
    print("selected %s" % selected)
    for row in selection:
        print("  %-8s fits %d  macro MAE %.6f  std %.6f  severe %.6f"
              % (row["model"], row["fit_count"],
                 row["primary_macro_MAE_mean"],
                 row["primary_macro_MAE_sample_std"],
                 row["severe_MAE_mean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
