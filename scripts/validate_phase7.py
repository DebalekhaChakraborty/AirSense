"""Read-only validator for AirSense V2 Phase 7.

Recomputes development-validation robustness metrics and spatial diagnostics,
checks the leave-one-station-out evidence, verifies every frozen hash, and
runs all upstream validators. It trains nothing and writes nothing.

Usage:
    .venv-v2/bin/python scripts/validate_phase7.py
"""

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
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
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
EDA = PROJECT_ROOT / "results" / "eda_training"

EXPECTED_FREEZE = (
    "0b6b0518182503425aff92a83a8f5d0f9f3b4721e8e667e052c87785339520c8"
)
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["GRU_R1", "TCN_R1", "SA_R3", "SA_R2"]
HORIZONS = [1, 6, 12, 24]


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))

    def render(self):
        width = 74
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


def sha256_of_file(path):
    with open(str(path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left, right, tolerance=5e-6):
    return abs(float(left) - float(right)) <= tolerance


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = values.argsort(kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    ranked[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranked)
    return (sums / counts)[inverse]


def spearman(left, right):
    left, right = ranks(left), ranks(right)
    left -= left.mean()
    right -= right.mean()
    denominator = np.sqrt((left * left).sum() * (right * right).sum())
    return float((left * right).sum() / denominator)


def haversine(first, second):
    lat1 = math.radians(float(first["latitude"]))
    lon1 = math.radians(float(first["longitude"]))
    lat2 = math.radians(float(second["latitude"]))
    lon2 = math.radians(float(second["longitude"]))
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    value = (math.sin(delta_lat / 2.0) ** 2
             + math.cos(lat1) * math.cos(lat2)
             * math.sin(delta_lon / 2.0) ** 2)
    return 2.0 * 6371.0088 * math.asin(math.sqrt(value))


def arrays():
    index = read_csv(PROCESSED / "sample_index" / "validation.csv")
    station = np.asarray([row["station"] for row in index])
    horizon = np.asarray([int(row["horizon_hours"]) for row in index],
                         dtype=np.int32)
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    return actual, station, horizon


def metric_values(actual, prediction, station, horizon):
    cells = M.cell_metrics(actual, prediction, station, horizon)
    severe = M.severe_metrics(actual, prediction, station, horizon)
    negative = M.negative_prediction_summary(prediction)
    return {
        "macro_station_horizon_MAE": M.macro_from_cells(cells),
        "micro_MAE": M.mae(actual, prediction),
        "macro_RMSE": M.macro_from_cells(cells, "rmse"),
        "micro_RMSE": M.rmse(actual, prediction),
        "macro_R2": M.macro_from_cells(cells, "r2"),
        "severe_MAE": severe["severe_mae"],
        "severe_RMSE": severe["severe_rmse"],
        "severe_mean_residual": severe["severe_mean_residual"],
        "severe_underprediction_pct":
            severe["severe_underprediction_pct"],
        "n_negative": negative["n_negative"],
        "min_prediction": negative["min_prediction"],
    }


def upstream_checks(report, candidate):
    validators = [
        ("validate_foundation.py", "python3"),
        ("validate_phase1.py", "python3"),
        ("validate_phase2.py", "python3"),
        ("validate_phase3.py", sys.executable),
        ("validate_phase4.py", sys.executable),
        ("validate_phase5.py", sys.executable),
        ("validate_phase6.py", sys.executable),
    ]
    for name, interpreter in validators:
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)
    report.add("Pre-analysis candidate freeze is byte-stable",
               sha256_of_file(ARTIFACTS / "phase7_candidate_freeze.json")
               == EXPECTED_FREEZE)
    drift = []
    for name, digest in candidate["upstream_manifest_sha256"].items():
        path = ARTIFACTS / name
        if not path.is_file() or sha256_of_file(path) != digest:
            drift.append(name)
    report.add("Phase 0-6 manifests match the pre-analysis freeze", not drift,
               "drifted: %s" % drift)


def candidate_checks(report, candidate):
    report.add("Candidate set is exactly the predeclared five",
               list(candidate["candidates"])
               == ["B0", "B3_R2", "GRU_R1", "TCN_R1", "SA_R3"])
    report.add("Stochastic seeds are exactly 42 through 46",
               candidate["robustness_seeds"] == SEEDS)
    report.add("Threshold and context remain frozen",
               candidate["severe_threshold"] == M.SEVERE_THRESHOLD == 244.0
               and candidate["context_hours"] == 48)
    bad = []
    for name, entry in candidate["candidates"].items():
        prediction = PROJECT_ROOT / entry["prediction_file"]
        if (not prediction.is_file()
                or sha256_of_file(prediction) != entry["prediction_sha256"]):
            bad.append(name + ":prediction")
        if entry.get("config_sha256"):
            base = SPATIAL if name.startswith("SA_") else (
                PROJECT_ROOT / "results" / "models" / "neural")
            config = base / (name + "_config.json")
            if sha256_of_file(config) != entry["config_sha256"]:
                bad.append(name + ":config")
    report.add("Frozen candidate predictions/configurations are unchanged",
               not bad, "drifted: %s" % bad)


def multiseed_checks(report, actual, station, horizon):
    rows = read_csv(OUT / "phase7_seed_metrics_complete.csv")
    receipt = load_json(ARTIFACTS /
                        "phase7_multiseed_execution_receipt.json")
    counts = Counter(row["model"] for row in rows)
    report.add("Exactly five runs exist for every stochastic arm/control",
               counts == Counter({name: 5 for name in MODELS})
               and {(row["model"], int(row["seed"])) for row in rows}
               == {(name, seed) for name in MODELS for seed in SEEDS})
    report.add("Every seed has an honest timing record",
               all((row["training_seconds"] and row["inference_seconds"])
                   or row["fit_inference_loop_elapsed_seconds"]
                   for row in rows))
    timing_mismatch = []
    for row in rows:
        seed = int(row["seed"])
        if seed == 42:
            if row["timing_provenance"] != "exact_frozen_phase_timing":
                timing_mismatch.append(row["model"] + "/42")
        else:
            expected = receipt["completion_log"][row["model"]][str(seed)][
                "fit_inference_loop_elapsed_seconds"]
            if (row["timing_provenance"] !=
                    "completion_log_upper_bound_after_timing_split_lost"
                    or not close(row["fit_inference_loop_elapsed_seconds"],
                                 expected, 1e-9)
                    or row["training_seconds"] or row["inference_seconds"]):
                timing_mismatch.append("%s/%d" % (row["model"], seed))
    report.add("Timing limitation and completion-log bounds are preserved",
               not timing_mismatch
               and receipt["reporting_failure"]["model_fit_failed"] is False
               and receipt["reporting_failure"][
                   "prediction_array_lost"] is False,
               "mismatches: %s" % timing_mismatch)
    mismatch = []
    for row in rows:
        name, seed = row["model"], int(row["seed"])
        path = (PRED / (name + "_validation.npy") if seed == 42 else
                OUT / ("%s_seed%d_validation.npy" % (name, seed)))
        if sha256_of_file(path) != row["prediction_sha256"]:
            mismatch.append("%s/%d hash" % (name, seed))
            continue
        prediction = np.load(str(path)).astype(np.float64)
        values = metric_values(actual, prediction, station, horizon)
        for field, value in values.items():
            if field == "n_negative":
                ok = int(float(row[field])) == int(value)
            else:
                ok = close(row[field], value)
            if not ok:
                mismatch.append("%s/%d %s" % (name, seed, field))
    report.add("All multi-seed metrics independently reproduce", not mismatch,
               "mismatches: %s" % mismatch[:8])

    horizon_rows = read_csv(OUT / "multiseed_metrics_by_horizon.csv")
    mismatch = []
    for row in horizon_rows:
        name, seed, hz = row["model"], int(row["seed"]), int(row["horizon"])
        path = (PRED / (name + "_validation.npy") if seed == 42 else
                OUT / ("%s_seed%d_validation.npy" % (name, seed)))
        prediction = np.load(str(path)).astype(np.float64)
        chosen = horizon == hz
        value = M.macro_station_mae(actual[chosen], prediction[chosen],
                                    station[chosen])
        if not close(row["macro_station_MAE"], value):
            mismatch.append("%s/%d/h%d" % (name, seed, hz))
    report.add("All horizon metrics independently reproduce", not mismatch,
               "mismatches: %s" % mismatch[:8])


def deterministic_checks(report, actual, station, horizon):
    rows = read_csv(OUT / "phase7_deterministic_metrics.csv")
    report.add("B0 and B3_R2 are recorded exactly once",
               [row["model"] for row in rows] == ["B0", "B3_R2"]
               and all(int(row["fit_count"]) == 1 for row in rows))
    mismatch = []
    for row in rows:
        prediction = np.load(str(PRED / (row["model"]
                                         + "_validation.npy"))).astype(
                                             np.float64)
        values = metric_values(actual, prediction, station, horizon)
        for field, value in values.items():
            if field == "n_negative":
                ok = int(float(row[field])) == int(value)
            else:
                ok = close(row[field], value)
            if not ok:
                mismatch.append(row["model"] + ":" + field)
    report.add("Deterministic metrics independently reproduce", not mismatch,
               "mismatches: %s" % mismatch)


def loso_checks(report, actual, station, horizon):
    rows = read_csv(OUT / "leave_one_station_out.csv")
    report.add("LOSO covers each frozen station exactly once",
               [row["held_out_station"] for row in rows] == list(STATIONS))
    mismatch = []
    normal = np.load(str(PRED / "SA_R3_validation.npy")).astype(np.float64)
    for row in rows:
        name = row["held_out_station"]
        path = OUT / ("SA_R3_loso_%s_validation.npy" % name)
        if sha256_of_file(path) != row["prediction_sha256"]:
            mismatch.append(name + ":hash")
            continue
        prediction = np.load(str(path)).astype(np.float64)
        chosen = station == name
        cells = M.cell_metrics(actual[chosen], prediction[chosen],
                               station[chosen], horizon[chosen])
        macro = M.macro_from_cells(cells)
        base_cells = M.cell_metrics(actual[chosen], normal[chosen],
                                    station[chosen], horizon[chosen])
        base_macro = M.macro_from_cells(base_cells)
        severe = chosen & (actual > M.SEVERE_THRESHOLD)
        severe_mae = M.mae(actual[severe], prediction[severe])
        if not close(row["loso_macro_over_horizon_MAE"], macro):
            mismatch.append(name + ":mae")
        if not close(row["in_training_macro_over_horizon_MAE"], base_macro):
            mismatch.append(name + ":base")
        if not close(row["loso_severe_MAE"], severe_mae):
            mismatch.append(name + ":severe")
    report.add("LOSO metrics and hashes independently reproduce", not mismatch,
               "mismatches: %s" % mismatch[:8])


def spatial_checks(report):
    coordinates_payload = load_json(ARTIFACTS /
                                    "phase7_station_coordinates.json")
    coordinates = {row["station"]: row
                   for row in coordinates_payload["stations"]}
    report.add("External coordinates exactly map all frozen stations",
               list(coordinates) == list(STATIONS)
               and all(coordinates[name]["source_station"] == name
                       for name in STATIONS))
    report.add("Coordinates remain analysis-only",
               coordinates_payload["model_input"] is False
               and coordinates_payload["graph_constructed"] is False)
    matrix = read_csv(OUT / "station_distance_matrix_km.csv")
    mismatch = []
    for row in matrix:
        target = row["target_station"]
        for source in STATIONS:
            if not close(row[source], haversine(coordinates[target],
                                                coordinates[source]), 1e-6):
                mismatch.append(target + "/" + source)
    report.add("Distance matrix independently reproduces", not mismatch,
               "mismatches: %s" % mismatch[:5])

    pair_rows = read_csv(OUT / "spatial_attention_pairs.csv")
    distance = np.asarray([float(row["distance_km"]) for row in pair_rows])
    attention = np.asarray([float(row["mean_attention_weight"])
                            for row in pair_rows])
    correlation = np.asarray([float(row["training_pm25_pearson_r"])
                              for row in pair_rows])
    summary = load_json(ARTIFACTS / "phase7_spatial_analysis.json")
    report.add("Spatial Spearman statistics independently reproduce",
               close(summary["spearman_distance_vs_attention"],
                     spearman(distance, attention), 1e-6)
               and close(summary[
                   "spearman_distance_vs_training_pm25_correlation"],
                   spearman(distance, correlation), 1e-6)
               and close(summary[
                   "spearman_attention_vs_training_pm25_correlation"],
                   spearman(attention, correlation), 1e-6))


def freeze_checks(report):
    freeze = load_json(ARTIFACTS / "phase7_pretest_freeze.json")
    report.add("Final pre-test artifact preserves the candidate freeze",
               freeze["candidate_freeze_sha256"] == EXPECTED_FREEZE)
    ranking = freeze["selection_ranking"]
    selected = min(ranking,
                   key=lambda row: row["primary_macro_MAE_mean"])["model"]
    report.add("Final selection follows the predeclared primary rule",
               freeze["selected_final_candidates"] == [selected]
               and selected in freeze["selected_configuration_and_artifacts"])
    drift = [path for path, digest in freeze["evidence_sha256"].items()
             if not (PROJECT_ROOT / path).is_file()
             or sha256_of_file(PROJECT_ROOT / path) != digest]
    report.add("All frozen Phase-7 evidence hashes match", not drift,
               "drifted: %s" % drift[:5])
    determinism = freeze["determinism_evidence"]
    report.add("Selected neural seed-42 reruns remain byte-identical",
               determinism[
                   "seed42_GRU_R1_prediction_rerun_byte_identical"] is True
               and determinism[
                   "seed42_TCN_R1_prediction_rerun_byte_identical"] is True
               and determinism[
                   "seed42_SA_R3_prediction_rerun_byte_identical"] is True
               and sha256_of_file(PRED / "GRU_R1_validation.npy")
               == sha256_of_file(PROJECT_ROOT / "results/models/neural/"
                                 "determinism/GRU_R1_validation.npy")
               and sha256_of_file(PRED / "TCN_R1_validation.npy")
               == sha256_of_file(PROJECT_ROOT / "results/models/neural/"
                                 "determinism/TCN_R1_validation.npy")
               and sha256_of_file(PRED / "SA_R3_validation.npy")
               == sha256_of_file(PROJECT_ROOT /
                                 "results/models/spatiotemporal/determinism/"
                                 "SA_R3_validation.npy"))
    report.add("Forbidden interventions remain absent",
               freeze["architecture_search_performed"] is False
               and freeze["features_changed"] is False
               and freeze["context_changed"] is False
               and freeze["severe_balancing"] is False
               and freeze["target_transformation"] is False
               and freeze["predictions_clipped"] is False)
    report.add("Final test remains sealed and untouched",
               freeze["final_test_status"] == "sealed"
               and freeze["test_data_accessed"] is False
               and freeze["test_targets_accessed"] is False
               and freeze["test_predictions_generated"] == 0
               and freeze["test_metrics_seen"] == 0
               and freeze["full_development_refit_performed"] is False)


def main():
    report = Report()
    candidate = load_json(ARTIFACTS / "phase7_candidate_freeze.json")
    upstream_checks(report, candidate)
    candidate_checks(report, candidate)
    actual, station, horizon = arrays()
    multiseed_checks(report, actual, station, horizon)
    deterministic_checks(report, actual, station, horizon)
    loso_checks(report, actual, station, horizon)
    spatial_checks(report)
    freeze_checks(report)
    report.render()
    if report.failed():
        print("\nPHASE 7 VALIDATION FAILED: %d gate(s)" % len(report.failed()))
        return 1
    print("\nPHASE 7 VALIDATION PASSED")
    print("Final test: SEALED; test predictions: 0; test metrics: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
