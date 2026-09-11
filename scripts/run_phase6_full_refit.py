"""Phase-6 full refit, external development validation and attention export.

Protocol Phase 6, sections 34-37, 45-46 and 52.

Order is the protocol's:

1. read the training-only internal selection and write the architecture
   selection freeze;
2. refit SA_R2 and SA_R3 from a clean seeded initialisation on the complete
   frozen Phase-2 training universe - never from candidate weights;
3. only then write the external-validation receipt and unseal validation
   history through the Phase-3 guarded rolling series;
4. predict every grouped validation origin and scatter the result back into
   the frozen canonical sample order - exactly 413,148 predictions per model;
5. export descriptive station-attention diagnostics for SA_R3.

``--determinism`` repeats steps 2 and 4 into a scratch location, compares byte
for byte, and reports the maximum prediction deviation if anything differs.

Usage:
    .venv-v2/bin/python scripts/run_phase6_full_refit.py
    .venv-v2/bin/python scripts/run_phase6_full_refit.py --determinism
"""

import argparse
import csv
import hashlib
import io
import json
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS, STATIONS         # noqa
from src.evaluation import metrics as M                               # noqa
from src.models import station_attention_grid as grid                 # noqa
from src.models.neural_features import static_channel_names           # noqa
from src.models.neural_training import configure_determinism          # noqa
from src.models.spatial_grouping import (                             # noqa
    GuardedNetworkHistory, NetworkBundle, build_groups)
from src.models.spatial_training import (                             # noqa
    attention_matrices, predict, train)
from src.models.station_attention_forecaster import (                 # noqa
    StationAttentionForecaster)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
OUT = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
CURVES = OUT / "training_curves"
DETERMINISM = OUT / "determinism"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
ARTIFACTS = PROJECT_ROOT / "artifacts"
SELECTION = OUT / "internal_candidate_selection.csv"
FREEZE = ARTIFACTS / "phase6_architecture_selection_freeze.json"
RECEIPT = ARTIFACTS / "phase6_external_validation_receipt.json"
BENCHMARK = ARTIFACTS / "phase6_runtime_benchmark.json"

START = time.time()
EXPECTED_VALIDATION_SAMPLES = 413148
EXPECTED_TRAINING_SAMPLES = 821184
MODES = OrderedDict([("SA_R2", False), ("SA_R3", True)])


def log(message):
    print("[%8.1fs] %s" % (time.time() - START, message), flush=True)


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def read_selection():
    rows = [r for r in csv.DictReader(open(str(SELECTION), encoding="utf-8"))
            if r["selected"] == "true"]
    if len(rows) != 1:
        raise SystemExit("internal selection marks %d winners" % len(rows))
    return rows[0]


def build(hidden_size, channels, statics, cross_station):
    return StationAttentionForecaster(
        channels, statics, hidden_size=hidden_size,
        heads=grid.FIXED["attention_heads"],
        attention_dropout=grid.FIXED["attention_dropout"],
        head_hidden=grid.FIXED["head_hidden"],
        head_dropout=grid.FIXED["head_dropout"],
        cross_station=cross_station)


def write_freeze(selected, runtime, batch_size, parameter_count):
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase6_architecture_selection_freeze"),
        ("architecture", "AirSense Station-Attention Forecaster"),
        ("implementation", "src/models/station_attention_forecaster.py"),
        ("not_an_implementation_of", ["Graph WaveNet", "DCRNN", "STGCN",
                                      "GAT", "ASTGCN"]),
        ("selected", OrderedDict([
            ("candidate_id", selected["candidate_id"]),
            ("hidden_size", int(selected["hidden_size"])),
            ("learning_rate", float(selected["learning_rate"])),
            ("mean_internal_macro_station_horizon_mae",
             float(selected["mean_macro_station_horizon_MAE"])),
            ("regimes_scored", int(selected["regimes_scored"])),
            ("parameter_count", parameter_count),
        ])),
        ("selection_rule", grid.SELECTION_CRITERION),
        ("tie_break_order", grid.TIE_BREAK_ORDER),
        ("comparison_tolerance", grid.COMPARISON_TOLERANCE),
        ("selected_from", "training-only internal core/tuning split"),
        ("internal_split_reused_from_phase4", True),
        ("external_validation_used_for_selection", False),
        ("same_configuration_used_for_both_modes", True),
        ("regimes", list(grid.REGIMES)),
        ("regime_meaning", OrderedDict(grid.REGIME_MEANING)),
        ("attention_masks", OrderedDict([
            ("SA_R2", "strict self-only; every off-diagonal edge forbidden"),
            ("SA_R3", "unmasked; all 12 stations reachable, no distance "
                      "mask, no chosen neighbour, no validation-derived edge"),
        ])),
        ("grouped_batch_size", batch_size),
        ("runtime_benchmark_sha256", sha256_of_file(BENCHMARK)),
        ("candidate_metrics_sha256",
         sha256_of_file(OUT / "internal_candidate_metrics.csv")),
        ("candidate_selection_sha256", sha256_of_file(SELECTION)),
        ("internal_split_sha256",
         sha256_of_file(ARTIFACTS / "phase4_internal_split.json")),
        ("dynamic_schema_sha256",
         sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
        ("static_schema_channels", static_channel_names()),
        ("model_source_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models"
            / "station_attention_forecaster.py")),
        ("grouping_source_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models" / "spatial_grouping.py")),
        ("grid_source_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models" / "station_attention_grid.py")),
        ("training_protocol", OrderedDict(grid.FIXED)),
        ("runtime", runtime),
        ("seed", 42),
        ("external_coordinates_used", False),
        ("external_phase6_validation_metrics_seen", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("frozen_after_this_point", [
            "hidden size", "learning rate", "attention heads", "epochs",
            "grouped batch size", "loss", "optimizer", "attention masks"]),
    ])
    write_json(FREEZE, payload)
    log("architecture-selection freeze %s" % sha256_of_file(FREEZE)[:16])


def write_receipt(groups):
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("event", "phase6_external_development_validation_opening"),
        ("phase3_validation_opening_receipt_sha256",
         sha256_of_file(ARTIFACTS
                        / "validation_development_opening_receipt.json")),
        ("phase6_architecture_selection_freeze_sha256",
         sha256_of_file(FREEZE)),
        ("sample_universe_manifest_sha256",
         sha256_of_file(ARTIFACTS / "sample_universe_manifest.json")),
        ("validation_sample_index_sha256",
         sha256_of_file(PROCESSED / "sample_index" / "validation.csv")),
        ("external_validation_interval", OrderedDict([
            ("start", PARTITION_BOUNDS["validation"][0].isoformat()),
            ("end", PARTITION_BOUNDS["validation"][1].isoformat()),
        ])),
        ("validation_groups", groups.size),
        ("expected_validation_predictions_per_model",
         EXPECTED_VALIDATION_SAMPLES),
        ("context_hours", 48),
        ("severe_threshold", 244.0),
        ("both_modes_selected_using_training_only", True),
        ("external_coordinates_used", False),
        ("validation_derived_graph_built", False),
        ("external_phase6_metrics_seen", 0),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("note", "Records the first station-attention access to external "
                 "development validation. It deliberately contains no "
                 "performance value."),
    ])
    write_json(RECEIPT, payload)
    log("external-validation receipt %s" % sha256_of_file(RECEIPT)[:16])


def export_attention(model, bundle, groups, batch_size, actual):
    """Descriptive diagnostics only - never a causal influence estimate."""
    overall, counted = attention_matrices(model, bundle,
                                          batch_size=batch_size)
    write_csv(OUT / "attention_matrix_overall.csv",
              ["target_station"] + list(STATIONS),
              [[STATIONS[i]] + [round(float(v), 8) for v in overall[i]]
               for i in range(12)])
    log("  attention_matrix_overall over %d groups" % counted)

    rows = []
    for horizon in (1, 6, 12, 24):
        selector = groups.horizons == horizon
        matrix, n = attention_matrices(model, bundle, batch_size=batch_size,
                                       selector=selector)
        for i in range(12):
            rows.append([horizon, STATIONS[i], n]
                        + [round(float(v), 8) for v in matrix[i]])
    write_csv(OUT / "attention_matrix_by_horizon.csv",
              ["horizon", "target_station", "groups"] + list(STATIONS), rows)

    severe_slot = np.zeros((groups.size, 12), dtype=bool)
    observed = groups.target_observed
    severe_flat = actual > M.SEVERE_THRESHOLD
    severe_slot[observed] = severe_flat[groups.sample_rows[observed]]
    selector = severe_slot.any(axis=1)
    matrix, n = attention_matrices(model, bundle, batch_size=batch_size,
                                   selector=selector)
    write_csv(OUT / "attention_matrix_severe.csv",
              ["target_station", "groups"] + list(STATIONS),
              [[STATIONS[i], n] + [round(float(v), 8) for v in matrix[i]]
               for i in range(12)])
    log("  attention_matrix_severe over %d groups with a severe target" % n)
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism", action="store_true")
    args = parser.parse_args()

    runtime = configure_determinism()
    log("runtime %s" % json.dumps(runtime))
    with open(str(BENCHMARK), encoding="utf-8") as handle:
        batch_size = json.load(handle)["frozen_grouped_batch_size"]
    selected = read_selection()
    hidden = int(selected["hidden_size"])
    learning_rate = float(selected["learning_rate"])
    log("selected %s (hidden %d, lr %s), grouped batch %d"
        % (selected["candidate_id"], hidden, learning_rate, batch_size))

    history = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=False)
    channels = history.dynamic_channel_count
    statics = len(static_channel_names())
    groups = build_groups(PROCESSED, "train",
                          labels_from=history.train_target_native)
    if groups.sample_total() != EXPECTED_TRAINING_SAMPLES:
        raise SystemExit("training universe %d" % groups.sample_total())
    bundle = NetworkBundle(groups, history)
    log("full training universe: %d groups, %d canonical samples; "
        "validation still sealed" % (groups.size, groups.sample_total()))

    if not args.determinism:
        probe = build(hidden, channels, statics, True)
        write_freeze(selected, runtime, batch_size, probe.parameter_count())
        del probe
    elif not FREEZE.is_file():
        raise SystemExit("determinism run requires the selection freeze")

    models = OrderedDict()
    diagnostics = []
    curves = {}
    for name, cross_station in MODES.items():
        configure_determinism()
        model = build(hidden, channels, statics, cross_station)
        curve, seconds = train(model, bundle, learning_rate,
                               epochs=grid.FIXED["epochs"],
                               batch_size=batch_size)
        models[name] = model
        curves[name] = [round(v, 6) for v in curve]
        diagnostics.append([name, "R2" if not cross_station else "R3", hidden,
                            learning_rate, model.parameter_count(),
                            round(seconds, 1), round(curve[0], 6),
                            round(curve[-1], 6)])
        log("  refit %s: %d params, final train L1 %.4f (%.0fs)"
            % (name, model.parameter_count(), curve[-1], seconds))
    counts = {m.parameter_count() for m in models.values()}
    if len(counts) != 1:
        raise SystemExit("paired design broken: parameter counts %s" % counts)
    log("paired design intact: both modes hold %d parameters" % counts.pop())

    v_groups = build_groups(PROCESSED, "validation")
    if not args.determinism:
        write_receipt(v_groups)

    unsealed = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=True)
    v_bundle = NetworkBundle(v_groups, unsealed)
    log("external validation unsealed through the guarded rolling series")

    destination = DETERMINISM if args.determinism else PRED
    destination.mkdir(parents=True, exist_ok=True)
    model_rows = []
    digests = OrderedDict()
    for name, model in models.items():
        grouped, inference_seconds = predict(model, v_bundle,
                                             batch_size=batch_size)
        flat = v_groups.scatter(grouped, EXPECTED_VALIDATION_SAMPLES)
        if flat.size != EXPECTED_VALIDATION_SAMPLES:
            raise SystemExit("%s produced %d predictions" % (name, flat.size))
        path = destination / ("%s_validation.npy" % name)
        np.save(str(path), flat.astype(np.float32))
        weight_path = ((DETERMINISM if args.determinism else OUT)
                       / ("%s.safetensors" % name))
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous()
                   for k, v in model.state_dict().items()}, str(weight_path))
        digests[name] = (sha256_of_file(path), sha256_of_file(weight_path))
        if not args.determinism:
            config_path = OUT / ("%s_config.json" % name)
            write_json(config_path, OrderedDict([
                ("model", name),
                ("architecture", "AirSense Station-Attention Forecaster"),
                ("regime", "R2" if name == "SA_R2" else "R3"),
                ("station_attention",
                 grid.REGIME_MEANING["R2" if name == "SA_R2" else "R3"]),
                ("hidden_size", hidden),
                ("learning_rate", learning_rate),
                ("candidate_id", selected["candidate_id"]),
                ("stations", len(STATIONS)),
                ("station_order", list(STATIONS)),
                ("dynamic_channels", channels),
                ("static_channels", statics),
                ("context_hours", 48),
                ("grouped_batch_size", batch_size),
                ("parameter_count", model.parameter_count()),
                ("external_coordinates_used", False),
                ("validation_derived_graph", False),
                ("pretrained_weights_used", False),
                ("training", OrderedDict(grid.FIXED)),
                ("trained_on", "complete frozen Phase-2 training universe"),
                ("initialised_from", "clean seeded initialisation (seed 42)"),
                ("output_constrained", False),
            ]))
            model_rows.append([
                name, "R2" if name == "SA_R2" else "R3",
                model.parameter_count(),
                str(config_path.relative_to(PROJECT_ROOT)),
                sha256_of_file(config_path),
                str(weight_path.relative_to(PROJECT_ROOT)), digests[name][1],
                weight_path.stat().st_size,
                str(path.relative_to(PROJECT_ROOT)), digests[name][0],
                "float32", round(inference_seconds, 1),
                round(flat.size / inference_seconds, 1)])
        log("  %s predicted %d canonical validation samples (%.0fs)"
            % (name, flat.size, inference_seconds))

    if args.determinism:
        problems = []
        for name, (prediction_digest, weight_digest) in digests.items():
            primary_prediction = PRED / ("%s_validation.npy" % name)
            primary_weight = OUT / ("%s.safetensors" % name)
            if sha256_of_file(primary_prediction) != prediction_digest:
                deviation = float(np.abs(
                    np.load(str(primary_prediction)).astype(np.float64)
                    - np.load(str(DETERMINISM
                                  / ("%s_validation.npy" % name)))
                    .astype(np.float64)).max())
                problems.append("%s: predictions differ, max deviation %.10f"
                                % (name, deviation))
            if sha256_of_file(primary_weight) != weight_digest:
                problems.append("%s: weight file differs" % name)
        for problem in problems:
            log("  DETERMINISM FAILURE %s" % problem)
        if problems:
            return 1
        log("determinism: every weight file and prediction array is "
            "byte-identical to the primary run")
        return 0

    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    export_attention(models["SA_R3"], v_bundle, v_groups, batch_size, actual)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_csv(OUT / "spatiotemporal_full_models.csv",
              ["model", "regime", "parameter_count", "config_file",
               "config_sha256", "weight_file", "weight_sha256",
               "weight_bytes", "prediction_file", "prediction_sha256",
               "prediction_dtype", "inference_seconds",
               "predictions_per_second"], model_rows)
    write_csv(OUT / "spatiotemporal_training_diagnostics.csv",
              ["model", "regime", "hidden_size", "learning_rate",
               "parameter_count", "training_seconds", "first_epoch_train_L1",
               "final_epoch_train_L1"], diagnostics)
    write_json(OUT / "compute_report.json", OrderedDict([
        ("grouped_batch_size", batch_size),
        ("hidden_size", hidden),
        ("parameter_count", diagnostics[0][4]),
        ("training_seconds", OrderedDict(
            (row[0], row[5]) for row in diagnostics)),
        ("inference_seconds", OrderedDict(
            (row[0], row[11]) for row in model_rows)),
        ("predictions_per_second", OrderedDict(
            (row[0], row[12]) for row in model_rows)),
        ("serialized_model_bytes", OrderedDict(
            (row[0], row[7]) for row in model_rows)),
        ("peak_rss_bytes", peak),
        ("peak_rss_gb", round(peak / 1e9, 3)),
        ("device", "cpu"),
        ("note", "CPU-only measurements on one host; no hardware-independent "
                 "efficiency claim is made."),
    ]))
    CURVES.mkdir(parents=True, exist_ok=True)
    with open(str(CURVES / "full_refit_curves.json"), "w",
              encoding="utf-8") as handle:
        json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
        handle.write("\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
