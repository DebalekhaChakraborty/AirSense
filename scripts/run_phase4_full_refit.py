"""Phase-4 full refit and external development validation.

Protocol Phase 4, Stage B.

Order is the protocol's:

1. read the internal selection and write the architecture-selection freeze;
2. refit the six models from a clean seeded initialisation on the **complete**
   frozen Phase-2 training universe - never from candidate weights;
3. only then write the external-validation receipt and unseal validation
   history through the Phase-3 guarded rolling interface;
4. predict all 413,148 canonical validation samples per model.

``--determinism`` repeats step 2 and 3-4 into a scratch location and compares,
without overwriting the primary artifacts.

Usage:
    .venv-v2/bin/python scripts/run_phase4_full_refit.py
    .venv-v2/bin/python scripts/run_phase4_full_refit.py --determinism
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS               # noqa: E402
from src.models import neural_grid                                # noqa: E402
from src.models.gru_forecaster import GRUForecaster               # noqa: E402
from src.models.neural_data import (                              # noqa: E402
    NeuralDataSource, load_sample_index)
from src.models.neural_features import (                          # noqa: E402
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import (                          # noqa: E402
    configure_determinism, predict, train)
from src.models.tcn_forecaster import TCNForecaster               # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
OUT = PROJECT_ROOT / "results" / "models" / "neural"
CURVES = OUT / "training_curves"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
ARTIFACTS = PROJECT_ROOT / "artifacts"
SELECTION = OUT / "internal_candidate_selection.csv"
FREEZE = ARTIFACTS / "phase4_architecture_selection_freeze.json"
RECEIPT = ARTIFACTS / "phase4_external_validation_receipt.json"

START = time.time()
EXPECTED_VALIDATION_SAMPLES = 413148


def log(message):
    print("[%7.1fs] %s" % (time.time() - START, message), flush=True)


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
    selected = {}
    with open(str(SELECTION), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["selected"] == "true":
                selected[row["architecture"]] = row
    if sorted(selected) != ["GRU", "TCN"]:
        raise SystemExit("internal selection incomplete")
    return selected


def build_model(architecture, capacity, dynamic, static):
    candidate = neural_grid.candidates(architecture)[0]
    if architecture == "GRU":
        return GRUForecaster(dynamic, static, hidden_size=capacity,
                             head_hidden=candidate["head_hidden"],
                             dropout=candidate["dropout"],
                             num_layers=candidate["num_layers"])
    return TCNForecaster(dynamic, static, channels=capacity,
                         head_hidden=candidate["head_hidden"],
                         dropout=candidate["dropout"],
                         kernel_size=candidate["kernel_size"],
                         dilations=candidate["dilations"])


def write_freeze(selected, runtime):
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase4_architecture_selection_freeze"),
        ("selected", OrderedDict(
            (architecture, OrderedDict([
                ("candidate_id", row["candidate_id"]),
                ("capacity", int(row["capacity"])),
                ("learning_rate", float(row["learning_rate"])),
                ("mean_internal_macro_station_horizon_mae",
                 float(row["mean_macro_station_horizon_MAE"])),
                ("regimes_scored", int(row["regimes_scored"])),
            ])) for architecture, row in sorted(selected.items()))),
        ("selection_rule", neural_grid.SELECTION_CRITERION),
        ("tie_break_order", neural_grid.TIE_BREAK_ORDER),
        ("comparison_tolerance", neural_grid.COMPARISON_TOLERANCE),
        ("selected_from", "training-only internal core/tuning split"),
        ("external_validation_used_for_selection", False),
        ("same_configuration_used_for_all_regimes", True),
        ("candidate_metrics_sha256",
         sha256_of_file(OUT / "internal_candidate_metrics.csv")),
        ("candidate_selection_sha256", sha256_of_file(SELECTION)),
        ("internal_split_sha256",
         sha256_of_file(ARTIFACTS / "phase4_internal_split.json")),
        ("neural_feature_schema_sha256",
         sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
        ("training_protocol", OrderedDict(neural_grid.FIXED)),
        ("runtime", runtime),
        ("seed", 42),
        ("external_neural_validation_metrics_seen", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("frozen_after_this_point", [
            "architecture capacity", "learning rate", "epochs", "batch size",
            "loss", "optimizer"]),
    ])
    write_json(FREEZE, payload)
    log("architecture-selection freeze %s" % sha256_of_file(FREEZE)[:16])


def write_receipt():
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("event", "phase4_external_development_validation_opening"),
        ("phase3_validation_opening_receipt_sha256",
         sha256_of_file(ARTIFACTS
                        / "validation_development_opening_receipt.json")),
        ("phase4_architecture_selection_freeze_sha256",
         sha256_of_file(FREEZE)),
        ("external_validation_interval", OrderedDict([
            ("start", PARTITION_BOUNDS["validation"][0].isoformat()),
            ("end", PARTITION_BOUNDS["validation"][1].isoformat()),
        ])),
        ("six_neural_models_fully_selected", True),
        ("hyperparameters_selected_from_training_only_internal_split", True),
        ("external_neural_validation_metrics_seen", 0),
        ("test_status", "sealed"),
        ("note", "Records the first neural access to external development "
                 "validation. It deliberately contains no performance "
                 "value."),
    ])
    write_json(RECEIPT, payload)
    log("external-validation receipt %s" % sha256_of_file(RECEIPT)[:16])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism", action="store_true")
    args = parser.parse_args()

    runtime = configure_determinism()
    selected = read_selection()
    log("selected GRU %s (capacity %s, lr %s) | TCN %s (capacity %s, lr %s)"
        % (selected["GRU"]["candidate_id"], selected["GRU"]["capacity"],
           selected["GRU"]["learning_rate"], selected["TCN"]["candidate_id"],
           selected["TCN"]["capacity"], selected["TCN"]["learning_rate"]))

    if not args.determinism:
        write_freeze(selected, runtime)
    elif not FREEZE.is_file():
        raise SystemExit("determinism run requires the selection freeze")

    stations, horizons, targets = load_sample_index(PROCESSED, "train")
    sealed_source = NeuralDataSource(PROJECT_ROOT, unseal_validation=False)
    log("full training universe: %d samples" % targets.size)

    models = OrderedDict()
    diagnostics = []
    curves = {}
    for architecture in ("GRU", "TCN"):
        capacity = int(selected[architecture]["capacity"])
        learning_rate = float(selected[architecture]["learning_rate"])
        for regime in neural_grid.REGIMES:
            name = "%s_%s" % (architecture, regime)
            bundle = sealed_source.bundle(regime, stations, horizons, targets)
            configure_determinism()
            model = build_model(architecture, capacity,
                                len(dynamic_channel_names(regime)),
                                len(static_channel_names()))
            curve, seconds = train(model, bundle, learning_rate)
            models[name] = model
            curves[name] = [round(v, 6) for v in curve]
            diagnostics.append([name, architecture, regime, capacity,
                                learning_rate, model.parameter_count(),
                                round(seconds, 1), round(curve[0], 6),
                                round(curve[-1], 6)])
            log("  refit %s: %d params, final train L1 %.4f (%.0fs)"
                % (name, model.parameter_count(), curve[-1], seconds))
            del bundle

    if not args.determinism:
        write_receipt()

    unsealed = NeuralDataSource(PROJECT_ROOT, unseal_validation=True)
    v_station, v_horizon, v_target = load_sample_index(PROCESSED, "validation")
    if v_target.size != EXPECTED_VALIDATION_SAMPLES:
        raise SystemExit("validation index size %d" % v_target.size)
    log("external validation unsealed through the guarded rolling interface")

    destination = (PRED if not args.determinism
                   else PROJECT_ROOT / "results" / "models" / "neural"
                   / "determinism")
    destination.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for name, model in models.items():
        regime = name.split("_")[1]
        bundle = unsealed.bundle(regime, v_station, v_horizon, v_target,
                                 pm25_source="rolling", with_labels=False)
        prediction, inference_seconds = predict(model, bundle)
        if prediction.size != EXPECTED_VALIDATION_SAMPLES:
            raise SystemExit("%s produced %d predictions" % (name,
                                                             prediction.size))
        path = destination / ("%s_validation.npy" % name)
        np.save(str(path), prediction.astype(np.float32))
        if not args.determinism:
            config_path = OUT / ("%s_config.json" % name)
            weight_path = OUT / ("%s.safetensors" % name)
            write_json(config_path, OrderedDict([
                ("model", name),
                ("architecture", name.split("_")[0]),
                ("regime", regime),
                ("capacity", int(selected[name.split("_")[0]]["capacity"])),
                ("learning_rate",
                 float(selected[name.split("_")[0]]["learning_rate"])),
                ("dynamic_channels", len(dynamic_channel_names(regime))),
                ("static_channels", len(static_channel_names())),
                ("context_hours", 48),
                ("parameter_count", model.parameter_count()),
                ("training", OrderedDict(neural_grid.FIXED)),
                ("trained_on", "complete frozen Phase-2 training universe"),
                ("initialised_from", "clean seeded initialisation (seed 42)"),
                ("output_constrained", False),
            ]))
            save_file({k: v.contiguous()
                       for k, v in model.state_dict().items()},
                      str(weight_path))
            model_rows.append([
                name, name.split("_")[0], regime, model.parameter_count(),
                str(config_path.relative_to(PROJECT_ROOT)),
                sha256_of_file(config_path),
                str(weight_path.relative_to(PROJECT_ROOT)),
                sha256_of_file(weight_path), weight_path.stat().st_size,
                str(path.relative_to(PROJECT_ROOT)), sha256_of_file(path),
                round(inference_seconds, 1),
                round(prediction.size / inference_seconds, 1)])
        log("  %s predicted %d validation samples (%.0fs)"
            % (name, prediction.size, inference_seconds))
        del bundle

    if not args.determinism:
        write_csv(OUT / "neural_full_models.csv",
                  ["model", "architecture", "regime", "parameter_count",
                   "config_file", "config_sha256", "weight_file",
                   "weight_sha256", "weight_bytes", "prediction_file",
                   "prediction_sha256", "inference_seconds",
                   "predictions_per_second"], model_rows)
        write_csv(OUT / "neural_training_diagnostics.csv",
                  ["model", "architecture", "regime", "capacity",
                   "learning_rate", "parameter_count", "training_seconds",
                   "first_epoch_train_L1", "final_epoch_train_L1"],
                  diagnostics)
        CURVES.mkdir(parents=True, exist_ok=True)
        with open(str(CURVES / "full_refit_curves.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
            handle.write("\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
