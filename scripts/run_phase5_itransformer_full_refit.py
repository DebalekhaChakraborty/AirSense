"""Phase-5 iTransformer full refit and external development validation.

Protocol Phase 5, Stage B.

Order is the protocol's, and it is the same order Phase 4 used:

1. read the training-only internal selection and write the selection freeze;
2. refit iTransformer_R1 and iTransformer_R2 from a clean seeded
   initialisation on the **complete** frozen Phase-2 training universe - never
   from candidate weights;
3. only then write the pre-validation freeze and unseal validation history
   through the Phase-3 guarded rolling interface;
4. predict all 413,148 canonical validation samples per model.

``--determinism`` repeats steps 2 and 4 into a scratch location, compares the
result byte for byte against the primary artifacts, and exits non-zero on any
difference. It never overwrites a primary artifact.

Chronos-2 is **not** run here and is not runnable from this script: the
pretraining-overlap audit returned classification C. See
``docs/CHRONOS2_PRETRAINING_AUDIT.md``.

Usage:
    .venv-v2/bin/python scripts/run_phase5_itransformer_full_refit.py
    .venv-v2/bin/python scripts/run_phase5_itransformer_full_refit.py \
        --determinism
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
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS               # noqa: E402
from src.models import itransformer_grid as grid                  # noqa: E402
from src.models.itransformer_forecaster import (                  # noqa: E402
    ITransformerForecaster)
from src.models.neural_data import (                              # noqa: E402
    NeuralDataSource, load_sample_index)
from src.models.neural_features import (                          # noqa: E402
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import (                          # noqa: E402
    configure_determinism, predict, train)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
OUT = PROJECT_ROOT / "results" / "models" / "itransformer"
CURVES = OUT / "training_curves"
DETERMINISM = OUT / "determinism"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
ARTIFACTS = PROJECT_ROOT / "artifacts"
SELECTION = OUT / "internal_candidate_selection.csv"
FREEZE = ARTIFACTS / "phase5_itransformer_selection_freeze.json"
PREVALIDATION = ARTIFACTS / "phase5_prevalidation_freeze.json"

START = time.time()
EXPECTED_VALIDATION_SAMPLES = 413148
EXPECTED_TRAINING_SAMPLES = 821184


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
    """The single configuration chosen on the training-only internal split."""
    rows = []
    with open(str(SELECTION), encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["selected"] == "true":
                rows.append(row)
    if len(rows) != 1:
        raise SystemExit("internal selection marks %d winners" % len(rows))
    return rows[0]


def build(regime, d_model):
    names = dynamic_channel_names(regime)
    fixed = grid.FIXED
    return ITransformerForecaster(
        len(names), len(static_channel_names()),
        target_index=names.index("value:PM2.5"), d_model=d_model,
        layers=fixed["encoder_layers"], heads=fixed["attention_heads"],
        ff_ratio=fixed["feedforward_ratio"], dropout=fixed["dropout"],
        head_hidden=fixed["head_hidden"])


def write_freeze(selected, runtime):
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase5_itransformer_selection_freeze"),
        ("architecture", "iTransformer"),
        ("implementation", "in-repository adaptation, "
                           "src/models/itransformer_forecaster.py"),
        ("selected", OrderedDict([
            ("candidate_id", selected["candidate_id"]),
            ("d_model", int(selected["d_model"])),
            ("learning_rate", float(selected["learning_rate"])),
            ("mean_internal_macro_station_horizon_mae",
             float(selected["mean_macro_station_horizon_MAE"])),
            ("regimes_scored", int(selected["regimes_scored"])),
        ])),
        ("selection_rule", grid.SELECTION_CRITERION),
        ("tie_break_order", grid.TIE_BREAK_ORDER),
        ("comparison_tolerance", grid.COMPARISON_TOLERANCE),
        ("selected_from", "training-only internal core/tuning split"),
        ("internal_split_reused_from_phase4", True),
        ("external_validation_used_for_selection", False),
        ("same_configuration_used_for_all_regimes", True),
        ("regimes", list(grid.REGIMES)),
        ("candidate_metrics_sha256",
         sha256_of_file(OUT / "internal_candidate_metrics.csv")),
        ("candidate_selection_sha256", sha256_of_file(SELECTION)),
        ("internal_split_sha256",
         sha256_of_file(ARTIFACTS / "phase4_internal_split.json")),
        ("neural_feature_schema_sha256",
         sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
        ("model_source_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models" / "itransformer_forecaster.py")),
        ("grid_source_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models" / "itransformer_grid.py")),
        ("training_protocol", OrderedDict(grid.FIXED)),
        ("runtime", runtime),
        ("seed", 42),
        ("external_validation_metrics_seen", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("frozen_after_this_point", [
            "d_model", "learning rate", "encoder layers", "attention heads",
            "epochs", "batch size", "loss", "optimizer"]),
    ])
    write_json(FREEZE, payload)
    log("selection freeze %s" % sha256_of_file(FREEZE)[:16])


def write_prevalidation(diagnostics):
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase5_prevalidation_freeze"),
        ("stage", "B_before_any_external_validation_metric"),
        ("event", "phase5_external_development_validation_opening"),
        ("models", OrderedDict(
            (row[0], OrderedDict([
                ("regime", row[1]),
                ("d_model", row[2]),
                ("learning_rate", row[3]),
                ("parameter_count", row[4]),
                ("dynamic_channels", len(dynamic_channel_names(row[1]))),
                ("static_channels", len(static_channel_names())),
                ("trained_on",
                 "complete frozen Phase-2 training universe"),
                ("initialised_from",
                 "clean seeded initialisation (seed 42)"),
            ])) for row in diagnostics)),
        ("training_samples", EXPECTED_TRAINING_SAMPLES),
        ("expected_validation_predictions_per_model",
         EXPECTED_VALIDATION_SAMPLES),
        ("pm25_history_interface",
         "src/data/rolling_history.py, Phase-3 guarded rolling series"),
        ("phase3_validation_opening_receipt_sha256",
         sha256_of_file(ARTIFACTS
                        / "validation_development_opening_receipt.json")),
        ("phase5_itransformer_selection_freeze_sha256",
         sha256_of_file(FREEZE)),
        ("upstream_manifest_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in
            ("v2_foundation_manifest.json", "v2_phase1_manifest.json",
             "v2_phase2_manifest.json", "v2_phase3_manifest.json",
             "v2_phase4_manifest.json",
             "verification_tooling_registry.json"))),
        ("external_validation_interval", OrderedDict([
            ("start", PARTITION_BOUNDS["validation"][0].isoformat()),
            ("end", PARTITION_BOUNDS["validation"][1].isoformat()),
        ])),
        ("regimes", list(grid.REGIMES)),
        ("horizons", [1, 6, 12, 24]),
        ("severe_threshold", 244.0),
        ("severe_threshold_source", "training_pooled_p95"),
        ("residual_convention", "residual = actual - prediction"),
        ("predictions_clipped", False),
        ("hyperparameters_selected_from_training_only_internal_split", True),
        ("foundation_model_executed", False),
        ("chronos2_executed", False),
        ("chronos2_gate", "pretraining-overlap audit classification C: "
                          "direct or material overlap with the sealed test "
                          "period; no checkpoint downloaded, no inference "
                          "run"),
        ("chronos2_audit_sha256", sha256_of_file(
            PROJECT_ROOT / "docs" / "CHRONOS2_PRETRAINING_AUDIT.md")),
        ("phase5_model_audit_sha256", sha256_of_file(
            PROJECT_ROOT / "docs" / "PHASE5_MODEL_AUDIT.md")),
        ("external_validation_metrics_seen", 0),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("note", "Records the first iTransformer access to external "
                 "development validation. It deliberately contains no "
                 "performance value."),
    ])
    write_json(PREVALIDATION, payload)
    log("pre-validation freeze %s" % sha256_of_file(PREVALIDATION)[:16])


def compare_determinism(prediction_digests, weight_digests):
    problems = []
    for name, digest in sorted(prediction_digests.items()):
        primary = PRED / ("%s_validation.npy" % name)
        if not primary.is_file():
            problems.append("%s: no primary prediction to compare" % name)
        elif sha256_of_file(primary) != digest:
            problems.append("%s: prediction digest differs" % name)
    for name, digest in sorted(weight_digests.items()):
        primary = OUT / ("%s.safetensors" % name)
        if not primary.is_file():
            problems.append("%s: no primary weight file to compare" % name)
        elif sha256_of_file(primary) != digest:
            problems.append("%s: weight digest differs" % name)
    for problem in problems:
        log("  DETERMINISM FAILURE %s" % problem)
    if problems:
        return 1
    log("determinism: every refit weight file and prediction array is "
        "byte-identical to the primary run")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism", action="store_true")
    args = parser.parse_args()

    runtime = configure_determinism()
    log("runtime %s" % json.dumps(runtime))
    selected = read_selection()
    d_model = int(selected["d_model"])
    learning_rate = float(selected["learning_rate"])
    log("selected %s (d_model %d, lr %s) from the training-only internal split"
        % (selected["candidate_id"], d_model, learning_rate))

    if not args.determinism:
        write_freeze(selected, runtime)
    elif not FREEZE.is_file():
        raise SystemExit("determinism run requires the selection freeze")

    stations, horizons, targets = load_sample_index(PROCESSED, "train")
    if targets.size != EXPECTED_TRAINING_SAMPLES:
        raise SystemExit("training index size %d" % targets.size)
    sealed_source = NeuralDataSource(PROJECT_ROOT, unseal_validation=False)
    log("full training universe: %d samples; validation still sealed"
        % targets.size)

    models = OrderedDict()
    diagnostics = []
    curves = {}
    for regime in grid.REGIMES:
        name = "iTransformer_%s" % regime
        bundle = sealed_source.bundle(regime, stations, horizons, targets)
        configure_determinism()
        model = build(regime, d_model)
        curve, seconds = train(model, bundle, learning_rate,
                               epochs=grid.FIXED["epochs"],
                               batch_size=grid.FIXED["batch_size"])
        models[name] = model
        curves[name] = [round(v, 6) for v in curve]
        diagnostics.append([name, regime, d_model, learning_rate,
                            model.parameter_count(), round(seconds, 1),
                            round(curve[0], 6), round(curve[-1], 6)])
        log("  refit %s: %d params, final train L1 %.4f (%.0fs)"
            % (name, model.parameter_count(), curve[-1], seconds))
        del bundle

    if not args.determinism:
        write_prevalidation(diagnostics)

    unsealed = NeuralDataSource(PROJECT_ROOT, unseal_validation=True)
    v_station, v_horizon, v_target = load_sample_index(PROCESSED,
                                                       "validation")
    if v_target.size != EXPECTED_VALIDATION_SAMPLES:
        raise SystemExit("validation index size %d" % v_target.size)
    log("external validation unsealed through the guarded rolling interface")

    destination = DETERMINISM if args.determinism else PRED
    destination.mkdir(parents=True, exist_ok=True)
    model_rows = []
    prediction_digests = OrderedDict()
    weight_digests = OrderedDict()
    for name, model in models.items():
        regime = name.split("_")[1]
        bundle = unsealed.bundle(regime, v_station, v_horizon, v_target,
                                 pm25_source="rolling", with_labels=False)
        prediction, inference_seconds = predict(model, bundle)
        if prediction.size != EXPECTED_VALIDATION_SAMPLES:
            raise SystemExit("%s produced %d predictions"
                             % (name, prediction.size))
        path = destination / ("%s_validation.npy" % name)
        np.save(str(path), prediction.astype(np.float32))
        prediction_digests[name] = sha256_of_file(path)
        weight_path = ((DETERMINISM if args.determinism else OUT)
                       / ("%s.safetensors" % name))
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  str(weight_path))
        weight_digests[name] = sha256_of_file(weight_path)
        if not args.determinism:
            config_path = OUT / ("%s_config.json" % name)
            write_json(config_path, OrderedDict([
                ("model", name),
                ("architecture", "iTransformer"),
                ("implementation",
                 "in-repository adaptation of the iTransformer principle; "
                 "not a byte-for-byte reproduction of thuml/iTransformer"),
                ("regime", regime),
                ("d_model", d_model),
                ("learning_rate", learning_rate),
                ("candidate_id", selected["candidate_id"]),
                ("dynamic_channels", len(dynamic_channel_names(regime))),
                ("variate_tokens", len(dynamic_channel_names(regime))),
                ("static_channels", len(static_channel_names())),
                ("attention_axis", "across variate tokens"),
                ("positional_encoding_across_variates", False),
                ("decoder", None),
                ("context_hours", 48),
                ("parameter_count", model.parameter_count()),
                ("pretrained_weights_used", False),
                ("training", OrderedDict(grid.FIXED)),
                ("trained_on", "complete frozen Phase-2 training universe"),
                ("initialised_from", "clean seeded initialisation (seed 42)"),
                ("output_constrained", False),
            ]))
            model_rows.append([
                name, "iTransformer", regime, model.parameter_count(),
                str(config_path.relative_to(PROJECT_ROOT)),
                sha256_of_file(config_path),
                str(weight_path.relative_to(PROJECT_ROOT)),
                weight_digests[name], weight_path.stat().st_size,
                str(path.relative_to(PROJECT_ROOT)), prediction_digests[name],
                round(inference_seconds, 1),
                round(prediction.size / inference_seconds, 1)])
        log("  %s predicted %d validation samples (%.0fs)"
            % (name, prediction.size, inference_seconds))
        del bundle

    if args.determinism:
        return compare_determinism(prediction_digests, weight_digests)

    write_csv(OUT / "itransformer_full_models.csv",
              ["model", "architecture", "regime", "parameter_count",
               "config_file", "config_sha256", "weight_file", "weight_sha256",
               "weight_bytes", "prediction_file", "prediction_sha256",
               "inference_seconds", "predictions_per_second"], model_rows)
    write_csv(OUT / "itransformer_training_diagnostics.csv",
              ["model", "regime", "d_model", "learning_rate",
               "parameter_count", "training_seconds", "first_epoch_train_L1",
               "final_epoch_train_L1"], diagnostics)
    CURVES.mkdir(parents=True, exist_ok=True)
    with open(str(CURVES / "full_refit_curves.json"), "w",
              encoding="utf-8") as handle:
        json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
        handle.write("\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
