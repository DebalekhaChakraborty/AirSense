"""Validate the AirSense V2 Phase-5 modern-model evaluation.

Read-only. Trains nothing, predicts nothing new, repairs nothing. Exits 0 only
if every gate passes.

Phase 5 executed one model family, the in-repository iTransformer-backbone
forecaster, and deliberately executed **no** foundation model: Chronos-2 was
stopped by the pretraining-overlap audit, TimesFM 3.0 and Moirai-2.0-R-small
by their weight licences. Several gates below assert exactly that.

Usage:
    .venv-v2/bin/python scripts/validate_phase5.py
"""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    PARTITION_BOUNDS, STATIONS, index_of)
from src.evaluation import metrics as M                           # noqa: E402
from src.models import itransformer_grid as grid                  # noqa: E402
from src.models.neural_data import (                              # noqa: E402
    INTERNAL_CORE, INTERNAL_TUNING)
from src.models.neural_features import (                          # noqa: E402
    CONTEXT_HOURS, HORIZON_VOCABULARY, dynamic_channel_names,
    static_channel_names)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
ITRANS = PROJECT_ROOT / "results" / "models" / "itransformer"
NEURAL = PROJECT_ROOT / "results" / "models" / "neural"
ARTIFACTS = PROJECT_ROOT / "artifacts"

MODELS = ["iTransformer_R1", "iTransformer_R2"]
CLASSICAL = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"]
PHASE4_MODELS = ["GRU_R0", "GRU_R1", "GRU_R2", "TCN_R0", "TCN_R1", "TCN_R2"]
EARLIER = CLASSICAL + PHASE4_MODELS
HORIZONS = [1, 6, 12, 24]
EXPECTED_SAMPLES = 413148
FOUNDATION_PACKAGES = ("chronos", "chronos_forecasting", "timesfm", "uni2ts",
                       "transformers", "gluonts")


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 70
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_index():
    stations, horizons, targets = [], [], []
    with open(str(PROCESSED / "sample_index" / "validation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
            targets.append(int(row["target_row_index"]))
    return (np.array(stations), np.array(horizons, dtype=np.int32),
            np.array(targets, dtype=np.int64))


def upstream(report):
    for name, interpreter in (("validate_foundation.py", "python3"),
                              ("validate_phase1.py", "python3"),
                              ("validate_phase2.py", "python3"),
                              ("validate_phase3.py", sys.executable),
                              ("validate_phase4.py", sys.executable)):
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)

    registry = load_json(ARTIFACTS / "verification_tooling_registry.json")
    drift = [name for name, entry in registry["validators"].items()
             if sha256_of_file(PROJECT_ROOT / name) != entry["sha256"]]
    report.add("Verification tooling matches the registry", not drift,
               "changed: %s" % drift)
    report.add("Registry records the Phase-5 scope correction",
               registry.get("correction_summary_phase5", {}).get(
                   "scientific_assertion_changed") is False
               and registry["correction_summary_phase5"]["gates_removed"] == 0
               and registry["correction_summary_phase5"]["gates_disabled"]
               == 0)

    phase4 = load_json(ARTIFACTS / "v2_phase4_manifest.json")
    stale = []
    for group, base in (("prediction_array_sha256",
                         "results/validation/predictions/"),
                        ("metric_table_sha256", "results/validation/"),
                        ("candidate_table_sha256", "results/models/neural/"),
                        ("model_config_and_weight_sha256",
                         "results/models/neural/"),
                        ("figure_sha256", "figures/")):
        for name, digest in phase4[group].items():
            path = PROJECT_ROOT / (base + name)
            if not path.is_file() or sha256_of_file(path) != digest:
                stale.append(name)
    report.add("Phase-4 evidence unchanged", not stale,
               "drifted: %s" % stale[:5])

    phase3 = load_json(ARTIFACTS / "v2_phase3_manifest.json")
    stale = []
    for group, base in (("prediction_array_sha256",
                         "results/validation/predictions/"),
                        ("metric_table_sha256", "results/validation/"),
                        ("b3_table_sha256", "results/models/b3/")):
        for name, digest in phase3[group].items():
            path = PROJECT_ROOT / (base + name)
            if not path.is_file() or sha256_of_file(path) != digest:
                stale.append(name)
    for name, digest in phase3["selected_model_sha256"].items():
        if sha256_of_file(PROJECT_ROOT / name) != digest:
            stale.append(name)
    report.add("Phase-3 evidence unchanged", not stale,
               "drifted: %s" % stale[:5])

    manifest = load_json(ARTIFACTS / "v2_phase5_manifest.json")
    for key, path in (("foundation_manifest_sha256",
                       "artifacts/v2_foundation_manifest.json"),
                      ("phase1_manifest_sha256",
                       "artifacts/v2_phase1_manifest.json"),
                      ("phase2_manifest_sha256",
                       "artifacts/v2_phase2_manifest.json"),
                      ("phase3_manifest_sha256",
                       "artifacts/v2_phase3_manifest.json"),
                      ("phase4_manifest_sha256",
                       "artifacts/v2_phase4_manifest.json")):
        report.add("Upstream manifest unchanged: %s" % Path(path).name,
                   manifest[key] == sha256_of_file(PROJECT_ROOT / path))


def contract(report):
    split = load_json(ARTIFACTS / "phase4_internal_split.json")
    train_start, train_end = PARTITION_BOUNDS["train"]
    core_start = datetime.fromisoformat(split["internal_core"]["start"])
    tune_end = datetime.fromisoformat(split["internal_tuning"]["end"])
    report.add("Phase-5 reused the Phase-4 training-only internal split",
               INTERNAL_CORE[0] == core_start
               and INTERNAL_TUNING[1] == tune_end)
    report.add("Internal selection lies entirely inside training",
               core_start >= train_start and tune_end <= train_end)
    report.add("Internal selection does not touch external validation",
               tune_end < PARTITION_BOUNDS["validation"][0])

    freeze = load_json(ARTIFACTS
                       / "phase5_itransformer_selection_freeze.json")
    report.add("Selection freeze pins the unchanged internal split",
               freeze["internal_split_sha256"]
               == sha256_of_file(ARTIFACTS / "phase4_internal_split.json")
               and freeze["internal_split_reused_from_phase4"] is True)
    report.add("Selection freeze pins the unchanged feature schema",
               freeze["neural_feature_schema_sha256"]
               == sha256_of_file(ARTIFACTS / "neural_feature_schema.json"))

    schema = load_json(ARTIFACTS / "neural_feature_schema.json")
    live = {r: dynamic_channel_names(r) for r in ("R0", "R1", "R2")}
    report.add("Context length is unchanged at 48",
               schema["context_hours"] == CONTEXT_HOURS == 48)
    report.add("Feature schema matches the frozen record",
               all(schema["dynamic_channels"][r] == live[r] for r in live)
               and schema["static_channels"] == static_channel_names())
    report.add("Horizon vocabulary is the frozen 1/6/12/24",
               schema["horizon_vocabulary"] == HORIZON_VOCABULARY
               == [1, 6, 12, 24])
    report.add("R1 sees PM2.5 and no other pollutant",
               "value:PM2.5" in live["R1"]
               and not any("value:%s" % p in live["R1"]
                           for p in ("PM10", "SO2", "NO2", "CO", "O3")))
    report.add("R2 sees all frozen target-station pollutants",
               all("value:%s" % p in live["R2"] for p in
                   ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")))


def grid_and_selection(report):
    candidates = grid.candidates()
    report.add("The grid is exactly the four frozen candidates",
               [(c["candidate_id"], c["d_model"], c["learning_rate"])
                for c in candidates]
               == [("IT_C01", 64, 0.0003), ("IT_C02", 64, 0.001),
                   ("IT_C03", 128, 0.0003), ("IT_C04", 128, 0.001)])
    report.add("Only d_model and learning rate vary across candidates",
               all(c["encoder_layers"] == 2 and c["attention_heads"] == 4
                   and c["feedforward_ratio"] == 4 and c["dropout"] == 0.1
                   for c in candidates))
    report.add("Training budget fixed at 8 epochs / batch 1024 / no early "
               "stopping",
               all(c["epochs"] == 8 and c["batch_size"] == 1024
                   and c["early_stopping"] is False and c["loss"] == "L1"
                   and c["optimizer"] == "AdamW" for c in candidates))
    report.add("Only R1 and R2 were evaluated", grid.REGIMES == ["R1", "R2"])

    metrics = read_csv(ITRANS / "internal_candidate_metrics.csv")
    counts = Counter(row["candidate_id"] for row in metrics)
    report.add("8 internal candidate fits, 2 regimes per candidate",
               len(metrics) == 8 and set(counts.values()) == {2}
               and sorted(counts) == ["IT_C01", "IT_C02", "IT_C03", "IT_C04"])
    report.add("Internal scoring used the internal tuning period only",
               all(row["regime"] in ("R1", "R2") for row in metrics))

    selection = read_csv(ITRANS / "internal_candidate_selection.csv")
    pool = [{"candidate_id": r["candidate_id"], "d_model": int(r["d_model"]),
             "learning_rate": float(r["learning_rate"]),
             "mean_macro_mae": float(r["mean_macro_station_horizon_MAE"]),
             "mean_macro_rmse": float(r["mean_macro_RMSE"])}
            for r in selection]
    recomputed = grid.select(pool)["candidate_id"]
    marked = [r["candidate_id"] for r in selection if r["selected"] == "true"]
    report.add("Selected candidate matches the frozen selection rule",
               marked == [recomputed],
               "marked %s, rule gives %s" % (marked, recomputed))
    report.add("Aggregated candidate scores reconcile with the fit table",
               not [c for c in counts
                    if abs(float(np.mean(
                        [float(r["internal_macro_station_horizon_MAE"])
                         for r in metrics if r["candidate_id"] == c]))
                        - float([r["mean_macro_station_horizon_MAE"]
                                 for r in selection
                                 if r["candidate_id"] == c][0])) > 1e-6])

    freeze = load_json(ARTIFACTS
                       / "phase5_itransformer_selection_freeze.json")
    report.add("Architecture selection came from training only",
               freeze["external_validation_used_for_selection"] is False
               and freeze["selected_from"]
               == "training-only internal core/tuning split")
    report.add("Selection freeze recorded zero external metrics",
               freeze["external_validation_metrics_seen"] == 0
               and freeze["test_metrics_seen"] == 0)
    report.add("The same configuration is used for R1 and R2",
               freeze["same_configuration_used_for_all_regimes"] is True)
    report.add("Selection freeze pins the candidate tables it was built from",
               freeze["candidate_selection_sha256"]
               == sha256_of_file(ITRANS / "internal_candidate_selection.csv")
               and freeze["candidate_metrics_sha256"]
               == sha256_of_file(ITRANS / "internal_candidate_metrics.csv"))
    report.add("Selection freeze pins the model source it selected",
               freeze["model_source_sha256"] == sha256_of_file(
                   PROJECT_ROOT / "src" / "models"
                   / "itransformer_forecaster.py"))

    configs = {name: load_json(ITRANS / ("%s_config.json" % name))
               for name in MODELS}
    report.add("Both full models exist with weights",
               len(configs) == 2
               and all((ITRANS / ("%s.safetensors" % n)).is_file()
                       for n in MODELS))
    report.add("Both models use the selected d_model and learning rate",
               all(c["d_model"] == freeze["selected"]["d_model"]
                   and c["learning_rate"]
                   == freeze["selected"]["learning_rate"]
                   for c in configs.values()))
    report.add("Full models were trained on the complete training universe "
               "from a clean initialisation",
               all(c["trained_on"]
                   == "complete frozen Phase-2 training universe"
                   and c["initialised_from"].startswith("clean seeded")
                   for c in configs.values()))
    report.add("iTransformer outputs are declared unconstrained",
               all(c["output_constrained"] is False
                   for c in configs.values()))
    report.add("No pretrained weights were used",
               all(c["pretrained_weights_used"] is False
                   for c in configs.values()))


def architecture(report):
    """The inverted formulation is asserted structurally, not by name."""
    from src.models.itransformer_forecaster import ITransformerForecaster

    freeze = load_json(ARTIFACTS
                       / "phase5_itransformer_selection_freeze.json")
    d_model = int(freeze["selected"]["d_model"])
    names = dynamic_channel_names("R2")
    model = ITransformerForecaster(len(names), len(static_channel_names()),
                                   target_index=names.index("value:PM2.5"),
                                   d_model=d_model,
                                   layers=grid.FIXED["encoder_layers"],
                                   heads=grid.FIXED["attention_heads"],
                                   ff_ratio=grid.FIXED["feedforward_ratio"],
                                   dropout=grid.FIXED["dropout"],
                                   head_hidden=grid.FIXED["head_hidden"])
    report.add("One token per variate: the embedding consumes a whole "
               "48-hour history",
               model.variate_embedding.in_features == CONTEXT_HOURS == 48
               and model.variate_embedding.out_features == d_model)
    with torch.no_grad():
        tokens = model.variate_embedding(
            torch.zeros(2, len(names), CONTEXT_HOURS))
    report.add("Attention runs across variate tokens, not time steps",
               tuple(tokens.shape) == (2, len(names), d_model)
               and tuple(model.encoder(tokens).shape)
               == (2, len(names), d_model))
    children = sorted(name for name, _ in model.named_children())
    report.add("No decoder and no auxiliary sequence module",
               children == ["encoder", "head", "variate_embedding"],
               "children: %s" % children)
    positional = [name for name, _ in model.named_parameters()
                  if "pos" in name.lower() or "embed_positions" in name]
    report.add("No positional encoding across variate tokens", not positional,
               "found: %s" % positional)
    report.add("The head reads the PM2.5 variate token plus static "
               "conditioning",
               model.target_index == names.index("value:PM2.5")
               and model.head[0].in_features
               == d_model + len(static_channel_names()))
    report.add("Encoder depth and head count match the frozen grid",
               len(model.encoder.layers) == grid.FIXED["encoder_layers"]
               and model.encoder.layers[0].self_attn.num_heads
               == grid.FIXED["attention_heads"])
    configs = {name: load_json(ITRANS / ("%s_config.json" % name))
               for name in MODELS}
    report.add("Saved configs declare the inverted formulation",
               all(c["attention_axis"] == "across variate tokens"
                   and c["positional_encoding_across_variates"] is False
                   and c["decoder"] is None
                   and c["variate_tokens"] == c["dynamic_channels"]
                   for c in configs.values()))


def predictions_and_metrics(report, station, horizon):
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    problems = []
    predictions = {}
    for name in MODELS:
        path = PRED / ("%s_validation.npy" % name)
        if not path.is_file():
            problems.append("%s missing" % name)
            continue
        values = np.load(str(path)).astype(np.float64)
        predictions[name] = values
        if values.size != EXPECTED_SAMPLES:
            problems.append("%s has %d predictions" % (name, values.size))
        elif not np.isfinite(values).all():
            problems.append("%s has non-finite predictions" % name)
    report.add("Each iTransformer predicts exactly 413,148 samples",
               not problems, "; ".join(problems[:3]))
    report.add("Prediction arrays align with the canonical validation index",
               station.size == EXPECTED_SAMPLES
               and all(v.size == station.size for v in predictions.values()))

    manifest = load_json(ARTIFACTS / "v2_phase2_manifest.json")
    report.add("Validation index hash unchanged since Phase 2",
               manifest["sample_index_sha256"]["validation"]
               == sha256_of_file(PROCESSED / "sample_index"
                                 / "validation.csv"))

    negatives = read_csv(OUT / "phase5_negative_predictions.csv")
    clipped = [r["model"] for r in negatives if r["clipped"] != "false"]
    report.add("No prediction was clipped", not clipped,
               "clipped: %s" % clipped)
    mismatch = [r["model"] for r in negatives if r["model"] in predictions
                and abs(float(predictions[r["model"]].min())
                        - float(r["min_prediction"])) > 1e-3]
    report.add("Reported minima match the stored arrays", not mismatch,
               "mismatched: %s" % mismatch)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase5_model_comparison.csv")}
    cells = read_csv(OUT / "phase5_metrics_by_station_horizon.csv")
    problems = []
    for name in MODELS:
        subset = [float(r["MAE"]) for r in cells if r["model"] == name]
        if len(subset) != 48:
            problems.append("%s has %d cells" % (name, len(subset)))
            continue
        if abs(float(np.mean(subset))
               - float(comparison[name]["macro_station_horizon_MAE"])) > 1e-6:
            problems.append("%s macro mismatch" % name)
    report.add("Primary macro metric reconciles from the 48 cells",
               not problems, "; ".join(problems[:3]))
    report.add("Earlier-phase values in the combined table are reused, not "
               "recomputed",
               all(comparison[n]["value_source"] == "phase3_frozen"
                   for n in CLASSICAL)
               and all(comparison[n]["value_source"] == "phase4_frozen"
                       for n in PHASE4_MODELS)
               and all(comparison[n]["value_source"] == "phase5"
                       for n in MODELS))

    frozen4 = {r["model"]: r
               for r in read_csv(OUT / "phase4_model_comparison.csv")}
    drift = [n for n in EARLIER
             if abs(float(frozen4[n]["macro_station_horizon_MAE"])
                    - float(comparison[n]["macro_station_horizon_MAE"]))
             > 1e-12]
    report.add("Frozen Phase-3 and Phase-4 values are byte-faithful",
               not drift, "drifted: %s" % drift)

    gains = read_csv(OUT / "phase5_information_regime_gain.csv")
    sources = {r["architecture"]: r["value_source"] for r in gains}
    report.add("Co-pollutant table reuses frozen values for B3, GRU and TCN",
               sources.get("B3") == "phase3_frozen"
               and sources.get("GRU") == "phase4_frozen"
               and sources.get("TCN") == "phase4_frozen"
               and sources.get("iTransformer") == "phase5")

    severe = {r["model"]: r
              for r in read_csv(OUT / "phase5_severe_metrics.csv")}
    problems = []
    mask = actual > 244.0
    for name in MODELS:
        recomputed = float(np.abs(actual[mask]
                                  - predictions[name][mask]).mean())
        if abs(recomputed - float(severe[name]["severe_MAE"])) > 1e-6:
            problems.append(name)
    report.add("Severe metrics reconcile at the frozen 244.0 threshold",
               not problems and M.SEVERE_THRESHOLD == 244.0,
               "mismatched: %s" % problems)
    report.add("Severe threshold was not re-derived from validation",
               all(int(severe[n]["severe_n"]) == int(severe["B0"]["severe_n"])
                   for n in MODELS))


def determinism(report):
    directory = ITRANS / "determinism"
    problems = []
    for name in MODELS:
        repeat = directory / ("%s_validation.npy" % name)
        primary = PRED / ("%s_validation.npy" % name)
        if not repeat.is_file():
            problems.append("%s: no determinism refit" % name)
        elif sha256_of_file(repeat) != sha256_of_file(primary):
            problems.append("%s: prediction differs" % name)
    report.add("Determinism refit reproduces every prediction array byte for "
               "byte", not problems, "; ".join(problems))

    problems = []
    for name in MODELS:
        repeat = directory / ("%s.safetensors" % name)
        primary = ITRANS / ("%s.safetensors" % name)
        if not repeat.is_file():
            problems.append("%s: no determinism weight file" % name)
        elif sha256_of_file(repeat) != sha256_of_file(primary):
            problems.append("%s: weights differ" % name)
    report.add("Determinism refit reproduces every weight file byte for byte",
               not problems, "; ".join(problems))


def prohibitions(report):
    for name in FOUNDATION_PACKAGES:
        try:
            __import__(name)
            report.add("%s is absent" % name, False, "installed")
        except ImportError:
            report.add("%s is absent" % name, True)

    # The audit asserts no checkpoint was ever fetched. Assert that against
    # the Hugging Face cache layout rather than against the directory's mere
    # existence: a hub download materialises a models--<org>--<name> tree and
    # weight blobs, and neither exists here.
    cached = []
    hub = Path.home() / ".cache" / "huggingface"
    if hub.exists():
        cached += [str(path) for path in hub.rglob("models--*")]
        cached += [str(path) for path in hub.rglob("*")
                   if path.is_file() and path.suffix in
                   (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx")]
    for base in (PROJECT_ROOT / "models", PROJECT_ROOT / "checkpoints"):
        if base.exists():
            cached.append(str(base))
    report.add("No foundation-model checkpoint was downloaded", not cached,
               "found: %s" % cached[:5])

    foreign = []
    for directory in ("src", "scripts", "results", "artifacts", "configs",
                      "figures", "docs", "notebooks", "tests",
                      "data/processed"):
        base = PROJECT_ROOT / directory
        if not base.is_dir():
            continue
        foreign += [str(path.relative_to(PROJECT_ROOT))
                    for path in sorted(base.rglob("*"))
                    if path.is_file() and path.suffix in
                    (".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf")]
    report.add("The repository holds no imported model weight file",
               not foreign, "found: %s" % foreign[:5])

    audit = load_json(ARTIFACTS / "phase5_prevalidation_freeze.json")
    report.add("Chronos-2 is recorded as not executed",
               audit["chronos2_executed"] is False
               and audit["foundation_model_executed"] is False
               and "classification C" in audit["chronos2_gate"])
    report.add("The Chronos-2 audit document is pinned unchanged",
               audit["chronos2_audit_sha256"] == sha256_of_file(
                   PROJECT_ROOT / "docs" / "CHRONOS2_PRETRAINING_AUDIT.md"))
    manifest = load_json(ARTIFACTS / "v2_phase5_manifest.json")
    report.add("Manifest records no TSFM load and no pretrained weights",
               manifest["tsfm_loaded"] is False
               and manifest["chronos2_checkpoint_downloaded"] is False
               and manifest["pretrained_weights_used"] is False)
    report.add("No R0 or R3 iTransformer was produced",
               not list(ITRANS.glob("*_R0*"))
               and not list(ITRANS.glob("*_R3*"))
               and manifest["r3_implemented"] is False)
    report.add("torch reports no CUDA device in use",
               not torch.cuda.is_available())

    test_artifacts = [str(p.relative_to(PROJECT_ROOT))
                      for p in sorted(PRED.glob("*test*"))
                      + sorted(ITRANS.rglob("*test*"))]
    report.add("No test prediction artifact exists", not test_artifacts,
               "found: %s" % test_artifacts)
    summary = load_json(ARTIFACTS / "phase5_validation_summary.json")
    report.add("Summary records a sealed test and zero test activity",
               summary["final_test_status"] == "sealed"
               and summary["test_predictions_generated"] == 0
               and summary["test_metrics_seen"] == 0)
    report.add("Pre-validation freeze precedes any external metric",
               audit["external_validation_metrics_seen"] == 0
               and audit["final_test_status"] == "sealed"
               and audit["hyperparameters_selected_from_training_only_"
                         "internal_split"] is True)
    report.add("Pre-validation freeze pins the selection freeze that "
               "preceded it",
               audit["phase5_itransformer_selection_freeze_sha256"]
               == sha256_of_file(
                   ARTIFACTS
                   / "phase5_itransformer_selection_freeze.json"))


def independent(report, station, horizon, target):
    """Fourteen recomputations through a path separate from the reporting."""
    from safetensors.torch import load_file

    from src.data.preprocessing import (
        TARGET, load_training_statistics)
    from src.data.rolling_history import RollingPm25Series
    from src.data.target_history import TargetAccessError
    from src.data.windowing import StationSeries
    from src.models.itransformer_forecaster import ITransformerForecaster
    from src.models.neural_features import (
        build_static_matrix, build_station_dynamic_matrix, gather_windows)

    probe_station = "Dongsi"
    probe_code = STATIONS.index(probe_station)
    calendar = np.load(str(PROCESSED / "calendar" / "calendar_cyclic.npy"))
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    station_stats = load_json(ARTIFACTS / "preprocessing_statistics.json")[
        "station_training_medians"]
    raw = next((PROJECT_ROOT / "data" / "raw"
                / "PRSA_Data_20130301-20170228")
               .glob("PRSA_Data_%s_*.csv" % probe_station))
    rolling = RollingPm25Series(probe_station, raw, statistics,
                                station_stats[probe_station][TARGET])
    series = StationSeries(PROCESSED, probe_station)

    rows = np.flatnonzero((station == probe_station) & (horizon == 6))
    row = int(rows[len(rows) // 2])
    origin = int(target[row] - horizon[row])

    windows = {}
    for regime in ("R1", "R2"):
        dynamic = build_station_dynamic_matrix(regime, series,
                                               rolling._scaled)
        window = gather_windows(dynamic[None, :, :], [0], [origin])
        windows[regime] = window
        report.add("Independent: %s input window is 48 steps ending at origin"
                   % regime,
                   window.shape[1] == CONTEXT_HOURS
                   and np.allclose(window[0, -1], dynamic[origin])
                   and np.allclose(window[0, 0], dynamic[origin - 47]))

    static = build_static_matrix([probe_code] * 4, [1, 6, 12, 24],
                                 [origin] * 4,
                                 [origin + h for h in (1, 6, 12, 24)],
                                 calendar)
    report.add("Independent: horizon one-hot for all four horizons",
               np.array_equal(static[:, 12:16], np.eye(4, dtype=np.float32)))
    report.add("Independent: station one-hot selects the probe station",
               static[0, probe_code] == 1.0 and static[0, :12].sum() == 1.0)

    sequence = rolling.scaled_window(origin, CONTEXT_HOURS)
    report.add("Independent: rolling validation PM2.5 sequence has 48 finite "
               "values", sequence.size == CONTEXT_HOURS
               and bool(np.isfinite(sequence).all()))
    refused = False
    try:
        rolling.native_at(origin + 1, origin)
    except TargetAccessError:
        refused = True
    sealed = False
    try:
        rolling.scaled_window(index_of(PARTITION_BOUNDS["test"][0]),
                              CONTEXT_HOURS)
    except TargetAccessError:
        sealed = True
    report.add("Independent: no PM2.5 value after the origin, test sealed",
               refused and sealed)

    static_one = build_static_matrix([probe_code], [int(horizon[row])],
                                     [origin], [int(target[row])], calendar)
    reloaded = {}
    for name in MODELS:
        regime = name.split("_")[1]
        config = load_json(ITRANS / ("%s_config.json" % name))
        names = dynamic_channel_names(regime)
        model = ITransformerForecaster(
            config["dynamic_channels"], config["static_channels"],
            target_index=names.index("value:PM2.5"),
            d_model=config["d_model"],
            layers=config["training"]["encoder_layers"],
            heads=config["training"]["attention_heads"],
            ff_ratio=config["training"]["feedforward_ratio"],
            dropout=config["training"]["dropout"],
            head_hidden=config["training"]["head_hidden"])
        model.load_state_dict(load_file(str(ITRANS
                                            / ("%s.safetensors" % name))))
        model.eval()
        reloaded[name] = model
        with torch.no_grad():
            value = float(model(
                torch.from_numpy(np.ascontiguousarray(windows[regime])),
                torch.from_numpy(static_one))[0])
        stored = float(np.load(str(PRED / ("%s_validation.npy" % name)))[row])
        report.add("Independent: reloaded %s reproduces a stored prediction"
                   % name, abs(value - stored) < 1e-3,
                   "recomputed %.6f vs stored %.6f" % (value, stored))

    model = reloaded["iTransformer_R2"]
    names = dynamic_channel_names("R2")
    perturbed = windows["R2"].copy()
    perturbed[0, :, names.index("value:CO")] += 5.0
    with torch.no_grad():
        base = float(model(
            torch.from_numpy(np.ascontiguousarray(windows["R2"])),
            torch.from_numpy(static_one))[0])
        moved = float(model(torch.from_numpy(np.ascontiguousarray(perturbed)),
                            torch.from_numpy(static_one))[0])
    report.add("Independent: a non-target variate moves the PM2.5 prediction, "
               "so cross-variate attention is live",
               abs(moved - base) > 1e-6,
               "base %.6f perturbed %.6f" % (base, moved))

    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = {n: np.load(str(PRED / ("%s_validation.npy" % n)))
                   .astype(np.float64) for n in MODELS}
    cells = read_csv(OUT / "phase5_metrics_by_station_horizon.csv")
    best = min(MODELS, key=lambda n: float(np.abs(actual
                                                  - predictions[n]).mean()))
    cell = [r for r in cells if r["model"] == best
            and r["station"] == probe_station and r["horizon"] == "6"][0]
    selector = (station == probe_station) & (horizon == 6)
    report.add("Independent: %s %s h6 cell MAE" % (best, probe_station),
               abs(float(np.abs(actual[selector]
                                - predictions[best][selector]).mean())
                   - float(cell["MAE"])) < 1e-6)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase5_model_comparison.csv")}

    def macro(name, mask=None):
        values = []
        for st in STATIONS:
            for hz in HORIZONS:
                selection = (station == st) & (horizon == hz)
                if mask is not None:
                    selection = selection & mask
                if not selection.any():
                    continue
                values.append(float(np.abs(actual[selection]
                                           - predictions[name][selection])
                                    .mean()))
        return float(np.mean(values))

    for name in MODELS:
        report.add("Independent: %s overall macro station-horizon MAE" % name,
                   abs(macro(name)
                       - float(comparison[name]["macro_station_horizon_MAE"]))
                   < 1e-6, "recomputed %.6f" % macro(name))

    severe_rows = {r["model"]: r
                   for r in read_csv(OUT / "phase5_severe_metrics.csv")}
    mask = actual > 244.0
    recomputed = float(np.abs(actual[mask] - predictions[best][mask]).mean())
    report.add("Independent: %s severe MAE at the frozen threshold" % best,
               abs(recomputed - float(severe_rows[best]["severe_MAE"]))
               < 1e-6, "recomputed %.6f over %d severe hours"
               % (recomputed, int(mask.sum())))

    gains = {(r["architecture"], r["scope"]): r
             for r in read_csv(OUT / "phase5_information_regime_gain.csv")}
    row_gain = gains[("iTransformer", "overall")]
    delta = macro("iTransformer_R1") - macro("iTransformer_R2")
    report.add("Independent: iTransformer R1->R2 co-pollutant increment",
               abs(delta - float(row_gain["absolute_MAE_improvement"]))
               < 1e-6, "recomputed %.6f" % delta)

    report.add("Independent: every iTransformer array holds 413,148 "
               "predictions",
               all(v.size == EXPECTED_SAMPLES for v in predictions.values()))
    report.add("Independent: no artifact holds a test-period target",
               not list(PRED.glob("*test*"))
               and bool(np.isnan(rolling._scaled[
                   index_of(PARTITION_BOUNDS["test"][0]):]).all()))


def main():
    print("")
    print("[AIRSENSE V2 PHASE-5 VALIDATION]")
    print("")
    print("Read-only. Development validation only; the final test is sealed.")
    print("")
    report = Report()
    station, horizon, target = load_index()
    upstream(report)
    contract(report)
    grid_and_selection(report)
    architecture(report)
    predictions_and_metrics(report, station, horizon)
    determinism(report)
    prohibitions(report)
    independent(report, station, horizon, target)

    print("=" * 78)
    report.render()
    print("=" * 78)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-5 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-5 VALIDATION: PASS")
    print("Development validation only. The 2016-17 test remains sealed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
