"""Phase-7 multi-seed robustness for the frozen candidate set.

Protocol Phase 7, sections 2 and 3.

Refits every stochastically-trained candidate under seeds 42-46 and scores each
fit on development validation. **Nothing about any model changes**: same
architecture, features, context, epochs, optimizer, learning rate, batch size
and severe threshold. Only the seed moves.

Seed 42 is **not retrained**. Its frozen Phase-4 and Phase-6 prediction arrays
are re-scored through this script's own code path, so the seed-42 row is the
published model rather than a lookalike.

``SA_R2`` is included as the **paired control** for the Phase-6 cross-station
contrast, because section 3A asks whether the h = 24 reversal survives
reseeding and that question is only answerable in pairs. It is *not* a
candidate for final selection and the Phase-7 candidate freeze excludes it.

B0 and B3_R2 are deterministic and are recorded once, from frozen artifacts.

Usage:
    .venv-v2/bin/python scripts/run_phase7_multiseed.py
    .venv-v2/bin/python scripts/run_phase7_multiseed.py --models SA_R3
"""

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import STATIONS                          # noqa
from src.evaluation import metrics as M                              # noqa
from src.models import neural_grid                                   # noqa
from src.models import station_attention_grid as sa_grid             # noqa
from src.models.gru_forecaster import GRUForecaster                  # noqa
from src.models.neural_data import NeuralDataSource, load_sample_index  # noqa
from src.models.neural_features import (                             # noqa
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import configure_determinism         # noqa
from src.models.neural_training import predict as flat_predict       # noqa
from src.models.neural_training import train as flat_train           # noqa
from src.models.spatial_grouping import (                            # noqa
    GuardedNetworkHistory, NetworkBundle, build_groups)
from src.models.spatial_training import predict as grouped_predict   # noqa
from src.models.spatial_training import train as grouped_train       # noqa
from src.models.station_attention_forecaster import (                # noqa
    StationAttentionForecaster)
from src.models.tcn_forecaster import TCNForecaster                  # noqa

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "models" / "robustness"
NEURAL = PROJECT_ROOT / "results" / "models" / "neural"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"

SEEDS = [42, 43, 44, 45, 46]
BASE_SEED = 42
HORIZONS = [1, 6, 12, 24]
EXPECTED_SAMPLES = 413148
MODELS = ["GRU_R1", "TCN_R1", "SA_R3", "SA_R2"]
ROLE = {"GRU_R1": "candidate", "TCN_R1": "candidate", "SA_R3": "candidate",
        "SA_R2": "paired control, not a candidate"}
# Declared rather than derived. Deriving them by scoring a short slice fails:
# a slice with no severe hour makes severe_metrics return only severe_n.
METRIC_NAMES = ["macro_station_horizon_MAE", "micro_MAE", "macro_RMSE",
                "micro_RMSE", "macro_R2", "severe_MAE", "severe_RMSE",
                "severe_mean_residual", "severe_underprediction_pct",
                "n_negative", "min_prediction"]

START = time.time()


def log(message):
    print("[%8.1fs] %s" % (time.time() - START, message), flush=True)


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed):
    """Phase-4 determinism configuration, then the seed this run wants."""
    runtime = configure_determinism()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime = OrderedDict(runtime)
    runtime["seed"] = seed
    return runtime


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
        ("severe_underprediction_pct", severe["severe_underprediction_pct"]),
        ("n_negative", negatives["n_negative"]),
        ("min_prediction", negatives["min_prediction"]),
    ])


class FlatRunner(object):
    """GRU_R1 and TCN_R1: one canonical sample per forward pass."""

    def __init__(self, name):
        self.name = name
        self.architecture = name.split("_")[0]
        self.regime = name.split("_")[1]
        config = load_json(NEURAL / ("%s_config.json" % name))
        self.capacity = config["capacity"]
        self.learning_rate = config["learning_rate"]
        self.candidate = neural_grid.candidates(self.architecture)[0]
        self.sealed = NeuralDataSource(PROJECT_ROOT, unseal_validation=False)
        self.unsealed = NeuralDataSource(PROJECT_ROOT, unseal_validation=True)
        self.t_station, self.t_horizon, self.t_target = load_sample_index(
            PROCESSED, "train")
        self.v_station, self.v_horizon, self.v_target = load_sample_index(
            PROCESSED, "validation")

    def build(self):
        dynamic = len(dynamic_channel_names(self.regime))
        static = len(static_channel_names())
        if self.architecture == "GRU":
            return GRUForecaster(dynamic, static, hidden_size=self.capacity,
                                 head_hidden=self.candidate["head_hidden"],
                                 dropout=self.candidate["dropout"],
                                 num_layers=self.candidate["num_layers"])
        return TCNForecaster(dynamic, static, channels=self.capacity,
                             head_hidden=self.candidate["head_hidden"],
                             dropout=self.candidate["dropout"],
                             kernel_size=self.candidate["kernel_size"],
                             dilations=self.candidate["dilations"])

    def run(self, seed):
        bundle = self.sealed.bundle(self.regime, self.t_station,
                                    self.t_horizon, self.t_target)
        model = self.build()
        curve, seconds = flat_train(model, bundle, self.learning_rate,
                                    seed=seed)
        del bundle
        v_bundle = self.unsealed.bundle(self.regime, self.v_station,
                                        self.v_horizon, self.v_target,
                                        pm25_source="rolling",
                                        with_labels=False)
        prediction, inference = flat_predict(model, v_bundle)
        del v_bundle
        return model, prediction, curve, seconds, inference


class GroupedRunner(object):
    """SA_R3 and SA_R2: one grouped origin per forward pass, 12 stations."""

    def __init__(self, name):
        self.name = name
        self.cross_station = name.endswith("R3")
        config = load_json(SPATIAL / ("%s_config.json" % name))
        self.hidden = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["grouped_batch_size"]
        self.sealed = GuardedNetworkHistory(PROJECT_ROOT,
                                            unseal_validation=False)
        self.unsealed = GuardedNetworkHistory(PROJECT_ROOT,
                                              unseal_validation=True)
        self.train_groups = build_groups(
            PROCESSED, "train", labels_from=self.sealed.train_target_native)
        self.v_groups = build_groups(PROCESSED, "validation")

    def build(self):
        return StationAttentionForecaster(
            self.sealed.dynamic_channel_count, len(static_channel_names()),
            hidden_size=self.hidden,
            heads=sa_grid.FIXED["attention_heads"],
            attention_dropout=sa_grid.FIXED["attention_dropout"],
            head_hidden=sa_grid.FIXED["head_hidden"],
            head_dropout=sa_grid.FIXED["head_dropout"],
            cross_station=self.cross_station)

    def run(self, seed):
        bundle = NetworkBundle(self.train_groups, self.sealed)
        model = self.build()
        curve, seconds = grouped_train(model, bundle, self.learning_rate,
                                       batch_size=self.batch_size, seed=seed)
        v_bundle = NetworkBundle(self.v_groups, self.unsealed)
        grouped, inference = grouped_predict(model, v_bundle,
                                             batch_size=self.batch_size)
        prediction = self.v_groups.scatter(grouped, EXPECTED_SAMPLES)
        return model, prediction, curve, seconds, inference


def frozen_prediction(name):
    return np.load(str(PRED / ("%s_validation.npy" % name))).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--from-saved", action="store_true",
                        help="rescore prediction arrays already on disk "
                             "instead of retraining anything")
    args = parser.parse_args()

    freeze = ARTIFACTS / "phase7_candidate_freeze.json"
    if not freeze.is_file():
        raise SystemExit("the Phase-7 candidate freeze must exist first")
    log("candidate freeze %s" % sha256_of_file(freeze)[:16])

    station = np.array([r["station"] for r in csv.DictReader(
        open(str(PROCESSED / "sample_index" / "validation.csv"),
             encoding="utf-8"))])
    horizon = np.array([int(r["horizon_hours"]) for r in csv.DictReader(
        open(str(PROCESSED / "sample_index" / "validation.csv"),
             encoding="utf-8"))], dtype=np.int32)
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    horizon_rows = []
    curves = {}
    for name in args.models:
        runner = None
        if not args.from_saved:
            runner = (GroupedRunner(name) if name.startswith("SA_")
                      else FlatRunner(name))
        log("%s ready (%s)" % (name, ROLE[name]))
        for seed in SEEDS:
            if seed == BASE_SEED:
                prediction = frozen_prediction(name)
                seconds = inference = None
                parameters = None
                source = "frozen_phase4" if name in ("GRU_R1", "TCN_R1") \
                    else "frozen_phase6"
                digest = sha256_of_file(PRED / ("%s_validation.npy" % name))
            elif args.from_saved:
                path = OUT / ("%s_seed%d_validation.npy" % (name, seed))
                prediction = np.load(str(path)).astype(np.float64)
                seconds = inference = parameters = None
                digest = sha256_of_file(path)
                source = "phase7_refit_rescored"
            else:
                runtime = seed_everything(seed)
                model, prediction, curve, seconds, inference = runner.run(seed)
                parameters = model.parameter_count()
                path = OUT / ("%s_seed%d_validation.npy" % (name, seed))
                np.save(str(path), prediction.astype(np.float32))
                digest = sha256_of_file(path)
                curves["%s|%d" % (name, seed)] = [round(v, 6) for v in curve]
                source = "phase7_refit"
                del model
            if prediction.size != EXPECTED_SAMPLES:
                raise SystemExit("%s seed %d produced %d predictions"
                                 % (name, seed, prediction.size))
            values = score(actual, prediction, station, horizon)
            rows.append([name, ROLE[name], seed, source, parameters]
                        + [round(values[k], 6) if isinstance(values[k], float)
                           else values[k] for k in values]
                        + [None if seconds is None else round(seconds, 1),
                           None if inference is None else round(inference, 1),
                           digest])
            for hz in HORIZONS:
                selector = horizon == hz
                horizon_rows.append([
                    name, seed, hz, int(selector.sum()),
                    round(M.macro_station_mae(actual[selector],
                                              prediction[selector],
                                              station[selector]), 6),
                    round(M.mae(actual[selector], prediction[selector]), 6),
                    round(M.rmse(actual[selector], prediction[selector]), 6)])
            log("  %-7s seed %d  macro MAE %10.6f  severe %8.3f  (%s)"
                % (name, seed, values["macro_station_horizon_MAE"],
                   values["severe_MAE"], source))
        del runner

    write_csv(OUT / "multiseed_metrics.csv",
              ["model", "role", "seed", "source", "parameter_count"]
              + METRIC_NAMES + ["training_seconds", "inference_seconds",
                                "prediction_sha256"], rows)
    write_csv(OUT / "multiseed_metrics_by_horizon.csv",
              ["model", "seed", "horizon", "n", "macro_station_MAE",
               "micro_MAE", "micro_RMSE"], horizon_rows)
    if curves:
        (OUT / "training_curves").mkdir(parents=True, exist_ok=True)
        with open(str(OUT / "training_curves" / "multiseed_curves.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
            handle.write("\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
