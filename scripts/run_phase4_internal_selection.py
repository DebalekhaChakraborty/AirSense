"""Phase-4 internal architecture selection, training-only.

Protocol Phase 4, Stage A.

Trains each of the eight frozen candidates on the internal **core** period and
scores it on the internal **tuning** period, under all three information
regimes - 24 fits. Both periods lie inside the frozen Phase-2 training
partition, so the external development validation set is not read at all here.

Usage:
    .venv-v2/bin/python scripts/run_phase4_internal_selection.py
"""

import csv
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import STATIONS                     # noqa: E402
from src.evaluation import metrics as M                         # noqa: E402
from src.models import neural_grid                              # noqa: E402
from src.models.gru_forecaster import GRUForecaster             # noqa: E402
from src.models.neural_data import (                            # noqa: E402
    INTERNAL_CORE, INTERNAL_TUNING, NeuralDataSource, internal_mask,
    load_sample_index)
from src.models.neural_features import (                        # noqa: E402
    dynamic_channel_names, static_channel_names)
from src.models.neural_training import (                        # noqa: E402
    configure_determinism, predict, train)
from src.models.tcn_forecaster import TCNForecaster             # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
OUT = PROJECT_ROOT / "results" / "models" / "neural"
CURVES = OUT / "training_curves"
ARTIFACTS = PROJECT_ROOT / "artifacts"

START = time.time()


def log(message):
    print("[%7.1fs] %s" % (time.time() - START, message), flush=True)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def build_model(candidate, dynamic, static):
    if candidate["architecture"] == "GRU":
        return GRUForecaster(dynamic, static,
                             hidden_size=candidate["capacity"],
                             head_hidden=candidate["head_hidden"],
                             dropout=candidate["dropout"],
                             num_layers=candidate["num_layers"])
    return TCNForecaster(dynamic, static, channels=candidate["capacity"],
                         head_hidden=candidate["head_hidden"],
                         dropout=candidate["dropout"],
                         kernel_size=candidate["kernel_size"],
                         dilations=candidate["dilations"])


def main():
    runtime = configure_determinism()
    log("runtime %s" % json.dumps(runtime))

    stations, horizons, targets = load_sample_index(PROCESSED, "train")
    core = internal_mask(targets, INTERNAL_CORE)
    tuning = internal_mask(targets, INTERNAL_TUNING)
    log("internal core %d | internal tuning %d"
        % (int(core.sum()), int(tuning.sum())))

    source = NeuralDataSource(PROJECT_ROOT, unseal_validation=False)
    log("canonical layer loaded; external validation NOT unsealed")

    rows = []
    curves = {}
    for regime in neural_grid.REGIMES:
        core_bundle = source.bundle(regime, stations[core], horizons[core],
                                    targets[core])
        tune_bundle = source.bundle(regime, stations[tuning],
                                    horizons[tuning], targets[tuning])
        tune_station = stations[tuning]
        tune_horizon = horizons[tuning]
        actual = tune_bundle.labels.astype(np.float64)
        log("%s bundles: core %d x %d channels | tuning %d"
            % (regime, core_bundle.size,
               len(dynamic_channel_names(regime)), tune_bundle.size))
        for architecture in ("GRU", "TCN"):
            for candidate in neural_grid.candidates(architecture):
                configure_determinism()
                model = build_model(candidate,
                                    len(dynamic_channel_names(regime)),
                                    len(static_channel_names()))
                curve, seconds = train(model, core_bundle,
                                       candidate["learning_rate"],
                                       epochs=candidate["epochs"],
                                       batch_size=candidate["batch_size"])
                prediction, inference_seconds = predict(model, tune_bundle)
                cells = M.cell_metrics(actual, prediction, tune_station,
                                       tune_horizon)
                macro_mae = M.macro_from_cells(cells)
                macro_rmse = M.macro_from_cells(cells, "rmse")
                rows.append([
                    architecture, candidate["candidate_id"], regime,
                    model.parameter_count(), candidate["learning_rate"],
                    candidate["capacity"], round(macro_mae, 6),
                    round(macro_rmse, 6),
                    round(M.mae(actual, prediction), 6),
                    round(M.rmse(actual, prediction), 6),
                    round(seconds, 1), round(inference_seconds, 1)])
                curves["%s|%s|%s" % (architecture, candidate["candidate_id"],
                                     regime)] = [round(v, 6) for v in curve]
                log("  %s %s %s macro-MAE %.6f (train %.0fs, infer %.0fs)"
                    % (regime, architecture, candidate["candidate_id"],
                       macro_mae, seconds, inference_seconds))
                del model
        del core_bundle, tune_bundle

    write_csv(OUT / "internal_candidate_metrics.csv",
              ["architecture", "candidate_id", "regime", "parameter_count",
               "learning_rate", "capacity",
               "internal_macro_station_horizon_MAE", "internal_macro_RMSE",
               "micro_MAE", "micro_RMSE", "training_seconds",
               "inference_seconds"], rows)

    CURVES.mkdir(parents=True, exist_ok=True)
    with open(str(CURVES / "internal_candidate_curves.json"), "w",
              encoding="utf-8") as handle:
        json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
        handle.write("\n")

    aggregate = []
    for architecture in ("GRU", "TCN"):
        for candidate in neural_grid.candidates(architecture):
            subset = [r for r in rows
                      if r[1] == candidate["candidate_id"]]
            aggregate.append(OrderedDict([
                ("architecture", architecture),
                ("candidate_id", candidate["candidate_id"]),
                ("capacity", candidate["capacity"]),
                ("learning_rate", candidate["learning_rate"]),
                ("mean_macro_mae", float(np.mean([r[6] for r in subset]))),
                ("mean_macro_rmse", float(np.mean([r[7] for r in subset]))),
                ("regimes_scored", len(subset)),
            ]))
    selected = {}
    for architecture in ("GRU", "TCN"):
        pool = [a for a in aggregate if a["architecture"] == architecture]
        selected[architecture] = neural_grid.select(pool)
    write_csv(OUT / "internal_candidate_selection.csv",
              ["architecture", "candidate_id", "capacity", "learning_rate",
               "regimes_scored", "mean_macro_station_horizon_MAE",
               "mean_macro_RMSE", "selected"],
              [[a["architecture"], a["candidate_id"], a["capacity"],
                a["learning_rate"], a["regimes_scored"],
                round(a["mean_macro_mae"], 6), round(a["mean_macro_rmse"], 6),
                "true" if a["candidate_id"]
                == selected[a["architecture"]]["candidate_id"] else "false"]
               for a in aggregate])
    log("selected GRU %s | TCN %s" % (selected["GRU"]["candidate_id"],
                                      selected["TCN"]["candidate_id"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
