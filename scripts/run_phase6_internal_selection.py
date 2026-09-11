"""Phase-6 station-attention internal architecture selection, training-only.

Protocol Phase 6, section 31.

Trains each of the four frozen candidates on the Phase-4 internal **core**
period and scores it on the internal **tuning** period, under SA_R2
(self-only attention) and SA_R3 (full cross-station attention) - 8 fits. Both
periods lie inside the frozen Phase-2 training partition, so the external
development validation set is not read at all here.

SA_R2 and SA_R3 differ only in which stations are reachable. Identical
architecture, identical parameter count, identical seed, identical budget.

Usage:
    .venv-v2/bin/python scripts/run_phase6_internal_selection.py
"""

import csv
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

from src.data.preprocessing import STATIONS                          # noqa
from src.evaluation import metrics as M                              # noqa
from src.models import station_attention_grid as grid                # noqa
from src.models.neural_data import (                                 # noqa
    INTERNAL_CORE, INTERNAL_TUNING, internal_mask)
from src.models.neural_features import static_channel_names          # noqa
from src.models.neural_training import configure_determinism         # noqa
from src.models.spatial_grouping import (                            # noqa
    GuardedNetworkHistory, NetworkBundle, StationGroups, build_groups)
from src.models.spatial_training import predict, train               # noqa
from src.models.station_attention_forecaster import (                # noqa
    StationAttentionForecaster)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
OUT = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
CURVES = OUT / "training_curves"
BENCHMARK = PROJECT_ROOT / "artifacts" / "phase6_runtime_benchmark.json"

START = time.time()


def log(message):
    print("[%8.1fs] %s" % (time.time() - START, message), flush=True)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def subset(groups, selector):
    return StationGroups(groups.partition, groups.target_indices[selector],
                         groups.horizons[selector],
                         groups.sample_rows[selector],
                         None if groups.labels is None
                         else groups.labels[selector])


def flatten(groups, grouped):
    """Grouped (G, 12) values into canonical sample-shaped arrays."""
    observed = groups.target_observed
    station_index = np.tile(np.arange(12), (groups.size, 1))[observed]
    horizon = np.repeat(groups.horizons[:, None], 12, axis=1)[observed]
    return (np.array(STATIONS)[station_index], horizon.astype(np.int32),
            groups.labels[observed].astype(np.float64),
            np.asarray(grouped)[observed].astype(np.float64))


def main():
    runtime = configure_determinism()
    log("runtime %s" % json.dumps(runtime))
    with open(str(BENCHMARK), encoding="utf-8") as handle:
        batch_size = json.load(handle)["frozen_grouped_batch_size"]
    log("frozen grouped batch size %d" % batch_size)

    history = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=False)
    log("canonical layer loaded; external validation NOT unsealed")
    all_groups = build_groups(PROCESSED, "train",
                              labels_from=history.train_target_native)
    core = subset(all_groups, internal_mask(all_groups.target_indices,
                                            INTERNAL_CORE))
    tuning = subset(all_groups, internal_mask(all_groups.target_indices,
                                              INTERNAL_TUNING))
    log("internal core %d groups (%d samples) | tuning %d groups (%d samples)"
        % (core.size, core.sample_total(), tuning.size,
           tuning.sample_total()))
    core_bundle = NetworkBundle(core, history)
    tune_bundle = NetworkBundle(tuning, history)
    channels = history.dynamic_channel_count
    statics = len(static_channel_names())

    rows = []
    curves = {}
    for candidate in grid.candidates():
        for regime in grid.REGIMES:
            configure_determinism()
            model = StationAttentionForecaster(
                channels, statics, hidden_size=candidate["hidden_size"],
                heads=candidate["attention_heads"],
                attention_dropout=candidate["attention_dropout"],
                head_hidden=candidate["head_hidden"],
                head_dropout=candidate["head_dropout"],
                cross_station=(regime == "R3"))
            curve, seconds = train(model, core_bundle,
                                   candidate["learning_rate"],
                                   epochs=candidate["epochs"],
                                   batch_size=batch_size)
            grouped, inference_seconds = predict(model, tune_bundle,
                                                 batch_size=batch_size)
            station, horizon, actual, prediction = flatten(tuning, grouped)
            cells = M.cell_metrics(actual, prediction, station, horizon)
            macro_mae = M.macro_from_cells(cells)
            macro_rmse = M.macro_from_cells(cells, "rmse")
            rows.append([
                candidate["candidate_id"], regime, candidate["hidden_size"],
                candidate["learning_rate"], model.parameter_count(),
                round(macro_mae, 6), round(macro_rmse, 6),
                round(M.mae(actual, prediction), 6),
                round(M.rmse(actual, prediction), 6), round(seconds, 1),
                round(inference_seconds, 1)])
            curves["%s|%s" % (candidate["candidate_id"], regime)] = [
                round(v, 6) for v in curve]
            log("  %s %s macro-MAE %.6f (%d params, train %.0fs, infer %.0fs)"
                % (candidate["candidate_id"], regime, macro_mae,
                   model.parameter_count(), seconds, inference_seconds))
            del model

    write_csv(OUT / "internal_candidate_metrics.csv",
              ["candidate_id", "regime", "hidden_size", "learning_rate",
               "parameter_count", "internal_macro_station_horizon_MAE",
               "internal_macro_RMSE", "micro_MAE", "micro_RMSE",
               "training_seconds", "inference_seconds"], rows)
    CURVES.mkdir(parents=True, exist_ok=True)
    with open(str(CURVES / "internal_candidate_curves.json"), "w",
              encoding="utf-8") as handle:
        json.dump(OrderedDict(sorted(curves.items())), handle, indent=2)
        handle.write("\n")

    aggregate = []
    for candidate in grid.candidates():
        picked = [r for r in rows if r[0] == candidate["candidate_id"]]
        aggregate.append(OrderedDict([
            ("candidate_id", candidate["candidate_id"]),
            ("hidden_size", candidate["hidden_size"]),
            ("learning_rate", candidate["learning_rate"]),
            ("mean_macro_mae", float(np.mean([r[5] for r in picked]))),
            ("mean_macro_rmse", float(np.mean([r[6] for r in picked]))),
            ("regimes_scored", len(picked)),
        ]))
    winner = grid.select(aggregate)
    write_csv(OUT / "internal_candidate_selection.csv",
              ["candidate_id", "hidden_size", "learning_rate",
               "regimes_scored", "mean_macro_station_horizon_MAE",
               "mean_macro_RMSE", "selected"],
              [[a["candidate_id"], a["hidden_size"], a["learning_rate"],
                a["regimes_scored"], round(a["mean_macro_mae"], 6),
                round(a["mean_macro_rmse"], 6),
                "true" if a["candidate_id"] == winner["candidate_id"]
                else "false"] for a in aggregate])
    log("selected %s (hidden %d, lr %s)"
        % (winner["candidate_id"], winner["hidden_size"],
           winner["learning_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
