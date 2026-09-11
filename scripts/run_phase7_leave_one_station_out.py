"""Phase-7 leave-one-station-out robustness for SA_R3.

Protocol Phase 7, section 5.

Asks whether the cross-station model learned **transferable regional
structure** or merely memorised each station's identity. For each of the 12
stations in turn, the model is trained with that station's targets removed
from the loss, then evaluated on exactly that station's canonical validation
samples.

The held-out station's *history* remains visible to the shared encoder, since
the station physically exists in the network and its readings are legitimate
inputs for its neighbours' forecasts. What is withheld is every training
**target** for that station, so the model never learns a fitted mapping for
it. This is the "unseen target station" protocol, and the caveat is recorded
rather than glossed: it is a test of target transfer, not of a station the
model has never observed at all.

Nothing else changes - architecture, features, context, epochs, optimizer,
learning rate, batch size and threshold are the frozen Phase-6 values, and
every fold uses seed 42.

Usage:
    .venv-v2/bin/python scripts/run_phase7_leave_one_station_out.py
"""

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
from src.models import station_attention_grid as sa_grid             # noqa
from src.models.neural_features import static_channel_names          # noqa
from src.models.neural_training import configure_determinism         # noqa
from src.models.spatial_grouping import (                            # noqa
    NO_SAMPLE, GuardedNetworkHistory, NetworkBundle, StationGroups,
    build_groups)
from src.models.spatial_training import predict, train               # noqa
from src.models.station_attention_forecaster import (                # noqa
    StationAttentionForecaster)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
OUT = PROJECT_ROOT / "results" / "models" / "robustness"

SEED = 42
HORIZONS = [1, 6, 12, 24]
EXPECTED_SAMPLES = 413148

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


def withhold(groups, station_index):
    """A copy whose loss mask excludes one station's targets entirely."""
    rows = groups.sample_rows.copy()
    rows[:, station_index] = NO_SAMPLE
    return StationGroups(groups.partition, groups.target_indices,
                         groups.horizons, rows, groups.labels)


def main():
    if not (ARTIFACTS / "phase7_candidate_freeze.json").is_file():
        raise SystemExit("the Phase-7 candidate freeze must exist first")
    config = json.load(open(str(SPATIAL / "SA_R3_config.json"),
                            encoding="utf-8"))
    hidden = config["hidden_size"]
    learning_rate = config["learning_rate"]
    batch_size = config["grouped_batch_size"]

    sealed = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=False)
    unsealed = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=True)
    train_groups = build_groups(PROCESSED, "train",
                                labels_from=sealed.train_target_native)
    v_groups = build_groups(PROCESSED, "validation")
    v_bundle = NetworkBundle(v_groups, unsealed)
    statics = len(static_channel_names())
    channels = sealed.dynamic_channel_count

    station = np.array([r["station"] for r in csv.DictReader(
        open(str(PROCESSED / "sample_index" / "validation.csv"),
             encoding="utf-8"))])
    horizon = np.array([int(r["horizon_hours"]) for r in csv.DictReader(
        open(str(PROCESSED / "sample_index" / "validation.csv"),
             encoding="utf-8"))], dtype=np.int32)
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    reference = np.load(str(PRED / "SA_R3_validation.npy")).astype(np.float64)

    log("SA_R3 leave-one-station-out: hidden %d, lr %s, batch %d, seed %d"
        % (hidden, learning_rate, batch_size, SEED))
    rows = []
    horizon_rows = []
    for index, held_out in enumerate(STATIONS):
        configure_determinism()
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        reduced = withhold(train_groups, index)
        bundle = NetworkBundle(reduced, sealed)
        model = StationAttentionForecaster(
            channels, statics, hidden_size=hidden,
            heads=sa_grid.FIXED["attention_heads"],
            attention_dropout=sa_grid.FIXED["attention_dropout"],
            head_hidden=sa_grid.FIXED["head_hidden"],
            head_dropout=sa_grid.FIXED["head_dropout"], cross_station=True)
        curve, seconds = train(model, bundle, learning_rate,
                               batch_size=batch_size, seed=SEED)
        grouped, inference = predict(model, v_bundle, batch_size=batch_size)
        prediction = v_groups.scatter(grouped, EXPECTED_SAMPLES)
        path = OUT / ("SA_R3_loso_%s_validation.npy" % held_out)
        np.save(str(path), prediction.astype(np.float32))

        selector = station == held_out
        cells = M.cell_metrics(actual[selector], prediction[selector],
                               station[selector], horizon[selector])
        macro = M.macro_from_cells(cells)
        severe_mask = selector & (actual > M.SEVERE_THRESHOLD)
        severe = float(np.abs(actual[severe_mask]
                              - prediction[severe_mask]).mean())
        under = float(100.0 * ((actual[severe_mask]
                                - prediction[severe_mask]) > 0).mean())
        base_cells = M.cell_metrics(actual[selector], reference[selector],
                                    station[selector], horizon[selector])
        base_macro = M.macro_from_cells(base_cells)
        base_severe = float(np.abs(actual[severe_mask]
                                   - reference[severe_mask]).mean())
        withheld = int(train_groups.target_observed[:, index].sum())
        rows.append([held_out, index, withheld,
                     int(selector.sum()), int(severe_mask.sum()),
                     round(macro, 6), round(base_macro, 6),
                     round(macro - base_macro, 6),
                     round(100.0 * (macro - base_macro) / base_macro, 4),
                     round(severe, 6), round(base_severe, 6),
                     round(severe - base_severe, 6), round(under, 4),
                     round(seconds, 1), round(inference, 1),
                     round(curve[-1], 6), sha256_of_file(path)])
        for hz in HORIZONS:
            pick = selector & (horizon == hz)
            horizon_rows.append([
                held_out, hz, int(pick.sum()),
                round(float(np.abs(actual[pick] - prediction[pick]).mean()),
                      6),
                round(float(np.abs(actual[pick] - reference[pick]).mean()),
                      6)])
        log("  %-14s held-out MAE %10.6f vs in-training %10.6f  (%+.2f%%)  "
            "severe %8.3f vs %8.3f"
            % (held_out, macro, base_macro,
               100.0 * (macro - base_macro) / base_macro, severe,
               base_severe))
        del model, bundle, reduced

    write_csv(OUT / "leave_one_station_out.csv",
              ["held_out_station", "station_index",
               "training_targets_withheld", "validation_samples",
               "severe_samples", "loso_macro_over_horizon_MAE",
               "in_training_macro_over_horizon_MAE", "absolute_penalty",
               "relative_penalty_pct", "loso_severe_MAE",
               "in_training_severe_MAE", "severe_penalty",
               "loso_severe_underprediction_pct", "training_seconds",
               "inference_seconds", "final_epoch_train_L1",
               "prediction_sha256"], rows)
    write_csv(OUT / "leave_one_station_out_by_horizon.csv",
              ["held_out_station", "horizon", "n", "loso_MAE",
               "in_training_MAE"], horizon_rows)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
