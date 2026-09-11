"""Deterministic CPU training for AirSense V2 temporal neural baselines.

Protocol Phase 4.

Everything that could vary between runs is pinned: Python, NumPy and torch
seeds; deterministic algorithms; a fixed thread count; a seeded permutation
generator; no workers; no AMP; no ``torch.compile``; no GPU kernels. Training
is float32 and metric accumulation is float64.

The training budget is **fixed in advance** - 8 epochs, batch 1024, AdamW with
weight decay 1e-4 and global-norm gradient clipping at 1.0, L1 loss, no
scheduler, no warmup, no early stopping. Every candidate receives exactly the
same budget, so no epoch count is ever chosen by looking at held-out data.
"""

import random
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from src.models.neural_features import CONTEXT_HOURS, gather_windows

SEED = 42
EPOCHS = 8
BATCH_SIZE = 1024
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_THREADS = 16
LOSS = "L1"
OPTIMIZER = "AdamW"


def configure_determinism(threads=NUM_THREADS):
    """Pin every seed and thread setting. Returns what was actually set."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    notes = []
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = True
    except Exception as error:                            # noqa: BLE001
        deterministic = False
        notes.append("use_deterministic_algorithms unavailable: %s" % error)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
        interop = 1
    except RuntimeError as error:
        interop = torch.get_num_interop_threads()
        notes.append("interop threads already initialised: %s" % error)
    return OrderedDict([
        ("seed", SEED),
        ("deterministic_algorithms", deterministic),
        ("num_threads", torch.get_num_threads()),
        ("num_interop_threads", interop),
        ("device", "cpu"),
        ("dtype", "float32"),
        ("dataloader_workers", 0),
        ("amp", False),
        ("torch_compile", False),
        ("notes", notes),
    ])


class SampleBundle(object):
    """Canonical samples plus the arrays needed to build their windows."""

    def __init__(self, station_codes, horizons, origin_indices,
                 target_indices, dynamic_by_station, static, labels=None):
        self.station_codes = np.asarray(station_codes, dtype=np.int64)
        self.horizons = np.asarray(horizons, dtype=np.int64)
        self.origin_indices = np.asarray(origin_indices, dtype=np.int64)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.dynamic_by_station = dynamic_by_station
        self.static = np.asarray(static, dtype=np.float32)
        self.labels = (None if labels is None
                       else np.asarray(labels, dtype=np.float32))
        self.size = self.station_codes.size

    def batch(self, rows):
        sequence = gather_windows(self.dynamic_by_station,
                                  self.station_codes[rows],
                                  self.origin_indices[rows])
        if not np.isfinite(sequence).all():
            raise ValueError("a window contains a non-finite value; a sealed "
                             "or unavailable position was reached")
        return (torch.from_numpy(np.ascontiguousarray(sequence)),
                torch.from_numpy(np.ascontiguousarray(self.static[rows])))


def train(model, bundle, learning_rate, epochs=EPOCHS,
          batch_size=BATCH_SIZE, seed=SEED):
    """Fixed-budget deterministic training. Returns per-epoch mean L1 loss."""
    if bundle.labels is None:
        raise ValueError("training requires labels")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    generator = np.random.default_rng(seed)
    labels = torch.from_numpy(bundle.labels)
    curve = []
    started = time.time()
    model.train()
    for epoch in range(epochs):
        order = generator.permutation(bundle.size)
        total = 0.0
        counted = 0
        for start in range(0, bundle.size, batch_size):
            rows = order[start:start + batch_size]
            sequence, static = bundle.batch(rows)
            target = labels[rows]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence, static)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach()) * rows.size
            counted += rows.size
        curve.append(total / counted)
    return curve, time.time() - started


def predict(model, bundle, batch_size=BATCH_SIZE):
    """Inference in canonical sample order. Never clips the output."""
    model.eval()
    out = np.empty(bundle.size, dtype=np.float64)
    started = time.time()
    with torch.no_grad():
        for start in range(0, bundle.size, batch_size):
            rows = np.arange(start, min(start + batch_size, bundle.size))
            sequence, static = bundle.batch(rows)
            out[rows] = model(sequence, static).numpy().astype(np.float64)
    return out, time.time() - started
