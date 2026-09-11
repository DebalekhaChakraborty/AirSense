"""Deterministic CPU training for the AirSense Phase-6 station-attention model.

Protocol Phase 6, sections 12 and 26-28.

The grouped representation means one forward pass emits 12 station
predictions, but only the canonical eligible ones exist as samples. The loss is
therefore a **masked L1** over observed canonical labels only: a station slot
with no canonical sample contributes nothing to the gradient, and no missing
label is ever imputed. A group is never dropped merely because one of its 12
station targets is missing.

Everything that could vary between runs is pinned by the Phase-4 determinism
configuration, which this module reuses unchanged: seeds, deterministic
algorithms, thread counts, a seeded permutation generator, no workers, no AMP,
no ``torch.compile``, no GPU kernels.

The budget is fixed in advance - 8 epochs, grouped batch 512, AdamW with
weight decay 1e-4 and global-norm clipping at 1.0, L1 loss, no scheduler, no
warmup, no early stopping - and is identical for SA_R2 and SA_R3.
"""

import time

import numpy as np
import torch

EPOCHS = 8
GROUPED_BATCH_SIZE = 512
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 42


def masked_l1(prediction, labels, mask):
    """Mean absolute error over observed canonical labels only."""
    counted = mask.sum()
    if counted <= 0:
        return prediction.sum() * 0.0
    return ((prediction - labels).abs() * mask).sum() / counted


def train(model, bundle, learning_rate, epochs=EPOCHS,
          batch_size=GROUPED_BATCH_SIZE, seed=SEED):
    """Fixed-budget deterministic training. Returns per-epoch masked L1."""
    groups = bundle.groups
    if groups.labels is None:
        raise ValueError("training requires grouped labels")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=WEIGHT_DECAY)
    generator = np.random.default_rng(seed)
    labels = torch.from_numpy(groups.labels)
    mask = torch.from_numpy(groups.target_observed.astype(np.float32))
    curve = []
    started = time.time()
    model.train()
    for _ in range(epochs):
        order = generator.permutation(groups.size)
        total = 0.0
        counted = 0.0
        for start in range(0, groups.size, batch_size):
            rows = order[start:start + batch_size]
            sequence, static = bundle.batch(rows)
            batch_mask = mask[rows]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(sequence),
                               torch.from_numpy(static))
            loss = masked_l1(prediction, labels[rows], batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            observed = float(batch_mask.sum())
            total += float(loss.detach()) * observed
            counted += observed
        curve.append(total / counted)
    return curve, time.time() - started


def predict(model, bundle, batch_size=GROUPED_BATCH_SIZE):
    """Grouped inference in canonical group order. Never clips the output."""
    model.eval()
    groups = bundle.groups
    out = np.empty((groups.size, 12), dtype=np.float64)
    started = time.time()
    with torch.no_grad():
        for start in range(0, groups.size, batch_size):
            rows = np.arange(start, min(start + batch_size, groups.size))
            sequence, static = bundle.batch(rows)
            out[rows] = model(torch.from_numpy(sequence),
                              torch.from_numpy(static)).numpy()
    return out, time.time() - started


def attention_matrices(model, bundle, batch_size=GROUPED_BATCH_SIZE,
                       selector=None):
    """Mean head-averaged station attention. Descriptive diagnostic only."""
    model.eval()
    groups = bundle.groups
    rows_all = (np.arange(groups.size) if selector is None
                else np.flatnonzero(selector))
    total = np.zeros((12, 12), dtype=np.float64)
    counted = 0
    with torch.no_grad():
        for start in range(0, rows_all.size, batch_size):
            rows = rows_all[start:start + batch_size]
            sequence, _ = bundle.batch(rows)
            weights = model.attention_weights(torch.from_numpy(sequence))
            total += weights.numpy().astype(np.float64).sum(axis=0)
            counted += rows.size
    if counted == 0:
        return np.full((12, 12), np.nan), 0
    return total / counted, counted
