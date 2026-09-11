"""Frozen Phase-4 neural candidate grids and selection rule.

Protocol Phase 4. **Frozen before any candidate fit.**

Four GRU candidates and four TCN candidates - capacity crossed with learning
rate, nothing else. Everything else is fixed across every candidate, including
the epoch budget, so no setting can be chosen by looking at held-out data.

A candidate is scored by the **equal-weight mean of its three regime-specific
internal-tuning macro station-horizon MAEs**. Scoring the same candidate under
R0, R1 and R2 keeps the search budget identical across regimes: no information
regime gets a wider search than another, and the winning configuration is then
used unchanged for all three.
"""

from collections import OrderedDict

GRU_CANDIDATES = [
    ("GRU_C01", 64, 0.0003),
    ("GRU_C02", 64, 0.001),
    ("GRU_C03", 128, 0.0003),
    ("GRU_C04", 128, 0.001),
]

TCN_CANDIDATES = [
    ("TCN_C01", 32, 0.0003),
    ("TCN_C02", 32, 0.001),
    ("TCN_C03", 64, 0.0003),
    ("TCN_C04", 64, 0.001),
]

FIXED = OrderedDict([
    ("epochs", 8),
    ("batch_size", 1024),
    ("optimizer", "AdamW"),
    ("weight_decay", 1e-4),
    ("grad_clip_global_norm", 1.0),
    ("loss", "L1"),
    ("head_hidden", 64),
    ("dropout", 0.1),
    ("seed", 42),
    ("scheduler", None),
    ("warmup", None),
    ("early_stopping", False),
])

GRU_FIXED = OrderedDict([("num_layers", 1), ("bidirectional", False)])
TCN_FIXED = OrderedDict([("blocks", 4), ("kernel_size", 3),
                         ("dilations", [1, 2, 4, 8])])

REGIMES = ["R0", "R1", "R2"]
SELECTION_CRITERION = ("equal-weight mean over R0/R1/R2 of the internal-tuning "
                       "macro station-horizon MAE")
TIE_BREAK_ORDER = [
    "lower mean macro station-horizon MAE",
    "lower mean macro station-horizon RMSE",
    "smaller architecture capacity",
    "lower learning rate",
    "lower candidate id",
]
COMPARISON_TOLERANCE = 1e-12


def candidates(architecture):
    grid = GRU_CANDIDATES if architecture == "GRU" else TCN_CANDIDATES
    key = "hidden_size" if architecture == "GRU" else "channels"
    extra = GRU_FIXED if architecture == "GRU" else TCN_FIXED
    out = []
    for candidate_id, capacity, learning_rate in grid:
        entry = OrderedDict([
            ("architecture", architecture),
            ("candidate_id", candidate_id),
            ("capacity", capacity),
            ("capacity_parameter", key),
            ("learning_rate", learning_rate),
        ])
        entry.update(extra)
        entry.update(FIXED)
        out.append(entry)
    return out


def select(rows):
    """Frozen tie-break order. `rows` carry the aggregated candidate scores."""
    def key(row):
        return (round(row["mean_macro_mae"] / COMPARISON_TOLERANCE),
                round(row["mean_macro_rmse"] / COMPARISON_TOLERANCE),
                row["capacity"],
                row["learning_rate"],
                row["candidate_id"])
    return sorted(rows, key=key)[0]
