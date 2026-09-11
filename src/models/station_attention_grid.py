"""Frozen Phase-6 station-attention candidate grid and selection rule.

Protocol Phase 6. **Frozen before any candidate fit.**

Four candidates - GRU hidden size crossed with learning rate, nothing else.
Every other setting is fixed across candidates, including the epoch budget, so
no setting can be chosen by looking at held-out data.

A candidate is scored by the **equal-weight mean of its SA_R2 and SA_R3
internal-tuning macro station-horizon MAEs**, computed on the Phase-4
training-only internal split. The winning configuration is then used unchanged
for both modes, so the cross-station mode never receives a wider search than
the self-only control it is measured against.
"""

from collections import OrderedDict

CANDIDATES = [
    ("SA_C01", 64, 0.0003),
    ("SA_C02", 64, 0.001),
    ("SA_C03", 128, 0.0003),
    ("SA_C04", 128, 0.001),
]

FIXED = OrderedDict([
    ("temporal_encoder", "shared unidirectional GRU"),
    ("gru_layers", 1),
    ("bidirectional", False),
    ("attention_axis", "station"),
    ("attention_heads", 4),
    ("attention_dropout", 0.1),
    ("attention_layers", 1),
    ("fusion", "LayerNorm(H + Attention(H))"),
    ("graph_convolution", None),
    ("station_count", 12),
    ("head_hidden", 64),
    ("head_dropout", 0.1),
    ("epochs", 8),
    ("optimizer", "AdamW"),
    ("weight_decay", 1e-4),
    ("grad_clip_global_norm", 1.0),
    ("loss", "L1"),
    ("seed", 42),
    ("scheduler", None),
    ("warmup", None),
    ("early_stopping", False),
    ("output_constrained", False),
    ("external_coordinates", False),
])

REGIMES = ["R2", "R3"]
REGIME_MEANING = OrderedDict([
    ("R2", "self-only station attention: a station may attend only to itself"),
    ("R3", "full cross-station attention over all 12 stations"),
])
SELECTION_CRITERION = ("equal-weight mean over SA_R2/SA_R3 of the "
                       "internal-tuning macro station-horizon MAE")
TIE_BREAK_ORDER = [
    "lower mean macro station-horizon MAE",
    "lower mean macro station-horizon RMSE",
    "smaller hidden size",
    "lower learning rate",
    "lower candidate id",
]
COMPARISON_TOLERANCE = 1e-12


def candidates():
    out = []
    for candidate_id, hidden_size, learning_rate in CANDIDATES:
        entry = OrderedDict([
            ("architecture", "StationAttention"),
            ("candidate_id", candidate_id),
            ("hidden_size", hidden_size),
            ("learning_rate", learning_rate),
        ])
        entry.update(FIXED)
        out.append(entry)
    return out


def select(rows):
    def key(row):
        return (round(row["mean_macro_mae"] / COMPARISON_TOLERANCE),
                round(row["mean_macro_rmse"] / COMPARISON_TOLERANCE),
                row["hidden_size"],
                row["learning_rate"],
                row["candidate_id"])
    return sorted(rows, key=key)[0]
