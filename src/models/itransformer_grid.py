"""Frozen Phase-5 iTransformer candidate grid and selection rule.

Protocol Phase 5. **Frozen before any candidate fit.**

Four candidates - d_model crossed with learning rate, nothing else. Every
other setting is fixed across candidates, including the epoch budget, so no
setting can be chosen by looking at held-out data.

A candidate is scored by the **equal-weight mean of its R1 and R2
internal-tuning macro station-horizon MAEs**, computed on the Phase-4
training-only internal split. The winning configuration is then used unchanged
for both regimes, so neither regime receives a wider search than the other.
"""

from collections import OrderedDict

CANDIDATES = [
    ("IT_C01", 64, 0.0003),
    ("IT_C02", 64, 0.001),
    ("IT_C03", 128, 0.0003),
    ("IT_C04", 128, 0.001),
]

FIXED = OrderedDict([
    ("encoder_layers", 2),
    ("attention_heads", 4),
    ("feedforward_ratio", 4),
    ("activation", "GELU"),
    ("normalization", "LayerNorm"),
    ("decoder", None),
    ("positional_encoding_across_variates", False),
    ("dropout", 0.1),
    ("head_hidden", 64),
    ("epochs", 8),
    ("batch_size", 1024),
    ("optimizer", "AdamW"),
    ("weight_decay", 1e-4),
    ("grad_clip_global_norm", 1.0),
    ("loss", "L1"),
    ("seed", 42),
    ("scheduler", None),
    ("warmup", None),
    ("early_stopping", False),
])

REGIMES = ["R1", "R2"]
SELECTION_CRITERION = ("equal-weight mean over R1/R2 of the internal-tuning "
                       "macro station-horizon MAE")
TIE_BREAK_ORDER = [
    "lower mean macro station-horizon MAE",
    "lower mean macro station-horizon RMSE",
    "smaller d_model",
    "lower learning rate",
    "lower candidate id",
]
COMPARISON_TOLERANCE = 1e-12


def candidates():
    out = []
    for candidate_id, d_model, learning_rate in CANDIDATES:
        entry = OrderedDict([
            ("architecture", "iTransformer"),
            ("candidate_id", candidate_id),
            ("d_model", d_model),
            ("learning_rate", learning_rate),
        ])
        entry.update(FIXED)
        out.append(entry)
    return out


def select(rows):
    def key(row):
        return (round(row["mean_macro_mae"] / COMPARISON_TOLERANCE),
                round(row["mean_macro_rmse"] / COMPARISON_TOLERANCE),
                row["d_model"],
                row["learning_rate"],
                row["candidate_id"])
    return sorted(rows, key=key)[0]
