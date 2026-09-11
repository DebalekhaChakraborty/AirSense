"""AirSense Station-Attention Forecaster.

Protocol Phase 6.

A purpose-built controlled spatiotemporal baseline. It is **not** Graph
WaveNet, DCRNN, STGCN, GAT or ASTGCN, does not implement any of them, and is
not claimed to.

The design exists to isolate *information* from *capacity*. One architecture is
evaluated in two modes:

* ``cross_station=False`` (**SA_R2**) - a station may attend only to itself;
* ``cross_station=True``  (**SA_R3**) - a station may attend to all 12.

The attention module, its parameters and every other component are identical
between the two. The self-only mask is a registered **buffer**, not a
parameter, so the two modes have exactly the same parameter count and the same
initialisation from the same seed. The only primary difference is which
stations are reachable.

Structure:

1. a **shared** single-layer unidirectional GRU encodes each station's frozen
   48-hour R2 window; the same weights process all 12 stations, so the model
   cannot smuggle in 12 station-specific temporal models;
2. multi-head attention across the **station** dimension, 4 heads;
3. residual fusion ``Z = LayerNorm(H + A)``, no graph convolution, no second
   attention layer;
4. a shared head over each fused station representation concatenated with the
   frozen static conditioning, producing all 12 station predictions at once.

No station coordinates, no distance, no manually chosen neighbours and no
validation-derived edge enter the model: the relational weighting is learned
from training data alone. The output is **unconstrained** - no ReLU, softplus
or clipping - so negative predictions stay visible.
"""

import torch
from torch import nn

STATION_COUNT = 12


class StationAttentionForecaster(nn.Module):

    def __init__(self, dynamic_channels, static_channels, hidden_size=64,
                 heads=4, attention_dropout=0.1, head_hidden=64,
                 head_dropout=0.1, num_layers=1, stations=STATION_COUNT,
                 cross_station=True):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.hidden_size = hidden_size
        self.heads = heads
        self.stations = stations
        self.num_layers = num_layers
        self.cross_station = bool(cross_station)
        self.encoder = nn.GRU(dynamic_channels, hidden_size,
                              num_layers=num_layers, batch_first=True,
                              bidirectional=False)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=attention_dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_channels, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
        # True marks a forbidden edge. The diagonal is always permitted, so no
        # attention row is ever fully masked.
        self.register_buffer(
            "self_only_mask",
            ~torch.eye(stations, dtype=torch.bool), persistent=False)

    def encode(self, sequence):
        """(B, S, T, C) -> (B, S, D) through the shared temporal encoder."""
        batch, stations, steps, channels = sequence.shape
        flat = sequence.reshape(batch * stations, steps, channels)
        _, hidden = self.encoder(flat)
        return hidden[-1].reshape(batch, stations, self.hidden_size)

    def fuse(self, representations, need_weights=False):
        mask = None if self.cross_station else self.self_only_mask
        attended, weights = self.attention(
            representations, representations, representations,
            attn_mask=mask, need_weights=need_weights,
            average_attn_weights=True)
        return self.norm(representations + attended), weights

    def forward(self, sequence, static):
        representations = self.encode(sequence)
        fused, _ = self.fuse(representations)
        return self.head(torch.cat([fused, static], dim=-1)).squeeze(-1)

    def attention_weights(self, sequence):
        """(B, S, S) head-averaged station attention. Diagnostic only."""
        representations = self.encode(sequence)
        _, weights = self.fuse(representations, need_weights=True)
        return weights

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
