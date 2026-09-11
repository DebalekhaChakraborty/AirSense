"""GRU temporal forecaster for AirSense V2.

Protocol Phase 4, family N1.

A single-layer unidirectional GRU over the canonical 48-hour window. The final
hidden state - the representation after the forecast origin's timestep, and
never after it - is concatenated with the static conditioning vector and passed
through a small head.

The output is **unconstrained**. No ReLU, softplus or clipping is applied, so
negative predictions stay visible rather than being silently repaired; a
non-negative head was not preregistered and must not appear after seeing
performance.
"""

import torch
from torch import nn


class GRUForecaster(nn.Module):

    def __init__(self, dynamic_channels, static_channels, hidden_size=64,
                 head_hidden=64, dropout=0.1, num_layers=1):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=dynamic_channels,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, sequence, static):
        output, hidden = self.gru(sequence)
        final = hidden[-1]
        return self.head(torch.cat([final, static], dim=1)).squeeze(-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
