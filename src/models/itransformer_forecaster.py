"""AirSense iTransformer-backbone forecaster.

Protocol Phase 5, family T1.

Implements the core principle of iTransformer (Liu et al., ICLR 2024,
arXiv 2310.06625): **time-series variates are tokens**. Each dynamic channel's
complete 48-hour history is projected into a single token, and self-attention
operates *across variate tokens* rather than across time steps, so the model
attends between variables instead of between hours.

This is an **in-repository adaptation, not a byte-for-byte reproduction** of
the official implementation, and it is named accordingly. The adaptations,
all deliberate:

* a regression head over the encoded **PM2.5 variate token**, concatenated
  with AirSense's frozen static conditioning (station, horizon, origin and
  target calendar) - the paper's head predicts a whole future sequence per
  variate, whereas AirSense predicts one scalar at one canonical horizon;
* **no decoder**, and no future sequence enters the encoder;
* **no positional encoding across variate tokens** - token order is channel
  order, which carries no sequential meaning; temporal order lives inside each
  token's 48-value embedding, exactly as the inverted formulation intends;
* AirSense's frozen 48-hour context and frozen channel order rather than the
  paper's benchmark configurations.

The output is **unconstrained**: no ReLU, softplus or clipping, so negative
predictions stay visible.
"""

import torch
from torch import nn


class ITransformerForecaster(nn.Module):

    def __init__(self, dynamic_channels, static_channels, target_index,
                 d_model=64, layers=2, heads=4, ff_ratio=4, dropout=0.1,
                 head_hidden=64, context_hours=48):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.target_index = target_index
        self.d_model = d_model
        self.layers = layers
        self.heads = heads
        self.context_hours = context_hours
        # One token per variate: its whole history is the token's content.
        self.variate_embedding = nn.Linear(context_hours, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads,
            dim_feedforward=d_model * ff_ratio, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + static_channels, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, sequence, static):
        # (N, T, C) -> (N, C, T): each row is now one variate's full history.
        variates = sequence.transpose(1, 2)
        tokens = self.variate_embedding(variates)      # (N, C, d_model)
        encoded = self.encoder(tokens)                 # attention across C
        target_token = encoded[:, self.target_index, :]
        return self.head(torch.cat([target_token, static], dim=1)).squeeze(-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
