"""Causal temporal convolutional forecaster for AirSense V2.

Protocol Phase 4, family N2.

Four residual blocks, two kernel-3 convolutions each, dilations 1, 2, 4, 8.
Padding is applied **only on the left**, so position *t* of every layer depends
on inputs at or before *t* and never on a future timestep. There is no
right-padding to trim and therefore no opportunity for the usual off-by-one
leak.

Receptive field:

    1 + sum over layers of (kernel - 1) * dilation
  = 1 + 2*(2*1) + 2*(2*2) + 2*(2*4) + 2*(2*8)
  = 1 + 4 + 8 + 16 + 32
  = 61 timesteps

which covers the whole 48-hour context with room to spare. The value is
asserted programmatically by ``receptive_field()`` and by a unit test that
perturbs a future input and requires the origin-position output to be
bit-identical.

The output is **unconstrained**, as for the GRU: negative predictions remain
visible.
"""

import torch
from torch import nn

KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8]
CONVS_PER_BLOCK = 2


def receptive_field(kernel_size=KERNEL_SIZE, dilations=DILATIONS,
                    convs_per_block=CONVS_PER_BLOCK):
    return 1 + sum((kernel_size - 1) * d * convs_per_block for d in dilations)


class CausalConv1d(nn.Module):
    """Conv1d with explicit left padding only."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.project = (nn.Conv1d(in_channels, channels, 1)
                        if in_channels != channels else None)

    def forward(self, x):
        out = self.drop1(self.relu(self.conv1(x)))
        out = self.drop2(self.relu(self.conv2(out)))
        residual = x if self.project is None else self.project(x)
        return out + residual


class TCNForecaster(nn.Module):

    def __init__(self, dynamic_channels, static_channels, channels=32,
                 head_hidden=64, dropout=0.1, kernel_size=KERNEL_SIZE,
                 dilations=None):
        super().__init__()
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = list(dilations or DILATIONS)
        blocks = []
        in_channels = dynamic_channels
        for dilation in self.dilations:
            blocks.append(ResidualBlock(in_channels, channels, kernel_size,
                                        dilation, dropout))
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels + static_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def receptive_field(self):
        return receptive_field(self.kernel_size, self.dilations,
                               CONVS_PER_BLOCK)

    def forward(self, sequence, static):
        # (N, T, C) -> (N, C, T) for convolution, final position is the origin.
        out = self.blocks(sequence.transpose(1, 2))
        final = out[:, :, -1]
        return self.head(torch.cat([final, static], dim=1)).squeeze(-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
