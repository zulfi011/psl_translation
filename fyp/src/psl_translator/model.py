"""
model.py
=========
3-block Temporal Convolutional Network (TCN) for sign-language recognition.
Not run directly — imported by train, evaluate, and realtime scripts.

Architecture
------------
  Input : (batch, 60, 258)   — 60 frames, 258 features/frame

  TCNBlock 1 :  258 → 64  channels
  TCNBlock 2 :   64 → 128 channels
  TCNBlock 3 :  128 → 256 channels

  Each TCNBlock contains:
    Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm
    + residual (skip) connection
    + final ReLU

  AdaptiveAvgPool1d(1)  — collapse time axis → (batch, 256)
  Dropout(p=0.4)        — regularisation before classifier
  Linear(256, num_classes)

  Output: (batch, num_classes)  raw logits

Weight initialisation: PyTorch default (Kaiming uniform for Conv1d) — correct
                       for ReLU networks, no manual init needed.
Activation function  : ReLU throughout.
"""

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    One TCN block: two Conv1d layers with BatchNorm and a residual skip.

    Parameters
    ----------
    in_ch  : input  channels (= feature dimension along the channel axis)
    out_ch : output channels
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()

        # first conv layer
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(out_ch)

        # second conv layer (same spatial size thanks to same-padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.relu  = nn.ReLU()

        # 1×1 conv skip so residual addition always has matching channel count
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        # x : (batch, in_ch, time)
        residual = self.skip(x)                   # skip path

        x = self.relu(self.bn1(self.conv1(x)))    # first conv + BN + ReLU
        x = self.bn2(self.conv2(x))               # second conv + BN

        x = self.relu(x + residual)               # residual add + ReLU
        return x                                  # (batch, out_ch, time)


class SignTCN(nn.Module):
    """
    Full 3-block TCN classifier.

    Parameters
    ----------
    input_size  : int – features per frame (258)
    num_classes : int – number of sign classes
    dropout     : float – dropout probability before the FC layer
    """

    def __init__(self, input_size=258, num_classes=5, dropout=0.4):
        super().__init__()

        # three stacked blocks; channels double at each stage
        self.block1 = TCNBlock(input_size, 64)     # 258 → 64
        self.block2 = TCNBlock(64,  128)           #  64 → 128
        self.block3 = TCNBlock(128, 256)           # 128 → 256

        # global average pooling collapses the time axis into one vector
        self.pool    = nn.AdaptiveAvgPool1d(1)

        # dropout before classifier to prevent overfitting on small dataset
        self.dropout = nn.Dropout(p=dropout)

        # final linear layer: 256-d feature → class logits
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        # x arrives as (batch, time=60, features=258)
        # Conv1d needs (batch, channels, time) → permute
        x = x.permute(0, 2, 1)       # → (batch, 258, 60)

        x = self.block1(x)            # → (batch,  64, 60)
        x = self.block2(x)            # → (batch, 128, 60)
        x = self.block3(x)            # → (batch, 256, 60)

        x = self.pool(x).squeeze(-1)  # → (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)                # → (batch, num_classes)
        return x
