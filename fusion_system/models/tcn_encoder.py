"""
models/tcn_encoder.py
Adapted from Darshan's mini-project-algo (TCNDefectClassifier).

Changes:
  - Classifier head removed → exposes the pooled 256-dim embedding
  - SE weights stored for attention-based explainability
  - Scalar features NOT handled here (done in TabularEncoder instead)
  - Output: (B, TCN_EMB_DIM=256) sequence embedding only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


class CausalConv1d(nn.Module):
    """
    Strictly causal 1-D convolution.
    Left-only padding ensures the model cannot see future timesteps.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel,
            dilation=dilation, padding=0,
        )
        nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SqueezeExcitation(nn.Module):
    """
    Channel attention module.
    Input / output: (B, C, T)
    Stores last weights for inspection / Grad-CAM.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.last_se_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)                          # (B, C)  global avg pool
        s = F.relu(self.fc1(s))
        w = torch.sigmoid(self.fc2(s))             # (B, C)
        self.last_se_weights = w.detach()
        return x * w.unsqueeze(2)                  # broadcast over time


class TCNBlock(nn.Module):
    """
    Two causal dilated convolutions + SE-attention + residual connection.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.se    = SqueezeExcitation(out_ch)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)

        # 1×1 conv for residual when channels change
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.se(out)
        return F.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network sequence encoder.

    Input : (B, in_channels, seq_len)  — resampled DXP time series
    Output: (B, TCN_EMB_DIM=256)       — pooled sequence embedding
    """

    def __init__(
        self,
        in_channels : int  = C.TCN_IN_CHANNELS,
        channel_list: list = C.TCN_CHANNEL_LIST,   # [64, 128, 256]
        kernel_size : int  = C.TCN_KERNEL_SIZE,
        dilations   : list = C.TCN_DILATIONS,
        dropout     : float= C.TCN_DROPOUT,
    ):
        super().__init__()

        # ---- build TCN blocks with exponential dilation --------------------
        blocks = []
        in_ch = in_channels
        for out_ch in channel_list:
            for d in dilations:
                blocks.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
                in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)

        # ---- project final channel to embedding dim -----------------------
        final_ch = channel_list[-1]
        self.proj = nn.Sequential(
            nn.Linear(final_ch, C.TCN_EMB_DIM),
            nn.LayerNorm(C.TCN_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    @property
    def receptive_field(self) -> int:
        k = C.TCN_KERNEL_SIZE
        total_dilation = sum(
            d for _ in C.TCN_CHANNEL_LIST for d in C.TCN_DILATIONS
        ) * 2   # × 2 for two convs per block
        return 1 + (k - 1) * total_dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        Returns: (B, TCN_EMB_DIM)
        """
        feat = self.tcn(x)              # (B, 256, T)
        pooled = feat.mean(dim=2)       # global average pool over time → (B, 256)
        return self.proj(pooled)        # (B, 256)
