"""
models/tabular_encoder.py
MLP encoder for tabular scalar features: SET / QUA / ENV / CALC / DOS / DRY / SIM
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


class TabularEncoder(nn.Module):
    """
    Three-layer MLP with BatchNorm + Dropout on tabular scalar features.

    Input : (B, n_features)
    Output: (B, TABULAR_EMB_DIM=192)
    """

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()

        out_dim = C.TABULAR_EMB_DIM   # 192

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 3 → output
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
