"""
models/thermal_encoder.py
Fixed and improved version of Ashwin's EfficientNet-B0 thermal branch.

Changes vs. Ashwin's original:
  1. Input: float32 CSV tensors (not PNG) — preserves sub-degree precision
  2. 3-channel input: [frame1, frame2, frame1−frame2] — physics-meaningful
  3. Correct thermal normalization (not ImageNet stats)
  4. Parallel ROI physics MLP head
  5. Output: single 512-dim embedding (fused CNN + ROI)
  6. Group-aware split enforced at Dataset level (not here)
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


class ROIPhysicsHead(nn.Module):
    """
    Small MLP that processes the 10 SKZ ROI temperature features:
    [Img1Temp{Sprue,Dome,EdgeHor,EdgeVer,Full}, Img2Temp{same}]

    Output: 64-dim embedding fused into the main thermal embedding.
    """

    def __init__(self, in_dim: int = 10, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        return self.net(roi)


class ThermalEncoder(nn.Module):
    """
    EfficientNet-B0 backbone adapted for 3-channel thermal input.

    Architecture:
        Input  : (B, 3, 224, 224)  float32 thermal stack
        Backbone: EfficientNet-B0  → (B, 1280)
        CNN head: Linear(1280→448) + ReLU + Dropout
        ROI head: MLP(10→64)
        Fusion  : concat [448 + 64] → Linear(512→512)
        Output : (B, 512)  thermal embedding
    """

    CNN_HIDDEN  = 448
    ROI_OUT     = 64
    FINAL_DIM   = C.THERMAL_EMB_DIM   # 512

    def __init__(self, freeze_backbone_epochs: int = 3):
        super().__init__()

        # ---- backbone -------------------------------------------------------
        # timm handles the 3-channel → thermal adaptation automatically;
        # we replace the classifier with identity to get raw features.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,          # removes the built-in head → outputs (B,1280)
            in_chans=3,
        )
        backbone_out = 1280         # EfficientNet-B0 feature dim

        # ---- CNN head -------------------------------------------------------
        self.cnn_head = nn.Sequential(
            nn.Linear(backbone_out, self.CNN_HIDDEN),
            nn.BatchNorm1d(self.CNN_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # ---- ROI physics head -----------------------------------------------
        self.roi_head = ROIPhysicsHead(
            in_dim=len(C.THERMAL_ROI_COLS),
            hidden=64,
            out_dim=self.ROI_OUT,
        )

        # ---- fusion linear --------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(self.CNN_HIDDEN + self.ROI_OUT, self.FINAL_DIM),
            nn.LayerNorm(self.FINAL_DIM),
            nn.ReLU(inplace=True),
        )

        # optional: freeze backbone for first N epochs
        self._freeze_backbone_epochs = freeze_backbone_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        for p in self.backbone.parameters():
            p.requires_grad = (epoch >= self._freeze_backbone_epochs)

    def forward(
        self,
        img: torch.Tensor,          # (B, 3, H, W) float32
        roi: torch.Tensor,          # (B, 10) ROI physics features
    ) -> torch.Tensor:
        cnn_feat = self.backbone(img)          # (B, 1280)
        cnn_emb  = self.cnn_head(cnn_feat)     # (B, 448)
        roi_emb  = self.roi_head(roi)          # (B, 64)
        fused    = torch.cat([cnn_emb, roi_emb], dim=1)   # (B, 512)
        return self.fusion(fused)              # (B, 512)
