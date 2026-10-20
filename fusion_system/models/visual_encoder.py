"""
models/visual_encoder.py
3-view CV visual encoder for the warpage shell close-up images.

Each part has 3 section images (left | mid | right of the Verzugsschale).
Architecture:
    Shared ResNet-50 backbone processes each section independently
    → 3 section embeddings of shape (B, 512)
    + learnable position embedding per section
    → cross-section self-attention (which section is most informative?)
    → pooled part-level embedding (B, 512)

Why shared weights?
    All 3 sections show the same material / same defect types under the
    same lighting. Sharing weights massively reduces parameters while
    letting the model specialise via position embeddings.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


class CrossSectionAttention(nn.Module):
    """
    Self-attention over 3 section tokens → weighted pool to one vector.

    Input:  (B, 3, D)   — 3 section embeddings
    Output: (B, D)      — attended part embedding
            (B, 3)      — attention weights (for explainability / Grad-CAM)
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        # learnable [CLS] token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)             # (B, 4, D)
        x   = self.norm(x)
        out, weights = self.attn(x, x, x)            # (B, 4, D), (B, 4, 4)
        cls_out = out[:, 0]                           # (B, D)  — CLS token output
        sec_weights = weights[:, 0, 1:]               # (B, 3)  — CLS→section weights
        return cls_out, sec_weights


class VisualEncoder(nn.Module):
    """
    3-view visual encoder for the injection-moulded warpage shell.

    Input:
        imgs : (B, 3, 1, H, W)  — 3 greyscale sections, each (1, H, W)
    Output:
        embedding    : (B, VISUAL_EMB_DIM=512)
        section_attn : (B, 3)  — per-section attention weights
    """

    SECTION_DIM = 512       # output dim per section from backbone
    FINAL_DIM   = C.VISUAL_EMB_DIM   # 512

    def __init__(self, freeze_backbone_epochs: int = 3):
        super().__init__()

        # ---- shared ResNet-50 backbone -------------------------------------
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Adapt first conv to accept 1-channel greyscale
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Initialise: average ImageNet RGB weights across channels
        with torch.no_grad():
            base.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # strip the ResNet classifier → keeps avgpool + all conv layers
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)

        # project 2048 → 512
        self.proj = nn.Sequential(
            nn.Flatten(),                           # (B, 2048)
            nn.Linear(2048, self.SECTION_DIM),
            nn.BatchNorm1d(self.SECTION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # ---- position embedding per section (1, 2, 3) ----------------------
        self.pos_emb = nn.Embedding(3, self.SECTION_DIM)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # ---- cross-section attention → final embedding ----------------------
        self.cross_attn = CrossSectionAttention(self.SECTION_DIM, n_heads=4)
        self.out_proj   = nn.Sequential(
            nn.Linear(self.SECTION_DIM, self.FINAL_DIM),
            nn.LayerNorm(self.FINAL_DIM),
            nn.ReLU(inplace=True),
        )

        self._freeze_backbone_epochs = freeze_backbone_epochs
        self._current_epoch          = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        for p in self.backbone.parameters():
            p.requires_grad = (epoch >= self._freeze_backbone_epochs)

    def _encode_section(self, img: torch.Tensor, pos_idx: int) -> torch.Tensor:
        """Encode a single section image. img: (B, 1, H, W)"""
        feat = self.backbone(img).squeeze(-1).squeeze(-1)   # (B, 2048)
        emb  = self.proj(feat)                               # (B, 512)
        emb  = emb + self.pos_emb(
            torch.tensor(pos_idx, device=emb.device)
        )
        return emb   # (B, 512)

    def forward(self, imgs: torch.Tensor):
        """
        imgs: (B, 3, 1, H, W)
              dim-1 indexes section (0=left, 1=mid, 2=right)
        """
        sections = []
        for i in range(3):
            sec_img = imgs[:, i]            # (B, 1, H, W)
            sec_emb = self._encode_section(sec_img, i)  # (B, 512)
            sections.append(sec_emb)

        tokens = torch.stack(sections, dim=1)   # (B, 3, 512)
        pooled, sec_attn = self.cross_attn(tokens)      # (B, 512), (B, 3)
        out = self.out_proj(pooled)                      # (B, 512)
        return out, sec_attn
