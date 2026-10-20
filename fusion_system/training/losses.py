"""
training/losses.py
All loss functions used in the fusion system.

  - MultiLabelFocalLoss     : primary defect detection loss
  - DANNLoss                : cross-entropy for domain-adversarial head
  - SensorConsistencyLoss   : MSE between thermal and TCN embeddings (cosine)
  - TotalLoss               : weighted sum of all above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


class MultiLabelFocalLoss(nn.Module):
    """
    Focal loss for multi-label binary classification.
    Addresses the extreme class imbalance in LBL_* labels.

    FL(p) = -alpha * (1 - p)^gamma * log(p)   for positives
    FL(p) = -(1 - alpha) * p^gamma * log(1-p) for negatives
    """

    def __init__(
        self,
        pos_weight : torch.Tensor = None,   # per-label class weight (B,)
        alpha      : float = C.FOCAL_ALPHA,
        gamma      : float = C.FOCAL_GAMMA,
        reduction  : str   = "mean",
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, n_labels) raw logits
        targets : (B, n_labels) float 0/1
        """
        probs = torch.sigmoid(logits)
        # standard BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )   # (B, n_labels)

        # focal weight: (1 - p_t)^gamma
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DANNLoss(nn.Module):
    """
    Cross-entropy loss for the domain-adversarial (experiment-number) head.
    The gradient reversal is applied in the model; here we just compute CE.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, experiment_labels: torch.Tensor) -> torch.Tensor:
        """
        logits             : (B, n_experiments)
        experiment_labels  : (B,) int64
        """
        return self.ce(logits, experiment_labels)


class SensorConsistencyLoss(nn.Module):
    """
    Encourages agreement between thermal and TCN embeddings.

    For parts where BOTH modalities are available:
        loss = 1 - cosine_similarity(thermal_emb, tcn_emb)

    For parts where one modality is missing: skip (no loss).
    This acts as a self-supervised signal and also trains the system
    to detect cross-modal disagreement at inference time.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        thermal_emb  : torch.Tensor,   # (B, D)
        tcn_emb      : torch.Tensor,   # (B, D)
        thermal_valid: torch.Tensor,   # (B,) bool
        seq_valid    : torch.Tensor,   # (B,) bool
    ) -> torch.Tensor:
        both_valid = thermal_valid & seq_valid   # (B,)
        if not both_valid.any():
            return torch.tensor(0.0, device=thermal_emb.device, requires_grad=True)

        t_sel = F.normalize(thermal_emb[both_valid], dim=1)
        s_sel = F.normalize(tcn_emb[both_valid],     dim=1)
        cos_sim = (t_sel * s_sel).sum(dim=1)       # (n_valid,)
        return (1.0 - cos_sim).mean()


class TotalLoss(nn.Module):
    """
    Weighted combination of all losses.

    total = λ_det * focal
          + λ_dann * dann       (only if dann_lambda > 0)
          + λ_cons * consistency
    """

    def __init__(
        self,
        pos_weight       : torch.Tensor = None,
        n_experiments    : int   = 47,
        lambda_detection : float = 1.0,
        lambda_dann      : float = 0.0,   # starts at 0, ramped up during training
        lambda_consistency: float = 0.1,
    ):
        super().__init__()
        self.lambda_det  = lambda_detection
        self.lambda_dann = lambda_dann
        self.lambda_cons = lambda_consistency

        self.focal_loss = MultiLabelFocalLoss(pos_weight=pos_weight)
        self.dann_loss  = DANNLoss()
        self.cons_loss  = SensorConsistencyLoss()

    def set_dann_lambda(self, lam: float) -> None:
        self.lambda_dann = lam

    def forward(
        self,
        outputs        : Dict[str, torch.Tensor],
        labels         : torch.Tensor,
        experiment_ids : torch.Tensor,
        thermal_valid  : torch.Tensor,
        seq_valid      : torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # 1. Multi-label focal loss (main)
        det_loss = self.focal_loss(outputs["defect_logits"], labels)

        # 2. DANN loss
        if self.lambda_dann > 0:
            dann_loss = self.dann_loss(outputs["dann_logits"], experiment_ids)
        else:
            dann_loss = torch.tensor(0.0, device=det_loss.device)

        # 3. Cross-modal consistency loss
        cons_loss = self.cons_loss(
            outputs["thermal_emb"], outputs["tcn_emb"],
            thermal_valid, seq_valid,
        )

        total = (
            self.lambda_det  * det_loss
            + self.lambda_dann * dann_loss
            + self.lambda_cons * cons_loss
        )

        return {
            "total"      : total,
            "detection"  : det_loss.detach(),
            "dann"       : dann_loss.detach(),
            "consistency": cons_loss.detach(),
        }


def compute_pos_weights(df, device: torch.device) -> torch.Tensor:
    """Compute per-label positive weights from training set label frequencies."""
    weights = []
    for col in C.LABEL_COLS:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        w = n_neg / (n_pos + 1e-6)
        weights.append(float(w))
    return torch.tensor(weights, dtype=torch.float32, device=device)
