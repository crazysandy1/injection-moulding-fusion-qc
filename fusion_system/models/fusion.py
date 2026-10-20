"""
models/fusion.py
Cross-Modal Transformer Fusion — the core of Sandeep's contribution.

Design:
    1. Each encoder outputs its embedding → projected to FUSION_TOKEN_DIM (256).
    2. A learnable modality embedding is added (like positional encoding but
       for modality identity: thermal, visual, sequence, tabular).
    3. If a modality is missing, its token is replaced with a learnable [MASK]
       token. This lets the model gracefully degrade without N/A values.
    4. A 2-layer Transformer encoder runs over the 4 tokens.
    5. Mean-pool of all (non-masked) tokens → final fused embedding (256-dim).

Domain-Adversarial (DANN) head:
    A gradient-reversal layer is applied before a head that predicts
    MET_ExperimentNumber. By reversing gradients, the encoder learns to
    produce features that are NOT predictive of which experiment was run.
    This breaks the setpoint-memorisation leakage documented in the artifacts.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from models.thermal_encoder  import ThermalEncoder
from models.visual_encoder   import VisualEncoder
from models.tcn_encoder      import TCNEncoder
from models.tabular_encoder  import TabularEncoder


# =============================================================================
# Gradient Reversal Layer (DANN)
# =============================================================================

class GradientReversal(torch.autograd.Function):
    """
    Reverses gradient sign during backward pass, scaled by lambda.
    Used for domain-adversarial training.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lam)


# =============================================================================
# Modality indices (fixed)
# =============================================================================
MOD_THERMAL  = 0
MOD_VISUAL   = 1
MOD_SEQUENCE = 2
MOD_TABULAR  = 3
N_MODALITIES = 4


# =============================================================================
# Cross-Modal Fusion Transformer
# =============================================================================

class CrossModalFusion(nn.Module):
    """
    Fuses 4 modality embeddings via Transformer.

    Args:
        thermal_dim  : output dim of ThermalEncoder
        visual_dim   : output dim of VisualEncoder
        tcn_dim      : output dim of TCNEncoder
        tabular_dim  : output dim of TabularEncoder
        token_dim    : all projected to this dim before Transformer
        n_heads      : Transformer attention heads
        n_layers     : Transformer depth
        dropout      : Transformer dropout

    Output:
        fused_emb   : (B, token_dim) — mean-pooled fused representation
        token_weights: (B, 4) — per-modality attention weights (from last layer)
    """

    def __init__(
        self,
        thermal_dim : int = C.THERMAL_EMB_DIM,
        visual_dim  : int = C.VISUAL_EMB_DIM,
        tcn_dim     : int = C.TCN_EMB_DIM,
        tabular_dim : int = C.TABULAR_EMB_DIM,
        token_dim   : int = C.FUSION_TOKEN_DIM,
        n_heads     : int = C.FUSION_N_HEADS,
        n_layers    : int = C.FUSION_N_LAYERS,
        dropout     : float = C.FUSION_DROPOUT,
    ):
        super().__init__()
        self.token_dim = token_dim

        # ---- modality projections -------------------------------------------
        self.proj_thermal  = nn.Linear(thermal_dim,  token_dim)
        self.proj_visual   = nn.Linear(visual_dim,   token_dim)
        self.proj_sequence = nn.Linear(tcn_dim,      token_dim)
        self.proj_tabular  = nn.Linear(tabular_dim,  token_dim)

        # ---- modality embeddings (learned, like positional encoding) --------
        self.mod_emb = nn.Embedding(N_MODALITIES, token_dim)
        nn.init.normal_(self.mod_emb.weight, std=0.02)

        # ---- [MASK] token for missing modalities ----------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # ---- Transformer encoder -------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm = more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out    = nn.LayerNorm(token_dim)

        # ---- attention weight head for interpretability ---------------------
        self.attn_head = nn.Linear(token_dim, 1)

        # store for explainability access
        self._last_token_weights: Optional[torch.Tensor] = None

    def _build_tokens(
        self,
        thermal_emb : Optional[torch.Tensor],
        visual_emb  : Optional[torch.Tensor],
        tcn_emb     : Optional[torch.Tensor],
        tabular_emb : Optional[torch.Tensor],
        thermal_valid: torch.Tensor,
        visual_valid : torch.Tensor,
        seq_valid    : torch.Tensor,
        tab_valid    : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build token sequence (B, 4, D) with [MASK] for missing modalities.
        Returns: tokens (B, 4, D), valid_mask (B, 4) bool
        """
        B = thermal_valid.shape[0]
        device = thermal_valid.device

        projections = [
            (self.proj_thermal(thermal_emb),   thermal_valid,  MOD_THERMAL),
            (self.proj_visual(visual_emb),      visual_valid,   MOD_VISUAL),
            (self.proj_sequence(tcn_emb),       seq_valid,      MOD_SEQUENCE),
            (self.proj_tabular(tabular_emb),    tab_valid,      MOD_TABULAR),
        ]

        tokens = []
        valid_flags = []

        for proj, valid_mask, mod_idx in projections:
            mod_e = self.mod_emb(torch.tensor(mod_idx, device=device))  # (D,)

            # For each sample: use projected emb if valid, else mask_token
            token = torch.where(
                valid_mask.unsqueeze(1),               # (B, 1)
                proj + mod_e,                          # valid → proj + mod_emb
                self.mask_token.squeeze(0) + mod_e,    # invalid → mask + mod_emb
            )                                          # (B, D)

            tokens.append(token)
            valid_flags.append(valid_mask)

        tokens = torch.stack(tokens, dim=1)             # (B, 4, D)
        valid  = torch.stack(valid_flags, dim=1).float() # (B, 4)
        return tokens, valid

    def forward(
        self,
        thermal_emb  : torch.Tensor,   # (B, thermal_dim)
        visual_emb   : torch.Tensor,   # (B, visual_dim)
        tcn_emb      : torch.Tensor,   # (B, tcn_dim)
        tabular_emb  : torch.Tensor,   # (B, tabular_dim)
        thermal_valid: torch.Tensor,   # (B,) bool
        visual_valid : torch.Tensor,
        seq_valid    : torch.Tensor,
        tab_valid    : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_emb    : (B, token_dim)
            token_weights: (B, 4)  — per-modality importance
        """
        tokens, valid = self._build_tokens(
            thermal_emb, visual_emb, tcn_emb, tabular_emb,
            thermal_valid, visual_valid, seq_valid, tab_valid,
        )

        # Transformer (all 4 tokens always attend to each other)
        out = self.transformer(tokens)   # (B, 4, D)
        out = self.norm_out(out)

        # Weighted pool: learn which tokens matter most
        attn_scores = self.attn_head(out).squeeze(-1)     # (B, 4)
        # mask out invalid tokens before softmax
        attn_scores = attn_scores.masked_fill(valid < 0.5, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 4)

        self._last_token_weights = attn_weights.detach()

        fused = (out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return fused, attn_weights


# =============================================================================
# Full 4-Modal System
# =============================================================================

class InjectionMouldingFusionModel(nn.Module):
    """
    The complete 4-modality fusion model.

    Encoders:
        ThermalEncoder  → 512-dim
        VisualEncoder   → 512-dim
        TCNEncoder      → 256-dim
        TabularEncoder  → 192-dim

    Fusion:
        CrossModalFusion → 256-dim fused embedding

    Heads:
        DefectHead  → 8 LBL_* logits (multi-label BCE)
        DANNHead    → n_experiment_groups logits (gradient reversal)
    """

    def __init__(
        self,
        tabular_in_dim   : int,
        n_dxp_channels   : int = C.TCN_IN_CHANNELS,
        n_experiments    : int = 47,   # number of unique MET_ExperimentNumber values
        dann_lambda      : float = 0.0,
    ):
        super().__init__()
        self.dann_lambda = dann_lambda

        # ---- encoders -------------------------------------------------------
        self.thermal_enc = ThermalEncoder()
        self.visual_enc  = VisualEncoder()
        self.tcn_enc     = TCNEncoder(in_channels=n_dxp_channels)
        self.tab_enc     = TabularEncoder(in_dim=tabular_in_dim)

        # ---- cross-modal fusion ---------------------------------------------
        self.fusion = CrossModalFusion()

        token_dim = C.FUSION_TOKEN_DIM   # 256

        # ---- main defect head -----------------------------------------------
        self.defect_head = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, C.N_LABELS),   # 8 outputs, raw logits
        )

        # ---- DANN domain head (gradient reversal) ---------------------------
        self.dann_head = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_experiments),
        )

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch counter to encoders for backbone unfreezing."""
        self.thermal_enc.set_epoch(epoch)
        self.visual_enc.set_epoch(epoch)

    def set_dann_lambda(self, lam: float) -> None:
        self.dann_lambda = lam

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        # --- validity flags --------------------------------------------------
        t_valid = batch["thermal_valid"].to(device)
        v_valid = batch["visual_valid"].to(device)
        s_valid = batch["sequence_valid"].to(device)
        b_valid = batch["tabular_valid"].to(device)

        # --- encode each modality -------------------------------------------
        thermal_emb = self.thermal_enc(
            batch["thermal_img"].to(device),
            batch["thermal_roi"].to(device),
        )
        visual_emb, sec_attn = self.visual_enc(batch["visual_imgs"].to(device))
        tcn_emb     = self.tcn_enc(batch["sequence"].to(device))
        tabular_emb = self.tab_enc(batch["tabular"].to(device))

        # --- fuse ------------------------------------------------------------
        fused, token_weights = self.fusion(
            thermal_emb, visual_emb, tcn_emb, tabular_emb,
            t_valid, v_valid, s_valid, b_valid,
        )

        # --- main defect head ------------------------------------------------
        defect_logits = self.defect_head(fused)   # (B, 8)

        # --- DANN head -------------------------------------------------------
        if self.dann_lambda > 0 and self.training:
            reversed_fused = grad_reverse(fused, self.dann_lambda)
            dann_logits    = self.dann_head(reversed_fused)
        else:
            dann_logits = self.dann_head(fused)

        return {
            "defect_logits" : defect_logits,        # (B, 8)
            "dann_logits"   : dann_logits,           # (B, n_exp)
            "fused_emb"     : fused,                 # (B, 256) for analysis
            "token_weights" : token_weights,         # (B, 4) modality importance
            "section_attn"  : sec_attn,              # (B, 3) visual section weights
            # individual embeddings for cross-modal consistency check
            "thermal_emb"   : thermal_emb,
            "tcn_emb"       : tcn_emb,
        }
