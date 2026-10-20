"""
analysis/explainability.py
Gradient-based saliency and attention maps for the 4-modal fusion model.

Methods implemented:
  1. GradCAM on ThermalEncoder   → 480×640 heatmap (where the IR camera "looked")
  2. GradCAM on VisualEncoder     → 2448×2048 heatmap per section
  3. Section attention weights    → which of left/mid/right matters most
  4. Modality token weights       → CrossModalFusion's per-modality importance
  5. TCN channel saliency         → which DXP signals drove the decision
  6. ROI temperature importance   → which SKZ region contributes most
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from models.fusion import InjectionMouldingFusionModel


# =============================================================================
# Hook helpers
# =============================================================================

class _ActivationHook:
    """Captures forward activation of a layer."""
    def __init__(self):
        self.activation: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, module: nn.Module):
        self._handle = module.register_forward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, inp, out):
        self.activation = out

    def remove(self):
        if self._handle:
            self._handle.remove()


class _GradHook:
    """Captures gradient w.r.t. a layer's output."""
    def __init__(self):
        self.gradient: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, module: nn.Module):
        self._handle = module.register_full_backward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def remove(self):
        if self._handle:
            self._handle.remove()


# =============================================================================
# GradCAM
# =============================================================================

class GradCAM:
    """
    Standard GradCAM for any nn.Module that outputs spatial feature maps.
    Assumes the target layer outputs (B, C, H, W).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.act_hook    = _ActivationHook().register(target_layer)
        self.grad_hook   = _GradHook().register(target_layer)

    def __call__(
        self,
        input_tensor: torch.Tensor,   # model-specific input
        class_idx   : int,            # label index to explain
        model_forward_fn,             # callable: (model, input_tensor) → logits (B, n_labels)
    ) -> np.ndarray:
        """
        Returns GradCAM heatmap (H, W) normalised to [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(False)

        with torch.enable_grad():
            logits = model_forward_fn(self.model, input_tensor)  # (B, n_labels)
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

        act  = self.act_hook.activation   # (B, C, H, W)
        grad = self.grad_hook.gradient    # (B, C, H, W)

        if act is None or grad is None:
            raise RuntimeError("GradCAM hooks did not fire — check target_layer.")

        weights = grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam     = (weights * act).sum(dim=1).squeeze(0)  # (H, W)
        cam     = F.relu(cam)

        # normalise
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-6:
            cam /= cam.max()
        return cam.astype(np.float32)

    def remove_hooks(self):
        self.act_hook.remove()
        self.grad_hook.remove()


# =============================================================================
# Thermal GradCAM helper
# =============================================================================

def thermal_gradcam(
    model   : InjectionMouldingFusionModel,
    batch   : Dict[str, torch.Tensor],
    device  : torch.device,
    label_idx: int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
) -> np.ndarray:
    """
    Run GradCAM on the last conv block of the ThermalEncoder backbone.

    Returns:
        cam_upsampled : (480, 640) float32 heatmap, values in [0,1]
    """
    # Locate the last convolutional block of EfficientNet-B0
    # timm's EfficientNet: model.thermal_enc.backbone.blocks[-1]
    target_layer = model.thermal_enc.backbone.blocks[-1]

    def forward_fn(m, inp):
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = m(b)
        return out["defect_logits"]

    gcam = GradCAM(model, target_layer)
    cam  = gcam(batch["thermal_img"].to(device), label_idx, forward_fn)
    gcam.remove_hooks()

    # upsample to thermal resolution (480, 640)
    cam_t  = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=(480, 640), mode="bilinear", align_corners=False)
    return cam_up.squeeze().numpy()


# =============================================================================
# Visual GradCAM helper
# =============================================================================

def visual_gradcam(
    model    : InjectionMouldingFusionModel,
    batch    : Dict[str, torch.Tensor],
    device   : torch.device,
    section  : int = 0,   # 0=left, 1=mid, 2=right
    label_idx: int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
) -> np.ndarray:
    """
    GradCAM on the last ResNet-50 conv block for one visual section.

    Returns:
        cam_upsampled : (2448, 2448) float32 heatmap (square crop)
    """
    # ResNet-50 layer4 is the standard GradCAM target
    target_layer = model.visual_enc.backbone.layer4[-1]

    # We need to forward only the selected section through the backbone.
    # The VisualEncoder processes all sections together, so we hook at the
    # shared backbone called per-section via loop.
    # Strategy: replace visual_imgs with only section `section` repeated 3 times,
    # then average — it's an approximation but standard for 3-view architectures.

    def forward_fn(m, inp):
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Isolate section: use same section for all 3 slots to focus saliency
        section_img = b["visual_imgs"][:, section:section+1, :, :, :]   # (B,1,1,H,W)
        b["visual_imgs"] = section_img.expand(-1, 3, -1, -1, -1)
        out = m(b)
        return out["defect_logits"]

    gcam = GradCAM(model, target_layer)
    cam  = gcam(batch["visual_imgs"].to(device), label_idx, forward_fn)
    gcam.remove_hooks()

    # upsample to full visual res (1:1 square crop used after resize)
    cam_t  = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=(C.VISUAL_RESIZE, C.VISUAL_RESIZE),
                           mode="bilinear", align_corners=False)
    return cam_up.squeeze().numpy()


# =============================================================================
# Modality token weights (from CrossModalFusion)
# =============================================================================

def get_modality_weights(
    model  : InjectionMouldingFusionModel,
    batch  : Dict,
    device : torch.device,
) -> np.ndarray:
    """
    Run a forward pass and return the CrossModalFusion attention weights.

    Returns:
        weights : (4,) float32 — [thermal, visual, sequence, tabular]
    """
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)
    # token_weights is (B, 4)
    return out["token_weights"][0].cpu().numpy()


# =============================================================================
# TCN channel saliency (gradient w.r.t. input sequence)
# =============================================================================

def tcn_channel_saliency(
    model    : InjectionMouldingFusionModel,
    batch    : Dict,
    device   : torch.device,
    label_idx: int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
) -> np.ndarray:
    """
    Computes |dL/d_input| summed over time → per-channel importance.

    Returns:
        saliency : (n_channels,) float32 — higher = more important
    """
    model.eval()
    seq = batch["sequence"].to(device).detach()   # (B, n_channels, T)
    seq.requires_grad_(True)

    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    b["sequence"] = seq

    with torch.enable_grad():
        out   = model(b)
        score = out["defect_logits"][0, label_idx]
        model.zero_grad()
        score.backward()

    grad = seq.grad.detach().abs()    # (B, n_channels, T)
    sal  = grad[0].mean(dim=1)        # (n_channels,)
    sal  = sal.cpu().numpy()
    sal  = sal / (sal.sum() + 1e-8)
    return sal


# =============================================================================
# ROI temperature importance (gradient on tabular input)
# =============================================================================

def roi_importance(
    model    : InjectionMouldingFusionModel,
    batch    : Dict,
    device   : torch.device,
    label_idx: int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
) -> np.ndarray:
    """
    Gradient of output w.r.t. the thermal ROI vector (10-dim).

    Returns:
        importance : (10,) float32, normalised sum=1
    """
    model.eval()
    roi = batch["thermal_roi"].to(device).detach()
    roi.requires_grad_(True)

    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    b["thermal_roi"] = roi

    with torch.enable_grad():
        out   = model(b)
        score = out["defect_logits"][0, label_idx]
        model.zero_grad()
        score.backward()

    grad = roi.grad.detach().abs()[0].cpu().numpy()   # (10,)
    grad = grad / (grad.sum() + 1e-8)
    return grad


# =============================================================================
# Visualisation helpers
# =============================================================================

SECTION_NAMES = ["Left", "Mid", "Right"]
MODALITY_NAMES = ["Thermal", "Visual", "Sequence", "Tabular"]

def _overlay_heatmap(
    base_img  : np.ndarray,   # (H, W) float32 or uint8
    cam       : np.ndarray,   # (H, W) float32 [0,1]
    alpha     : float = 0.45,
    cmap      : str   = "jet",
) -> np.ndarray:
    """Overlay GradCAM heatmap on greyscale base image. Returns (H, W, 3) uint8."""
    if base_img.dtype != np.uint8:
        base_img = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base_img, base_img, base_img], axis=2)

    colormap = cm.get_cmap(cmap)
    heatmap  = (colormap(cam)[:, :, :3] * 255).astype(np.uint8)

    overlay = (alpha * heatmap + (1 - alpha) * rgb).astype(np.uint8)
    return overlay


def save_thermal_explanation(
    model      : InjectionMouldingFusionModel,
    batch      : Dict,
    device     : torch.device,
    out_path   : Path,
    label_names: List[str] = C.LABEL_COLS,
    label_idx  : int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
):
    """
    Save a 3-panel thermal explanation figure:
      [Frame1 | Frame2 | GradCAM overlay]
    """
    cam = thermal_gradcam(model, batch, device, label_idx=label_idx)

    imgs = batch["thermal_img"][0].cpu().numpy()   # (3, 480, 640)
    frame1 = imgs[0]
    frame2 = imgs[1]

    # Normalise frames to [0,1] for display
    def norm01(x):
        x = x - x.min()
        return x / (x.max() + 1e-6)

    f1n = norm01(frame1)
    f2n = norm01(frame2)

    overlay = _overlay_heatmap(f1n, cam, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(f1n, cmap="inferno")
    axes[0].set_title("Thermal Frame 1")
    axes[0].axis("off")

    axes[1].imshow(f2n, cmap="inferno")
    axes[1].set_title("Thermal Frame 2")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"GradCAM → {label_names[label_idx]}")
    axes[2].axis("off")

    # ROI importance bar
    roi_imp = roi_importance(model, batch, device, label_idx=label_idx)
    roi_labels = [c.replace("THERMAL_", "") for c in C.THERMAL_ROI_COLS]

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.bar(range(len(roi_imp)), roi_imp, color="steelblue")
    ax2.set_xticks(range(len(roi_imp)))
    ax2.set_xticklabels(roi_labels[:len(roi_imp)], rotation=45, ha="right")
    ax2.set_ylabel("Gradient Importance")
    ax2.set_title("ROI Temperature Importance")
    fig2.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig2.savefig(out_path.parent / (out_path.stem + "_roi.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


def save_visual_explanation(
    model    : InjectionMouldingFusionModel,
    batch    : Dict,
    device   : torch.device,
    out_path : Path,
    label_idx: int = C.LABEL_COLS.index(C.PRIMARY_TARGET),
):
    """
    Save a 2-row figure: [raw section images] / [GradCAM overlays]
    + section attention bar chart.
    """
    section_attn = batch.get("section_attn")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def norm01(x):
        x = x - x.min()
        return x / (x.max() + 1e-6)

    for sec_i in range(3):
        raw_img = batch["visual_imgs"][0, sec_i, 0].cpu().numpy()  # (H, W)
        raw_disp = norm01(raw_img)
        cam = visual_gradcam(model, batch, device, section=sec_i, label_idx=label_idx)

        # resize raw to VISUAL_RESIZE for overlay
        raw_resized = np.array(
            Image.fromarray((raw_disp * 255).astype(np.uint8)).resize(
                (C.VISUAL_RESIZE, C.VISUAL_RESIZE), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0

        overlay = _overlay_heatmap(raw_resized, cam, alpha=0.45)

        axes[0, sec_i].imshow(raw_resized, cmap="gray")
        axes[0, sec_i].set_title(f"{SECTION_NAMES[sec_i]} — Raw")
        axes[0, sec_i].axis("off")

        axes[1, sec_i].imshow(overlay)
        axes[1, sec_i].set_title(f"{SECTION_NAMES[sec_i]} — GradCAM")
        axes[1, sec_i].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # section attention (from last forward pass, stored in model)
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)

    sec_w = out["section_attn"][0].cpu().numpy()   # (3,)
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.bar(SECTION_NAMES, sec_w, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax3.set_ylabel("Attention Weight")
    ax3.set_title("Cross-Section Attention")
    fig3.tight_layout()
    fig3.savefig(out_path.parent / (out_path.stem + "_section_attn.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


def save_fusion_explanation(
    model    : InjectionMouldingFusionModel,
    batch    : Dict,
    device   : torch.device,
    out_path : Path,
    dxp_cols : List[str] = None,
):
    """
    Save a summary figure with:
      - Modality token weights (bar chart)
      - TCN channel saliency (bar chart)
    """
    mod_w = get_modality_weights(model, batch, device)
    tcn_s = tcn_channel_saliency(model, batch, device)

    col_names = dxp_cols if dxp_cols else [f"ch{i}" for i in range(len(tcn_s))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(MODALITY_NAMES, mod_w, color=["#E91E63", "#2196F3", "#4CAF50", "#FF9800"])
    ax1.set_ylabel("Token Attention Weight")
    ax1.set_title("Cross-Modal Fusion: Modality Importance")
    ax1.set_ylim(0, 1)

    ax2.bar(range(len(tcn_s)), tcn_s, color="teal")
    ax2.set_xticks(range(len(tcn_s)))
    ax2.set_xticklabels([c.replace("DXP_", "") for c in col_names], rotation=45, ha="right")
    ax2.set_ylabel("Gradient Saliency (normalised)")
    ax2.set_title("TCN: DXP Channel Importance")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")


# =============================================================================
# Batch explainability runner
# =============================================================================

def explain_sample(
    model    : InjectionMouldingFusionModel,
    batch    : Dict,
    device   : torch.device,
    out_dir  : Path,
    sample_id: str,
    dxp_cols : List[str] = None,
):
    """
    Full explanation suite for one sample — thermal + visual + fusion.
    Saves all figures under out_dir/sample_id/.
    """
    sid_dir = out_dir / sample_id
    sid_dir.mkdir(parents=True, exist_ok=True)

    if batch.get("thermal_valid", torch.zeros(1))[0]:
        save_thermal_explanation(model, batch, device,
                                 sid_dir / "thermal_gradcam.png")

    if batch.get("visual_valid", torch.zeros(1))[0]:
        save_visual_explanation(model, batch, device,
                                sid_dir / "visual_gradcam.png")

    save_fusion_explanation(model, batch, device,
                            sid_dir / "fusion_weights.png",
                            dxp_cols=dxp_cols)
