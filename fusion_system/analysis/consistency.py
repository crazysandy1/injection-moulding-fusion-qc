"""
analysis/consistency.py
Cross-modal consistency checker and sensor-fault detector.

Motivation:
  In production, sensors can fail silently:
    - Thermal camera loses calibration  → emb drifts uncorrelated with TCN
    - CV section lamp flickers          → one section emb is noise
    - DXP probe disconnects             → flat-line time series

  The fusion model is robust to *missing* modalities (uses [MASK] token),
  but it cannot detect a *present-but-wrong* modality.

  This module:
    1. Measures cross-modal consistency: cos_sim(thermal_emb, tcn_emb)
    2. Flags single-modality anomalies: embedding Z-score vs reference distribution
    3. Detects flat-line DXP channels: variance threshold
    4. Provides a per-part fault score (0 = fully consistent, 1 = severe mismatch)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from models.fusion import InjectionMouldingFusionModel


# =============================================================================
# Cross-modal cosine consistency
# =============================================================================

def cross_modal_cosine(
    thermal_emb: torch.Tensor,   # (B, D)
    tcn_emb    : torch.Tensor,   # (B, D)
) -> torch.Tensor:
    """
    Per-sample cosine similarity between thermal and sequence embeddings.
    Expected: both embeddings should agree on injection quality.

    Returns:
        similarity : (B,) values in [-1, 1]
    """
    t_norm = F.normalize(thermal_emb, dim=1)
    s_norm = F.normalize(tcn_emb,     dim=1)
    return (t_norm * s_norm).sum(dim=1)


def consistency_score(cos_sim: torch.Tensor) -> torch.Tensor:
    """
    Convert cosine similarity → inconsistency score in [0, 1].
    0 = perfectly consistent, 1 = fully anti-correlated.
    """
    return (1.0 - cos_sim) / 2.0


# =============================================================================
# Embedding anomaly detector (single modality)
# =============================================================================

class EmbeddingAnomalyDetector:
    """
    Fits a reference distribution (mean ± std) from training embeddings
    and flags production embeddings that are statistical outliers.

    Uses Mahalanobis distance (diagonal covariance for efficiency).
    """

    def __init__(self, reference_embeddings: np.ndarray):
        """
        Args:
            reference_embeddings : (N_ref, D) from the training set
        """
        self.mean = reference_embeddings.mean(axis=0)       # (D,)
        self.std  = reference_embeddings.std(axis=0) + 1e-6  # (D,)
        # Store n_sigma threshold calibrated on reference (99th percentile)
        z_scores  = np.abs((reference_embeddings - self.mean) / self.std)
        mahal     = z_scores.mean(axis=1)    # (N_ref,) mean per-dim Z
        self.threshold = float(np.percentile(mahal, 99))

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Returns per-sample anomaly score (mean absolute Z-score).
        Higher = more anomalous.

        Args:
            embeddings : (N, D)

        Returns:
            scores : (N,) float32
        """
        z = np.abs((embeddings - self.mean) / self.std)
        return z.mean(axis=1).astype(np.float32)

    def flag_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Returns boolean array: True = anomalous.
        """
        return self.score(embeddings) > self.threshold

    @classmethod
    def from_model_outputs(
        cls,
        model       : InjectionMouldingFusionModel,
        reference_loader,
        device      : torch.device,
        modality    : str = "thermal",   # "thermal", "visual", "tcn", "tabular", "fused"
    ) -> "EmbeddingAnomalyDetector":
        """
        Collect reference embeddings from a DataLoader and build detector.
        """
        model.eval()
        embs = []
        with torch.no_grad():
            for batch in reference_loader:
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(b)

                if modality == "thermal":
                    e = out["thermal_emb"]
                elif modality == "tcn":
                    e = out["tcn_emb"]
                elif modality == "fused":
                    e = out["fused_emb"]
                else:
                    e = out["fused_emb"]

                embs.append(e.cpu().numpy())
        ref_embs = np.concatenate(embs, axis=0)
        return cls(ref_embs)


# =============================================================================
# DXP flat-line detector
# =============================================================================

def detect_flatline_channels(
    sequence     : np.ndarray,    # (n_channels, T)
    var_threshold: float = 1e-4,
) -> Dict[int, bool]:
    """
    Detects channels with near-zero variance (disconnected probe / constant output).

    Returns:
        {channel_idx: is_flatline}
    """
    channel_vars = sequence.var(axis=1)   # (n_channels,)
    return {i: bool(v < var_threshold) for i, v in enumerate(channel_vars)}


def dxp_quality_score(
    sequence   : np.ndarray,
    dxp_cols   : List[str],
    var_threshold: float = 1e-4,
) -> Dict:
    """
    Full DXP quality assessment: flat-line detection + variance summary.

    Returns dict with per-channel quality flags and overall score.
    """
    flatlines    = detect_flatline_channels(sequence, var_threshold)
    n_flatlines  = sum(flatlines.values())
    channel_vars = sequence.var(axis=1)

    quality_score = 1.0 - (n_flatlines / max(len(dxp_cols), 1))

    return {
        "flatline_flags"  : {dxp_cols[i]: v for i, v in flatlines.items() if i < len(dxp_cols)},
        "n_flatline_ch"   : n_flatlines,
        "channel_variance": {
            dxp_cols[i]: float(channel_vars[i])
            for i in range(min(len(dxp_cols), len(channel_vars)))
        },
        "dxp_quality"     : float(quality_score),
        "is_ok"           : n_flatlines == 0,
    }


# =============================================================================
# Per-part consistency report
# =============================================================================

@torch.no_grad()
def consistency_report(
    model               : InjectionMouldingFusionModel,
    batch               : Dict,
    device              : torch.device,
    thermal_detector    : Optional[EmbeddingAnomalyDetector] = None,
    tcn_detector        : Optional[EmbeddingAnomalyDetector] = None,
    fused_detector      : Optional[EmbeddingAnomalyDetector] = None,
    dxp_cols            : Optional[List[str]] = None,
    cos_alert_threshold : float = 0.3,   # below this → YELLOW
    cos_fault_threshold : float = 0.1,   # below this → RED
) -> Dict:
    """
    Full consistency check for one batch (typically batch_size=1 for per-part analysis).

    Returns a structured report dict ready for logging or API response.
    """
    model.eval()
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    out = model(b)

    # ---- cross-modal cosine ------------------------------------------------
    thermal_emb = out["thermal_emb"]    # (B, D)
    tcn_emb     = out["tcn_emb"]        # (B, D)
    cos_sim_    = cross_modal_cosine(thermal_emb, tcn_emb)
    incons_     = consistency_score(cos_sim_)

    thermal_valid = b["thermal_valid"][0].item()
    seq_valid     = b["sequence_valid"][0].item()

    cross_modal = {}
    if thermal_valid and seq_valid:
        c = float(cos_sim_[0])
        cross_modal = {
            "cosine_similarity"   : c,
            "inconsistency_score" : float(incons_[0]),
            "alert"               : (
                "GREEN" if c >= cos_alert_threshold else
                "YELLOW" if c >= cos_fault_threshold else
                "RED"
            ),
        }
    else:
        cross_modal = {
            "cosine_similarity"   : None,
            "inconsistency_score" : None,
            "alert"               : "N/A (modality missing)",
        }

    # ---- embedding anomaly scores ------------------------------------------
    anomaly = {}
    if thermal_detector and thermal_valid:
        t_score = thermal_detector.score(thermal_emb[0:1].cpu().numpy())[0]
        anomaly["thermal"] = {
            "score"     : float(t_score),
            "is_anomaly": bool(t_score > thermal_detector.threshold),
        }
    if tcn_detector and seq_valid:
        s_score = tcn_detector.score(tcn_emb[0:1].cpu().numpy())[0]
        anomaly["tcn"] = {
            "score"     : float(s_score),
            "is_anomaly": bool(s_score > tcn_detector.threshold),
        }
    if fused_detector:
        f_score = fused_detector.score(out["fused_emb"][0:1].cpu().numpy())[0]
        anomaly["fused"] = {
            "score"     : float(f_score),
            "is_anomaly": bool(f_score > fused_detector.threshold),
        }

    # ---- DXP flat-line check ------------------------------------------------
    dxp_report = {}
    if seq_valid and dxp_cols:
        seq_np = b["sequence"][0].cpu().numpy()  # (n_channels, T)
        dxp_report = dxp_quality_score(seq_np, dxp_cols)

    # ---- modality token weights for context --------------------------------
    token_weights = out["token_weights"][0].cpu().numpy().tolist()

    # ---- overall fault score -----------------------------------------------
    fault_signals = 0
    if cross_modal.get("alert") == "RED":
        fault_signals += 2
    elif cross_modal.get("alert") == "YELLOW":
        fault_signals += 1

    for mod_name, anom in anomaly.items():
        if anom.get("is_anomaly"):
            fault_signals += 1

    if not dxp_report.get("is_ok", True):
        fault_signals += dxp_report.get("n_flatline_ch", 0)

    max_signals = 5
    fault_score = min(fault_signals / max_signals, 1.0)

    return {
        "cross_modal_consistency": cross_modal,
        "embedding_anomaly"      : anomaly,
        "dxp_quality"            : dxp_report,
        "modality_weights"       : {
            "thermal" : token_weights[0],
            "visual"  : token_weights[1],
            "sequence": token_weights[2],
            "tabular" : token_weights[3],
        },
        "fault_score"            : float(fault_score),
        "global_alert"           : (
            "RED"    if fault_score >= 0.6 else
            "YELLOW" if fault_score >= 0.3 else
            "GREEN"
        ),
    }


# =============================================================================
# Batch consistency sweep
# =============================================================================

def batch_consistency_sweep(
    model               : InjectionMouldingFusionModel,
    loader,
    device              : torch.device,
    thermal_detector    : Optional[EmbeddingAnomalyDetector] = None,
    tcn_detector        : Optional[EmbeddingAnomalyDetector] = None,
    fused_detector      : Optional[EmbeddingAnomalyDetector] = None,
    dxp_cols            : Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run consistency_report for every batch in a DataLoader.
    Returns list of per-part reports.
    """
    reports = []
    for batch in loader:
        report = consistency_report(
            model, batch, device,
            thermal_detector=thermal_detector,
            tcn_detector=tcn_detector,
            fused_detector=fused_detector,
            dxp_cols=dxp_cols,
        )
        # attach cycle_id if present
        if "cycle_id" in batch:
            cycle_ids = batch["cycle_id"]
            if isinstance(cycle_ids, (list, tuple)):
                report["cycle_id"] = cycle_ids[0]
            else:
                report["cycle_id"] = str(cycle_ids)
        reports.append(report)
    return reports


def summarise_sweep(reports: List[Dict]) -> Dict:
    """Aggregate consistency sweep into a summary."""
    n = len(reports)
    if n == 0:
        return {}

    alerts = [r.get("global_alert", "GREEN") for r in reports]
    fault_scores = [r.get("fault_score", 0.0) for r in reports]
    cos_sims = [
        r["cross_modal_consistency"].get("cosine_similarity")
        for r in reports
        if r["cross_modal_consistency"].get("cosine_similarity") is not None
    ]
    flatline_parts = [
        r for r in reports
        if not r.get("dxp_quality", {}).get("is_ok", True)
    ]

    return {
        "n_parts"             : n,
        "red_alerts"          : alerts.count("RED"),
        "yellow_alerts"       : alerts.count("YELLOW"),
        "green_alerts"        : alerts.count("GREEN"),
        "mean_fault_score"    : float(np.mean(fault_scores)),
        "max_fault_score"     : float(np.max(fault_scores)),
        "mean_cross_modal_cos": float(np.mean(cos_sims)) if cos_sims else None,
        "min_cross_modal_cos" : float(np.min(cos_sims)) if cos_sims else None,
        "n_flatline_dxp_parts": len(flatline_parts),
        "pct_flatline"        : len(flatline_parts) / n,
    }
