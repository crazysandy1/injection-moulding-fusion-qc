"""
analysis/drift.py
Production drift monitoring for the 4-modal fusion system.

Detects distribution shift BEFORE defects appear, enabling predictive alerts.

Three complementary detectors:
  1. PSI (Population Stability Index) — tabular / thermal-ROI features
  2. KS-test (Kolmogorov-Smirnov) — per-feature statistical significance
  3. Embedding drift — cosine drift on the fused latent space

Alert levels:
  - GREEN  : PSI < 0.1   (negligible drift)
  - YELLOW : 0.1 ≤ PSI < 0.25  (moderate drift, monitor)
  - RED    : PSI ≥ 0.25  (significant drift, alert operator)

Reference: ProBayes dataset uses Mar 2022 PP Taguchi run as nominal production.
The Z01-Diff (frame2 − frame1 thermal) features encode injection dynamics
and degrade slowly with tool wear — ideal early-warning signal.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


# =============================================================================
# PSI
# =============================================================================

def compute_psi(
    reference : np.ndarray,   # 1-D array, training distribution
    production: np.ndarray,   # 1-D array, new production window
    n_bins    : int = 10,
    eps       : float = 1e-8,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = sum[(P_prod - P_ref) * ln(P_prod / P_ref)]

    Interpretation:
        PSI < 0.1  → negligible shift
        0.1 – 0.25 → moderate shift
        > 0.25     → significant shift
    """
    # Use reference quantiles as bin edges
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  -= 1e-6
    bin_edges[-1] += 1e-6

    ref_counts  = np.histogram(reference,  bins=bin_edges)[0].astype(float) + eps
    prod_counts = np.histogram(production, bins=bin_edges)[0].astype(float) + eps

    ref_pct  = ref_counts  / ref_counts.sum()
    prod_pct = prod_counts / prod_counts.sum()

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


def psi_alert_level(psi: float) -> str:
    if psi < 0.1:
        return "GREEN"
    elif psi < 0.25:
        return "YELLOW"
    else:
        return "RED"


# =============================================================================
# KS-test
# =============================================================================

def ks_test_features(
    reference : np.ndarray,   # (N_ref, n_features)
    production: np.ndarray,   # (N_prod, n_features)
    alpha     : float = 0.05,
) -> Dict[int, Dict]:
    """
    Per-feature two-sample KS test.

    Returns dict keyed by feature index:
        {statistic, p_value, drifted}
    """
    results = {}
    n_feats = reference.shape[1]
    for i in range(n_feats):
        stat, pval = stats.ks_2samp(reference[:, i], production[:, i])
        results[i] = {
            "statistic": float(stat),
            "p_value"  : float(pval),
            "drifted"  : bool(pval < alpha),
        }
    return results


# =============================================================================
# Embedding drift
# =============================================================================

def embedding_drift(
    ref_embeddings : np.ndarray,   # (N_ref, D)
    prod_embeddings: np.ndarray,   # (N_prod, D)
) -> Dict:
    """
    Compare mean embedding between reference and production window.

    Metrics:
      - cosine_similarity : cos(μ_ref, μ_prod)
      - euclidean_distance: ||μ_ref - μ_prod||_2
      - mean_l2_shift     : average ||z_prod - μ_ref||_2
    """
    mu_ref  = ref_embeddings.mean(axis=0)
    mu_prod = prod_embeddings.mean(axis=0)

    # cosine
    cos_sim = float(
        np.dot(mu_ref, mu_prod) /
        (np.linalg.norm(mu_ref) * np.linalg.norm(mu_prod) + 1e-8)
    )
    # euclidean
    l2_dist = float(np.linalg.norm(mu_ref - mu_prod))

    # per-sample l2 drift from reference mean
    mean_shift = float(
        np.linalg.norm(prod_embeddings - mu_ref[np.newaxis, :], axis=1).mean()
    )

    alert = "GREEN"
    if 1 - cos_sim > 0.1 or l2_dist > 5.0:
        alert = "YELLOW"
    if 1 - cos_sim > 0.25 or l2_dist > 15.0:
        alert = "RED"

    return {
        "cosine_similarity"  : cos_sim,
        "euclidean_distance" : l2_dist,
        "mean_l2_shift"      : mean_shift,
        "alert"              : alert,
    }


# =============================================================================
# Thermal Z01-Diff drift detector (most sensitive early warning)
# =============================================================================

class ThermalDiffDriftDetector:
    """
    Monitors the Z01-Diff (frame1 – frame2) thermal features.

    The diff image encodes injection dynamics: slow-filling defects (sink marks,
    underfill) appear as cold/hot anomalies in the diff channel.
    Tool wear gradually changes the diff distribution before visible defects.

    Uses a sliding window over recent cycles and compares to the reference
    (first N_REF cycles of the production run).
    """

    N_REF    : int   = 50    # number of reference parts to baseline against
    WINDOW   : int   = 20    # sliding window for production comparison
    ROI_COLS : List[str] = C.THERMAL_ROI_COLS

    def __init__(self, reference_df: pd.DataFrame):
        """
        Args:
            reference_df : subset of parquet (nominal, e.g. first 50 parts of shift)
        """
        avail = [c for c in self.ROI_COLS if c in reference_df.columns]
        self.feat_cols = avail
        self._ref_vals = reference_df[self.feat_cols].dropna().values.astype(np.float32)
        if len(self._ref_vals) == 0:
            raise ValueError("No thermal ROI data in reference_df.")
        self._history: List[np.ndarray] = []   # sliding window

    def update(self, row_vals: np.ndarray) -> Optional[Dict]:
        """
        Add one new cycle's thermal ROI vector and check for drift.

        Args:
            row_vals : (n_roi,) float32 ROI temperatures

        Returns:
            drift report dict if window is full, else None
        """
        self._history.append(row_vals)
        if len(self._history) < self.WINDOW:
            return None

        window = np.stack(self._history[-self.WINDOW:], axis=0)  # (WINDOW, n_roi)
        report = self._assess(window)
        return report

    def _assess(self, window: np.ndarray) -> Dict:
        """Run all drift tests on the current window vs reference."""
        psi_per_feat = {}
        for i, col in enumerate(self.feat_cols):
            ref_col  = self._ref_vals[:, i]
            prod_col = window[:, i]
            psi_per_feat[col] = compute_psi(ref_col, prod_col)

        ks_results = ks_test_functions_for_window(self._ref_vals, window)

        max_psi   = max(psi_per_feat.values())
        alert_lvl = psi_alert_level(max_psi)

        # top-3 drifting features
        sorted_feats = sorted(psi_per_feat.items(), key=lambda x: x[1], reverse=True)

        return {
            "alert"            : alert_lvl,
            "max_psi"          : max_psi,
            "psi_per_feature"  : psi_per_feat,
            "top_drift_features": [f for f, _ in sorted_feats[:3]],
            "ks_results"       : ks_results,
            "window_size"      : self.WINDOW,
        }

    def batch_assess(self, production_df: pd.DataFrame) -> List[Dict]:
        """
        Assess drift over an entire production dataframe in sliding-window fashion.
        Returns a list of drift reports (one per completed window).
        """
        reports = []
        for _, row in production_df.iterrows():
            vals = np.array([
                float(row.get(c, 0.0)) for c in self.feat_cols
            ], dtype=np.float32)
            report = self.update(vals)
            if report is not None:
                reports.append(report)
        return reports


def ks_test_functions_for_window(ref: np.ndarray, prod: np.ndarray) -> Dict:
    """Wrapper: per-feature KS test returning only drifted flags."""
    result = ks_test_features(ref, prod)
    return {
        f"feat_{i}": {"drifted": v["drifted"], "p_value": v["p_value"]}
        for i, v in result.items()
    }


# =============================================================================
# Full drift monitor
# =============================================================================

class DriftMonitor:
    """
    Orchestrates all three drift detectors.

    Usage:
        monitor = DriftMonitor(reference_df, tabular_cols, ref_embeddings)
        report  = monitor.check(production_df, prod_embeddings)
        if report["global_alert"] == "RED":
            send_operator_alert(report)
    """

    def __init__(
        self,
        reference_df     : pd.DataFrame,
        tabular_cols     : List[str],
        ref_embeddings   : Optional[np.ndarray] = None,
    ):
        self.tabular_cols  = tabular_cols
        self.ref_tab       = reference_df[
            [c for c in tabular_cols if c in reference_df.columns]
        ].fillna(0).values.astype(np.float32)
        self.ref_embeddings = ref_embeddings
        self.thermal_detector = ThermalDiffDriftDetector(reference_df)

    def check(
        self,
        production_df     : pd.DataFrame,
        prod_embeddings   : Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full drift check on a production batch.

        Returns global alert level + per-detector breakdown.
        """
        results: Dict = {}

        # --- tabular PSI / KS -----------------------------------------------
        prod_tab = production_df[
            [c for c in self.tabular_cols if c in production_df.columns]
        ].fillna(0).values.astype(np.float32)

        tab_psi = {}
        for i, col in enumerate([c for c in self.tabular_cols if c in production_df.columns]):
            if i < self.ref_tab.shape[1]:
                tab_psi[col] = compute_psi(self.ref_tab[:, i], prod_tab[:, i])

        max_tab_psi = max(tab_psi.values()) if tab_psi else 0.0
        results["tabular"] = {
            "alert"          : psi_alert_level(max_tab_psi),
            "max_psi"        : max_tab_psi,
            "psi_per_feature": tab_psi,
        }

        # --- thermal ROI drift ----------------------------------------------
        thermal_reports = self.thermal_detector.batch_assess(production_df)
        if thermal_reports:
            worst_thermal = max(thermal_reports, key=lambda r: r["max_psi"])
            results["thermal"] = worst_thermal
        else:
            results["thermal"] = {"alert": "GREEN", "max_psi": 0.0, "psi_per_feature": {}}

        # --- embedding drift ------------------------------------------------
        if self.ref_embeddings is not None and prod_embeddings is not None:
            emb_result = embedding_drift(self.ref_embeddings, prod_embeddings)
            results["embedding"] = emb_result
        else:
            results["embedding"] = {"alert": "GREEN"}

        # --- global alert level (worst across all detectors) -----------------
        level_order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        global_alert = max(
            results["tabular"]["alert"],
            results["thermal"]["alert"],
            results["embedding"]["alert"],
            key=lambda a: level_order.get(a, 0),
        )
        results["global_alert"]       = global_alert
        results["n_production_parts"] = len(production_df)

        return results

    def save_report(self, report: Dict, out_path: Path) -> None:
        """Serialise drift report to JSON."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def format_summary(self, report: Dict) -> str:
        """Human-readable one-page summary."""
        g = report["global_alert"]
        colour = {"GREEN": "\033[92m", "YELLOW": "\033[93m", "RED": "\033[91m"}.get(g, "")
        reset  = "\033[0m"

        lines = [
            "=" * 60,
            f"  DRIFT MONITOR REPORT  [{colour}{g}{reset}]",
            f"  Parts checked: {report.get('n_production_parts', '?')}",
            "=" * 60,
        ]

        for name in ["tabular", "thermal", "embedding"]:
            det = report.get(name, {})
            lvl = det.get("alert", "N/A")
            psi = det.get("max_psi", det.get("euclidean_distance", "—"))
            lines.append(f"  {name:12s}  {lvl:8s}  worst={psi}")

        if g != "GREEN":
            lines.append("\nTop drifting tabular features:")
            tab_psi = report.get("tabular", {}).get("psi_per_feature", {})
            for col, psi in sorted(tab_psi.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"  {col:35s}  PSI={psi:.3f}  {psi_alert_level(psi)}")

        lines.append("=" * 60)
        return "\n".join(lines)
