"""
analysis/counterfactual.py
Counterfactual / nearest-OK recommender for injection moulding setpoints.

Causal model:
  - Use the Taguchi DOE grid (47 cells × SET_* columns) as the "treatment space".
  - For a NOK part, find the nearest OK experiment cell in setpoint space and
    return the delta: "change SET_InjectionSpeed by +12 mm/s, etc."
  - Also run a gradient-based counterfactual: perturb SET_* inputs until model
    predicts LBL_NOK = 0 (probability below threshold), while minimising
    Euclidean change in setpoint space.

Rationale:
  This is a direct research contribution. Existing systems only predict defects;
  this module explains HOW to fix them, making the system actionable for operators.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from models.fusion import InjectionMouldingFusionModel


# =============================================================================
# Taguchi nearest-OK recommender
# =============================================================================

class TaguchiBestCell:
    """
    Grid-search the Taguchi DOE cells to find the nearest OK setpoint.

    Algorithm:
      1. Aggregate each experiment cell: mean SET_* features, NOK rate.
      2. Mark cells where NOK rate ≤ nok_threshold as "OK cells".
      3. For a query part, compute Euclidean distance in standardised SET_ space
         to all OK cells.
      4. Return the closest OK cell and the delta from query setpoints.
    """

    SET_COLS: List[str] = [
        "SET_InjectionSpeed",
        "SET_HoldPressure",
        "SET_CoolingTime",
        "SET_MeltTemperature",
        "SET_MouldTemperature",
        "SET_BackPressure",
        "SET_ScrewSpeed",
        "SET_ToolTemperature",
    ]

    def __init__(
        self,
        df            : pd.DataFrame,
        nok_threshold : float = 0.2,
    ):
        """
        Build the cell index from training data.

        Args:
            df            : full dataset (train+val, group column present)
            nok_threshold : cells with NOK rate <= this are "OK cells"
        """
        available_cols = [c for c in self.SET_COLS if c in df.columns]
        if not available_cols:
            raise ValueError("No SET_* columns found in dataframe.")
        self.set_cols = available_cols

        # Aggregate per experiment cell
        grp = df.groupby(C.GROUP_COL, as_index=False)
        cell_set_means = grp[self.set_cols].mean()
        cell_nok_rate  = grp[C.PRIMARY_TARGET].mean()
        self.cell_df   = cell_set_means.merge(
            cell_nok_rate.rename(columns={C.PRIMARY_TARGET: "nok_rate"}),
            on=C.GROUP_COL,
        )

        # Standardise SET features
        self._mu  = self.cell_df[self.set_cols].values.mean(axis=0)
        self._std = self.cell_df[self.set_cols].values.std(axis=0) + 1e-6

        self.ok_cells = self.cell_df[
            self.cell_df["nok_rate"] <= nok_threshold
        ].copy().reset_index(drop=True)

        if len(self.ok_cells) == 0:
            warnings.warn(
                f"No experiment cells with NOK rate <= {nok_threshold}. "
                "Lowering threshold to 50th percentile."
            )
            median_rate = self.cell_df["nok_rate"].median()
            self.ok_cells = self.cell_df[
                self.cell_df["nok_rate"] <= median_rate
            ].copy().reset_index(drop=True)

        self.ok_set_norm = (
            self.ok_cells[self.set_cols].values - self._mu
        ) / self._std

    def recommend(
        self,
        query_row  : pd.Series,
        top_k      : int = 3,
    ) -> List[Dict]:
        """
        Find top-k nearest OK cells for a query part.

        Returns:
            List of dicts: {
                'experiment'    : str  — target experiment ID,
                'nok_rate'      : float,
                'distance'      : float,
                'deltas'        : {SET_col: delta_value},
                'recommended'   : {SET_col: target_value},
            }
        """
        query_vals = np.array([
            float(query_row.get(c, self._mu[i]))
            for i, c in enumerate(self.set_cols)
        ])
        query_norm = (query_vals - self._mu) / self._std

        dists = np.linalg.norm(
            self.ok_set_norm - query_norm[np.newaxis, :], axis=1
        )
        top_idx = np.argsort(dists)[:top_k]

        results = []
        for idx in top_idx:
            cell = self.ok_cells.iloc[idx]
            deltas = {
                c: float(cell[c] - query_row.get(c, self._mu[i]))
                for i, c in enumerate(self.set_cols)
            }
            results.append({
                "experiment"  : cell[C.GROUP_COL],
                "nok_rate"    : float(cell["nok_rate"]),
                "distance"    : float(dists[idx]),
                "deltas"      : deltas,
                "recommended" : {c: float(cell[c]) for c in self.set_cols},
            })
        return results

    def describe_recommendation(self, rec: Dict) -> str:
        """Human-readable summary of a single recommendation."""
        lines = [
            f"Nearest OK cell: {rec['experiment']}  (NOK rate: {rec['nok_rate']:.1%})",
            "Recommended setpoint changes:",
        ]
        for col, delta in rec["deltas"].items():
            sign   = "+" if delta >= 0 else ""
            target = rec["recommended"][col]
            short  = col.replace("SET_", "")
            lines.append(f"  {short:22s}  {sign}{delta:+.2f}  → {target:.2f}")
        return "\n".join(lines)


# =============================================================================
# Gradient-based counterfactual (model-guided)
# =============================================================================

class GradientCounterfactual:
    """
    Given a NOK part, perturb the tabular (SET_*) features via gradient descent
    until the model no longer predicts NOK, while penalising large deviations
    from the original setpoints.

    This is the model-in-the-loop approach — it uses the actual fusion model's
    predictions, unlike the grid-search approach above.

    Loss:
        L = BCE(model(x'), 0) + lambda_reg * ||x'_set - x_set||^2

    x' is the modified tabular vector (only SET_ indices are free; everything
    else is frozen).
    """

    def __init__(
        self,
        model       : InjectionMouldingFusionModel,
        tabular_cols: List[str],
        set_indices : Optional[List[int]] = None,   # which tabular indices are SET_
        lambda_reg  : float = 0.5,
        lr          : float = 0.05,
        max_steps   : int   = 200,
        target_prob : float = 0.3,   # succeed when NOK prob drops below this
    ):
        self.model        = model
        self.tabular_cols = tabular_cols
        self.lambda_reg   = lambda_reg
        self.lr           = lr
        self.max_steps    = max_steps
        self.target_prob  = target_prob

        # Identify which indices in tabular_cols correspond to SET_ columns
        if set_indices is not None:
            self.set_idx = set_indices
        else:
            self.set_idx = [
                i for i, c in enumerate(tabular_cols) if c.startswith("SET_")
            ]

        if not self.set_idx:
            warnings.warn("No SET_ columns found in tabular_cols — all features will be perturbed.")
            self.set_idx = list(range(len(tabular_cols)))

        self.nok_idx = C.LABEL_COLS.index(C.PRIMARY_TARGET)

    def run(
        self,
        batch  : Dict[str, torch.Tensor],
        device : torch.device,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run counterfactual optimisation.

        Returns:
            cf_delta      : (n_set_cols,) change in SET_ values
            info          : {success, nok_prob_original, nok_prob_final, n_steps}
        """
        self.model.eval()

        # Deep-copy batch, detach everything
        b = {k: (v.to(device).clone() if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

        orig_tab = b["tabular"].clone()          # (B, n_tab)
        cf_tab   = b["tabular"].clone()

        # Free only the SET_ columns
        set_mask = torch.zeros(cf_tab.shape[1], dtype=torch.bool, device=device)
        for i in self.set_idx:
            set_mask[i] = True

        delta = torch.zeros_like(cf_tab, requires_grad=False)
        delta_free = torch.zeros(
            (cf_tab.shape[0], len(self.set_idx)), dtype=torch.float32, device=device,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([delta_free], lr=self.lr)

        # Original NOK probability
        with torch.no_grad():
            b["tabular"] = orig_tab
            out0 = self.model(b)
            orig_prob = torch.sigmoid(out0["defect_logits"][0, self.nok_idx]).item()

        success  = False
        n_steps  = 0
        final_prob = orig_prob

        for step in range(self.max_steps):
            optimizer.zero_grad()

            # Build modified tabular
            mod_tab = orig_tab.clone()
            for j, idx in enumerate(self.set_idx):
                mod_tab[:, idx] = orig_tab[:, idx] + delta_free[:, j]
            b["tabular"] = mod_tab

            out    = self.model(b)
            logit  = out["defect_logits"][0, self.nok_idx]
            # We want NOK to go to 0 → target = 0
            bce    = nn.functional.binary_cross_entropy_with_logits(
                logit.unsqueeze(0),
                torch.zeros(1, device=device),
            )
            reg    = self.lambda_reg * (delta_free ** 2).mean()
            loss   = bce + reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                final_prob = torch.sigmoid(logit).item()

            n_steps = step + 1
            if final_prob < self.target_prob:
                success = True
                break

        # Extract final deltas for SET_ columns
        cf_delta = delta_free[0].detach().cpu().numpy()  # (n_set_cols,)

        return cf_delta, {
            "success"           : success,
            "nok_prob_original" : orig_prob,
            "nok_prob_final"    : final_prob,
            "n_steps"           : n_steps,
        }

    def describe(
        self,
        cf_delta    : np.ndarray,
        scaler_mean : np.ndarray,
        scaler_std  : np.ndarray,
    ) -> str:
        """
        Convert normalised delta back to physical units and format for display.

        Args:
            cf_delta     : (n_set_cols,) delta in *normalised* space
            scaler_mean  : (n_tab,) StandardScaler mean
            scaler_std   : (n_tab,) StandardScaler scale
        """
        lines = ["Gradient Counterfactual — suggested SET_* changes:"]
        for j, idx in enumerate(self.set_idx):
            col   = self.tabular_cols[idx]
            delta_phys = cf_delta[j] * scaler_std[idx]
            sign  = "+" if delta_phys >= 0 else ""
            lines.append(f"  {col.replace('SET_', ''):22s}  {sign}{delta_phys:.3f}")
        return "\n".join(lines)


# =============================================================================
# Combined counterfactual report
# =============================================================================

def counterfactual_report(
    model       : InjectionMouldingFusionModel,
    batch       : Dict[str, torch.Tensor],
    query_row   : pd.Series,
    training_df : pd.DataFrame,
    tabular_cols: List[str],
    device      : torch.device,
    scaler_mean : np.ndarray,
    scaler_std  : np.ndarray,
) -> str:
    """
    Combine both approaches into a single actionable report.

    Returns a multi-line string suitable for printing or logging.
    """
    lines = ["=" * 60, "DEFECT ROOT CAUSE & RECOMMENDATION REPORT", "=" * 60]

    # --- model prediction summary ----------------------------------------
    model.eval()
    with torch.no_grad():
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)
        probs = torch.sigmoid(out["defect_logits"][0]).cpu().numpy()

    lines.append("\nModel predictions:")
    for i, col in enumerate(C.LABEL_COLS):
        flag = " *** NOK ***" if col == C.PRIMARY_TARGET and probs[i] > 0.5 else ""
        lines.append(f"  {col:30s}  {probs[i]:.3f}{flag}")

    # --- Taguchi nearest-OK -----------------------------------------------
    lines.append("\n--- TAGUCHI NEAREST-OK CELLS ---")
    try:
        recommender = TaguchiBestCell(training_df)
        recs = recommender.recommend(query_row, top_k=3)
        for rank, rec in enumerate(recs, 1):
            lines.append(f"\nOption {rank} (distance={rec['distance']:.3f}):")
            lines.append(recommender.describe_recommendation(rec))
    except Exception as e:
        lines.append(f"[Taguchi recommender unavailable: {e}]")

    # --- Gradient counterfactual ------------------------------------------
    lines.append("\n--- GRADIENT COUNTERFACTUAL ---")
    try:
        gcf = GradientCounterfactual(model, tabular_cols)
        cf_delta, info = gcf.run(batch, device)
        lines.append(f"Optimisation: {'SUCCESS' if info['success'] else 'DID NOT CONVERGE'}")
        lines.append(f"NOK probability: {info['nok_prob_original']:.3f} → {info['nok_prob_final']:.3f}")
        lines.append(gcf.describe(cf_delta, scaler_mean, scaler_std))
    except Exception as e:
        lines.append(f"[Gradient counterfactual unavailable: {e}]")

    lines.append("=" * 60)
    return "\n".join(lines)
