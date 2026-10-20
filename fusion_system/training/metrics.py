"""
training/metrics.py
All evaluation metrics.

Headline metric: group-aware macro-F1 (group = MET_ExperimentNumber)
Also tracks: per-label F1, AUC-ROC, AUC-PR, confusion matrix.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


def find_best_thresholds(
    probs   : np.ndarray,   # (N, n_labels) sigmoid probabilities
    targets : np.ndarray,   # (N, n_labels) 0/1
    grid    : List[float] = C.THRESHOLD_GRID,
) -> np.ndarray:
    """
    Grid-search per-label threshold that maximises label-level F1.
    Returns array of shape (n_labels,).
    """
    best_thresholds = np.full(probs.shape[1], 0.5)
    for i in range(probs.shape[1]):
        best_f1 = -1.0
        for thr in grid:
            preds = (probs[:, i] >= thr).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = thr
    return best_thresholds


def compute_metrics(
    probs      : np.ndarray,         # (N, n_labels)
    targets    : np.ndarray,         # (N, n_labels)
    thresholds : Optional[np.ndarray] = None,
    label_names: List[str] = C.LABEL_COLS,
) -> Dict:
    """
    Compute full metrics dict for one split.
    """
    if thresholds is None:
        thresholds = np.full(probs.shape[1], 0.5)

    preds = (probs >= thresholds).astype(int)

    # macro / micro F1
    macro_f1 = f1_score(targets, preds, average="macro",  zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro",  zero_division=0)

    # per-label F1
    per_label_f1 = f1_score(targets, preds, average=None, zero_division=0)

    # AUC-ROC per label (only for labels with both classes present)
    roc_aucs = []
    pr_aucs  = []
    for i in range(probs.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            roc_aucs.append(np.nan)
            pr_aucs.append(np.nan)
        else:
            roc_aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
            pr_aucs.append(average_precision_score(targets[:, i], probs[:, i]))

    # NOK binary metrics
    nok_idx = C.LABEL_COLS.index(C.PRIMARY_TARGET)
    nok_cm  = confusion_matrix(targets[:, nok_idx], preds[:, nok_idx], labels=[0, 1])

    return {
        "macro_f1"      : float(macro_f1),
        "micro_f1"      : float(micro_f1),
        "mean_roc_auc"  : float(np.nanmean(roc_aucs)),
        "mean_pr_auc"   : float(np.nanmean(pr_aucs)),
        "per_label_f1"  : {k: float(v) for k, v in zip(label_names, per_label_f1)},
        "per_label_auc" : {k: float(v) for k, v in zip(label_names, roc_aucs)},
        "thresholds"    : {k: float(v) for k, v in zip(label_names, thresholds)},
        "nok_confusion" : nok_cm.tolist(),
    }


def group_aware_evaluation(
    probs       : np.ndarray,
    targets     : np.ndarray,
    groups      : np.ndarray,      # MET_ExperimentNumber per sample
    thresholds  : Optional[np.ndarray] = None,
    label_names : List[str] = C.LABEL_COLS,
) -> Dict:
    """
    Per-group breakdown of F1 and AUC.
    Shows which experimental conditions the model generalises to.

    Returns:
        per_group : {group_name: {macro_f1, micro_f1, ...}}
        summary   : {mean_group_f1, std_group_f1, min_group_f1, max_group_f1}
    """
    unique_groups = np.unique(groups)
    per_group     = {}
    group_f1s     = []

    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue
        m = compute_metrics(
            probs[mask], targets[mask], thresholds=thresholds,
            label_names=label_names,
        )
        per_group[g] = m
        group_f1s.append(m["macro_f1"])

    group_f1s = np.array(group_f1s)

    return {
        "per_group"       : per_group,
        "mean_group_f1"   : float(group_f1s.mean()),
        "std_group_f1"    : float(group_f1s.std()),
        "min_group_f1"    : float(group_f1s.min()),
        "max_group_f1"    : float(group_f1s.max()),
        "n_groups"        : len(group_f1s),
    }


def build_headline_table(results: Dict) -> pd.DataFrame:
    """
    Build the paper-ready evaluation table comparing:
        - Random split (naive, optimistic)
        - Time-aware split
        - Group k-fold mean ± std  ← HEADLINE
        - Material OOD (PP → ABS)

    results: dict keyed by split_name → compute_metrics output
    """
    rows = []
    for split_name, m in results.items():
        row = {
            "Split"        : split_name,
            "Macro-F1"     : f"{m['macro_f1']:.3f}",
            "Micro-F1"     : f"{m['micro_f1']:.3f}",
            "ROC-AUC"      : f"{m['mean_roc_auc']:.3f}",
            "PR-AUC"       : f"{m['mean_pr_auc']:.3f}",
        }
        rows.append(row)
    return pd.DataFrame(rows)
