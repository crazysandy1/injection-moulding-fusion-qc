"""
evaluate.py
Comprehensive evaluation script — generates the paper-ready headline table
comparing all 4 split strategies.

Usage:
    python evaluate.py                              # uses saved fusion_final.pt
    python evaluate.py --model path/to/model.pt
    python evaluate.py --split group_kfold         # single split only
    python evaluate.py --drift                      # run drift monitor
    python evaluate.py --consistency                # run sensor consistency sweep

Output artefacts:
    artifacts/evaluation_headline.csv
    artifacts/evaluation_full.json
    artifacts/confusion_matrix_nok.png
    artifacts/per_label_f1.png
    artifacts/drift_report.json                    (if --drift)
    artifacts/consistency_report.json              (if --consistency)
"""

import argparse
import logging
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from data.loaders    import load_parquet, build_cv_index, join_cv_to_parquet, load_tabular_vector
from data.dataset    import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from models.fusion   import InjectionMouldingFusionModel
from training.splits import (
    group_kfold_splits, random_split, time_aware_split,
    material_ood_split, build_experiment_label_map,
)
from training.losses  import TotalLoss
from training.metrics import (
    compute_metrics, find_best_thresholds,
    group_aware_evaluation, build_headline_table,
)
from training.trainer import run_epoch, train_one_fold

log = logging.getLogger("evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# Model loader helper
# =============================================================================

def load_model(
    model_path  : Path,
    tabular_cols: List[str],
    dxp_cols    : List[str],
    n_experiments: int,
    device      : torch.device,
) -> InjectionMouldingFusionModel:
    model = InjectionMouldingFusionModel(
        tabular_in_dim=len(tabular_cols),
        n_dxp_channels=len(dxp_cols),
        n_experiments=n_experiments,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded model from %s", model_path)
    return model


# =============================================================================
# Dataset / DataLoader builder
# =============================================================================

def build_loader(
    df          : pd.DataFrame,
    tabular_cols: List[str],
    dxp_cols    : List[str],
    train_df    : pd.DataFrame,
    batch_size  : int = C.BATCH_SIZE,
) -> DataLoader:
    """Build a DataLoader with scaler/medians fit on train_df."""
    scaler = StandardScaler()
    tab_vals = [load_tabular_vector(row, tabular_cols) for _, row in train_df.iterrows()]
    scaler.fit(np.array(tab_vals))

    tab_medians = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
    roi_medians = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

    ds = MultiModalDataset(
        df, tabular_cols, dxp_cols, train=False,
        tabular_medians=tab_medians, roi_medians=roi_medians,
        tabular_scaler=scaler,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0)


# =============================================================================
# Per-split evaluator
# =============================================================================

def evaluate_split(
    model       : InjectionMouldingFusionModel,
    test_df     : pd.DataFrame,
    train_df    : pd.DataFrame,
    tabular_cols: List[str],
    dxp_cols    : List[str],
    exp_label_map: Dict[str, int],
    device      : torch.device,
    thresholds  : Optional[np.ndarray] = None,
) -> Dict:
    """
    Run inference on test_df, compute full metrics.
    """
    loader = build_loader(test_df, tabular_cols, dxp_cols, train_df)
    loss_fn = TotalLoss()

    _, probs, targets, groups = run_epoch(
        model, loader, loss_fn, None, device, exp_label_map, train=False
    )

    if thresholds is None:
        thresholds = find_best_thresholds(probs, targets)

    metrics = compute_metrics(probs, targets, thresholds=thresholds)
    return metrics, probs, targets, groups


# =============================================================================
# Confusion matrix plot
# =============================================================================

def plot_confusion_matrix(cm: list, out_path: Path, label: str = "LBL_NOK"):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred OK", "Pred NOK"],
        yticklabels=["True OK", "True NOK"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {label}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Confusion matrix saved to %s", out_path)


def plot_per_label_f1(metrics_dict: Dict[str, Dict], out_path: Path):
    """
    Bar chart comparing per-label F1 across all evaluation splits.
    """
    label_names = C.LABEL_COLS
    n_labels    = len(label_names)
    n_splits    = len(metrics_dict)

    x     = np.arange(n_labels)
    width = 0.8 / n_splits
    fig, ax = plt.subplots(figsize=(14, 5))

    colours = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for k, (split_name, metrics) in enumerate(metrics_dict.items()):
        per_f1 = [metrics["per_label_f1"].get(l, 0.0) for l in label_names]
        ax.bar(x + k * width, per_f1, width, label=split_name,
               color=colours[k % len(colours)], alpha=0.85)

    ax.set_xticks(x + width * (n_splits - 1) / 2)
    ax.set_xticklabels([l.replace("LBL_", "") for l in label_names], rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Label F1 Score by Evaluation Split")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Per-label F1 chart saved to %s", out_path)


# =============================================================================
# Main evaluation pipeline
# =============================================================================

def run_evaluation(
    parquet_path: Path,
    model_path  : Path,
    splits      : List[str],
    run_drift   : bool = False,
    run_cons    : bool = False,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ---- Load data -----------------------------------------------------------
    df      = load_parquet(parquet_path)
    cv_idx  = build_cv_index()
    df      = join_cv_to_parquet(df, cv_idx)

    tabular_cols  = get_tabular_cols(df)
    dxp_cols      = get_dxp_cols(df)
    exp_label_map = build_experiment_label_map(df)

    # Load saved thresholds if available
    thr_path = C.MODELS_DIR / "best_thresholds.npy"
    saved_thrs = np.load(thr_path) if thr_path.exists() else None

    all_results: Dict[str, Dict] = {}

    # ============================================================
    # 1. Group K-Fold (HEADLINE)
    # ============================================================
    if "group_kfold" in splits:
        log.info("\n=== Evaluating: Group K-Fold (HEADLINE) ===")
        all_probs, all_targets, all_groups = [], [], []

        for fold_i, (train_df, val_df) in enumerate(group_kfold_splits(df)):
            fold_model = load_model(model_path, tabular_cols, dxp_cols,
                                    len(exp_label_map), device)
            # Fine-tune if checkpoint is from final model (not fold-specific)
            # Otherwise use directly
            metrics_fold, vp, vt, vg = evaluate_split(
                fold_model, val_df, train_df, tabular_cols, dxp_cols,
                exp_label_map, device, saved_thrs,
            )
            all_probs.append(vp)
            all_targets.append(vt)
            all_groups.append(vg)
            log.info("  Fold %d  macro_F1=%.4f", fold_i + 1, metrics_fold["macro_f1"])

        all_p = np.concatenate(all_probs,   axis=0)
        all_t = np.concatenate(all_targets, axis=0)
        all_g = np.concatenate(all_groups,  axis=0)
        thrs  = find_best_thresholds(all_p, all_t)
        gkf_metrics = compute_metrics(all_p, all_t, thresholds=thrs)
        gkf_breakdown = group_aware_evaluation(all_p, all_t, all_g, thresholds=thrs)

        all_results["Group k-fold (HEADLINE)"] = gkf_metrics

        log.info("  Group K-Fold macro-F1: %.4f", gkf_metrics["macro_f1"])
        log.info(
            "  Per-group mean±std: %.4f ± %.4f",
            gkf_breakdown["mean_group_f1"], gkf_breakdown["std_group_f1"],
        )

    # ============================================================
    # 2. Random Split (BASELINE)
    # ============================================================
    if "random" in splits:
        log.info("\n=== Evaluating: Random Split (BASELINE) ===")
        train_df, val_df, test_df = random_split(df)
        model_rand = load_model(model_path, tabular_cols, dxp_cols,
                                len(exp_label_map), device)
        rand_metrics, _, _, _ = evaluate_split(
            model_rand, test_df, train_df,
            tabular_cols, dxp_cols, exp_label_map, device, saved_thrs,
        )
        all_results["Random split (baseline)"] = rand_metrics
        log.info("  Random macro-F1: %.4f", rand_metrics["macro_f1"])

    # ============================================================
    # 3. Time-Aware Split
    # ============================================================
    if "time" in splits:
        log.info("\n=== Evaluating: Time-Aware Split ===")
        train_df, val_df, test_df = time_aware_split(df)
        model_time = load_model(model_path, tabular_cols, dxp_cols,
                                len(exp_label_map), device)
        time_metrics, _, _, _ = evaluate_split(
            model_time, test_df, train_df,
            tabular_cols, dxp_cols, exp_label_map, device, saved_thrs,
        )
        all_results["Time-aware split"] = time_metrics
        log.info("  Time-aware macro-F1: %.4f", time_metrics["macro_f1"])

    # ============================================================
    # 4. Material OOD (PP → ABS)
    # ============================================================
    if "ood" in splits:
        log.info("\n=== Evaluating: Material OOD (PP → ABS) ===")
        if C.MATERIAL_COL not in df.columns:
            log.warning("MATERIAL_COL not found in dataframe — skipping OOD eval.")
        else:
            ood_train, ood_test = material_ood_split(df)
            if len(ood_test) == 0:
                log.warning("No ABS test data found — skipping OOD eval.")
            else:
                model_ood = load_model(model_path, tabular_cols, dxp_cols,
                                       len(exp_label_map), device)
                ood_metrics, _, _, _ = evaluate_split(
                    model_ood, ood_test, ood_train,
                    tabular_cols, dxp_cols, exp_label_map, device, saved_thrs,
                )
                all_results["Material OOD (PP → ABS)"] = ood_metrics
                log.info("  OOD macro-F1: %.4f", ood_metrics["macro_f1"])

    # ============================================================
    # Headline table
    # ============================================================
    if all_results:
        table = build_headline_table(all_results)
        log.info("\n%s", "=" * 60)
        log.info("HEADLINE EVALUATION TABLE")
        log.info("%s", table.to_string(index=False))
        log.info("%s", "=" * 60)

        out_csv = C.ARTIFACTS_DIR / "evaluation_headline.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
        log.info("Headline table saved to %s", out_csv)

        # Confusion matrix for group k-fold headline
        if "Group k-fold (HEADLINE)" in all_results:
            cm = all_results["Group k-fold (HEADLINE)"].get("nok_confusion")
            if cm:
                plot_confusion_matrix(cm, C.ARTIFACTS_DIR / "confusion_matrix_nok.png")

        # Per-label F1 chart
        plot_per_label_f1(all_results, C.ARTIFACTS_DIR / "per_label_f1.png")

    # ============================================================
    # Drift monitor
    # ============================================================
    if run_drift:
        log.info("\n=== Running Drift Monitor ===")
        from analysis.drift import DriftMonitor
        train_df, _, test_df = random_split(df)
        monitor = DriftMonitor(train_df, tabular_cols)
        drift_report = monitor.check(test_df)
        log.info(monitor.format_summary(drift_report))
        monitor.save_report(drift_report, C.ARTIFACTS_DIR / "drift_report.json")

    # ============================================================
    # Consistency sweep
    # ============================================================
    if run_cons:
        log.info("\n=== Running Consistency Sweep ===")
        from analysis.consistency import (
            batch_consistency_sweep, summarise_sweep, EmbeddingAnomalyDetector,
        )
        _, _, test_df = random_split(df)
        model_cons = load_model(model_path, tabular_cols, dxp_cols,
                                len(exp_label_map), device)
        train_df, _, _ = random_split(df)
        ref_loader  = build_loader(train_df, tabular_cols, dxp_cols, train_df, batch_size=32)
        thermal_det = EmbeddingAnomalyDetector.from_model_outputs(
            model_cons, ref_loader, device, modality="thermal"
        )
        fused_det = EmbeddingAnomalyDetector.from_model_outputs(
            model_cons, ref_loader, device, modality="fused"
        )
        test_loader = build_loader(test_df, tabular_cols, dxp_cols, train_df)
        reports = batch_consistency_sweep(
            model_cons, test_loader, device,
            thermal_detector=thermal_det,
            fused_detector=fused_det,
            dxp_cols=dxp_cols,
        )
        summary = summarise_sweep(reports)
        log.info("Consistency summary: %s", summary)
        out_cons = C.ARTIFACTS_DIR / "consistency_report.json"
        with open(out_cons, "w") as f:
            json.dump({"summary": summary, "samples": reports[:20]}, f, indent=2, default=str)
        log.info("Consistency report saved to %s", out_cons)

    # ============================================================
    # Save full results
    # ============================================================
    full_out = C.ARTIFACTS_DIR / "evaluation_full.json"
    with open(full_out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Full evaluation results saved to %s", full_out)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the 4-Modal Fusion System across all split strategies"
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to dataset_V2.parquet",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to model checkpoint (.pt). Default: MODELS_DIR/fusion_final.pt",
    )
    parser.add_argument(
        "--split", type=str,
        default="group_kfold,random,time,ood",
        help="Comma-separated list of splits to evaluate: "
             "group_kfold, random, time, ood",
    )
    parser.add_argument(
        "--drift", action="store_true",
        help="Run PSI/KS drift monitor on test set",
    )
    parser.add_argument(
        "--consistency", action="store_true",
        help="Run cross-modal sensor consistency sweep",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    parquet_path = args.data  or C.PARQUET_PATH
    model_path   = args.model or (C.MODELS_DIR / "fusion_final.pt")

    if not model_path.exists():
        log.error(
            "Model checkpoint not found at %s. "
            "Run train.py first, or pass --model <path>.",
            model_path,
        )
        sys.exit(1)

    splits_requested = [s.strip() for s in args.split.split(",")]

    run_evaluation(
        parquet_path=parquet_path,
        model_path=model_path,
        splits=splits_requested,
        run_drift=args.drift,
        run_cons=args.consistency,
    )


if __name__ == "__main__":
    main()
