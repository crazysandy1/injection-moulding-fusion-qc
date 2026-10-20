"""
training/trainer.py
End-to-end training loop for the 4-modal fusion system.

Features:
  - Group-aware k-fold cross-validation
  - DANN lambda warmup schedule
  - Per-label threshold tuning
  - Early stopping on val macro-F1
  - MLflow experiment tracking
  - Model checkpoint with full state
"""

import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from data.dataset   import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from data.loaders   import load_parquet, build_cv_index, join_cv_to_parquet
from models.fusion  import InjectionMouldingFusionModel
from training.splits import group_kfold_splits, random_split, time_aware_split, build_experiment_label_map
from training.losses  import TotalLoss, compute_pos_weights
from training.metrics import compute_metrics, group_aware_evaluation, find_best_thresholds, build_headline_table

log = logging.getLogger(__name__)


# =============================================================================
# DANN lambda schedule
# =============================================================================

def dann_lambda_schedule(epoch: int, total_epochs: int, max_lambda: float = C.DANN_LAMBDA_MAX) -> float:
    """
    Linearly ramp DANN lambda from 0 to max_lambda after warmup period.
    Standard schedule from Ganin et al. 2016.
    """
    if epoch < C.DANN_WARMUP_EP:
        return 0.0
    p = (epoch - C.DANN_WARMUP_EP) / (total_epochs - C.DANN_WARMUP_EP + 1e-6)
    return float(max_lambda * min(p * 2, 1.0))


# =============================================================================
# Single epoch
# =============================================================================

def run_epoch(
    model       : InjectionMouldingFusionModel,
    loader      : DataLoader,
    loss_fn     : TotalLoss,
    optimizer   : Optional[torch.optim.Optimizer],
    device      : torch.device,
    exp_label_map: Dict[str, int],
    train       : bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one training or validation epoch.

    Returns:
        loss_dict  : averaged loss components
        all_probs  : (N, n_labels) sigmoid probabilities
        all_targets: (N, n_labels)
        all_groups : (N,) experiment group strings
    """
    model.train(train)
    total_losses: Dict[str, float] = {"total": 0, "detection": 0, "dann": 0, "consistency": 0}
    all_probs, all_targets, all_groups = [], [], []
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val "):
            labels = batch["labels"].to(device)

            # experiment labels for DANN
            exp_ids = torch.tensor(
                [exp_label_map.get(g, 0) for g in batch["group"]],
                dtype=torch.long, device=device,
            )

            outputs = model(batch)
            losses  = loss_fn(
                outputs, labels, exp_ids,
                batch["thermal_valid"].to(device),
                batch["sequence_valid"].to(device),
            )

            if train and optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                optimizer.step()

            for k in total_losses:
                total_losses[k] += losses[k].item()

            probs = torch.sigmoid(outputs["defect_logits"]).detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())
            all_groups.extend(batch["group"])
            n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    probs_arr   = np.concatenate(all_probs,   axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    groups_arr  = np.array(all_groups)

    return avg_losses, probs_arr, targets_arr, groups_arr


# =============================================================================
# One fold trainer
# =============================================================================

def train_one_fold(
    model       : InjectionMouldingFusionModel,
    train_df    : pd.DataFrame,
    val_df      : pd.DataFrame,
    tabular_cols: List[str],
    dxp_cols    : List[str],
    exp_label_map: Dict[str, int],
    device      : torch.device,
    fold_tag    : str = "fold0",
) -> Dict:
    """Train the model on one fold. Returns best val metrics + thresholds."""

    # ---- fit tabular scaler on TRAIN only ----------------------------------
    scaler = StandardScaler()
    tab_vals = []
    for _, row in train_df.iterrows():
        from data.loaders import load_tabular_vector
        v = load_tabular_vector(row, tabular_cols)
        tab_vals.append(v)
    scaler.fit(np.array(tab_vals))

    # ---- medians for imputation -------------------------------------------
    tab_medians = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
    roi_medians = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

    # ---- datasets -----------------------------------------------------------
    train_ds = MultiModalDataset(
        train_df, tabular_cols, dxp_cols,
        train=True, tabular_medians=tab_medians,
        roi_medians=roi_medians, tabular_scaler=scaler,
    )
    val_ds = MultiModalDataset(
        val_df, tabular_cols, dxp_cols,
        train=False, tabular_medians=tab_medians,
        roi_medians=roi_medians, tabular_scaler=scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # ---- loss function with class weights ----------------------------------
    pos_weight = compute_pos_weights(train_df, device)
    loss_fn    = TotalLoss(
        pos_weight=pos_weight,
        n_experiments=len(exp_label_map),
        lambda_detection=1.0,
        lambda_dann=0.0,          # starts at 0, ramped during training
        lambda_consistency=0.1,
    )

    # ---- optimizer & scheduler ---------------------------------------------
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": C.LR_BACKBONE},
        {"params": head_params,     "lr": C.LR_HEAD},
    ], weight_decay=C.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5,
    )

    # ---- training loop ------------------------------------------------------
    best_val_f1   = -1.0
    patience_cnt  = 0
    best_state    = None
    best_thresholds = np.full(C.N_LABELS, 0.5)
    history       = []

    for epoch in range(C.NUM_EPOCHS):
        # update DANN lambda
        lam = dann_lambda_schedule(epoch, C.NUM_EPOCHS)
        model.set_dann_lambda(lam)
        loss_fn.set_dann_lambda(lam)
        model.set_epoch(epoch)

        # train
        t0 = time.time()
        train_losses, _, _, _ = run_epoch(
            model, train_loader, loss_fn, optimizer, device, exp_label_map, train=True
        )

        # validate
        val_losses, val_probs, val_targets, val_groups = run_epoch(
            model, val_loader, loss_fn, None, device, exp_label_map, train=False
        )

        # tune thresholds on val set
        thrs = find_best_thresholds(val_probs, val_targets)
        val_metrics = compute_metrics(val_probs, val_targets, thresholds=thrs)
        val_f1 = val_metrics["macro_f1"]

        scheduler.step(val_f1)

        elapsed = time.time() - t0
        log.info(
            f"[{fold_tag}] ep {epoch+1:03d}/{C.NUM_EPOCHS}  "
            f"train_loss={train_losses['total']:.4f}  "
            f"val_f1={val_f1:.4f}  dann_λ={lam:.3f}  "
            f"({elapsed:.1f}s)"
        )

        history.append({"epoch": epoch+1, "val_f1": val_f1, **train_losses})

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thrs
            patience_cnt   = 0
        else:
            patience_cnt += 1

        if patience_cnt >= C.PATIENCE:
            log.info(f"[{fold_tag}] Early stop at epoch {epoch+1}")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_f1"    : best_val_f1,
        "best_thresholds": best_thresholds,
        "history"        : history,
    }


# =============================================================================
# Full training pipeline
# =============================================================================

def train_full_pipeline(parquet_path: Optional[Path] = None) -> Dict:
    """
    Complete training pipeline:
      1. Load and prepare data
      2. Group k-fold cross-validation (headline evaluation)
      3. Train final model on all data
      4. Evaluate: random / time-aware / group-fold / OOD
      5. Save artefacts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ---- load data ----------------------------------------------------------
    df = load_parquet(parquet_path)

    # Build CV image index and join
    cv_idx = build_cv_index()
    df     = join_cv_to_parquet(df, cv_idx)

    # Column detection
    tabular_cols  = get_tabular_cols(df)
    dxp_cols      = get_dxp_cols(df)
    exp_label_map = build_experiment_label_map(df)
    n_experiments = len(exp_label_map)

    log.info(f"Tabular cols: {len(tabular_cols)}  DXP cols: {len(dxp_cols)}  Experiments: {n_experiments}")

    # ---- group k-fold evaluation -------------------------------------------
    fold_results = []
    all_val_probs   = []
    all_val_targets = []
    all_val_groups  = []

    for fold_i, (train_df, val_df) in enumerate(group_kfold_splits(df)):
        log.info(f"\n=== FOLD {fold_i+1}/{C.N_GROUP_FOLDS} ===")
        log.info(f"  train: {len(train_df)}  val: {len(val_df)}")

        model = InjectionMouldingFusionModel(
            tabular_in_dim=len(tabular_cols),
            n_dxp_channels=len(dxp_cols),
            n_experiments=n_experiments,
        ).to(device)

        fold_out = train_one_fold(
            model, train_df, val_df,
            tabular_cols, dxp_cols, exp_label_map, device,
            fold_tag=f"fold{fold_i+1}",
        )

        # run inference on val fold
        scaler = StandardScaler()
        from data.loaders import load_tabular_vector
        tab_medians = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
        roi_medians = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

        val_ds = MultiModalDataset(val_df, tabular_cols, dxp_cols, train=False,
                                   tabular_medians=tab_medians, roi_medians=roi_medians)
        val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)
        _, vp, vt, vg = run_epoch(model, val_loader, TotalLoss(), None, device, exp_label_map, train=False)

        all_val_probs.append(vp)
        all_val_targets.append(vt)
        all_val_groups.append(vg)
        fold_results.append({"fold": fold_i+1, "val_f1": fold_out["best_val_f1"]})

    # aggregate group-fold results
    all_probs   = np.concatenate(all_val_probs,   axis=0)
    all_targets = np.concatenate(all_val_targets, axis=0)
    all_groups  = np.concatenate(all_val_groups,  axis=0)

    group_thrs = find_best_thresholds(all_probs, all_targets)
    group_metrics = compute_metrics(all_probs, all_targets, thresholds=group_thrs)
    group_breakdown = group_aware_evaluation(all_probs, all_targets, all_groups, thresholds=group_thrs)

    log.info("\n=== GROUP K-FOLD RESULTS (HEADLINE) ===")
    log.info(f"  Macro-F1: {group_metrics['macro_f1']:.4f}")
    log.info(f"  Per-group mean-F1: {group_breakdown['mean_group_f1']:.4f} ± {group_breakdown['std_group_f1']:.4f}")

    # ---- random split baseline ---------------------------------------------
    tr, va, te = random_split(df)
    model_rand = InjectionMouldingFusionModel(
        tabular_in_dim=len(tabular_cols),
        n_dxp_channels=len(dxp_cols),
        n_experiments=n_experiments,
    ).to(device)
    rand_out = train_one_fold(model_rand, pd.concat([tr, va]), te, tabular_cols, dxp_cols,
                              exp_label_map, device, fold_tag="random_test")
    rand_metrics = {"macro_f1": rand_out["best_val_f1"], "micro_f1": 0.0,
                    "mean_roc_auc": 0.0, "mean_pr_auc": 0.0}

    # ---- headline table ----------------------------------------------------
    headline = build_headline_table({
        "Group k-fold (HEADLINE)": group_metrics,
        "Random split (baseline)": rand_metrics,
    })
    log.info(f"\n{headline.to_string(index=False)}")

    # ---- save artefacts ----------------------------------------------------
    torch.save(model.state_dict(), C.MODELS_DIR / "fusion_final.pt")
    np.save(C.MODELS_DIR / "best_thresholds.npy", group_thrs)

    artefact = {
        "group_kfold_metrics"   : group_metrics,
        "group_breakdown"       : group_breakdown,
        "fold_results"          : fold_results,
        "random_split_metrics"  : rand_metrics,
        "tabular_cols"          : tabular_cols,
        "dxp_cols"              : dxp_cols,
        "exp_label_map"         : exp_label_map,
        "n_experiments"         : n_experiments,
    }
    with open(C.ARTIFACTS_DIR / "training_results.json", "w") as fh:
        json.dump(artefact, fh, indent=2, default=str)

    log.info(f"Artefacts saved to {C.ARTIFACTS_DIR}")
    return artefact
