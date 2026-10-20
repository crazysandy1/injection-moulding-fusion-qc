"""
train.py
Main training entry-point for the 4-Modal Injection Moulding Fusion System.

Usage:
    python train.py                         # default parquet path from config
    python train.py --data path/to/data.parquet
    python train.py --epochs 80 --batch 32
    python train.py --dann_sweep            # run DANN lambda ablation (paper experiment)

Outputs (all under ARTIFACTS_DIR / MODELS_DIR from config.py):
    models/fusion_final.pt
    models/best_thresholds.npy
    artifacts/training_results.json
    artifacts/dann_sweep_results.json       (if --dann_sweep)
    artifacts/explainability/               (sample explanations)
"""

import argparse
import logging
import json
import sys
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from training.trainer import train_full_pipeline
from training.splits  import group_kfold_splits, random_split, build_experiment_label_map
from training.metrics import compute_metrics, find_best_thresholds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# =============================================================================
# DANN lambda sweep (research ablation)
# =============================================================================

def run_dann_sweep(parquet_path: Path) -> Dict:
    """
    Train with λ_dann ∈ {0.0, 0.1, 0.2, 0.3, 0.5} and compare
    random-split F1 vs group-kfold F1.

    Demonstrates the leakage-fixing effect of DANN: λ=0 has high random but
    low group F1 (setpoint memorisation); λ>0 narrows the gap.
    """
    import config as C_mod
    from training.trainer import train_full_pipeline

    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    sweep_results = []

    log.info("Starting DANN lambda sweep: %s", lambda_values)

    for lam in lambda_values:
        log.info("=== DANN lambda = %.2f ===", lam)
        # Temporarily override config
        original_max = C_mod.DANN_LAMBDA_MAX
        C_mod.DANN_LAMBDA_MAX = lam

        t0 = time.time()
        results = train_full_pipeline(parquet_path)
        elapsed = time.time() - t0

        sweep_results.append({
            "dann_lambda"         : lam,
            "group_kfold_macro_f1": results["group_kfold_metrics"]["macro_f1"],
            "random_split_macro_f1": results["random_split_metrics"]["macro_f1"],
            "gap"                 : (
                results["random_split_metrics"]["macro_f1"]
                - results["group_kfold_metrics"]["macro_f1"]
            ),
            "training_time_s"     : elapsed,
        })

        C_mod.DANN_LAMBDA_MAX = original_max

        log.info(
            "lambda=%.2f  group_f1=%.4f  random_f1=%.4f  gap=%.4f",
            lam,
            sweep_results[-1]["group_kfold_macro_f1"],
            sweep_results[-1]["random_split_macro_f1"],
            sweep_results[-1]["gap"],
        )

    # Save
    out_path = C.ARTIFACTS_DIR / "dann_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    log.info("DANN sweep saved to %s", out_path)

    # Print summary table
    log.info("\n=== DANN SWEEP SUMMARY ===")
    log.info(f"{'lambda':>8}  {'group_F1':>10}  {'random_F1':>10}  {'gap':>8}")
    for r in sweep_results:
        log.info(
            f"{r['dann_lambda']:>8.2f}  {r['group_kfold_macro_f1']:>10.4f}  "
            f"{r['random_split_macro_f1']:>10.4f}  {r['gap']:>8.4f}"
        )

    return {"sweep": sweep_results}


# =============================================================================
# Sample explainability run
# =============================================================================

def run_explainability_samples(parquet_path: Path, n_samples: int = 5):
    """
    Load the saved model and produce explanation figures for n_samples parts.
    """
    from data.loaders   import load_parquet, build_cv_index, join_cv_to_parquet
    from data.dataset   import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
    from models.fusion  import InjectionMouldingFusionModel
    from training.splits import build_experiment_label_map
    from analysis.explainability import explain_sample
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_parquet(parquet_path)
    cv_idx = build_cv_index()
    df     = join_cv_to_parquet(df, cv_idx)

    tabular_cols  = get_tabular_cols(df)
    dxp_cols      = get_dxp_cols(df)
    exp_label_map = build_experiment_label_map(df)

    # Use full dataset for explanation (no split needed)
    scaler = StandardScaler()
    from data.loaders import load_tabular_vector
    tab_vals = [load_tabular_vector(row, tabular_cols) for _, row in df.iterrows()]
    scaler.fit(np.array(tab_vals))

    tab_medians = {c: float(df[c].median()) for c in tabular_cols if c in df.columns}
    roi_medians = {c: float(df[c].median()) for c in C.THERMAL_ROI_COLS if c in df.columns}

    ds = MultiModalDataset(df, tabular_cols, dxp_cols, train=False,
                           tabular_medians=tab_medians, roi_medians=roi_medians,
                           tabular_scaler=scaler)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = InjectionMouldingFusionModel(
        tabular_in_dim=len(tabular_cols),
        n_dxp_channels=len(dxp_cols),
        n_experiments=len(exp_label_map),
    ).to(device)

    ckpt_path = C.MODELS_DIR / "fusion_final.pt"
    if not ckpt_path.exists():
        log.warning("Model checkpoint not found at %s — skipping explainability.", ckpt_path)
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    out_dir = C.ARTIFACTS_DIR / "explainability"
    count   = 0

    for i, batch in enumerate(loader):
        cycle_id = batch.get("cycle_id", [f"sample_{i}"])
        sid = cycle_id[0] if isinstance(cycle_id, (list, tuple)) else str(cycle_id)

        explain_sample(
            model, batch, device,
            out_dir=out_dir,
            sample_id=str(sid),
            dxp_cols=dxp_cols,
        )
        count += 1
        log.info("Explained sample %d/%d  (cycle_id=%s)", count, n_samples, sid)

        if count >= n_samples:
            break

    log.info("Explanation figures saved to %s", out_dir)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 4-Modal Injection Moulding Fusion System"
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to dataset_V2.parquet (default: config.PARQUET_PATH)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override NUM_EPOCHS from config",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override BATCH_SIZE from config",
    )
    parser.add_argument(
        "--dann_sweep", action="store_true",
        help="Run DANN lambda ablation study (long, trains 5 models)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="After training, produce explainability figures for 5 sample parts",
    )
    parser.add_argument(
        "--n_explain", type=int, default=5,
        help="Number of sample parts to explain (used with --explain)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override config if flags provided
    if args.epochs is not None:
        C.NUM_EPOCHS = args.epochs
        log.info("Overriding NUM_EPOCHS to %d", C.NUM_EPOCHS)
    if args.batch is not None:
        C.BATCH_SIZE = args.batch
        log.info("Overriding BATCH_SIZE to %d", C.BATCH_SIZE)

    parquet_path = args.data or C.PARQUET_PATH

    # Ensure output dirs exist
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- DANN sweep (paper experiment) ----
    if args.dann_sweep:
        log.info("Running DANN lambda sweep...")
        run_dann_sweep(parquet_path)
        return

    # ---- Normal training ----
    log.info("Starting full training pipeline...")
    log.info("  Data     : %s", parquet_path)
    log.info("  Epochs   : %d", C.NUM_EPOCHS)
    log.info("  Batch    : %d", C.BATCH_SIZE)
    log.info("  Device   : %s", "cuda" if torch.cuda.is_available() else "cpu")
    log.info("  N folds  : %d", C.N_GROUP_FOLDS)

    t0 = time.time()
    results = train_full_pipeline(parquet_path)
    elapsed = time.time() - t0

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE in %.1f seconds", elapsed)
    log.info("Group k-fold macro-F1: %.4f", results["group_kfold_metrics"]["macro_f1"])
    log.info("Random split macro-F1: %.4f", results["random_split_metrics"]["macro_f1"])
    log.info("Models saved to: %s", C.MODELS_DIR)
    log.info("=" * 60)

    # ---- Optional explainability ----
    if args.explain:
        log.info("Generating explainability figures...")
        run_explainability_samples(parquet_path, n_samples=args.n_explain)


if __name__ == "__main__":
    main()
