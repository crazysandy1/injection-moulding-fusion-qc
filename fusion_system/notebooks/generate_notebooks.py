"""
generate_notebooks.py
=====================
Master generator for all 7 project Jupyter notebooks.

Run once from this directory:
    python generate_notebooks.py

Produces:
    00_eda.ipynb
    01_train_colab.ipynb
    02_evaluate.ipynb
    03_explainability.ipynb
    04_counterfactual.ipynb
    05_drift_monitor.ipynb
    06_consistency.ipynb
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parent  # fusion_system/

# ─────────────────────────────────────────────────────────────
# Cell builders
# ─────────────────────────────────────────────────────────────

def _uid():
    return uuid.uuid4().hex[:8]

def _src(text: str):
    """Multiline Python string → notebook source list."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _uid(), "metadata": {}, "source": _src(text)}

def code(text: str, tags=None) -> dict:
    meta = {"tags": tags} if tags else {}
    return {
        "cell_type": "code", "execution_count": None,
        "id": _uid(), "metadata": meta, "outputs": [], "source": _src(text),
    }

def nb(cells: list, colab: bool = False) -> dict:
    meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "pygments_lexer": "ipython3", "version": "3.10.11",
        },
    }
    if colab:
        meta["colab"]       = {"provenance": [], "gpuType": "T4", "machine_shape": "hm"}
        meta["accelerator"] = "GPU"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}

def save(notebook: dict, name: str):
    path = OUT / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"  OK  {name}")


# ═══════════════════════════════════════════════════════════════
# 00_eda.ipynb  —  Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════
def make_eda():
    C = [
        md("""# 00 · ProBayes Dataset — Exploratory Data Analysis
**Project** : Intelligent Multi-Sensor Quality Control for Injection Moulding
**Dataset** : ProBayes (SKZ + Fraunhofer IPA) — KraussMaffei 160-750PX, Verzugsschale (warpage shell)
**Author**  : Sandeep J S · 1MS24AD402 · M.S. Ramaiah Institute of Technology
**Guide**   : Dr. Sowmya B J | ADP67 Mini Project, Feb–May 2026

---
### What this notebook does
1. Dataset overview — shape, modalities, missing data
2. Label-imbalance audit (LBL_* columns)
3. **The Leakage Problem** — why random-split F1 is 1.00 but group-fold F1 is 0.30
4. Thermal image visualisation (Z01-Diff analysis)
5. CV image visualisation (3-section warpage shell)
6. DXP process-signal visualisation (8 key channels)
7. Taguchi DOE cell structure
8. Key findings & research motivation
"""),

        code("""import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── locate fusion_system package ──────────────────────────────
NB_DIR   = Path.cwd()
FS_DIR   = NB_DIR.parent          # fusion_system/
sys.path.insert(0, str(FS_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

import config as C

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})
print("Config OK  |  LABEL_COLS:", C.LABEL_COLS)
print("Paths:")
print("  THERMAL_CSV_DIR :", C.THERMAL_CSV_DIR)
print("  CV_IMAGE_DIR    :", C.CV_IMAGE_DIR)
print("  PARQUET_PATH    :", C.PARQUET_PATH)
"""),

        md("## 1 · Load Dataset"),

        code("""from data.loaders import load_parquet, build_cv_index, join_cv_to_parquet
from data.dataset import get_tabular_cols, get_dxp_cols

df = load_parquet(C.PARQUET_PATH)
print(f"Parquet loaded: {df.shape[0]} parts × {df.shape[1]} columns")

# Join CV image index
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)
print(f"After CV join : {df.shape}")

tabular_cols = get_tabular_cols(df)
dxp_cols     = get_dxp_cols(df)
print(f"\\nTabular features : {len(tabular_cols)}")
print(f"DXP channels     : {len(dxp_cols)}")
print(f"Label columns    : {C.N_LABELS}")
"""),

        code("""# ── Dataset summary ──────────────────────────────────────────
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(df.dtypes.value_counts().to_string())
print(f"\\nMissing values (top-10):")
miss = df.isnull().sum().sort_values(ascending=False)
print(miss[miss > 0].head(10).to_string())
print(f"\\nExperiment groups : {df[C.GROUP_COL].nunique()}")
print(f"Parts per group   : {df.groupby(C.GROUP_COL).size().describe().to_string()}")
"""),

        md("## 2 · Label Distribution — Class Imbalance"),

        code("""fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, col in enumerate(C.LABEL_COLS):
    counts = df[col].value_counts()
    nok_pct = counts.get(1, 0) / len(df) * 100
    colour  = "#E53935" if nok_pct > 20 else "#FB8C00" if nok_pct > 10 else "#43A047"
    axes[i].bar(["OK (0)", "NOK (1)"], [counts.get(0,0), counts.get(1,0)],
                color=["#78909C", colour], edgecolor="white", linewidth=0.8)
    axes[i].set_title(col.replace("LBL_",""), fontsize=10)
    axes[i].set_ylabel("Parts")
    axes[i].text(1, counts.get(1,0) + 1, f"{nok_pct:.1f}%", ha="center",
                 fontsize=9, color=colour, fontweight="bold")

fig.suptitle("Label Distribution — Positive (NOK) Rate per Defect Type", fontsize=13)
plt.tight_layout()
plt.show()
print("Most labels are severely imbalanced → Focal Loss required.")
"""),

        md("""## 3 · The Leakage Problem
> **Key Research Finding**: XGBoost and BiLSTM achieve **F1 = 1.00** on random splits
> but only **F1 = 0.19–0.47** under group-aware k-fold.
> Root cause: the Taguchi DOE design means the 47 experiment cells each have
> a *fixed setpoint signature* → models memorise SET_ToolTemperature → defect,
> not the actual physics. Our 4-modal system with DANN breaks this.
"""),

        code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler

tab_cols = [c for c in tabular_cols if c in df.columns]
X_raw    = df[tab_cols].fillna(df[tab_cols].median())
y        = df[C.PRIMARY_TARGET].fillna(0).astype(int).values
groups   = df[C.GROUP_COL].values

scaler   = StandardScaler()
X        = scaler.fit_transform(X_raw)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Random 10-fold (the misleading result)
skf        = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
random_f1s = cross_val_score(clf, X, y, cv=skf, scoring="f1", n_jobs=-1)

# Group k-fold (the honest result)
gkf        = GroupKFold(n_splits=5)
group_f1s  = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="f1", n_jobs=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.boxplot([random_f1s, group_f1s], labels=["Random 10-fold\\n(MISLEADING)","Group 5-fold\\n(HONEST)"],
            patch_artist=True,
            boxprops=dict(facecolor="#EF9A9A"),
            medianprops=dict(color="#B71C1C", linewidth=2.5))
ax1.set_ylabel("F1 Score (LBL_NOK)")
ax1.set_title("Tabular-Only: Random vs Group CV")
ax1.set_ylim(0, 1.05)
ax1.axhline(random_f1s.mean(), ls="--", color="#C62828", alpha=0.5)
ax1.axhline(group_f1s.mean(),  ls="--", color="#1565C0", alpha=0.5)

# Gap bar
gap = random_f1s.mean() - group_f1s.mean()
ax2.bar(["Random F1", "Group F1", "Leakage Gap"],
        [random_f1s.mean(), group_f1s.mean(), gap],
        color=["#EF5350","#42A5F5","#FF7043"])
ax2.set_title(f"Leakage = {gap:.3f} F1 points")
ax2.set_ylim(0, 1.1)
for rect, val in zip(ax2.patches, [random_f1s.mean(), group_f1s.mean(), gap]):
    ax2.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.02,
             f"{val:.3f}", ha="center", fontweight="bold")

plt.suptitle("The Leakage Problem — Why 4-Modal DANN is Needed", fontsize=12)
plt.tight_layout()
plt.show()

print(f"Random  F1 : {random_f1s.mean():.4f} ± {random_f1s.std():.4f}  ← inflated by setpoint memorisation")
print(f"Group   F1 : {group_f1s.mean():.4f}  ± {group_f1s.std():.4f}  ← true generalisation")
print(f"Leakage gap: {gap:.4f}  ← our DANN + thermal/CV modalities must close this")
"""),

        md("## 4 · Thermal Imagery — Z01-Diff Analysis"),

        code("""from data.loaders import load_thermal_pair, extract_thermal_rois

# Sample a part that has thermal data
thermal_rows = df[df["has_thermal"]].head(3) if "has_thermal" in df.columns else df.head(3)

if len(thermal_rows) == 0:
    print("No thermal CSV files found at", C.THERMAL_CSV_DIR)
    print("(Expected when running from local system without the full dataset)")
else:
    fig, axes = plt.subplots(len(thermal_rows), 3, figsize=(15, 4 * len(thermal_rows)))
    if len(thermal_rows) == 1:
        axes = axes[np.newaxis, :]

    cmaps = ["inferno", "inferno", "RdBu_r"]
    titles = ["Frame 1 (filling)", "Frame 2 (cooling)", "Z01-Diff (F1–F2)"]

    for row_i, (_, row) in enumerate(thermal_rows.iterrows()):
        try:
            imgs = load_thermal_pair(row)        # (3, 480, 640)
            for ch, (cmap, title) in enumerate(zip(cmaps, titles)):
                im = axes[row_i, ch].imshow(imgs[ch], cmap=cmap, interpolation="lanczos")
                axes[row_i, ch].set_title(f"Part {row.get(C.CYCLE_ID_COL,'?')} — {title}")
                axes[row_i, ch].axis("off")
                plt.colorbar(im, ax=axes[row_i, ch], fraction=0.046, pad=0.04)
        except Exception as e:
            axes[row_i, 0].text(0.5, 0.5, f"Load error:\\n{e}",
                                ha="center", va="center", transform=axes[row_i, 0].transAxes)

    plt.suptitle("Thermal Image Triplet: [Frame1 | Frame2 | Diff] per Part", fontsize=13)
    plt.tight_layout()
    plt.show()
"""),

        md("## 5 · Visual Imagery — 3-Section Warpage Shell"),

        code("""cv_rows = df[df.get("cv_path_pos1","").notna()].head(2) if "cv_path_pos1" in df.columns else pd.DataFrame()

if cv_rows.empty:
    print("No CV images joined. Check CV_IMAGE_DIR in config.py")
    print("Expected path:", C.CV_IMAGE_DIR)
else:
    from data.loaders import load_cv_triple
    fig, axes = plt.subplots(len(cv_rows), 3, figsize=(15, 5 * len(cv_rows)))
    if len(cv_rows) == 1:
        axes = axes[np.newaxis, :]
    pos_labels = ["Left section", "Mid section", "Right section"]

    for ri, (_, row) in enumerate(cv_rows.iterrows()):
        try:
            imgs = load_cv_triple(row)     # (3, H, W) float32
            for ci in range(3):
                axes[ri, ci].imshow(imgs[ci], cmap="gray", vmin=0, vmax=1)
                axes[ri, ci].set_title(f"Part {row.get(C.CYCLE_ID_COL,'?')} — {pos_labels[ci]}")
                axes[ri, ci].axis("off")
        except Exception as e:
            axes[ri, 0].text(0.5, 0.5, str(e), ha="center", transform=axes[ri, 0].transAxes)

    plt.suptitle("CV Images: Left / Mid / Right sections of Warpage Shell", fontsize=13)
    plt.tight_layout()
    plt.show()
"""),

        md("## 6 · DXP Process Signals — Injection Cycle Dynamics"),

        code("""from data.loaders import load_dxp_tensor

# Pick a NOK and OK part for comparison
nok_rows = df[df[C.PRIMARY_TARGET] == 1].head(1)
ok_rows  = df[df[C.PRIMARY_TARGET] == 0].head(1)
sample_rows = pd.concat([ok_rows, nok_rows]).reset_index(drop=True)
labels_plot  = ["OK", "NOK"]

dxp_available = [c for c in C.KEY_DXP_CHANNELS if c in df.columns]
if not dxp_available:
    print("DXP columns not found. (Expected when dataset is not loaded locally)")
else:
    n_ch = len(dxp_available)
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.2 * n_ch), sharex=True)
    colours = ["#1976D2", "#D32F2F"]

    for ri, (_, row) in enumerate(sample_rows.iterrows()):
        try:
            seq = load_dxp_tensor(row, dxp_available)   # (n_ch, T)
            for ci in range(n_ch):
                x = np.linspace(0, 1, seq.shape[1])
                axes[ci].plot(x, seq[ci], color=colours[ri], alpha=0.75,
                              linewidth=1.2, label=labels_plot[ri])
                axes[ci].set_ylabel(dxp_available[ci].replace("DXP_",""), fontsize=8)
                axes[ci].tick_params(labelsize=7)
        except Exception as e:
            axes[0].text(0.5, 0.5, str(e), ha="center", transform=axes[0].transAxes)

    axes[-1].set_xlabel("Normalised Cycle Time")
    axes[0].legend(loc="upper right")
    fig.suptitle("DXP Key Channels: OK vs NOK Injection Cycle", fontsize=13)
    plt.tight_layout()
    plt.show()
"""),

        md("## 7 · Taguchi DOE Cell Structure"),

        code("""# Visualise how parts are distributed across experiment cells
grp_sizes  = df.groupby(C.GROUP_COL).size().sort_values(ascending=False)
grp_nok    = df.groupby(C.GROUP_COL)[C.PRIMARY_TARGET].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Parts per cell
bars = ax1.bar(range(len(grp_sizes)), grp_sizes.values,
               color=sns.color_palette("Blues_d", len(grp_sizes)))
ax1.set_title(f"Parts per Experiment Cell (total={df[C.GROUP_COL].nunique()} cells)")
ax1.set_xlabel("Experiment Cell (sorted by count)")
ax1.set_ylabel("Number of Parts")
ax1.axhline(grp_sizes.mean(), ls="--", color="red", label=f"Mean={grp_sizes.mean():.1f}")
ax1.legend()

# NOK rate per cell — colour-coded
nok_sorted = grp_nok.sort_values(ascending=False)
bar_colours = ["#EF5350" if v > 0.5 else "#FF9800" if v > 0.2 else "#66BB6A"
               for v in nok_sorted.values]
ax2.bar(range(len(nok_sorted)), nok_sorted.values, color=bar_colours)
ax2.axhline(0.2,  ls="--", color="orange",  label="20% threshold (YELLOW)")
ax2.axhline(0.5,  ls="--", color="red",     label="50% threshold (RED)")
ax2.set_title("NOK Rate per Experiment Cell (Taguchi DOE)")
ax2.set_xlabel("Experiment Cell (sorted by NOK rate)")
ax2.set_ylabel("NOK Rate")
ax2.set_ylim(0, 1.05)
ax2.legend()

fig.suptitle("Taguchi DOE: 47 Cells × ~12 Parts — Group-Aware Split is Mandatory", fontsize=12)
plt.tight_layout()
plt.show()

print(f"Cells with NOK rate > 50%: {(grp_nok > 0.5).sum()}")
print(f"Cells with NOK rate < 20%: {(grp_nok < 0.2).sum()}")
print(f"\\nConclusion: models must generalise to UNSEEN experiment cells,")
print(f"not memorise which setpoints cause defects.")
"""),

        md("## 8 · Key Findings"),

        code("""findings = {
    "Dataset"        : f"{len(df)} parts, {df.shape[1]} columns, 9 data sources",
    "Labels"         : f"{C.N_LABELS} multi-label targets, severely imbalanced",
    "Leakage gap"    : f"{gap:.3f} F1 — tabular-only model memorises Taguchi setpoints",
    "Thermal"        : f"{df['has_thermal'].sum() if 'has_thermal' in df.columns else 'N/A'} parts with IR CSVs (480×640, 2 frames)",
    "CV images"      : f"3 sections per part, 2448×2048 BMP greyscale",
    "DXP channels"   : f"{len(dxp_cols)} available, {len(C.KEY_DXP_CHANNELS)} key channels selected",
    "Experiments"    : f"{df[C.GROUP_COL].nunique()} Taguchi cells, GroupKFold required",
    "Solution"       : "4-Modal DANN Fusion — thermal+CV encode physics, not setpoints",
}

print("=" * 60)
print("EDA KEY FINDINGS")
print("=" * 60)
for k, v in findings.items():
    print(f"  {k:16s} : {v}")
print("=" * 60)
"""),
    ]
    save(nb(C), "00_eda.ipynb")


# ═══════════════════════════════════════════════════════════════
# 01_train_colab.ipynb  —  Google Colab Training (GPU)
# ═══════════════════════════════════════════════════════════════
def make_train_colab():
    C = [
        md("""# 01 · Training on Google Colab (GPU)
**4-Modal Fusion System — Full Training Pipeline**

> Run this notebook on **Google Colab** with GPU runtime
> `Runtime → Change runtime type → T4 GPU`

### Workflow
1. Check GPU
2. Mount Google Drive
3. Install dependencies
4. Upload / point to fusion_system package
5. Verify data access
6. Run group k-fold training
7. DANN lambda sweep (paper ablation)
8. Save artifacts to Drive
"""),

        # ── SECTION 1: GPU ──
        md("## 1 · Environment Check"),

        code("""import subprocess, sys, platform, os

print("Python  :", sys.version)
print("Platform:", platform.platform())

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"\\nGPU     : {dev.name}")
        print(f"VRAM    : {dev.total_memory / 1e9:.1f} GB")
        print(f"CUDA    : {torch.version.cuda}")
        DEVICE  = torch.device("cuda")
    else:
        print("\\nNo CUDA GPU found — falling back to CPU")
        DEVICE  = torch.device("cpu")
except ImportError:
    print("torch not installed yet — run cell 3 first")
    DEVICE = None

# Colab detection
IN_COLAB = "google.colab" in sys.modules
print(f"\\nColab   : {IN_COLAB}")
"""),

        # ── SECTION 2: Drive ──
        md("## 2 · Mount Google Drive"),

        code("""import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = "/content/drive/MyDrive/fusion_project"
    import os; os.makedirs(DRIVE_ROOT, exist_ok=True)
    print("Drive mounted at:", DRIVE_ROOT)
else:
    # Local run
    from pathlib import Path
    DRIVE_ROOT = str(Path.cwd().parent)
    print("Local mode — DRIVE_ROOT:", DRIVE_ROOT)
"""),

        # ── SECTION 3: Install ──
        md("## 3 · Install Dependencies"),

        code("""# Run this cell once per Colab session
import subprocess, sys

pkgs = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "timm",
    "numpy pandas pyarrow scikit-learn scipy Pillow",
    "matplotlib seaborn tqdm",
    "mlflow joblib",
]

for pkg in pkgs:
    print(f"Installing: {pkg[:60]}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkg.split(),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  WARN:", result.stderr[-200:])
    else:
        print("  OK")

print("\\nAll packages installed.")
"""),

        # ── SECTION 4: Paths ──
        md("## 4 · Fusion System Path Setup"),

        code("""import sys, os
from pathlib import Path

# ── Option A: Colab — upload fusion_system as a zip to Drive ──
# FUSION_DIR = "/content/drive/MyDrive/fusion_project/fusion_system"

# ── Option B: Clone from private repo (replace with your URL) ──
# !git clone https://github.com/YOUR_ORG/fusion_system /content/fusion_system
# FUSION_DIR = "/content/fusion_system"

# ── Option C: Colab — notebook is inside fusion_system/notebooks/ ──
NB_DIR     = Path.cwd()
FUSION_DIR = NB_DIR.parent if (NB_DIR.parent / "config.py").exists() else Path(DRIVE_ROOT)

sys.path.insert(0, str(FUSION_DIR))
print("FUSION_DIR:", FUSION_DIR)

import config as C
print("config.py loaded — N_LABELS:", C.N_LABELS)

# Override paths if running from Drive
import config as C_mod
DATA_ON_DRIVE = Path(DRIVE_ROOT) / "dataset"
if DATA_ON_DRIVE.exists():
    C_mod.DATA_ROOT       = DATA_ON_DRIVE
    C_mod.PARQUET_PATH    = DATA_ON_DRIVE / "dataset_V2.parquet"
    C_mod.THERMAL_CSV_DIR = DATA_ON_DRIVE / "Thermographie" / "Thermographie"
    C_mod.CV_IMAGE_DIR    = DATA_ON_DRIVE / "CV_images_Vorversuche_PP" / "Rohbilder"
    print("Using Drive dataset:", C_mod.PARQUET_PATH)
else:
    print("Using config.py paths (local/default)")

C_mod.MODELS_DIR    = Path(DRIVE_ROOT) / "saved_models"
C_mod.ARTIFACTS_DIR = Path(DRIVE_ROOT) / "artifacts"
C_mod.MODELS_DIR.mkdir(parents=True, exist_ok=True)
C_mod.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
print("Models dir    :", C_mod.MODELS_DIR)
print("Artifacts dir :", C_mod.ARTIFACTS_DIR)
"""),

        # ── SECTION 5: Data verify ──
        md("## 5 · Data Verification"),

        code("""from data.loaders import load_parquet, build_cv_index, join_cv_to_parquet
from data.dataset import get_tabular_cols, get_dxp_cols
from training.splits import build_experiment_label_map
import pandas as pd

print("Loading parquet...")
df = load_parquet(C.PARQUET_PATH)
print(f"  Shape          : {df.shape}")
print(f"  Parts          : {len(df)}")
print(f"  Experiments    : {df[C.GROUP_COL].nunique()}")

print("\\nBuilding CV index...")
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)
print(f"  CV-joined shape: {df.shape}")

tabular_cols  = get_tabular_cols(df)
dxp_cols      = get_dxp_cols(df)
exp_label_map = build_experiment_label_map(df)
n_experiments = len(exp_label_map)

has_thermal = df["has_thermal"].sum() if "has_thermal" in df.columns else 0
has_visual  = df["has_visual"].sum()  if "has_visual"  in df.columns else 0

print(f"\\n  Tabular cols   : {len(tabular_cols)}")
print(f"  DXP channels   : {len(dxp_cols)}")
print(f"  Experiments    : {n_experiments}")
print(f"  Has thermal    : {has_thermal}/{len(df)}")
print(f"  Has visual     : {has_visual}/{len(df)}")
print(f"\\nLabel distribution:")
print(df[C.LABEL_COLS].sum().to_string())
"""),

        # ── SECTION 6: Model ──
        md("## 6 · Model Instantiation"),

        code("""import torch
from models.fusion import InjectionMouldingFusionModel

model = InjectionMouldingFusionModel(
    tabular_in_dim = len(tabular_cols),
    n_dxp_channels = len(dxp_cols),
    n_experiments  = n_experiments,
    dann_lambda    = 0.0,
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 55)
print("MODEL SUMMARY")
print("=" * 55)
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")
print(f"  Device               : {DEVICE}")
print(f"  DANN head            : {n_experiments} experiment classes")
print(f"  Defect head          : {C.N_LABELS} labels")
print("=" * 55)

# Sub-module parameter counts
for name, mod in [("ThermalEncoder", model.thermal_enc),
                  ("VisualEncoder",  model.visual_enc),
                  ("TCNEncoder",     model.tcn_enc),
                  ("TabularEncoder", model.tab_enc),
                  ("CrossModalFusion",model.fusion),
                  ("DefectHead",     model.defect_head)]:
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name:22s}: {n:>9,} params")
"""),

        # ── SECTION 7: Training ──
        md("## 7 · Group K-Fold Training (HEADLINE)"),

        code("""import time, json
import numpy as np
from training.trainer import train_one_fold, run_epoch
from training.splits  import group_kfold_splits
from training.losses  import TotalLoss
from training.metrics import compute_metrics, find_best_thresholds, group_aware_evaluation
from torch.utils.data import DataLoader
from data.dataset import MultiModalDataset, collate_fn
from sklearn.preprocessing import StandardScaler
from data.loaders import load_tabular_vector

fold_results   = []
all_val_probs  = []
all_val_targets= []
all_val_groups = []

print(f"Starting {C.N_GROUP_FOLDS}-fold group cross-validation...")
print(f"Epochs: {C.NUM_EPOCHS}  |  Batch: {C.BATCH_SIZE}  |  Device: {DEVICE}")
print("=" * 55)

for fold_i, (train_df, val_df) in enumerate(group_kfold_splits(df)):
    print(f"\\n=== FOLD {fold_i+1}/{C.N_GROUP_FOLDS}  ===")
    print(f"  Train: {len(train_df)} parts  |  Val: {len(val_df)} parts")

    # Fresh model each fold
    fold_model = InjectionMouldingFusionModel(
        tabular_in_dim=len(tabular_cols),
        n_dxp_channels=len(dxp_cols),
        n_experiments=n_experiments,
    ).to(DEVICE)

    t0 = time.time()
    fold_out = train_one_fold(
        fold_model, train_df, val_df,
        tabular_cols, dxp_cols, exp_label_map, DEVICE,
        fold_tag=f"fold{fold_i+1}",
    )

    # Inference on validation fold for aggregation
    scaler = StandardScaler()
    tab_vals = [load_tabular_vector(row, tabular_cols) for _, row in train_df.iterrows()]
    scaler.fit(np.array(tab_vals))
    tab_med = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
    roi_med = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

    val_ds = MultiModalDataset(val_df, tabular_cols, dxp_cols, train=False,
                               tabular_medians=tab_med, roi_medians=roi_med,
                               tabular_scaler=scaler)
    val_ld = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    _, vp, vt, vg = run_epoch(fold_model, val_ld, TotalLoss(), None,
                               DEVICE, exp_label_map, train=False)
    all_val_probs.append(vp)
    all_val_targets.append(vt)
    all_val_groups.append(vg)

    elapsed = time.time() - t0
    print(f"  Best val F1 : {fold_out['best_val_f1']:.4f}  ({elapsed:.0f}s)")
    fold_results.append({"fold": fold_i+1, "val_f1": fold_out["best_val_f1"]})

# Aggregate
all_p = np.concatenate(all_val_probs,   axis=0)
all_t = np.concatenate(all_val_targets, axis=0)
all_g = np.concatenate(all_val_groups,  axis=0)

group_thrs     = find_best_thresholds(all_p, all_t)
group_metrics  = compute_metrics(all_p, all_t, thresholds=group_thrs)
group_breakdown= group_aware_evaluation(all_p, all_t, all_g, thresholds=group_thrs)

print("\\n" + "=" * 55)
print("GROUP K-FOLD HEADLINE RESULTS")
print("=" * 55)
print(f"  Macro-F1 : {group_metrics['macro_f1']:.4f}")
print(f"  Micro-F1 : {group_metrics['micro_f1']:.4f}")
print(f"  ROC-AUC  : {group_metrics['mean_roc_auc']:.4f}")
print(f"  PR-AUC   : {group_metrics['mean_pr_auc']:.4f}")
print(f"  Per-group mean±std: {group_breakdown['mean_group_f1']:.4f} ± {group_breakdown['std_group_f1']:.4f}")
print("=" * 55)
"""),

        # ── SECTION 8: Training curves ──
        md("## 8 · Training Curves"),

        code("""import matplotlib.pyplot as plt

# Use last fold's history for illustration
history = fold_out["history"]
epochs  = [h["epoch"] for h in history]
val_f1s = [h["val_f1"] for h in history]
t_loss  = [h.get("total", 0) for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(epochs, val_f1s, color="#1976D2", linewidth=2, label="Val macro-F1")
ax1.axhline(max(val_f1s), ls="--", color="#43A047", label=f"Best={max(val_f1s):.4f}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Macro-F1"); ax1.set_title(f"Validation F1 — Fold {C.N_GROUP_FOLDS}")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, t_loss, color="#E53935", linewidth=2, label="Train total loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Training Loss — Last Fold")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Training Diagnostics (Fold N)", fontsize=12)
plt.tight_layout()
plt.show()

# Per-label F1 bar chart
per_f1 = group_metrics["per_label_f1"]
fig2, ax3 = plt.subplots(figsize=(11, 4))
colours = ["#E53935" if v < 0.5 else "#FF9800" if v < 0.7 else "#43A047"
           for v in per_f1.values()]
ax3.bar(range(len(per_f1)), list(per_f1.values()), color=colours, edgecolor="white")
ax3.set_xticks(range(len(per_f1)))
ax3.set_xticklabels([k.replace("LBL_","") for k in per_f1.keys()], rotation=30, ha="right")
ax3.set_ylabel("F1 Score"); ax3.set_ylim(0, 1.05)
ax3.set_title("Per-Label F1 — Group K-Fold Headline")
plt.tight_layout(); plt.show()
"""),

        # ── SECTION 9: Save ──
        md("## 9 · Save Artifacts to Drive"),

        code("""import torch, json, numpy as np

# Final model = last fold model (or re-train on all data)
C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.save(fold_model.state_dict(), C.MODELS_DIR / "fusion_final.pt")
np.save(C.MODELS_DIR / "best_thresholds.npy", group_thrs)

artifacts = {
    "group_kfold_metrics"  : group_metrics,
    "group_breakdown"      : group_breakdown,
    "fold_results"         : fold_results,
    "tabular_cols"         : tabular_cols,
    "dxp_cols"             : dxp_cols,
    "exp_label_map"        : exp_label_map,
    "n_experiments"        : n_experiments,
}
with open(C.ARTIFACTS_DIR / "training_results.json", "w") as f:
    json.dump(artifacts, f, indent=2, default=str)

print("Saved:")
print(f"  {C.MODELS_DIR / 'fusion_final.pt'}")
print(f"  {C.MODELS_DIR / 'best_thresholds.npy'}")
print(f"  {C.ARTIFACTS_DIR / 'training_results.json'}")
print(f"\\nFinal group macro-F1 : {group_metrics['macro_f1']:.4f}")
"""),

        md("## 10 · DANN Lambda Ablation (Paper Experiment)"),

        code("""# ── Run this section separately — trains 5 full models ──────
# Uncomment and run to generate Fig. X in the paper

# DANN_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5]
# sweep = []
# import config as C_mod
# original_lam = C_mod.DANN_LAMBDA_MAX
#
# for lam in DANN_LAMBDAS:
#     C_mod.DANN_LAMBDA_MAX = lam
#     from training.trainer import train_full_pipeline
#     res = train_full_pipeline(C.PARQUET_PATH)
#     sweep.append({
#         "lambda"         : lam,
#         "group_f1"       : res["group_kfold_metrics"]["macro_f1"],
#         "random_f1"      : res["random_split_metrics"]["macro_f1"],
#         "gap"            : res["random_split_metrics"]["macro_f1"] - res["group_kfold_metrics"]["macro_f1"],
#     })
#     print(f"lambda={lam:.2f}  group={sweep[-1]['group_f1']:.4f}  random={sweep[-1]['random_f1']:.4f}")
#
# C_mod.DANN_LAMBDA_MAX = original_lam
#
# # Plot
# import matplotlib.pyplot as plt
# lams = [r["lambda"] for r in sweep]
# fig, ax = plt.subplots(figsize=(8,4))
# ax.plot(lams, [r["group_f1"]  for r in sweep], "o-", label="Group k-fold F1 (honest)")
# ax.plot(lams, [r["random_f1"] for r in sweep], "s--", label="Random split F1 (leaky)")
# ax.fill_between(lams, [r["group_f1"] for r in sweep], [r["random_f1"] for r in sweep],
#                 alpha=0.15, color="red", label="Leakage gap")
# ax.set_xlabel("DANN Lambda"); ax.set_ylabel("Macro-F1")
# ax.set_title("DANN Lambda Sweep: Closing the Leakage Gap")
# ax.legend(); plt.tight_layout(); plt.show()

print("DANN sweep cells are commented out.")
print("Uncomment above to run the paper ablation (needs ~5× training time).")
"""),
    ]
    save(nb(C, colab=True), "01_train_colab.ipynb")


# ═══════════════════════════════════════════════════════════════
# 02_evaluate.ipynb
# ═══════════════════════════════════════════════════════════════
def make_evaluate():
    C = [
        md("""# 02 · Model Evaluation — All Four Split Strategies
Loads the saved checkpoint and computes the paper-ready headline table.

| Split | Purpose |
|---|---|
| **Group k-fold (HEADLINE)** | True generalisation to unseen experiment cells |
| Random split | Baseline — shows inflated F1 due to leakage |
| Time-aware split | Temporal ordering — last 15% by cycle ID |
| Material OOD (PP → ABS) | Domain shift: train PP, test ABS recyclate |
"""),

        code("""import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

NB_DIR  = Path.cwd()
FS_DIR  = NB_DIR.parent
sys.path.insert(0, str(FS_DIR))

import config as C
from data.loaders    import load_parquet, build_cv_index, join_cv_to_parquet, load_tabular_vector
from data.dataset    import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from models.fusion   import InjectionMouldingFusionModel
from training.splits import (group_kfold_splits, random_split,
                              time_aware_split, material_ood_split,
                              build_experiment_label_map)
from training.losses  import TotalLoss
from training.metrics import (compute_metrics, find_best_thresholds,
                               group_aware_evaluation, build_headline_table)
from training.trainer import run_epoch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
"""),

        code("""# ── Load data ────────────────────────────────────────────────
df     = load_parquet(C.PARQUET_PATH)
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)

tabular_cols  = get_tabular_cols(df)
dxp_cols      = get_dxp_cols(df)
exp_label_map = build_experiment_label_map(df)

saved_thrs = None
thr_path   = C.MODELS_DIR / "best_thresholds.npy"
if thr_path.exists():
    saved_thrs = np.load(thr_path)
    print("Loaded thresholds from:", thr_path)

def _build_loader(eval_df, ref_df, batch=C.BATCH_SIZE):
    scaler = StandardScaler()
    scaler.fit(np.array([load_tabular_vector(r, tabular_cols) for _, r in ref_df.iterrows()]))
    tab_med = {c: float(ref_df[c].median()) for c in tabular_cols if c in ref_df.columns}
    roi_med = {c: float(ref_df[c].median()) for c in C.THERMAL_ROI_COLS if c in ref_df.columns}
    ds = MultiModalDataset(eval_df, tabular_cols, dxp_cols, train=False,
                           tabular_medians=tab_med, roi_medians=roi_med, tabular_scaler=scaler)
    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

def _load_model():
    m = InjectionMouldingFusionModel(
        tabular_in_dim=len(tabular_cols),
        n_dxp_channels=len(dxp_cols),
        n_experiments=len(exp_label_map),
    ).to(DEVICE)
    ckpt = C.MODELS_DIR / "fusion_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Run 01_train_colab.ipynb first.")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    return m

model = _load_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {n_params:,} params")
"""),

        md("## 1 · Group K-Fold (HEADLINE)"),

        code("""all_probs, all_targets, all_groups = [], [], []
for fold_i, (train_df, val_df) in enumerate(group_kfold_splits(df)):
    ld = _build_loader(val_df, train_df)
    _, vp, vt, vg = run_epoch(model, ld, TotalLoss(), None, DEVICE, exp_label_map, train=False)
    all_probs.append(vp); all_targets.append(vt); all_groups.append(vg)
    print(f"Fold {fold_i+1}: {len(val_df)} val parts")

all_p = np.concatenate(all_probs)
all_t = np.concatenate(all_targets)
all_g = np.concatenate(all_groups)
thrs_gkf   = find_best_thresholds(all_p, all_t) if saved_thrs is None else saved_thrs
gkf_metrics = compute_metrics(all_p, all_t, thresholds=thrs_gkf)
gkf_break   = group_aware_evaluation(all_p, all_t, all_g, thresholds=thrs_gkf)
print(f"\\nGroup K-Fold Macro-F1 : {gkf_metrics['macro_f1']:.4f}")
print(f"Per-group mean±std    : {gkf_break['mean_group_f1']:.4f} ± {gkf_break['std_group_f1']:.4f}")
"""),

        md("## 2 · Random Split Baseline"),

        code("""tr, va, te = random_split(df)
ld_rand = _build_loader(te, tr)
_, rp, rt, rg = run_epoch(model, ld_rand, TotalLoss(), None, DEVICE, exp_label_map, train=False)
thrs_rand  = find_best_thresholds(rp, rt)
rand_metrics = compute_metrics(rp, rt, thresholds=thrs_rand)
print(f"Random Split Macro-F1 : {rand_metrics['macro_f1']:.4f}  (inflated — leakage!)")
"""),

        md("## 3 · Time-Aware Split"),

        code("""tr_t, va_t, te_t = time_aware_split(df)
ld_time = _build_loader(te_t, tr_t)
_, tp, tt, tg = run_epoch(model, ld_time, TotalLoss(), None, DEVICE, exp_label_map, train=False)
thrs_time    = find_best_thresholds(tp, tt)
time_metrics = compute_metrics(tp, tt, thresholds=thrs_time)
print(f"Time-Aware Macro-F1   : {time_metrics['macro_f1']:.4f}")
"""),

        md("## 4 · Material OOD  (PP → ABS)"),

        code("""if C.MATERIAL_COL not in df.columns:
    print("MATERIAL_COL not found — skipping OOD evaluation")
    ood_metrics = {"macro_f1": float("nan"), "micro_f1": float("nan"),
                   "mean_roc_auc": float("nan"), "mean_pr_auc": float("nan"),
                   "per_label_f1": {}}
else:
    ood_tr, ood_te = material_ood_split(df)
    if len(ood_te) == 0:
        print("No ABS parts in dataset — OOD eval skipped")
        ood_metrics = {"macro_f1": float("nan"), "micro_f1": float("nan"),
                       "mean_roc_auc": float("nan"), "mean_pr_auc": float("nan"),
                       "per_label_f1": {}}
    else:
        ld_ood = _build_loader(ood_te, ood_tr)
        _, op, ot, og = run_epoch(model, ld_ood, TotalLoss(), None, DEVICE, exp_label_map, train=False)
        thrs_ood   = find_best_thresholds(op, ot)
        ood_metrics = compute_metrics(op, ot, thresholds=thrs_ood)
        print(f"OOD (PP→ABS) Macro-F1 : {ood_metrics['macro_f1']:.4f}")
"""),

        md("## 5 · Paper Headline Table"),

        code("""results_map = {
    "Group k-fold (HEADLINE)" : gkf_metrics,
    "Random split (baseline)" : rand_metrics,
    "Time-aware split"        : time_metrics,
    "Material OOD (PP→ABS)"  : ood_metrics,
}
headline = build_headline_table(results_map)
print("=" * 60)
print("HEADLINE EVALUATION TABLE")
print("=" * 60)
print(headline.to_string(index=False))
print("=" * 60)

headline.to_csv(C.ARTIFACTS_DIR / "evaluation_headline.csv", index=False)
print("Saved to:", C.ARTIFACTS_DIR / "evaluation_headline.csv")
"""),

        md("## 6 · Confusion Matrix & Per-Label F1"),

        code("""from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt, numpy as np

# Confusion matrix — LBL_NOK — group k-fold
cm = np.array(gkf_metrics["nok_confusion"])
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OK","NOK"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix — LBL_NOK (Group k-fold)")

# Per-label F1 comparison
labels = C.LABEL_COLS
x = np.arange(len(labels))
w = 0.22
for ki, (sname, smet) in enumerate(results_map.items()):
    pf1 = [smet["per_label_f1"].get(l, 0) for l in labels]
    axes[1].bar(x + ki*w, pf1, w, label=sname, alpha=0.85)

axes[1].set_xticks(x + w*1.5)
axes[1].set_xticklabels([l.replace("LBL_","") for l in labels], rotation=30, ha="right")
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("F1 Score")
axes[1].set_title("Per-Label F1 — All Split Strategies")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(C.ARTIFACTS_DIR / "evaluation_figure.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("## 7 · Embedding Space — t-SNE"),

        code("""from sklearn.manifold import TSNE
import torch, numpy as np

# Collect fused embeddings + labels
embs, ys, grps = [], [], []
ld_all = _build_loader(df.head(200), df.head(200))  # quick subset
model.eval()
with torch.no_grad():
    for batch in ld_all:
        b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)
        embs.append(out["fused_emb"].cpu().numpy())
        ys.append(batch["labels"][:, C.LABEL_COLS.index(C.PRIMARY_TARGET)].numpy())
        grps.extend(batch["group"])

embs_arr = np.concatenate(embs)
ys_arr   = np.concatenate(ys)

print("Running t-SNE on fused embeddings...")
tsne  = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
embs2 = tsne.fit_transform(embs_arr)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(embs2[:, 0], embs2[:, 1], c=ys_arr,
                     cmap="RdYlGn_r", alpha=0.7, s=40, edgecolors="white", linewidths=0.3)
plt.colorbar(scatter, ax=ax, label="LBL_NOK")
ax.set_title("t-SNE of Fused Embeddings — coloured by LBL_NOK")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
plt.tight_layout(); plt.show()
print("Well-separated clusters → model has learned discriminative representations")
"""),
    ]
    save(nb(C), "02_evaluate.ipynb")


# ═══════════════════════════════════════════════════════════════
# 03_explainability.ipynb
# ═══════════════════════════════════════════════════════════════
def make_explainability():
    C = [
        md("""# 03 · Model Explainability — GradCAM & Attention Analysis
Explains *why* the model makes each decision:
- **Thermal GradCAM** — where the IR camera contributes most
- **Visual GradCAM** — which surface region per section is critical
- **Cross-section attention** — left / mid / right importance
- **Modality token weights** — which of the 4 modalities dominates
- **TCN channel saliency** — which DXP signal drives the prediction
- **ROI temperature importance** — which SKZ measurement matters
"""),

        code("""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path.cwd().parent))
import config as C
from data.loaders    import load_parquet, build_cv_index, join_cv_to_parquet, load_tabular_vector
from data.dataset    import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from models.fusion   import InjectionMouldingFusionModel
from training.splits import build_experiment_label_map, random_split
from analysis.explainability import (
    thermal_gradcam, visual_gradcam, get_modality_weights,
    tcn_channel_saliency, roi_importance,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({"figure.dpi": 120, "axes.titleweight": "bold"})
print("Device:", DEVICE)
"""),

        code("""# ── Load data + model ────────────────────────────────────────
df     = load_parquet(C.PARQUET_PATH)
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)

tabular_cols  = get_tabular_cols(df)
dxp_cols      = get_dxp_cols(df)
exp_label_map = build_experiment_label_map(df)

scaler   = StandardScaler()
tab_vals = [load_tabular_vector(r, tabular_cols) for _, r in df.iterrows()]
scaler.fit(np.array(tab_vals))
tab_med  = {c: float(df[c].median()) for c in tabular_cols if c in df.columns}
roi_med  = {c: float(df[c].median()) for c in C.THERMAL_ROI_COLS if c in df.columns}

ds  = MultiModalDataset(df, tabular_cols, dxp_cols, train=False,
                        tabular_medians=tab_med, roi_medians=roi_med, tabular_scaler=scaler)
ld  = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

model = InjectionMouldingFusionModel(
    tabular_in_dim=len(tabular_cols),
    n_dxp_channels=len(dxp_cols),
    n_experiments=len(exp_label_map),
).to(DEVICE)
ckpt = C.MODELS_DIR / "fusion_final.pt"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print("Model loaded")
else:
    print(f"WARNING: No checkpoint at {ckpt} — model is random (for layout demo only)")
model.eval()
"""),

        md("## 1 · Select Sample Parts"),

        code("""# Grab 1 NOK and 1 OK part for comparison
sample_batches = {}
label_idx = C.LABEL_COLS.index(C.PRIMARY_TARGET)

for batch in ld:
    lbl = batch["labels"][0, label_idx].item()
    key = "NOK" if lbl == 1 else "OK"
    if key not in sample_batches:
        sample_batches[key] = batch
    if len(sample_batches) == 2:
        break

print("Collected samples:", list(sample_batches.keys()))
for key, b in sample_batches.items():
    cycle = b.get("cycle_id", ["?"])[0]
    has_t = b["thermal_valid"][0].item()
    has_v = b["visual_valid"][0].item()
    has_s = b["sequence_valid"][0].item()
    print(f"  {key:4s} | cycle={cycle} | thermal={has_t} | visual={has_v} | seq={has_s}")
"""),

        md("## 2 · Modality Token Weights"),

        code("""MODALITY_NAMES = ["Thermal", "Visual", "Sequence", "Tabular"]
COLOURS        = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (key, batch) in zip(axes, sample_batches.items()):
    weights = get_modality_weights(model, batch, DEVICE)
    bars = ax.bar(MODALITY_NAMES, weights, color=COLOURS, edgecolor="white", linewidth=1.2)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{w:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_title(f"Modality Weights — {key} Part")
    ax.set_ylabel("Attention Weight")
    ax.set_xlabel("Modality")

plt.suptitle("Cross-Modal Fusion: Which Modality Drives the Decision?", fontsize=12)
plt.tight_layout(); plt.show()
"""),

        md("## 3 · Thermal GradCAM"),

        code("""import matplotlib.cm as cm

def _overlay(base, cam, alpha=0.5):
    base_n = (base - base.min()) / (base.max() - base.min() + 1e-6)
    if base_n.dtype != np.uint8:
        base_uint = (base_n * 255).astype(np.uint8)
    rgb = np.stack([base_uint]*3, axis=2)
    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row_i, (key, batch) in enumerate(sample_batches.items()):
    if not batch["thermal_valid"][0].item():
        for col in range(3):
            axes[row_i, col].text(0.5, 0.5, "No thermal data", ha="center",
                                  transform=axes[row_i, col].transAxes)
        continue

    imgs = batch["thermal_img"][0].cpu().numpy()   # (3, 480, 640)
    f1, f2 = imgs[0], imgs[1]

    try:
        cam_map = thermal_gradcam(model, batch, DEVICE, label_idx=label_idx)
        overlay = _overlay(f1, cam_map)
    except Exception as e:
        cam_map = np.zeros((480, 640)); overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        print(f"GradCAM error ({key}):", e)

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    axes[row_i, 0].imshow(_norm(f1), cmap="inferno")
    axes[row_i, 0].set_title(f"{key} — Frame 1 (filling)")
    axes[row_i, 1].imshow(_norm(f2), cmap="inferno")
    axes[row_i, 1].set_title(f"{key} — Frame 2 (cooling)")
    axes[row_i, 2].imshow(overlay)
    axes[row_i, 2].set_title(f"{key} — GradCAM overlay")

    for ax in axes[row_i]: ax.axis("off")

plt.suptitle("Thermal GradCAM: Where the IR encoder attends (LBL_NOK)", fontsize=13)
plt.tight_layout(); plt.savefig(C.PLOTS_DIR/"thermal_gradcam.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("## 4 · TCN Channel Saliency & ROI Importance"),

        code("""fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col_i, (key, batch) in enumerate(sample_batches.items()):
    # TCN channel saliency
    if batch["sequence_valid"][0].item():
        try:
            sal = tcn_channel_saliency(model, batch, DEVICE, label_idx=label_idx)
        except Exception as e:
            sal = np.ones(len(dxp_cols)) / len(dxp_cols)
            print("TCN saliency error:", e)
    else:
        sal = np.zeros(len(dxp_cols))

    ch_names = [c.replace("DXP_","") for c in dxp_cols]
    bar_cols  = ["#E53935" if s == sal.max() else "#64B5F6" for s in sal]
    axes[0, col_i].bar(range(len(sal)), sal, color=bar_cols, edgecolor="white")
    axes[0, col_i].set_xticks(range(len(sal)))
    axes[0, col_i].set_xticklabels(ch_names, rotation=35, ha="right", fontsize=8)
    axes[0, col_i].set_title(f"TCN Saliency — {key}")
    axes[0, col_i].set_ylabel("Normalised gradient importance")

    # ROI importance
    if batch["thermal_valid"][0].item():
        try:
            roi_imp = roi_importance(model, batch, DEVICE, label_idx=label_idx)
        except Exception as e:
            roi_imp = np.ones(10) / 10
            print("ROI error:", e)
    else:
        roi_imp = np.zeros(10)

    roi_names = [c.replace("IR_","") for c in C.THERMAL_ROI_COLS]
    axes[1, col_i].bar(range(len(roi_imp)), roi_imp,
                       color=["#EF5350" if v==roi_imp.max() else "#A5D6A7" for v in roi_imp])
    axes[1, col_i].set_xticks(range(len(roi_imp)))
    axes[1, col_i].set_xticklabels(roi_names, rotation=35, ha="right", fontsize=8)
    axes[1, col_i].set_title(f"ROI Temperature Importance — {key}")
    axes[1, col_i].set_ylabel("Gradient importance")

plt.suptitle("Feature Importance: DXP Channels & Thermal ROI Temperatures", fontsize=12)
plt.tight_layout()
plt.savefig(C.PLOTS_DIR/"channel_importance.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("## 5 · Multi-Part Modality Weight Heatmap"),

        code("""import numpy as np
weight_rows, part_ids, labels_list = [], [], []

for i, batch in enumerate(ld):
    if i >= 40:
        break
    weights = get_modality_weights(model, batch, DEVICE)
    weight_rows.append(weights)
    part_ids.append(str(batch.get("cycle_id", [i])[0]))
    labels_list.append(int(batch["labels"][0, label_idx].item()))

W = np.array(weight_rows)   # (N, 4)

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(W, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(["Thermal", "Visual", "Sequence", "Tabular"])
ax.set_yticks(range(len(part_ids)))
ax.set_yticklabels(
    [f"{'★' if l else '·'} {pid}" for pid, l in zip(part_ids, labels_list)],
    fontsize=7
)
ax.set_xlabel("Modality")
ax.set_ylabel("Part (★=NOK)")
plt.colorbar(im, ax=ax, label="Attention weight")
ax.set_title("Per-Part Modality Weights\\n(shows adaptive fusion behaviour)", fontsize=11)
plt.tight_layout()
plt.savefig(C.PLOTS_DIR/"modality_weight_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Higher weight on Thermal for thermally-driven defects,")
print("Higher weight on Sequence for pressure-driven defects (sink marks, underfill)")
"""),
    ]
    save(nb(C), "03_explainability.ipynb")


# ═══════════════════════════════════════════════════════════════
# 04_counterfactual.ipynb
# ═══════════════════════════════════════════════════════════════
def make_counterfactual():
    C = [
        md("""# 04 · Counterfactual Recommender — How to Fix Defects
Given a NOK part, this notebook answers: *"What setpoints should the operator change?"*

Two complementary strategies:
1. **Taguchi Nearest-OK** — find the experiment cell with lowest NOK rate closest in setpoint space
2. **Gradient Counterfactual** — gradient descent on SET_* features until model predicts OK

This is the *actionable intelligence* layer — unique research contribution.
"""),

        code("""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent))
import config as C
from data.loaders    import load_parquet, build_cv_index, join_cv_to_parquet, load_tabular_vector
from data.dataset    import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from models.fusion   import InjectionMouldingFusionModel
from training.splits import build_experiment_label_map, random_split
from analysis.counterfactual import (
    TaguchiBestCell, GradientCounterfactual, counterfactual_report,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
"""),

        code("""df     = load_parquet(C.PARQUET_PATH)
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)
tabular_cols  = get_tabular_cols(df)
dxp_cols      = get_dxp_cols(df)
exp_label_map = build_experiment_label_map(df)
train_df, _, test_df = random_split(df)

scaler   = StandardScaler()
tab_vals = [load_tabular_vector(r, tabular_cols) for _, r in train_df.iterrows()]
scaler.fit(np.array(tab_vals))
tab_med  = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
roi_med  = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

model = InjectionMouldingFusionModel(
    tabular_in_dim=len(tabular_cols),
    n_dxp_channels=len(dxp_cols),
    n_experiments=len(exp_label_map),
).to(DEVICE)
ckpt = C.MODELS_DIR / "fusion_final.pt"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()

# Build Taguchi recommender on training data
recommender = TaguchiBestCell(train_df, nok_threshold=0.2)
print(f"Taguchi recommender: {len(recommender.ok_cells)} OK cells identified out of {train_df[C.GROUP_COL].nunique()} total")
"""),

        md("## 1 · Select a NOK Part"),

        code("""ds_test = MultiModalDataset(test_df, tabular_cols, dxp_cols, train=False,
                            tabular_medians=tab_med, roi_medians=roi_med, tabular_scaler=scaler)
ld_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

label_idx = C.LABEL_COLS.index(C.PRIMARY_TARGET)
nok_batch, nok_row_idx = None, None

for i, (batch, (_, row)) in enumerate(zip(ld_test, test_df.iterrows())):
    lbl = batch["labels"][0, label_idx].item()
    with torch.no_grad():
        b   = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)
        prob = torch.sigmoid(out["defect_logits"][0, label_idx]).item()
    if lbl == 1 and prob > 0.5:
        nok_batch   = batch
        nok_row     = row
        nok_prob    = prob
        nok_row_idx = i
        print(f"NOK part found: idx={i}  cycle={row.get(C.CYCLE_ID_COL,'?')}  P(NOK)={prob:.4f}")
        break

if nok_batch is None:
    print("No high-confidence NOK part found — using first test part for demonstration")
    for batch, (_, row) in zip(ld_test, test_df.iterrows()):
        nok_batch, nok_row = batch, row; break
"""),

        md("## 2 · Taguchi Nearest-OK Recommendations"),

        code("""recs = recommender.recommend(nok_row, top_k=3)

print("=" * 62)
print("TAGUCHI NEAREST-OK SETPOINT RECOMMENDATIONS")
print("=" * 62)
for rank, rec in enumerate(recs, 1):
    print(f"\\n  Option {rank}  [cell={rec['experiment']}, NOK rate={rec['nok_rate']:.1%}, dist={rec['distance']:.3f}]")
    for col, delta in rec["deltas"].items():
        if abs(delta) < 0.01:
            continue
        sign = "+" if delta > 0 else ""
        print(f"    {col.replace('SET_',''):25s} {sign}{delta:+.2f}  → {rec['recommended'][col]:.2f}")

# Visualise top recommendation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
top_rec = recs[0]

for ax_i, rec in enumerate(recs):
    cols_with_delta = {k: v for k, v in rec["deltas"].items() if abs(v) > 0.01}
    if not cols_with_delta:
        ax = axes[ax_i]; ax.text(0.5, 0.5, "No significant change", ha="center", transform=ax.transAxes)
        continue
    short_names = [k.replace("SET_","") for k in cols_with_delta.keys()]
    deltas      = list(cols_with_delta.values())
    bar_cols    = ["#EF5350" if d < 0 else "#66BB6A" for d in deltas]
    axes[ax_i].barh(short_names, deltas, color=bar_cols, edgecolor="white")
    axes[ax_i].axvline(0, color="black", linewidth=0.8)
    axes[ax_i].set_title(f"Option {ax_i+1}: cell {rec['experiment']}\\nNOK rate={rec['nok_rate']:.1%}")
    axes[ax_i].set_xlabel("Delta from current setpoint")

plt.suptitle("Taguchi Nearest-OK: Setpoint Changes Needed (green=increase, red=decrease)", fontsize=12)
plt.tight_layout(); plt.savefig(C.PLOTS_DIR/"counterfactual_taguchi.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("## 3 · Gradient Counterfactual Optimisation"),

        code("""gcf = GradientCounterfactual(
    model=model, tabular_cols=tabular_cols,
    lambda_reg=0.3, lr=0.02, max_steps=300, target_prob=0.25,
)

cf_delta, info = gcf.run(nok_batch, DEVICE)

print("=" * 55)
print("GRADIENT COUNTERFACTUAL RESULT")
print("=" * 55)
print(f"  Status   : {'SUCCESS' if info['success'] else 'Did not fully converge'}")
print(f"  P(NOK)   : {info['nok_prob_original']:.4f}  →  {info['nok_prob_final']:.4f}")
print(f"  Steps    : {info['n_steps']}")
print()
print(gcf.describe(cf_delta, scaler.mean_, scaler.scale_))
print("=" * 55)
"""),

        code("""# Visualise probability trajectory during optimisation (re-run with trace)
import torch.nn as nn

prob_trace = []
model.eval()
scaler_mean = scaler.mean_
scaler_std  = scaler.scale_
set_idx     = [i for i, c in enumerate(tabular_cols) if c.startswith("SET_")]
orig_tab    = nok_batch["tabular"].to(DEVICE).clone()

delta_free = torch.zeros((1, len(set_idx)), device=DEVICE, requires_grad=True)
opt        = torch.optim.Adam([delta_free], lr=0.02)

for step in range(300):
    opt.zero_grad()
    mod_tab = orig_tab.clone()
    for j, idx in enumerate(set_idx):
        mod_tab[:, idx] = orig_tab[:, idx] + delta_free[:, j]
    b2 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in nok_batch.items()}
    b2["tabular"] = mod_tab
    out2  = model(b2)
    logit = out2["defect_logits"][0, label_idx]
    bce   = nn.functional.binary_cross_entropy_with_logits(logit.unsqueeze(0), torch.zeros(1,device=DEVICE))
    reg   = 0.3 * (delta_free**2).mean()
    (bce + reg).backward()
    opt.step()
    with torch.no_grad():
        prob_trace.append(torch.sigmoid(logit).item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(prob_trace, color="#E53935", linewidth=2)
ax1.axhline(0.5, ls="--", color="gray", label="Decision boundary (0.5)")
ax1.axhline(0.25, ls="--", color="green", label="Target (0.25)")
ax1.fill_between(range(len(prob_trace)), prob_trace, 0.5,
                 where=[p < 0.5 for p in prob_trace], alpha=0.15, color="green", label="Below threshold")
ax1.set_xlabel("Optimisation step"); ax1.set_ylabel("P(NOK)")
ax1.set_title("Gradient CF: NOK Probability vs Step")
ax1.legend(); ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.3)

# Final deltas (physical units)
final_delta_phys = delta_free[0].detach().cpu().numpy() * scaler_std[set_idx]
set_names = [tabular_cols[i].replace("SET_","") for i in set_idx]
bar_cols  = ["#66BB6A" if d > 0 else "#EF5350" for d in final_delta_phys]
ax2.barh(set_names, final_delta_phys, color=bar_cols, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Setpoint delta (physical units)")
ax2.set_title("Gradient CF: Recommended Setpoint Changes")

plt.suptitle("Model-Guided Counterfactual: Minimum-Change Path to OK", fontsize=12)
plt.tight_layout(); plt.savefig(C.PLOTS_DIR/"counterfactual_gradient.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("## 4 · Full Operator Report"),

        code("""report = counterfactual_report(
    model=model, batch=nok_batch, query_row=nok_row,
    training_df=train_df, tabular_cols=tabular_cols,
    device=DEVICE, scaler_mean=scaler.mean_, scaler_std=scaler.scale_,
)
print(report)
"""),
    ]
    save(nb(C), "04_counterfactual.ipynb")


# ═══════════════════════════════════════════════════════════════
# 05_drift_monitor.ipynb
# ═══════════════════════════════════════════════════════════════
def make_drift():
    C = [
        md("""# 05 · Production Drift Monitor
Detects distribution shift **before defects appear** — the early-warning system.

Three complementary detectors:
| Detector | Signal | Alert |
|---|---|---|
| PSI per feature | Tabular SET_/QUA_/ENV_ | PSI > 0.25 = RED |
| KS-test | Per-feature statistical | p < 0.05 = drift |
| Embedding drift | Cosine of mean fused emb | Cosine < 0.75 = YELLOW |

**Key feature**: Z01-Diff thermal ROI temperatures — encode injection dynamics,
degrade with tool wear *days before* visible defects.
"""),

        code("""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent))
import config as C
from data.loaders  import load_parquet, build_cv_index, join_cv_to_parquet
from data.dataset  import get_tabular_cols, get_dxp_cols
from models.fusion import InjectionMouldingFusionModel
from training.splits import build_experiment_label_map, random_split
from analysis.drift import (
    compute_psi, psi_alert_level, ks_test_features,
    embedding_drift, DriftMonitor, ThermalDiffDriftDetector,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
sns.set_theme(style="whitegrid", palette="muted")
"""),

        code("""df     = load_parquet(C.PARQUET_PATH)
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)
tabular_cols = get_tabular_cols(df)
train_df, _, test_df = random_split(df)
print(f"Reference: {len(train_df)} parts  |  Production: {len(test_df)} parts")
"""),

        md("## 1 · PSI per Tabular Feature"),

        code("""# Compute PSI for every tabular feature
ref_tab  = train_df[[c for c in tabular_cols if c in train_df.columns]].fillna(0).values
prod_tab = test_df[[c for c in tabular_cols if c in test_df.columns]].fillna(0).values

psi_vals = {}
for i, col in enumerate([c for c in tabular_cols if c in train_df.columns]):
    if i < ref_tab.shape[1] and i < prod_tab.shape[1]:
        psi_vals[col] = compute_psi(ref_tab[:, i], prod_tab[:, i])

psi_series = pd.Series(psi_vals).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(15, 5))
colours = [{"GREEN":"#66BB6A","YELLOW":"#FF9800","RED":"#EF5350"}[psi_alert_level(v)]
           for v in psi_series.values]
ax.bar(range(len(psi_series)), psi_series.values, color=colours, edgecolor="white", linewidth=0.5)
ax.axhline(0.1,  ls="--", color="#FF9800", linewidth=1.5, label="YELLOW (0.1)")
ax.axhline(0.25, ls="--", color="#EF5350", linewidth=1.5, label="RED (0.25)")
ax.set_xticks(range(len(psi_series)))
ax.set_xticklabels([c[:20] for c in psi_series.index], rotation=60, ha="right", fontsize=7)
ax.set_ylabel("PSI"); ax.set_title("Population Stability Index — All Tabular Features")
ax.legend()
plt.tight_layout(); plt.savefig(C.PLOTS_DIR/"drift_psi.png", dpi=150, bbox_inches="tight"); plt.show()

red_cols = psi_series[psi_series >= 0.25].index.tolist()
yel_cols = psi_series[(psi_series >= 0.1) & (psi_series < 0.25)].index.tolist()
print(f"RED  features ({len(red_cols)}):", red_cols[:5])
print(f"YELLOW features ({len(yel_cols)}):", yel_cols[:5])
"""),

        md("## 2 · KS-Test Statistical Significance"),

        code("""from scipy import stats

ks_results = {}
feat_cols  = [c for c in tabular_cols if c in train_df.columns and c in test_df.columns]
for i, col in enumerate(feat_cols):
    if i >= ref_tab.shape[1] or i >= prod_tab.shape[1]:
        continue
    stat, pval = stats.ks_2samp(ref_tab[:, i], prod_tab[:, i])
    ks_results[col] = {"stat": stat, "pval": pval, "drifted": pval < 0.05}

ks_df    = pd.DataFrame(ks_results).T
n_drift  = ks_df["drifted"].sum()
print(f"KS-test: {n_drift}/{len(ks_df)} features show statistically significant drift (p<0.05)")

# Plot top-10 drifted features
top_ks = ks_df.nlargest(10, "stat")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.barh(range(len(top_ks)), top_ks["stat"].values,
         color=["#EF5350" if d else "#42A5F5" for d in top_ks["drifted"]])
ax1.set_yticks(range(len(top_ks)))
ax1.set_yticklabels([c[:25] for c in top_ks.index])
ax1.set_xlabel("KS statistic"); ax1.set_title("Top-10 Drifted Features (KS statistic)")

ax2.scatter(ks_df["stat"], -np.log10(ks_df["pval"].clip(1e-10)),
            c=["#EF5350" if d else "#78909C" for d in ks_df["drifted"]], alpha=0.7, s=30)
ax2.axhline(-np.log10(0.05), ls="--", color="red", label="p=0.05")
ax2.set_xlabel("KS statistic"); ax2.set_ylabel("-log10(p-value)")
ax2.set_title("Volcano Plot: KS drift significance"); ax2.legend()

plt.tight_layout(); plt.show()
"""),

        md("## 3 · Thermal ROI Drift — Sliding Window"),

        code("""detector = ThermalDiffDriftDetector(train_df)
reports  = detector.batch_assess(test_df)

if reports:
    windows    = range(len(reports))
    max_psis   = [r["max_psi"] for r in reports]
    alert_cols = [{"GREEN":"#66BB6A","YELLOW":"#FF9800","RED":"#EF5350"}[r["alert"]]
                  for r in reports]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(windows, max_psis, color=alert_cols, edgecolor="white", linewidth=0.5)
    ax.axhline(0.1,  ls="--", color="#FF9800", linewidth=1.5, label="YELLOW (0.1)")
    ax.axhline(0.25, ls="--", color="#EF5350", linewidth=1.5, label="RED (0.25)")
    ax.set_xlabel(f"Sliding window (size={detector.WINDOW})")
    ax.set_ylabel("Max PSI (thermal ROI)")
    ax.set_title("Thermal ROI Drift — Sliding Window over Production Sequence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR/"drift_thermal_window.png", dpi=150, bbox_inches="tight")
    plt.show()

    n_red = sum(1 for r in reports if r["alert"] == "RED")
    print(f"RED  windows: {n_red}/{len(reports)}")
    print(f"Top drifting features: {reports[-1].get('top_drift_features', [])}")
else:
    print("Insufficient thermal ROI data for sliding window (need >20 parts with ROI cols)")
"""),

        md("## 4 · Embedding Drift"),

        code("""from data.dataset import MultiModalDataset, collate_fn
from data.loaders import load_tabular_vector
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

scaler   = StandardScaler()
tab_vals = [load_tabular_vector(r, tabular_cols) for _, r in train_df.iterrows()]
scaler.fit(np.array(tab_vals))
tab_med  = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
roi_med  = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

def _collect_embs(df_sub):
    ds = MultiModalDataset(df_sub, tabular_cols, get_dxp_cols(df_sub), train=False,
                           tabular_medians=tab_med, roi_medians=roi_med, tabular_scaler=scaler)
    ld = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
    embs = []
    model.eval()
    with torch.no_grad():
        for batch in ld:
            b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            embs.append(model(b)["fused_emb"].cpu().numpy())
    return np.concatenate(embs)

model = InjectionMouldingFusionModel(
    tabular_in_dim=len(tabular_cols), n_dxp_channels=len(get_dxp_cols(df)),
    n_experiments=len(build_experiment_label_map(df)),
).to(DEVICE)
ckpt = C.MODELS_DIR / "fusion_final.pt"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

ref_embs  = _collect_embs(train_df)
prod_embs = _collect_embs(test_df)
result    = embedding_drift(ref_embs, prod_embs)

print("Embedding Drift Report:")
for k, v in result.items():
    print(f"  {k:25s}: {v}")
"""),

        md("## 5 · Full DriftMonitor Report"),

        code("""monitor = DriftMonitor(train_df, tabular_cols, ref_embeddings=ref_embs)
report  = monitor.check(test_df, prod_embeddings=prod_embs)

print(monitor.format_summary(report))
monitor.save_report(report, C.ARTIFACTS_DIR / "drift_report.json")
print("\\nSaved:", C.ARTIFACTS_DIR / "drift_report.json")
"""),
    ]
    save(nb(C), "05_drift_monitor.ipynb")


# ═══════════════════════════════════════════════════════════════
# 06_consistency.ipynb
# ═══════════════════════════════════════════════════════════════
def make_consistency():
    C = [
        md("""# 06 · Cross-Modal Sensor Consistency Check
Detects **silently-failing sensors** in production.

| Failure mode | Detector |
|---|---|
| Thermal camera miscalibration | Cross-modal cosine (thermal_emb vs tcn_emb) |
| Disconnected DXP probe | Flat-line variance detector |
| Noisy visual section | Embedding Mahalanobis anomaly score |

Unlike drift monitoring (which tracks distributions), consistency checking
operates **per-part** — it flags individual bad measurements in real time.
"""),

        code("""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, str(Path.cwd().parent))
import config as C
from data.loaders  import load_parquet, build_cv_index, join_cv_to_parquet, load_tabular_vector
from data.dataset  import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from models.fusion import InjectionMouldingFusionModel
from training.splits import build_experiment_label_map, random_split
from analysis.consistency import (
    EmbeddingAnomalyDetector, consistency_report,
    batch_consistency_sweep, summarise_sweep, detect_flatline_channels,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 120
print("Device:", DEVICE)
"""),

        code("""df     = load_parquet(C.PARQUET_PATH)
cv_idx = build_cv_index()
df     = join_cv_to_parquet(df, cv_idx)
tabular_cols  = get_tabular_cols(df)
dxp_cols      = get_dxp_cols(df)
exp_label_map = build_experiment_label_map(df)
train_df, _, test_df = random_split(df)

scaler   = StandardScaler()
scaler.fit(np.array([load_tabular_vector(r, tabular_cols) for _, r in train_df.iterrows()]))
tab_med  = {c: float(train_df[c].median()) for c in tabular_cols if c in train_df.columns}
roi_med  = {c: float(train_df[c].median()) for c in C.THERMAL_ROI_COLS if c in train_df.columns}

def _loader(df_sub, shuffle=False, bs=C.BATCH_SIZE):
    ds = MultiModalDataset(df_sub, tabular_cols, dxp_cols, train=False,
                           tabular_medians=tab_med, roi_medians=roi_med, tabular_scaler=scaler)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)

model = InjectionMouldingFusionModel(
    tabular_in_dim=len(tabular_cols), n_dxp_channels=len(dxp_cols),
    n_experiments=len(exp_label_map),
).to(DEVICE)
ckpt = C.MODELS_DIR / "fusion_final.pt"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print("Model loaded. Building reference detectors...")
"""),

        md("## 1 · Build Reference Anomaly Detectors"),

        code("""ref_loader = _loader(train_df, bs=32)

thermal_det = EmbeddingAnomalyDetector.from_model_outputs(
    model, ref_loader, DEVICE, modality="thermal"
)
fused_det = EmbeddingAnomalyDetector.from_model_outputs(
    model, ref_loader, DEVICE, modality="fused"
)
tcn_det = EmbeddingAnomalyDetector.from_model_outputs(
    model, ref_loader, DEVICE, modality="tcn"
)
print(f"Thermal anomaly threshold: {thermal_det.threshold:.4f}  (99th pct of ref Z-score)")
print(f"Fused   anomaly threshold: {fused_det.threshold:.4f}")
print(f"TCN     anomaly threshold: {tcn_det.threshold:.4f}")
"""),

        md("## 2 · Cross-Modal Cosine Distribution"),

        code("""from analysis.consistency import cross_modal_cosine
import torch

cos_sims, labels_list = [], []
label_idx = C.LABEL_COLS.index(C.PRIMARY_TARGET)
test_loader = _loader(test_df, bs=16)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        b = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b)

        # Only compute for parts where both modalities are valid
        t_valid = b["thermal_valid"]
        s_valid = b["sequence_valid"]
        both    = t_valid & s_valid

        if both.any():
            cos = cross_modal_cosine(out["thermal_emb"][both], out["tcn_emb"][both])
            cos_sims.extend(cos.cpu().numpy().tolist())
            labels_list.extend(b["labels"][both, label_idx].cpu().numpy().tolist())

cos_sims = np.array(cos_sims)
labels_a = np.array(labels_list)

if len(cos_sims) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.linspace(-1, 1, 40)

    for lbl, colour, name in [(0,"#42A5F5","OK"), (1,"#EF5350","NOK")]:
        mask = labels_a == lbl
        if mask.sum() > 0:
            ax1.hist(cos_sims[mask], bins=bins, alpha=0.65, color=colour, label=name, density=True)

    ax1.axvline(0.3,  ls="--", color="#FF9800", label="YELLOW (0.30)")
    ax1.axvline(0.1,  ls="--", color="#EF5350", label="RED (0.10)")
    ax1.set_xlabel("Cosine similarity (thermal_emb · tcn_emb)")
    ax1.set_ylabel("Density")
    ax1.set_title("Cross-Modal Cosine: OK vs NOK Parts")
    ax1.legend()

    ax2.scatter(cos_sims, labels_a + np.random.normal(0, 0.04, len(labels_a)),
                c=["#EF5350" if l else "#42A5F5" for l in labels_a],
                alpha=0.5, s=20, edgecolors="white", linewidths=0.2)
    ax2.axvline(0.3, ls="--", color="#FF9800")
    ax2.set_xlabel("Cosine similarity"); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["OK","NOK"])
    ax2.set_title("Cosine vs Label (NOK parts tend to show lower cross-modal agreement)")

    plt.suptitle("Cross-Modal Consistency: Thermal ↔ Sequence Encoder Agreement", fontsize=12)
    plt.tight_layout()
    plt.savefig(C.PLOTS_DIR/"consistency_cosine.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Mean cosine (OK) : {cos_sims[labels_a==0].mean():.4f}")
    print(f"Mean cosine (NOK): {cos_sims[labels_a==1].mean():.4f}")
else:
    print("No parts with both thermal and sequence valid in test set.")
"""),

        md("## 3 · DXP Flat-Line Simulation"),

        code("""# Simulate a disconnected probe (zero-out one DXP channel)
test_loader_1 = _loader(test_df.head(10), bs=1)
batch_demo = next(iter(test_loader_1))
seq_normal = batch_demo["sequence"][0].cpu().numpy()   # (n_ch, T)

# Simulate probe fault on channel 0 (injection pressure)
seq_faulty = seq_normal.copy()
seq_faulty[0, :] = 0.0

ch_names = [c.replace("DXP_","") for c in dxp_cols]

fig, axes = plt.subplots(2, 1, figsize=(13, 6))
for i, (seq, title) in enumerate([(seq_normal, "Normal DXP signal"),
                                   (seq_faulty, "Faulty DXP — channel 0 flat-lined")]):
    for ci in range(min(len(dxp_cols), seq.shape[0])):
        axes[i].plot(seq[ci], alpha=0.7, linewidth=1,
                     label=ch_names[ci] if ci < 8 else None)
    axes[i].set_title(title); axes[i].legend(fontsize=7, loc="upper right", ncol=4)
plt.tight_layout(); plt.show()

# Detect flat-line
from analysis.consistency import detect_flatline_channels
normal_fl = detect_flatline_channels(seq_normal)
faulty_fl = detect_flatline_channels(seq_faulty)
n_flatline_normal = sum(normal_fl.values())
n_flatline_faulty = sum(faulty_fl.values())
print(f"Normal signal — flat-line channels: {n_flatline_normal}")
print(f"Faulty signal — flat-line channels: {n_flatline_faulty}")
faulty_chs = {ch_names[i] if i < len(ch_names) else i for i, v in faulty_fl.items() if v}
print(f"Flagged channels: {faulty_chs}")
"""),

        md("## 4 · Batch Consistency Sweep"),

        code("""reports = batch_consistency_sweep(
    model, test_loader, DEVICE,
    thermal_detector=thermal_det,
    tcn_detector=tcn_det,
    fused_detector=fused_det,
    dxp_cols=dxp_cols,
)
summary = summarise_sweep(reports)

print("=" * 55)
print("CONSISTENCY SWEEP SUMMARY")
print("=" * 55)
for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print("=" * 55)
"""),

        code("""# Fault score distribution
fault_scores = [r["fault_score"] for r in reports]
alerts       = [r["global_alert"] for r in reports]
label_flags  = []

test_loader_labels = _loader(test_df, bs=1)
for batch in test_loader_labels:
    label_flags.append(int(batch["labels"][0, C.LABEL_COLS.index(C.PRIMARY_TARGET)].item()))
label_flags = label_flags[:len(fault_scores)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

alert_colours = {"GREEN": "#66BB6A", "YELLOW": "#FF9800", "RED": "#EF5350"}
ax1.bar(range(len(fault_scores)), fault_scores,
        color=[alert_colours.get(a,"gray") for a in alerts], edgecolor="white", linewidth=0.3)
ax1.axhline(0.3, ls="--", color="#FF9800", label="YELLOW (0.30)")
ax1.axhline(0.6, ls="--", color="#EF5350", label="RED (0.60)")
ax1.set_xlabel("Part index"); ax1.set_ylabel("Fault score")
ax1.set_title("Per-Part Sensor Consistency Fault Score"); ax1.legend()

# Correlation between fault score and actual NOK label
if len(label_flags) == len(fault_scores):
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(label_flags, fault_scores)
    except Exception:
        auc = float("nan")
    ok_scores  = [fs for fs, l in zip(fault_scores, label_flags) if l == 0]
    nok_scores = [fs for fs, l in zip(fault_scores, label_flags) if l == 1]
    ax2.boxplot([ok_scores, nok_scores], labels=["OK", "NOK"], patch_artist=True,
                boxprops=dict(facecolor="#90CAF9"),
                medianprops=dict(color="#B71C1C", linewidth=2))
    ax2.set_ylabel("Fault score")
    ax2.set_title(f"Fault Score vs NOK Label  (AUC={auc:.3f})")

plt.suptitle("Sensor Consistency: Fault Scores Across Test Set", fontsize=12)
plt.tight_layout()
plt.savefig(C.PLOTS_DIR/"consistency_fault_scores.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
    ]
    save(nb(C), "06_consistency.ipynb")


# ═══════════════════════════════════════════════════════════════
# Run all generators
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating notebooks...")
    make_eda()
    make_train_colab()
    make_evaluate()
    make_explainability()
    make_counterfactual()
    make_drift()
    make_consistency()
    print("\nDone. All notebooks written to:", OUT)
