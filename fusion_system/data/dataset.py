"""
data/dataset.py
MultiModalDataset — returns all four modalities per sample.

Missing-modality handling:
  Each modality has a corresponding boolean flag in the returned dict.
  If a modality is unavailable the tensor is zeroed-out and the flag is False.
  The fusion model uses these flags to apply a [MASK] token instead of
  encoding the (meaningless) zero tensor.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C
from data.loaders import (
    load_thermal_pair, extract_thermal_rois,
    load_cv_triple, load_dxp_tensor, load_tabular_vector,
)
from data.transforms import (
    get_thermal_transforms, get_visual_transforms,
    get_sequence_transforms, get_tabular_transforms,
)

log = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    One sample = one injection moulding cycle (one row in the parquet).

    Returns a dict with keys:
        thermal_img   : FloatTensor (3, H, W) or zeros
        thermal_roi   : FloatTensor (10,)
        thermal_valid : bool
        visual_imgs   : FloatTensor (3, C_img, H_img, W_img) or zeros
        visual_valid  : bool
        sequence      : FloatTensor (n_channels, T) or zeros
        sequence_valid: bool
        tabular       : FloatTensor (n_tab,) or zeros
        tabular_valid : bool
        labels        : FloatTensor (N_LABELS,)
        group         : str  (MET_ExperimentNumber — for stratified split)
        cycle_id      : int  (MET_MachineCycleID)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_cols: List[str],
        dxp_cols: List[str],
        train: bool = True,
        tabular_medians: Optional[Dict[str, float]] = None,
        roi_medians: Optional[Dict[str, float]] = None,
        tabular_scaler=None,           # fitted sklearn StandardScaler
    ):
        self.df             = df.reset_index(drop=True)
        self.tabular_cols   = tabular_cols
        self.dxp_cols       = dxp_cols
        self.train          = train
        self.tab_medians    = tabular_medians or {}
        self.roi_medians    = roi_medians or {}
        self.scaler         = tabular_scaler

        # transforms
        self.tf_thermal  = get_thermal_transforms(train)
        self.tf_visual   = get_visual_transforms(train)
        self.tf_sequence = get_sequence_transforms(train)
        self.tf_tabular  = get_tabular_transforms(train)

        # zero fallback tensors
        self._zero_thermal   = torch.zeros(3, C.THERMAL_INPUT_SIZE, C.THERMAL_INPUT_SIZE)
        self._zero_visual    = torch.zeros(3, 1, C.CV_INPUT_SIZE, C.CV_INPUT_SIZE)
        self._zero_sequence  = torch.zeros(len(dxp_cols), C.TCN_TARGET_LEN)
        self._zero_tabular   = torch.zeros(len(tabular_cols))
        self._zero_roi       = torch.zeros(len(C.THERMAL_ROI_COLS))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # ------------------------------------------------------------------ #
        # 1. Labels                                                            #
        # ------------------------------------------------------------------ #
        labels = torch.tensor(
            [row.get(c, 0.0) for c in C.LABEL_COLS],
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------ #
        # 2. Tabular scalars                                                   #
        # ------------------------------------------------------------------ #
        tab_vec = load_tabular_vector(row, self.tabular_cols, self.tab_medians)
        if self.scaler is not None:
            tab_vec = self.scaler.transform(tab_vec.reshape(1, -1)).flatten().astype(np.float32)
        tab_tensor = torch.from_numpy(tab_vec)
        tab_tensor = self.tf_tabular(tab_tensor)
        tab_valid  = torch.isfinite(tab_tensor).all().item()

        # ------------------------------------------------------------------ #
        # 3. DXP time-series                                                   #
        # ------------------------------------------------------------------ #
        seq_arr   = load_dxp_tensor(row, self.dxp_cols)
        seq_valid = seq_arr is not None
        if seq_valid:
            seq_tensor = torch.from_numpy(seq_arr)
            seq_tensor = self.tf_sequence(seq_tensor)
        else:
            seq_tensor = self._zero_sequence.clone()

        # ------------------------------------------------------------------ #
        # 4. Thermal (IR frames + ROI features)                               #
        # ------------------------------------------------------------------ #
        thermal_arr = load_thermal_pair(row)
        thermal_valid = thermal_arr is not None
        if thermal_valid:
            thermal_tensor = torch.from_numpy(thermal_arr)   # (3, H, W) float32
            thermal_tensor = self.tf_thermal(thermal_tensor)
        else:
            thermal_tensor = self._zero_thermal.clone()

        roi_arr    = extract_thermal_rois(row, self.roi_medians)
        roi_tensor = torch.from_numpy(roi_arr)

        # ------------------------------------------------------------------ #
        # 5. CV visual (3 BMP sections)                                        #
        # ------------------------------------------------------------------ #
        cv_arr   = load_cv_triple(row)
        cv_valid = cv_arr is not None
        if cv_valid:
            # transform each section independently → stack
            sections = []
            for i in range(3):
                sec = (cv_arr[i] * 255).astype(np.uint8)   # back to uint8 for PIL
                sec_t = self.tf_visual(sec)                 # (1, H, W)
                sections.append(sec_t)
            visual_tensor = torch.stack(sections, dim=0)   # (3, 1, H, W)
        else:
            visual_tensor = self._zero_visual.clone()

        return {
            # modality tensors
            "thermal_img"   : thermal_tensor,
            "thermal_roi"   : roi_tensor,
            "thermal_valid" : thermal_valid,
            "visual_imgs"   : visual_tensor,
            "visual_valid"  : cv_valid,
            "sequence"      : seq_tensor,
            "sequence_valid": seq_valid,
            "tabular"       : tab_tensor,
            "tabular_valid" : tab_valid,
            # targets & metadata
            "labels"        : labels,
            "group"         : str(row.get(C.GROUP_COL, "unknown")),
            "cycle_id"      : int(row.get(C.CYCLE_ID_COL, -1)),
            "material"      : str(row.get(C.MATERIAL_COL, "unknown")),
        }


# =============================================================================
# Collate function — handles mixed valid/invalid samples in a batch
# =============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and keeps lists for strings."""
    out: Dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], bool):
            out[k] = torch.tensor(vals, dtype=torch.bool)
        elif isinstance(vals[0], (int, float)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals   # list of strings

    return out


# =============================================================================
# Helper: build tabular columns list from parquet
# =============================================================================

def get_tabular_cols(df: pd.DataFrame) -> List[str]:
    """
    Return all scalar numeric tabular columns (SET_, QUA_, ENV_, CALC_, DOS_, DRY_, SIM_).
    Excludes sequence (array-valued) columns and label columns.
    """
    candidates = [
        c for c in df.columns
        if any(c.startswith(p) for p in C.TABULAR_PREFIXES)
        and c not in C.LABEL_COLS
    ]
    # keep only columns that are actual scalars (not arrays)
    scalar_cols = []
    sample = df.iloc[0]
    for c in candidates:
        v = sample[c]
        if isinstance(v, (int, float, np.integer, np.floating)) or pd.isna(v):
            scalar_cols.append(c)
    return scalar_cols


def get_dxp_cols(df: pd.DataFrame) -> List[str]:
    """
    Return DXP column names that exist in parquet AND are in C.KEY_DXP_CHANNELS.
    Falls back to all DXP_ columns if keys are missing.
    """
    available = [c for c in C.KEY_DXP_CHANNELS if c in df.columns]
    if len(available) < 3:
        # fallback: use all DXP_ columns that contain array data
        available = [
            c for c in df.columns
            if c.startswith(C.SEQUENCE_PREFIX) and c not in C.LABEL_COLS
        ]
    return available
