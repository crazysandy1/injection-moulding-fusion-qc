"""
data/loaders.py
Low-level I/O for every modality.

Thermal  → float32 numpy array (H, W)  from raw CSV
Visual   → uint8  numpy array  (H, W)  from BMP
Parquet  → pandas DataFrame with all 334 features + labels
"""

import os
import re
import struct
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C

log = logging.getLogger(__name__)


# =============================================================================
# Parquet loader
# =============================================================================

def load_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dataset_V2.parquet.  Cleans column types and annotates which
    modalities are available per row.
    """
    path = path or C.PARQUET_PATH
    df = pd.read_parquet(path)
    log.info(f"Parquet loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # ---- availability flags ------------------------------------------------
    df["has_thermal"] = df[C.IR_IMG1_COL].notna() & df[C.IR_IMG2_COL].notna()
    df["has_visual"]  = df[C.CV_IMG1_COL].notna() if C.CV_IMG1_COL in df.columns else False

    # ---- labels: cast to float32 int-like ---------------------------------
    for col in C.LABEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

    log.info(
        f"  has_thermal: {df['has_thermal'].sum()}  "
        f"has_visual: {df['has_visual'].sum()}  "
        f"NOK rate: {df[C.PRIMARY_TARGET].mean():.2%}"
    )
    return df


# =============================================================================
# Thermal CSV loader
# =============================================================================

def load_thermal_csv(csv_path: Path) -> np.ndarray:
    """
    Load a single Thermographie CSV file into a float32 (H, W) array.

    Format quirks:
      - UTF-8 BOM
      - semicolon separator
      - comma decimal  (European: "17,5" → 17.5)
      - CR-only line endings  (old Mac style)
    Returns shape (480, 640) float32, temperature in °C.
    """
    with open(csv_path, encoding="utf-8-sig") as fh:
        raw = fh.read()

    rows = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rows = [r for r in rows if r.strip()]

    data = []
    for row in rows:
        vals = [float(v.replace(",", ".")) for v in row.split(";") if v.strip()]
        if vals:
            data.append(vals)

    arr = np.array(data, dtype=np.float32)   # (480, 640)
    return arr


def load_thermal_pair(
    row: pd.Series,
    csv_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Load frames 1 & 2 for a single part and return a 3-channel float32
    array of shape (3, H, W):
        ch0 = frame1  (raw °C, normalised 0-1 over dataset stats)
        ch1 = frame2
        ch2 = frame1 - frame2  (cooling difference — physics signal)

    Returns None when files are missing.
    """
    csv_dir = csv_dir or C.THERMAL_CSV_DIR
    name1 = row.get(C.IR_IMG1_COL)
    name2 = row.get(C.IR_IMG2_COL)

    if not isinstance(name1, str) or not isinstance(name2, str):
        return None

    p1 = csv_dir / name1
    p2 = csv_dir / name2

    if not p1.exists() or not p2.exists():
        return None

    f1 = load_thermal_csv(p1)
    f2 = load_thermal_csv(p2)

    # normalise each frame independently to [0, 1] using robust percentiles
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    f1n = _norm(f1)
    f2n = _norm(f2)
    diff = np.clip(f1n - f2n, -1.0, 1.0) * 0.5 + 0.5   # shift to [0,1]

    return np.stack([f1n, f2n, diff], axis=0).astype(np.float32)  # (3, H, W)


def extract_thermal_rois(row: pd.Series, df_stats: Optional[Dict] = None) -> np.ndarray:
    """
    Extract the 10-dimensional ROI physics vector from the parquet row.
    Columns: IR_Img{1,2}Temp{Sprue,Dome,EdgeHor,EdgeVer,Full}

    Missing values are imputed with the training set median (passed in df_stats).
    Returns shape (10,) float32.
    """
    vals = []
    for col in C.THERMAL_ROI_COLS:
        v = row.get(col, np.nan)
        if pd.isna(v) and df_stats is not None:
            v = df_stats.get(col, 0.0)
        vals.append(float(v) if not pd.isna(v) else 0.0)
    return np.array(vals, dtype=np.float32)


# =============================================================================
# CV Visual BMP loader
# =============================================================================

def _bmp_info(path: Path) -> Tuple[int, int, int]:
    """Fast BMP header parse. Returns (width, height, bpp)."""
    with open(path, "rb") as fh:
        header = fh.read(54)
    w   = struct.unpack_from("<i", header, 18)[0]
    h   = struct.unpack_from("<i", header, 22)[0]
    bpp = struct.unpack_from("<H", header, 28)[0]
    return w, h, bpp


def load_bmp(path: Path) -> np.ndarray:
    """
    Load a single BMP file as uint8 numpy array (H, W).
    Uses PIL for cross-platform reliability.
    """
    from PIL import Image
    img = Image.open(path).convert("L")      # force greyscale
    return np.array(img, dtype=np.uint8)


def _parse_xml_cycle_id(xml_path: Path) -> Optional[float]:
    """Extract CycleID float from a CV measurement XML file."""
    try:
        with open(xml_path, "r", errors="ignore") as fh:
            content = fh.read()
        m = re.search(r'CycleID.*?>([0-9.]+)<', content)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def build_cv_index(xml_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Walk the Rohbilder directory and build a lookup table:
        cycle_id (float)  →  {1: bmp_path, 2: bmp_path, 3: bmp_path}

    The cycle_id in the XML corresponds to QUA_CycleTime in the parquet.
    Returns a DataFrame with cols: [cycle_id, pos1_path, pos2_path, pos3_path]
    """
    xml_dir = xml_dir or C.CV_XML_DIR
    records: Dict[float, Dict[int, Path]] = {}

    for xml_file in sorted(xml_dir.glob("*.xml")):
        cid = _parse_xml_cycle_id(xml_file)
        if cid is None or cid == 0.0:
            continue
        bmp = xml_file.with_suffix(".bmp")
        if not bmp.exists():
            continue

        # extract Position from XML
        try:
            with open(xml_file, "r", errors="ignore") as fh:
                content = fh.read()
            pm = re.search(r'Position.*?>(\d)<', content)
            pos = int(pm.group(1)) if pm else 1
        except Exception:
            pos = 1

        if cid not in records:
            records[cid] = {}
        records[cid][pos] = bmp

    rows = []
    for cid, pos_map in records.items():
        rows.append({
            "cv_cycle_id": cid,
            "pos1_path": str(pos_map.get(1, "")),
            "pos2_path": str(pos_map.get(2, "")),
            "pos3_path": str(pos_map.get(3, "")),
        })

    df = pd.DataFrame(rows).sort_values("cv_cycle_id").reset_index(drop=True)
    log.info(f"CV index built: {len(df)} unique cycles")
    return df


def join_cv_to_parquet(
    parquet_df: pd.DataFrame,
    cv_index: pd.DataFrame,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Fuzzy-join parquet rows to CV images via QUA_CycleTime ↔ cv_cycle_id.
    Uses merge_asof with tolerance.

    Adds columns: pos1_path, pos2_path, pos3_path
    """
    if "QUA_CycleTime" not in parquet_df.columns:
        log.warning("QUA_CycleTime not in parquet — CV join skipped")
        parquet_df["pos1_path"] = ""
        parquet_df["pos2_path"] = ""
        parquet_df["pos3_path"] = ""
        return parquet_df

    left  = parquet_df.copy().sort_values("QUA_CycleTime")
    right = cv_index.sort_values("cv_cycle_id")

    merged = pd.merge_asof(
        left, right,
        left_on="QUA_CycleTime",
        right_on="cv_cycle_id",
        tolerance=tolerance,
        direction="nearest",
    )
    n_matched = (merged["pos1_path"].fillna("") != "").sum()
    log.info(f"CV join: {n_matched}/{len(merged)} rows matched (tol={tolerance}s)")
    return merged


def load_cv_triple(row: pd.Series) -> Optional[np.ndarray]:
    """
    Load the 3 section images for a part.
    Returns float32 array shape (3, H, W) with values in [0, 1].
    Returns None if any image is missing.
    """
    paths = [row.get("pos1_path", ""), row.get("pos2_path", ""), row.get("pos3_path", "")]
    if not all(p and Path(p).exists() for p in paths):
        return None

    sections = []
    for p in paths:
        arr = load_bmp(Path(p)).astype(np.float32) / 255.0
        sections.append(arr)

    return np.stack(sections, axis=0)   # (3, H, W)


# =============================================================================
# Sequence / tabular helpers
# =============================================================================

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1-D or 2-D (channels, time) sequence to target_len via
    linear interpolation.  Handles variable-length DXP arrays.
    """
    from scipy.interpolate import interp1d
    if seq.ndim == 1:
        x_old = np.linspace(0, 1, len(seq))
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, seq, kind="linear", fill_value="extrapolate")
        return f(x_new).astype(np.float32)

    # (channels, time)
    out = np.zeros((seq.shape[0], target_len), dtype=np.float32)
    for i in range(seq.shape[0]):
        out[i] = resample_sequence(seq[i], target_len)
    return out


def load_dxp_tensor(row: pd.Series, channel_cols: List[str]) -> Optional[np.ndarray]:
    """
    Extract DXP time-series channels from a parquet row.
    Each cell in the parquet is a numpy array (variable length).
    Returns shape (n_channels, TCN_TARGET_LEN) float32, or None.
    """
    channels = []
    for col in channel_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if not hasattr(val, "__len__"):
            return None
        arr = np.array(val, dtype=np.float32)
        channels.append(resample_sequence(arr, C.TCN_TARGET_LEN))

    return np.stack(channels, axis=0)   # (n_channels, TCN_TARGET_LEN)


def load_tabular_vector(
    row: pd.Series,
    tabular_cols: List[str],
    medians: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract tabular scalar features, imputing missing with medians.
    Returns float32 vector of length len(tabular_cols).
    """
    vals = []
    for col in tabular_cols:
        v = row.get(col, np.nan)
        if pd.isna(v):
            v = medians.get(col, 0.0) if medians else 0.0
        vals.append(float(v))
    return np.array(vals, dtype=np.float32)
