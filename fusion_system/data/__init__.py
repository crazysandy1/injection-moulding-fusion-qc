"""data — dataset loading, transforms, and multi-modal dataset."""
from .loaders   import (
    load_parquet, load_thermal_csv, load_thermal_pair,
    extract_thermal_rois, build_cv_index, join_cv_to_parquet,
    load_cv_triple, resample_sequence, load_dxp_tensor,
    load_tabular_vector,
)
from .dataset   import MultiModalDataset, collate_fn, get_tabular_cols, get_dxp_cols
from .transforms import (
    ThermalNormalize, ThermalAugment, get_thermal_transforms,
    get_visual_transforms, SequenceAugment, TabularNoise,
)

__all__ = [
    "load_parquet", "load_thermal_csv", "load_thermal_pair",
    "extract_thermal_rois", "build_cv_index", "join_cv_to_parquet",
    "load_cv_triple", "resample_sequence", "load_dxp_tensor",
    "load_tabular_vector",
    "MultiModalDataset", "collate_fn", "get_tabular_cols", "get_dxp_cols",
    "ThermalNormalize", "ThermalAugment", "get_thermal_transforms",
    "get_visual_transforms", "SequenceAugment", "TabularNoise",
]
