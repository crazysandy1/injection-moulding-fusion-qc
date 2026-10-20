"""
training/splits.py
Group-aware, time-aware, and random splits for the ProBayes dataset.

The headline metric for this project uses GROUP K-FOLD by MET_ExperimentNumber.
Random-split results are computed as a "naive baseline" for comparison only.

Why group-aware?
    The ProBayes dataset uses a Taguchi DOE: 47 experiment cells, 12 parts each.
    Random splits leak setpoint information — the model can cheat by recognising
    a setpoint it saw in training. Group splits force generalisation to UNSEEN
    experimental conditions, which is the real-world requirement.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
from sklearn.model_selection import GroupKFold, train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


def random_split(
    df: pd.DataFrame,
    val_frac: float  = C.VAL_FRACTION,
    test_frac: float = C.TEST_FRACTION,
    seed: int        = C.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard random stratified split.
    BASELINE ONLY — do not use as headline metric.
    """
    rest, test = train_test_split(
        df, test_size=test_frac,
        stratify=df[C.PRIMARY_TARGET],
        random_state=seed,
    )
    val_frac_adj = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        rest, test_size=val_frac_adj,
        stratify=rest[C.PRIMARY_TARGET],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def time_aware_split(
    df: pd.DataFrame,
    val_frac: float  = C.VAL_FRACTION,
    test_frac: float = C.TEST_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by MET_JobCycleID order (temporal).
    Test = last cycles in the shift; val = next-to-last; train = remainder.
    Ensures no future data leaks into training.
    """
    sorted_df = df.sort_values(C.JOB_CYCLE_COL).reset_index(drop=True)
    n = len(sorted_df)
    test_start = int(n * (1.0 - test_frac))
    val_start  = int(n * (1.0 - test_frac - val_frac))

    train = sorted_df.iloc[:val_start]
    val   = sorted_df.iloc[val_start:test_start]
    test  = sorted_df.iloc[test_start:]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = C.N_GROUP_FOLDS,
    group_col: str = C.GROUP_COL,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train_df, val_df) for each fold.
    Groups = unique values of MET_ExperimentNumber.
    All 12 parts from an experiment cell land in the SAME fold.

    This is the HEADLINE evaluation metric.
    """
    groups = df[group_col].values
    gkf    = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in gkf.split(df, groups=groups):
        yield (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
        )


def material_ood_split(
    df: pd.DataFrame,
    train_material: str = "PP",
    test_material : str = "ABS",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on PP experiments, test on ABS recyclate (Dec 15, 2021).
    Measures domain-shift robustness across materials.
    """
    train = df[df[C.MATERIAL_COL] == train_material].reset_index(drop=True)
    test  = df[df[C.MATERIAL_COL] == test_material].reset_index(drop=True)
    return train, test


def build_experiment_label_map(df: pd.DataFrame) -> dict:
    """
    Returns {experiment_str: int_label} for the DANN head.
    E.g. {"A01": 0, "A02": 1, ...}
    """
    experiments = sorted(df[C.GROUP_COL].dropna().unique())
    return {exp: i for i, exp in enumerate(experiments)}
