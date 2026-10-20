"""training — losses, metrics, splits, and trainer."""
from .losses   import (
    MultiLabelFocalLoss, DANNLoss, SensorConsistencyLoss,
    TotalLoss, compute_pos_weights,
)
from .metrics  import (
    find_best_thresholds, compute_metrics,
    group_aware_evaluation, build_headline_table,
)
from .splits   import (
    random_split, time_aware_split, group_kfold_splits,
    material_ood_split, build_experiment_label_map,
)
from .trainer  import (
    dann_lambda_schedule, run_epoch,
    train_one_fold, train_full_pipeline,
)

__all__ = [
    "MultiLabelFocalLoss", "DANNLoss", "SensorConsistencyLoss",
    "TotalLoss", "compute_pos_weights",
    "find_best_thresholds", "compute_metrics",
    "group_aware_evaluation", "build_headline_table",
    "random_split", "time_aware_split", "group_kfold_splits",
    "material_ood_split", "build_experiment_label_map",
    "dann_lambda_schedule", "run_epoch",
    "train_one_fold", "train_full_pipeline",
]
