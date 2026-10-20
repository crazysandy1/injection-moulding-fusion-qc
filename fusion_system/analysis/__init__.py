"""analysis — explainability, counterfactual recommender, drift monitor, consistency checker."""
from .explainability import (
    GradCAM, thermal_gradcam, visual_gradcam,
    get_modality_weights, tcn_channel_saliency, roi_importance,
    save_thermal_explanation, save_visual_explanation,
    save_fusion_explanation, explain_sample,
)
from .counterfactual import (
    TaguchiBestCell, GradientCounterfactual, counterfactual_report,
)
from .drift import (
    compute_psi, psi_alert_level, ks_test_features, embedding_drift,
    ThermalDiffDriftDetector, DriftMonitor,
)
from .consistency import (
    cross_modal_cosine, consistency_score,
    EmbeddingAnomalyDetector, detect_flatline_channels, dxp_quality_score,
    consistency_report, batch_consistency_sweep, summarise_sweep,
)

__all__ = [
    # explainability
    "GradCAM", "thermal_gradcam", "visual_gradcam",
    "get_modality_weights", "tcn_channel_saliency", "roi_importance",
    "save_thermal_explanation", "save_visual_explanation",
    "save_fusion_explanation", "explain_sample",
    # counterfactual
    "TaguchiBestCell", "GradientCounterfactual", "counterfactual_report",
    # drift
    "compute_psi", "psi_alert_level", "ks_test_features", "embedding_drift",
    "ThermalDiffDriftDetector", "DriftMonitor",
    # consistency
    "cross_modal_cosine", "consistency_score",
    "EmbeddingAnomalyDetector", "detect_flatline_channels", "dxp_quality_score",
    "consistency_report", "batch_consistency_sweep", "summarise_sweep",
]
