"""models — all encoder and fusion model definitions."""
from .thermal_encoder  import ThermalEncoder
from .visual_encoder   import VisualEncoder
from .tcn_encoder      import TCNEncoder
from .tabular_encoder  import TabularEncoder
from .fusion           import (
    InjectionMouldingFusionModel,
    CrossModalFusion,
    GradientReversal,
    grad_reverse,
)

__all__ = [
    "ThermalEncoder",
    "VisualEncoder",
    "TCNEncoder",
    "TabularEncoder",
    "InjectionMouldingFusionModel",
    "CrossModalFusion",
    "GradientReversal",
    "grad_reverse",
]
