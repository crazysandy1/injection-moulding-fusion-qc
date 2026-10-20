"""Smoke tests: the full fusion model runs forward and backward on synthetic data.

Pretrained backbones are replaced by randomly initialised ones so the tests need
no network access and no dataset.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "fusion_system"
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_pretrained(monkeypatch):
    import timm
    import torchvision.models as tvm

    orig_create = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda *a, **k: orig_create(*a, **{**k, "pretrained": False}))
    orig_resnet = tvm.resnet50
    monkeypatch.setattr(tvm, "resnet50", lambda *a, **k: orig_resnet(weights=None))


def make_batch(batch_size, tab_dim, thermal_valid):
    import config as C

    return {
        "thermal_img": torch.randn(batch_size, 3, 224, 224),
        "thermal_roi": torch.randn(batch_size, 10),
        "visual_imgs": torch.randn(batch_size, 3, 1, 224, 224),
        "sequence": torch.randn(batch_size, C.TCN_IN_CHANNELS, C.TCN_TARGET_LEN),
        "tabular": torch.randn(batch_size, tab_dim),
        "thermal_valid": torch.tensor(thermal_valid).bool(),
        "visual_valid": torch.ones(batch_size).bool(),
        "sequence_valid": torch.ones(batch_size).bool(),
        "tabular_valid": torch.ones(batch_size).bool(),
    }


def test_forward_backward_with_missing_modality():
    import config as C
    from models.fusion import InjectionMouldingFusionModel

    model = InjectionMouldingFusionModel(tabular_in_dim=40, n_experiments=47, dann_lambda=0.3)
    model.train()
    out = model(make_batch(2, 40, [1, 0]))

    assert out["defect_logits"].shape == (2, C.N_LABELS)
    assert out["dann_logits"].shape == (2, 47)
    assert out["fused_emb"].shape == (2, C.FUSION_TOKEN_DIM)
    assert out["token_weights"].shape == (2, 4)

    (out["defect_logits"].sum() + out["dann_logits"].sum()).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_gradient_reversal_flips_sign():
    from models.fusion import grad_reverse

    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))
