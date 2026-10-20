"""
data/transforms.py
Per-modality augmentation pipelines.

Key design rules:
  - Thermal : NO colour jitter (meaningless on temperature data).
              Valid ops = flips, small rotations, Gaussian noise, random crop.
  - Visual  : Full photometric augmentations valid (it IS a real-light camera).
  - Sequence: Gaussian noise injection + random time-warping.
  - Tabular : Gaussian noise on continuous features only.
"""

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


# =============================================================================
# Thermal transforms
# =============================================================================

class ThermalNormalize:
    """
    Normalise each channel of a (3, H, W) float32 thermal tensor using
    physics-derived stats (mean / std computed from training set).
    Defaults are conservative mid-range values; replace with actual stats
    after running data/compute_stats.py.
    """
    MEAN = [0.50, 0.48, 0.50]   # ch0=frame1, ch1=frame2, ch2=diff
    STD  = [0.12, 0.12, 0.06]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(self.MEAN, dtype=torch.float32).view(3, 1, 1)
        s = torch.tensor(self.STD,  dtype=torch.float32).view(3, 1, 1)
        return (x - m) / (s + 1e-6)


class ThermalAugment:
    """Training augmentation for (3, H, W) float32 thermal tensor."""

    def __init__(
        self,
        flip_p: float = 0.5,
        rot_deg: float = 10.0,
        noise_std: float = 0.01,   # ~0.1 °C effective at normalised scale
        crop_scale: Tuple[float, float] = (0.85, 1.0),
    ):
        self.flip_p    = flip_p
        self.rot_deg   = rot_deg
        self.noise_std = noise_std
        self.crop_scale = crop_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # random horizontal flip
        if torch.rand(1).item() < self.flip_p:
            x = TF.hflip(x)

        # random vertical flip
        if torch.rand(1).item() < self.flip_p * 0.5:
            x = TF.vflip(x)

        # small rotation
        angle = (torch.rand(1).item() * 2 - 1) * self.rot_deg
        x = TF.rotate(x, angle)

        # random resized crop
        i, j, h, w = T.RandomResizedCrop.get_params(
            x, scale=self.crop_scale, ratio=(0.9, 1.1)
        )
        x = TF.resized_crop(x, i, j, h, w, (C.THERMAL_INPUT_SIZE, C.THERMAL_INPUT_SIZE))

        # Gaussian noise (temperature jitter)
        x = x + torch.randn_like(x) * self.noise_std

        return x


def get_thermal_transforms(train: bool):
    size = C.THERMAL_INPUT_SIZE
    norm = ThermalNormalize()

    if train:
        return T.Compose([
            T.Resize((size, size)),
            ThermalAugment(),
            norm,
        ])
    else:
        return T.Compose([
            T.Resize((size, size)),
            norm,
        ])


# =============================================================================
# Visual (CV) transforms
# =============================================================================

def get_visual_transforms(train: bool):
    """
    Transforms for a single-section greyscale BMP (H, W) → tensor (1, size, size).
    Each of the 3 sections is transformed independently then stacked.
    """
    size = C.CV_INPUT_SIZE

    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(int(size * 1.15)),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),                              # (1, H, W) uint8→float
            T.Normalize(mean=[0.5], std=[0.25]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.25]),
        ])


# =============================================================================
# Sequence transforms
# =============================================================================

class SequenceAugment:
    """
    Augment a (C, T) float32 sequence tensor.
    Valid ops on time-series sensor data:
      - Gaussian noise injection
      - Random channel dropout (simulate sensor dropout)
    """

    def __init__(self, noise_std: float = 0.02, drop_channel_p: float = 0.1):
        self.noise_std = noise_std
        self.drop_p    = drop_channel_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # additive noise
        x = x + torch.randn_like(x) * self.noise_std

        # random channel zero-out (simulate a sensor going offline)
        if torch.rand(1).item() < self.drop_p:
            ch = torch.randint(0, x.shape[0], (1,)).item()
            x[ch] = 0.0

        return x


def get_sequence_transforms(train: bool):
    if train:
        return SequenceAugment()
    return lambda x: x   # identity


# =============================================================================
# Tabular transforms
# =============================================================================

class TabularNoise:
    """Add small Gaussian noise to tabular features during training."""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


def get_tabular_transforms(train: bool):
    if train:
        return TabularNoise(std=0.01)
    return lambda x: x
