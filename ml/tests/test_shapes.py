"""Output shapes for odd grid sizes, and padding inside the dataset (PRD_AI_ML §5.2, §5.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.dataset import OceanScopeDataset
from oceanscope_ml.models.unet import UNet


@pytest.mark.parametrize("height, width", [(9, 11), (13, 7), (5, 5), (16, 16)])
def test_unet_returns_input_size(height: int, width: int) -> None:
    model = UNet(in_channels=5, out_channels=15, base_channels=8, depth_levels=4)
    x = torch.randn(2, 5, height, width)
    assert model(x).shape == (2, 15, height, width)


def test_dataset_pads_inputs_and_keeps_target_size(store: Path, config: ExperimentConfig) -> None:
    ds = OceanScopeDataset(store, config, split="train")
    sample = ds[0]
    # 9 x 11 grid padded to the next multiple of 2**4 = 16 for the U-Net.
    assert sample["x"].shape == (122, 16, 16)
    assert sample["y"].shape == (15, 9, 11)
    assert sample["mask"].shape == (15, 9, 11)
    assert torch.isfinite(sample["x"]).all()
