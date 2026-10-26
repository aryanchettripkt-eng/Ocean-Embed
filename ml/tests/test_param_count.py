"""Parameter cap (PRD_AI_ML §5.3: at most 10 M parameters)."""

from __future__ import annotations

import pytest

from oceanscope_ml.config import load_config
from oceanscope_ml.models.unet import MAX_PARAMETERS, UNet
from oceanscope_ml.train import build_model
from tests.synthetic_store import EXPERIMENT_YAML


def test_default_model_is_under_the_cap() -> None:
    config = load_config(EXPERIMENT_YAML)
    model = build_model(config, n_channels=122)
    counted = sum(p.numel() for p in model.parameters())
    assert model.n_parameters == counted
    assert counted <= MAX_PARAMETERS


def test_oversized_model_is_refused() -> None:
    with pytest.raises(ValueError, match="parameters"):
        UNet(in_channels=122, base_channels=128, depth_levels=4)


def test_base_channels_must_suit_group_norm() -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        UNet(in_channels=122, base_channels=12)
