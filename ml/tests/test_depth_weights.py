"""Depth weights for the loss (PRD_AI_ML §5.4)."""

from __future__ import annotations

import torch

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.models.loss import depth_weight_vector
from tests.synthetic_store import EXPERIMENT_YAML


def test_real_config_weights_double_the_50_to_300_m_band() -> None:
    config = load_config(EXPERIMENT_YAML)
    weights = depth_weight_vector(config)
    expected = [2.0 if 50.0 <= d <= 300.0 else 1.0 for d in config.depths_m]
    assert weights.tolist() == expected
    # 65.8 to 266.0 m. The 47.4 m level is just outside the band.
    assert int((weights == 2.0).sum()) == 6


def test_band_edges_are_inclusive(config: ExperimentConfig) -> None:
    config.depths_m = [49.9, 50.0, 300.0, 300.5]
    weights = depth_weight_vector(config)
    assert weights.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert weights.dtype == torch.float32


def test_custom_band_and_weights(config: ExperimentConfig) -> None:
    config.target["depth_weights"] = {"band_m": [0, 100], "in_band": 3.0, "out_band": 0.5}
    config.depths_m = [5.0, 100.0, 150.0]
    assert depth_weight_vector(config).tolist() == [3.0, 3.0, 0.5]
