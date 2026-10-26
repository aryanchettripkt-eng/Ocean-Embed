"""Shared fixtures: a tiny config and a synthetic Zarr store built from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from oceanscope_ml.config import ExperimentConfig
from tests.synthetic_store import build_synthetic_store, synthetic_config


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return synthetic_config(tmp_path)


@pytest.fixture
def store(tmp_path: Path, config: ExperimentConfig) -> Path:
    return build_synthetic_store(tmp_path / "store.zarr", config)
