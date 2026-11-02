"""Training resume: a run split in two must end with the same weights as one uninterrupted run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.train import train_run


def _runs_rows(config: ExperimentConfig) -> int:
    runs_csv = Path(config.paths.outputs) / "metrics" / "runs.csv"
    with runs_csv.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1  # minus the header


def _last_checkpoint(config: ExperimentConfig, run_id: str) -> dict[str, torch.Tensor]:
    path = Path(config.paths.checkpoints) / run_id / "last.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def test_resumed_run_matches_uninterrupted_run(config: ExperimentConfig, store: Path) -> None:
    straight = train_run(config, store, "straight", device_choice="cpu")
    assert straight.finished and straight.epochs_completed == 2

    first = train_run(config, store, "split", device_choice="cpu", epochs=1)
    assert first.epochs_completed == 1 and not first.finished

    resumed = train_run(config, store, "split", device_choice="cpu")
    assert resumed.finished and resumed.epochs_completed == 2

    expected = _last_checkpoint(config, "straight")["model"]
    actual = _last_checkpoint(config, "split")["model"]
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name

    # Only the two finished runs add rows. The part-way run does not.
    assert _runs_rows(config) == 2


def test_resume_refuses_a_changed_config(config: ExperimentConfig, store: Path) -> None:
    train_run(config, store, "changed", device_choice="cpu", epochs=1)
    config.train["lr"] = 2.0e-3
    config.raw_dict["train"]["lr"] = 2.0e-3
    with pytest.raises(ValueError, match="config hash"):
        train_run(config, store, "changed", device_choice="cpu")
