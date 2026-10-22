"""Pytest fixtures and configuration for backend tests."""

from pathlib import Path

import pytest

import app.main as app_main
from app.data import loader as loader_module
from app.data.loader import DataLoader

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"


@pytest.fixture(autouse=True)
def configure_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure DATA_DIR to point to committed sample outputs fixtures by default."""
    monkeypatch.setenv("DATA_DIR", str(FIXTURES_DIR))
    fixtures_loader = DataLoader(data_dir=FIXTURES_DIR)
    monkeypatch.setattr(loader_module, "loader", fixtures_loader)
    monkeypatch.setattr(app_main, "loader", fixtures_loader)
