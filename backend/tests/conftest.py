"""Pytest fixtures and configuration for backend tests."""

from pathlib import Path

import pytest

import app.api.v1.data as data_module
import app.api.v1.meta as meta_module
import app.api.v1.validation as validation_module
import app.data.loader as loader_module
import app.data.store as store_module
from app.data.loader import DataLoader
from app.data.store import OutputStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"


@pytest.fixture(autouse=True)
def configure_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure DATA_DIR to point to committed sample outputs fixtures by default."""
    monkeypatch.setenv("DATA_DIR", str(FIXTURES_DIR))
    fixtures_loader = DataLoader(data_dir=FIXTURES_DIR)
    fixtures_store = OutputStore()
    monkeypatch.setattr(loader_module, "loader", fixtures_loader)
    monkeypatch.setattr(store_module, "store", fixtures_store)
    monkeypatch.setattr(meta_module, "loader", fixtures_loader)
    monkeypatch.setattr(meta_module, "store", fixtures_store)
    monkeypatch.setattr(data_module, "store", fixtures_store)
    monkeypatch.setattr(validation_module, "store", fixtures_store)
