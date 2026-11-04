"""Pytest fixtures and configuration for backend tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest

import app.api.v1.data as data_module
import app.api.v1.meta as meta_module
import app.api.v1.validation as validation_module
import app.data.events as events_module
import app.data.loader as loader_module
import app.data.store as store_module
from app.data.loader import DataLoader
from app.data.store import OutputStore

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"


@pytest.fixture(autouse=True)
def configure_test_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Configure DATA_DIR to point to committed sample outputs fixtures by default."""
    monkeypatch.setenv("DATA_DIR", str(FIXTURES_DIR))
    # The committed events are dated for the real run. Fixture outputs cover 2020 only, so tests
    # start with no events file; tests that need events point EVENTS_PATH at their own file.
    monkeypatch.setattr(events_module, "EVENTS_PATH", tmp_path / "no-events.json")
    fixtures_loader = DataLoader(data_dir=FIXTURES_DIR)
    fixtures_store = OutputStore()
    monkeypatch.setattr(loader_module, "loader", fixtures_loader)
    monkeypatch.setattr(store_module, "store", fixtures_store)
    monkeypatch.setattr(meta_module, "loader", fixtures_loader)
    monkeypatch.setattr(meta_module, "store", fixtures_store)
    monkeypatch.setattr(data_module, "store", fixtures_store)
    monkeypatch.setattr(validation_module, "store", fixtures_store)


@pytest.fixture(autouse=True)
def configure_meta_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """A test credit budget with a throwaway ledger, so http-mode tests (which only ever talk
    to httpx.MockTransport) pass the fail-closed budget check. Prices are test values."""
    from app.meta import budget as budget_module
    from app.settings import settings

    monkeypatch.setattr(settings, "META_BUDGET_USD", 100.0)
    monkeypatch.setattr(settings, "META_DAILY_BUDGET_USD", 10.0)
    monkeypatch.setattr(settings, "META_BUDGET_LEDGER_PATH", str(tmp_path / "ledger.json"))
    monkeypatch.setattr(settings, "META_BUDGET_HF_REPO", None)
    monkeypatch.setattr(settings, "META_PRICE_INPUT_PER_MTOK", 1.0)
    monkeypatch.setattr(settings, "META_PRICE_OUTPUT_PER_MTOK", 4.0)
    monkeypatch.setattr(settings, "META_PRICE_PER_AUDIO_MIN", 0.01)
    monkeypatch.setattr(settings, "META_PRICE_PER_IMAGE", 0.05)
    monkeypatch.setattr(settings, "META_PRICE_PER_SEGMENT", 0.01)
    budget_module.reset_budget_for_tests()
    yield
    budget_module.reset_budget_for_tests()
