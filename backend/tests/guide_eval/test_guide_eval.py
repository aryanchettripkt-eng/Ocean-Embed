"""P7-R harness as tests: mock guide safety on the 20 questions, and the backend against
worst-case model scripts. No test calls the Meta API."""

from __future__ import annotations

import pytest

from app.meta.client import MockMetaModelClient

from .harness import Case, load_cases, make_store, run_case
from .worst_case import SEALED, WorstCase, run_worst_case, worst_cases

CASES = load_cases()
WORST = worst_cases()


def test_question_set_has_twenty_cases_with_ten_tricks() -> None:
    assert len(CASES) == 20
    assert len({c.id for c in CASES}) == 20
    assert sum(c.kind != "ordinary" for c in CASES) == 10


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_mock_guide_is_safe(case: Case) -> None:
    result = run_case(case, MockMetaModelClient(), make_store())
    assert result.safety_ok, [c for c in result.failed("safety")]
    assert "Offline mock guide" in result.answer


@pytest.mark.parametrize(
    "wc",
    [
        pytest.param(
            wc,
            id=wc.id,
            marks=[pytest.mark.xfail(strict=True, reason=f"known gap: {wc.title}")]
            if wc.expect == "gap"
            else [],
        )
        for wc in WORST
    ],
)
def test_backend_stops_worst_case_model(wc: WorstCase) -> None:
    result = run_worst_case(wc, make_store(), make_store(sealed_test=SEALED))
    assert result.safety_ok, [c for c in result.failed("safety")]


def test_upstream_error_is_a_quality_failure_and_stops_a_real_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.meta.client import MetaUpstreamError

    from . import run as run_module

    class Redirecting:
        calls = 0

        def chat(self, *a: object, **k: object) -> None:
            Redirecting.calls += 1
            raise MetaUpstreamError("Meta API returned HTTP 307")

    result = run_case(CASES[0], Redirecting(), make_store())  # type: ignore[arg-type]
    assert result.upstream_error and not result.quality_ok

    monkeypatch.setattr(run_module, "HttpMetaModelClient", lambda *a, **k: Redirecting())
    monkeypatch.setattr(run_module, "prices_configured", lambda kind: True)
    for key, value in {
        "META_API_BASE_URL": "https://meta.invalid",
        "META_API_KEY": "k",
        "META_CHAT_MODEL": "m",
    }.items():
        monkeypatch.setattr(run_module.settings, key, value)
    Redirecting.calls = 0
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        code = run_module.main(
            [
                "--client",
                "http",
                "--allow-real-meta-api",
                "--ledger",
                str(Path(d) / "l.json"),
                "--out",
                str(Path(d) / "r.md"),
            ]
        )
    assert code == 1
    assert Redirecting.calls == run_module.MAX_UPSTREAM_ERRORS_IN_A_ROW
