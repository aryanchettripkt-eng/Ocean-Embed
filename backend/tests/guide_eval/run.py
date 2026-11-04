"""Run the P7-R guide harness and write a Markdown report.

    python -m tests.guide_eval.run --client mock          # offline mock guide (default)
    python -m tests.guide_eval.run --client worst-case    # scripted bad model vs the backend
    python -m tests.guide_eval.run --client http --allow-real-meta-api   # Aryan only, real key

`--client http` calls the real Meta Model API with META_API_* from the environment, so it
spends credits and is never run by tests or CI (AGENTS.md rule 11). Use `--data-dir` to point
at real published outputs; the default is the committed synthetic fixtures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.meta.budget import Budget, BudgetedClient, FileLedgerStore, prices_configured
from app.meta.client import HttpMetaModelClient, MetaModelClient, MockMetaModelClient
from app.settings import settings

from .harness import FIXTURES_DIR, CaseResult, load_cases, make_store, render_report, run_case
from .worst_case import SEALED, run_worst_case, worst_cases

# A real run stops after this many failed calls in a row, so a bad URL or key cannot use
# up the budget on calls that never reach the model.
MAX_UPSTREAM_ERRORS_IN_A_ROW = 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--client", choices=["mock", "worst-case", "http"], default="mock")
    ap.add_argument("--allow-real-meta-api", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=FIXTURES_DIR)
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=2.0,
        help="hard cap in USD for --client http, across runs sharing --ledger (default 2.00)",
    )
    ap.add_argument("--ledger", type=Path, default=Path("p7r-budget-ledger.json"))
    ap.add_argument("--out", type=Path, default=None, help="report path (default: stdout)")
    args = ap.parse_args(argv)

    store = make_store(args.data_dir)
    results: list[CaseResult]
    if args.client == "worst-case":
        sealed = make_store(args.data_dir, sealed_test=SEALED)
        results = [run_worst_case(wc, store, sealed) for wc in worst_cases()]
        title = "Guide harness: worst-case model scripts"
    else:
        client: MetaModelClient
        if args.client == "http":
            if not args.allow_real_meta_api:
                print(
                    "Refusing: --client http calls the real Meta API. Add "
                    "--allow-real-meta-api if you mean it.",
                    file=sys.stderr,
                )
                return 2
            if not (
                settings.META_API_BASE_URL and settings.META_API_KEY and settings.META_CHAT_MODEL
            ):
                print(
                    "META_API_BASE_URL, META_API_KEY and META_CHAT_MODEL must be set.",
                    file=sys.stderr,
                )
                return 2
            if not prices_configured("chat"):
                print(
                    "Set META_PRICE_INPUT_PER_MTOK and META_PRICE_OUTPUT_PER_MTOK (USD per "
                    "million tokens, from the Meta dashboard) so the run can be capped.",
                    file=sys.stderr,
                )
                return 2
            # Same hard cap as production: every call is reserved against --budget-usd and
            # recorded in --ledger, which persists between runs.
            budget = Budget(FileLedgerStore(args.ledger), args.budget_usd, args.budget_usd)
            print(
                f"budget: ${budget.spent_usd:.4f} of ${args.budget_usd:.2f} already used "
                f"(ledger {args.ledger})",
                file=sys.stderr,
            )
            client = BudgetedClient(
                HttpMetaModelClient(
                    settings.META_API_BASE_URL,
                    settings.META_API_KEY,
                    settings.META_CHAT_MODEL,
                    settings.META_CHAT_FALLBACK_MODEL,
                ),
                budget,
            )
            title = f"Guide harness: real model ({settings.META_CHAT_MODEL})"
        else:
            client = MockMetaModelClient()
            title = "Guide harness: offline mock client"
        results = []
        streak = 0
        for case in load_cases():
            result = run_case(case, client, store)
            results.append(result)
            streak = streak + 1 if result.upstream_error else 0
            if args.client == "http" and streak >= MAX_UPSTREAM_ERRORS_IN_A_ROW:
                print(
                    f"stopping: {streak} upstream errors in a row ({result.answer}); "
                    "fix the connection before spending more",
                    file=sys.stderr,
                )
                break
        if isinstance(client, BudgetedClient):
            b = client.budget
            print(f"budget: ${b.spent_usd:.4f} of ${b.total_usd:.2f} used", file=sys.stderr)

    report = render_report(title, results)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    if args.client == "worst-case":
        return 0
    ok = all(r.safety_ok and not r.upstream_error for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
