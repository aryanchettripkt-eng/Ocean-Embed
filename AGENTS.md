# AGENTS.md — rules for every coding agent on OceanScope AI

This file is mandatory reading before any change. It applies to every agent (Gemini, Claude, or any other) and overrides anything an agent "thinks is better". If a rule blocks you, stop and ask; do not work around it.

## 1. What this project is
OceanScope AI reconstructs daily subsurface ocean temperature (15 depths, 0.25° grid, North Indian Ocean) from surface data with a small U-Net, and publishes the results through a FastAPI backend and a Next.js website.

## 2. Documents and their authority (highest first)
1. The owner's latest explicit instruction (Aryan).
2. `docs/DECISIONS.md` — dated decisions that change anything below.
3. `docs/PHASES.md` — execution order, task IDs and exit gates.
4. The PRD for the area you are working in:
   - `docs/PRD_AI_ML.md` → everything in `ml/`
   - `docs/PRD_BACKEND.md` → everything in `backend/`
   - `docs/PRD_FRONTEND.md` → everything in `frontend/`
5. `OceanScope_AI_Project_Handoff.md` (repo root) — background and science rationale only. Its Streamlit dashboard plan is **superseded** by the backend and frontend PRDs.

If two documents conflict, follow the higher one and add a note to `docs/OPEN_QUESTIONS.md`.

## 3. Repository layout (do not reorganize)
```
/
├── AGENTS.md  CLAUDE.md  GEMINI.md  README.md  OceanScope_AI_Project_Handoff.md
├── docs/            PRDs, DECISIONS.md, OPEN_QUESTIONS.md, PROGRESS.md
├── ml/              Python package `oceanscope_ml` (data, models, training, evaluation, inference, publish), configs/, notebooks/, tests/
├── backend/         FastAPI app, Dockerfile, openapi.json, tests/
├── frontend/        Next.js app
├── fixtures/
│   └── sample_outputs/   synthetic, contract-exact outputs (committed, < 5 MB)
└── .github/workflows/    CI for ml, backend, frontend
```
Never commit: raw or processed datasets, model checkpoints, `.env` files, notebooks with output cells containing tokens, anything > 5 MB.

## 4. Contracts (frozen)
These are shared between areas and may only change with the owner's approval plus an entry in `docs/DECISIONS.md`:
- ML output layout, `manifest.json` schema, prediction store and metrics files — `PRD_AI_ML.md` §6.
- HTTP API — `PRD_BACKEND.md` §4 and `backend/openapi.json`.
If you work in one area and need a contract change, **stop** and ask. Do not "temporarily" change the other side.

## 5. Hard rules
1. **No invented data or results.** Never hardcode temperatures, metrics, depths, dates or claims of accuracy. Numbers come from files produced by code or from the API.
2. **Test split is sealed** (see `PRD_AI_ML.md` §0). No agent may evaluate on test data unless the owner says "run the final test".
3. **Synthetic data stays labelled.** Anything generated (fixtures, mocks) carries `synthetic: true` and is never described as real.
4. **Secrets only in env vars / platform secrets.** If you see a secret in code or logs, remove it and tell the owner.
5. **Stay in scope.** Do only the task you were given. No drive-by refactors, renames, reformatting of unrelated files, or dependency upgrades.
6. **Don't weaken checks.** Never delete, skip or loosen a test, lint rule, type check or CI step to make something pass. Fix the cause or ask.
7. **No new dependencies** without saying why in the commit message; no paid services.
8. **Never force-push `main`, never rewrite shared history, never commit directly to `main`.**
9. **Ask instead of guessing** when requirements are unclear: add a dated entry to `docs/OPEN_QUESTIONS.md` and stop that task.

## 6. Workflow for every task
1. Read this file, the relevant PRD section, and `docs/PROGRESS.md`.
2. Create a branch `<area>/<short-task>` (e.g. `backend/map-endpoint`).
3. Implement the smallest change that meets the task's acceptance criteria in the PRD.
4. Run the area's checks locally and make them pass:
   - `ml/`: `ruff check . && ruff format --check . && mypy src && pytest -q`
   - `backend/`: `ruff check . && ruff format --check . && mypy app && pytest -q && python -m app.export_openapi --check`
   - `frontend/`: `pnpm typecheck && pnpm lint && pnpm test && pnpm build`
5. Commit with a message: `<area>: <what> (<PRD section>)`, body lists any decisions or deviations.
6. Update `docs/PROGRESS.md`: task, status, branch, what's verified, what's left.
7. Report to the owner: what changed, how it was verified (the command output), and anything uncertain. Never say "done" if a check failed or was not run.

## 7. Division of labour between agents
- **Claude** writes plans, PRD updates, task breakdowns and reviews; it may also implement.
- **Gemini** implements tasks from the plan.
- One agent per branch at a time. Before starting, check `docs/PROGRESS.md` for a task marked in progress by another agent.
- Reviews check the diff against the PRD acceptance criteria and this file's hard rules.

## 8. Environments
- ML training runs on Google Colab / Kaggle (see `PRD_AI_ML.md` §4.1, §11). Code must run there unchanged: paths from env vars, resumable checkpoints.
- Backend deploys to Hugging Face Spaces (Docker). Frontend deploys to Vercel.
- Python 3.11. Node 20 LTS with pnpm.
