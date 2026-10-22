# OceanScope AI

AI-powered 3D subsurface ocean temperature reconstruction for the North Indian Ocean.

## Overview
OceanScope AI reconstructs daily subsurface ocean temperature fields across 15 depth levels (surface to ~1,000 m) on a 0.25° grid using satellite surface dynamics (DUACS SLA, OSCAR currents, CCMP winds, SST, SSS) trained against GLORYS12 reanalysis and evaluated against independent ARGO float observations.

## Repository Layout
- `ml/`: Python package `oceanscope_ml`, configuration, training pipelines, and evaluation.
- `backend/`: FastAPI application serving precomputed predictions, metrics, and chat.
- `frontend/`: Next.js web application with MapLibre canvas overlays and interactive ocean exploration.
- `fixtures/sample_outputs/`: Contract-exact synthetic sample outputs for frontend and backend development.
- `docs/`: Product Requirements Documents (PRDs), decision logs, phase execution roadmaps, and progress tracking.

## Development Checks
- **ML**: `cd ml && ruff check . && ruff format --check . && mypy src && pytest -q`
- **Backend**: `cd backend && ruff check . && ruff format --check . && mypy app && pytest -q && python -m app.export_openapi --check`
- **Frontend**: `cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build`
