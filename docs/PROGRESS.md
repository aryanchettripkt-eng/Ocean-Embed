# Progress

| Date | Area | Task | Agent | Branch | Status | Verified by |
|---|---|---|---|---|---|---|
| 2026-10-22 | setup | P0-2, P0-3, P0-5: Repo scaffolding, CI, manifest schema, sample outputs | Gemini | setup/phase0-scaffold | complete | ruff, mypy, pytest, openapi check, pnpm typecheck/lint/test/build |
| 2026-10-22 | setup | P0 review fixes (A1-A8, B1-B7): xarray zarr store, fixture tests, CI path filters, mypy working-dir, pnpm 11 lockfile, gitignore | Gemini | setup/p0-fixes | complete | ruff, mypy, pytest (incl test_sample_outputs), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-22 | setup | P0 re-review fixes (R1-R4): Node 22 CI & engines, ML gitignore rules & test, remove backend loader fallback + 503 tests, zarr/xarray dependency floors | Gemini | setup/p0-fixes | complete | ruff, mypy, pytest (ML & Backend), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-22 | setup | P0 re-review fixes (R5): anchor data/outputs/checkpoints gitignore, @types/node 22, commit PRD v5 & review docs | Gemini | setup/p0-fixes | complete | ruff, mypy, pytest (ML & Backend), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | backend | P1-B0: manifest schema headline, v5 fixtures (headline CSV, GREP/ARMOR3D ARGO cols, live dates, 1993/2018 splits) | Gemini | backend/p1-b0-fixtures | complete | ruff, mypy, pytest, openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | backend | P1-B1: Settings, loader, manifest validation, /health, /meta, openapi export | Gemini | backend/p1-b1-skeleton | complete | ruff, mypy, pytest, openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | backend | P1-B2: /map, /profile, /timeseries, /region-summary + grid snapping | Gemini | backend/p1-b2-data-endpoints | complete | ruff, mypy, pytest (19 passed incl warm p95 benchmark), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | backend | P1-B3: /metrics, /argo, /argo/profiles, /headline, caveat strings | Gemini | backend/p1-b3-validation-endpoints | complete | ruff, mypy, pytest (26 passed), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | backend | P1-B4: Live-date handling, LIVE_CAVEAT, periodic manifest reload | Gemini | backend/p1-b4-live-and-reload | complete | ruff, mypy, pytest (29 passed), openapi check, pnpm typecheck/lint/test/build |
| 2026-10-23 | frontend | P1-F1: Next.js skeleton, Tailwind, theme, layout, generated API types, MSW mocks, CI | Gemini | frontend/p1-f1-skeleton | complete | pnpm typecheck, pnpm lint, pnpm test (21 passed), pnpm build |
| 2026-10-23 | frontend | P1-F2: Explorer map: MapLibre, canvas grid layer, controls, colorbar, coverage legend, URL state | Gemini | frontend/p1-f2-explorer-map | complete | pnpm typecheck, pnpm lint, pnpm test (22 passed), pnpm build |
| 2026-10-23 | frontend | P1-F3: Details panel: profile + time-series charts, depth accuracy line | Gemini | frontend/p1-f3-details-panel | complete | pnpm typecheck, pnpm lint, pnpm test (26 passed), pnpm build |
| 2026-10-23 | frontend | P1-F4: /validation (headline card, metrics tables, ARGO matchups) and /about pages, caveats, sealed test status | Gemini | frontend/p1-f4-validation-about | complete | pnpm typecheck, pnpm lint, pnpm test (30 passed), pnpm build |
| 2026-10-23 | ml | P1-M1: download.py CLI with dry-run planning, product registry, auth checks, indices & depth subsetting | Gemini | ml/p1-m1-m2-download-inspect | in progress | ruff, mypy, pytest (9 passed), download CLI (blocked on P0-4 accounts) |
| 2026-10-23 | ml | P1-M2: inspect.py NetCDF/Zarr coordinate, depth & variable inspector | Gemini | ml/p1-m1-m2-download-inspect | blocked on P0-4 accounts | ruff, mypy, pytest (9 passed), synthetic file inspect test |
| 2026-10-23 | full-stack | P1 review fixes (A1-A8, B1-B6, B10-B12): remove fabricated inspection & synthetic floats, real inspect.py, product table fixes, resilient loader, 422 date validation, Starlette 404 envelope, HealthResponse, dynamic grid res & science text, HeadlineCard n/a support & period filtering, msw worker init & export_mocks, vitest component tests, Playwright E2E tests, CI checks & permissions | Gemini | p1-fixes | complete | ml (ruff, format, mypy, pytest 9 passed), backend (ruff, format, mypy, pytest 32 passed, export_openapi --check, export_mocks --check), frontend (typecheck, lint, test 39 passed, build, e2e) |
| 2026-10-23 | full-stack | P1 re-review fixes: real downloads (Copernicus/OSCAR/CCMP/ERA5/GEBCO/WOA/ARGO with auth/exit checks), stratified regional ARGO holdout (BoB holdout per year & region counts), GLORYS-only depths, fixture generator test split rows, CI E2E test run & robust tests (date step, map cell click, theme toggle), backend /meta precomputed ranges, Next.js bundle budget (<350 kB, down to 128 kB), light mode, panel error boundaries, playback prefetch, HF remote commit tracking, Niño 3.4 weekly anomaly | Gemini | p1-fixes | complete | ml (ruff, format, mypy, 11 tests), backend (ruff, format, mypy, 32 tests, openapi/mocks check), frontend (typecheck, lint, 39 tests, build, 8 E2E tests) |





