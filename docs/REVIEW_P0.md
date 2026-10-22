# Review of Gemini's P0 work (P0-2, P0-3, P0-5)

Reviewer: Claude · 2026-10-22 · commit `e0b0998` ("setup: scaffold ml, backend, frontend, CI, and sample outputs")

**Verdict: not done.** The scaffold is mostly sound, but the sample fixtures can't be opened by xarray, CI doesn't do what P0-3 asked for, and the `.gitignore` will silently drop a frontend folder later. Fix the items in section A on a new branch `setup/p0-fixes`, run every area's checks, and update `docs/PROGRESS.md`.

How this was checked: every committed file was read; the manifest was validated against the schema; the zarr store was decoded chunk by chunk and opened with xarray 2026.9 / zarr 3.4; the git reflog and index were inspected. CI on GitHub could **not** be checked (no repo access from the reviewer), so "CI green on main" is unverified.

---

## A. Must fix (blocks P1)

### A1. The prediction store can't be opened by xarray (contract break, PRD_AI_ML §6.2)
`fixtures/sample_outputs/predictions/predictions_synthetic_v1.zarr` was written with the raw `zarr` API, so no array has the `_ARRAY_DIMENSIONS` attribute. Result:
```
xr.open_zarr(...)  ->  KeyError: Zarr object is missing the attribute `_ARRAY_DIMENSIONS` ...
```
The backend reads this store with xarray, so B1 can't even start on these fixtures.
**Fix:** in `make_sample_outputs.py`, build an `xarray.Dataset` (coords `time` as datetime64[ns], `depth`, `lat`, `lon`; vars `temp_pred`, `temp_glorys`, `coverage` int8) and write it with `ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2)` with encoding chunks `(1, 15, nlat, nlon)` / `(1, nlat, nlon)`. Regenerate the fixtures.

### A2. No test proves the fixtures are valid (P0 exit gate)
The exit gate says the fixtures must be "valid against the schema". Nothing in CI checks that, which is how A1 slipped through.
**Fix:** add `ml/tests/test_sample_outputs.py` that (a) runs the generator into a tmp dir, (b) validates `manifest.json` against `manifest.schema.json`, (c) opens the store with `xr.open_zarr` and asserts variable names, dims, dtypes, chunking (one day per chunk), `coverage` values ⊂ {0,1,2}, `temp_pred` is NaN wherever coverage ≠ 1, and the root attrs from §6.2, (d) checks every CSV has the §6.3 columns plus a `depth_m = all` row (not required for the ARGO CSV), and (e) checks the committed `fixtures/sample_outputs/` matches a fresh generation (or at least validates the committed copy too).

### A3. CI has no path filters (P0-3 requirement)
`ci.yml` runs all three jobs on every push. P0-3 explicitly says "path filters".
**Fix:** use `dorny/paths-filter` (or per-job `if:` on changed paths) so `ml/**` triggers ml, `backend/**` + `fixtures/**` + the ml schema trigger backend, `frontend/**` + `backend/openapi.json` trigger frontend. Keep the jobs required on PRs (a skipped job must still report success).

### A4. mypy in CI ignores the project's mypy settings
CI runs `mypy ml/src` and `mypy backend/app` **from the repo root**. mypy only looks for config in the current directory (and upwards), so `ml/pyproject.toml` (`disallow_untyped_defs`) and `backend/pyproject.toml` (`strict = true`) are never loaded in CI. CI is therefore weaker than the local checks AGENTS.md §6 lists. That counts as weakening a check (AGENTS.md hard rule 6).
**Fix:** give each Python job `defaults: run: working-directory: ml` (resp. `backend`) and run exactly the AGENTS.md §6 commands: `ruff check . && ruff format --check . && mypy src && pytest -q`, etc. Install with `pip install -e ".[dev]"` instead of the hand-written package lists, so CI uses the declared dev deps.

### A5. Frontend CI: pnpm version mismatch and unfrozen lockfile
- Locally Gemini used pnpm 10+/11 (`.pnpm-store/v11`, and `pnpm-workspace.yaml` uses the newer `allowBuilds` key). CI pins **pnpm 9**. With a `pnpm-workspace.yaml` that has no `packages:` field, pnpm 9 is likely to fail with "packages field missing or empty", and `allowBuilds` isn't a pnpm 9 setting.
- `pnpm install --no-frozen-lockfile` lets CI rewrite the lockfile instead of failing when it's out of date.

**Fix:** add `"packageManager": "pnpm@<exact local version>"` to `frontend/package.json`, let `pnpm/action-setup` read it (drop `version: 9`), and use `pnpm install --frozen-lockfile`. If `pnpm-workspace.yaml` is only there for build-script approval, keep it but make sure the CI pnpm version understands it.

### A6. `.gitignore` will silently drop `frontend/src/lib/`
The Python template line `lib/` matches **any** folder named `lib`, including `frontend/src/lib/`, where PRD_FRONTEND §5 puts `api/types.ts`, `api/hooks.ts` and `colormaps.ts`. Those files would never be committed and Vercel builds would break. `downloads/`, `parts/`, `var/` and `build/` have the same problem.
**Fix:** anchor them to Python build output, e.g. `/ml/build/`, `/backend/build/`, or just delete `lib/`, `lib64/`, `parts/`, `var/`, `downloads/`. Also add `*.tsbuildinfo`.

### A7. Build artefact committed
`frontend/tsconfig.tsbuildinfo` (139 KB, generated) is in the commit. Remove it from git (`git rm --cached`) after A6 ignores it.

### A8. Fixture grid uses cell edges, not cell centres
The fixture grid is `lat = 0, 1, …, 30` (31 points) and `lon = 40, …, 100` (61 points), i.e. points **on** the bbox edges. The real grid is cell centres (`experiment.yaml`: `lon_min + 0.125, …`; PRD_BACKEND `/meta` `grid_shape` 120×240 for 0.25°). The backend and frontend will be built against an off-by-one, edge-aligned grid.
**Fix:** at 1° use centres `0.5 … 29.5` (30) and `40.5 … 99.5` (60).

---

## B. Should fix (cheap, do in the same branch)

- **B1. `/health` lies.** `backend/app/main.py` hardcodes `"data_loaded": True` and `"run_id": "initial_skeleton"`. PRD_BACKEND §4.1 says it must return 503 if data failed to load, and the run_id must come from the manifest. For the skeleton, return `data_loaded: false` / `run_id: null` with 503 until the loader exists, or implement the loader as part of B1. Don't ship a hardcoded "true" (AGENTS hard rule 1).
- **B2. Synthetic metrics don't match the synthetic data.** The CSVs hardcode the same MAE/RMSE at every depth (e.g. 0.28/0.38 everywhere), while the actual `temp_pred − temp_glorys` difference in the store varies from −0.05 to +0.34 °C by depth. Compute the fixture metrics from the fixture arrays so the frontend's numbers are self-consistent. The `depth_m = all` `n` should also equal the sum of the per-depth rows.
- **B3. `split_of_each_date` format.** PRD_AI_ML §6.2 says "JSON mapping date ranges → split"; the fixture maps each single date. Use ranges, e.g. `{"2020-01-01/2020-01-10": "val"}`, or raise it in OPEN_QUESTIONS if per-date was intentional.
- **B4. Dependency floor is wrong.** The generator uses the zarr 3 API (`create_array`, `zarr_format=`), but `ml/pyproject.toml` and `backend/pyproject.toml` allow `zarr>=2.14.2`. Moot if A1 switches to xarray; otherwise pin `zarr>=3`.
- **B5. Schema is looser than the contract.** `depths_m` should be `minItems: 15, maxItems: 15`, and `created_utc`'s `format: date-time` isn't enforced unless the validator gets a `FormatChecker`.
- **B6. `.gitignore` vs fixtures.** `*.zarr/` excludes the directory, and git can't re-include files under an excluded directory, so `!fixtures/sample_outputs/**` does nothing. The current files are tracked (they were force-added), but a regenerated fixture with new chunk files will be silently left out. Replace with `/data/**/*.zarr/` and `/outputs/` patterns instead of a global `*.zarr/`.
- **B7. `ml/README.md` describes modules that don't exist** (data, models, training, …). Say they're planned, or create empty packages. AGENTS.md §3 also lists a root `README.md` and `ml/notebooks/`, which are missing.

---

## C. Process

- **Merged to `main` without a PR.** The reflog shows `setup/phase0-scaffold` was fast-forwarded into `main` locally and then pushed to `origin/main`. AGENTS.md hard rule 8: never commit directly to `main`. If Aryan did this deliberately, fine; if Gemini did, it broke the rules. Either way, protect `main` on GitHub (P0-1) so it can't happen again.
- **PROGRESS.md says "complete"** while the exit gate (CI green on `main`, fixtures valid) wasn't actually demonstrated: there's no fixture validation test, and no link to a green CI run.

## D. Not Gemini's fault
`ml/configs/experiment.yaml` `inputs.core` still lists only `sla, u/v_current, u/v_wind`. That matches the PRD Gemini was given. The v3 PRD (SST, SSS, ADT, wind-stress curl, ERA5, static priors, history) landed afterwards and will be applied in P1.

## What's good
Manifest validates against the schema; ML and backend schema copies are identical and CI diffs them; `synthetic: true` and the SYNTHETIC banner text are in place; coverage masking in the store is correct (pred is NaN wherever coverage ≠ 1); fixtures are about 1 MB, well under the 5 MB limit; secrets are env-only; the Dockerfile runs as non-root on port 7860; the OpenAPI export/check works.

---

# Re-review of `setup/p0-fixes` (commit `2ece8e8`)

Reviewer: Claude · 2026-10-22. I re-ran the ML and backend checks in a clean copy (ruff, ruff format, mypy, pytest, openapi check, schema parity: **all pass**), opened the new fixture store with xarray 2026.9 / zarr 3.4, and recomputed the fixture metrics from the arrays.

**Fixed and verified:** A1 (store opens in xarray, consolidated, correct dims/dtypes, daily chunks), A2 (`test_sample_outputs.py` checks generated and committed fixtures), A3 (path filters), A4 (per-package working dirs, `.[dev]` installs, strict mypy now applies), A6 (`lib/` gone), A7 (`tsbuildinfo` removed and ignored), A8 (cell-centre grid 30×60), B2 (metrics now computed from the arrays and match), B3 (date-range split map), B5 (15 depths, format checker), B7 (README, root README, `ml/notebooks/`). `main` was not touched; the work is on its own branch.

## Still wrong (fix before merging)

### R1. Frontend CI will fail: pnpm 11 needs Node 22, CI uses Node 20
`frontend/package.json` now pins `pnpm@11.25.0`, whose `engines` field is `node >=22.13`. CI still sets up Node 20, so pnpm refuses to run. (It passed locally because the local Node is newer.) Node 20 reached end of life in April 2026 anyway.
**Fix:** set `node-version: "22"` in `ci.yml`, add `"engines": {"node": ">=22.13"}` to `frontend/package.json`, set Node 22 in Vercel project settings, and change "Node 20 LTS" to "Node 22 LTS" in AGENTS.md §8.

### R2. `.gitignore` no longer ignores data written by the ML pipeline (regression)
The data rules were anchored to the repo root (`/data/raw/`, `/outputs/`, `/checkpoints/`, `/data/**/*.nc`). But every ML stage runs **from `ml/`** (PRD_AI_ML §4) with relative paths `data/raw`, `outputs`, `checkpoints`, so files land in `ml/data/raw/`, `ml/outputs/`, `ml/checkpoints/`, none of which are ignored now. One `git add .` after a download would commit gigabytes of NetCDF.
**Fix:** use `ml/data/`, `ml/outputs/`, `ml/checkpoints/`, plus unanchored `*.nc` and `*.zarr/` with a working exception for fixtures: either `!fixtures/**/*.zarr/` on the directory itself, or keep fixtures un-ignored by only ignoring `*.zarr/` under `ml/`. Add a test or CI step: `git check-ignore -q ml/data/raw/x.nc && git check-ignore -q ml/outputs/x && ! git check-ignore -q fixtures/sample_outputs/predictions/predictions_synthetic_v1.zarr/.zattrs`.

### R3. Backend silently serves synthetic data when real data is missing
`backend/app/data/loader.py` falls back to `../fixtures/sample_outputs` whenever `DATA_DIR` has no manifest. I ran the app with `DATA_DIR=/nonexistent` and got `200 {"status":"ok","run_id":"synthetic_v1","data_loaded":true}`. PRD_BACKEND §4.1 says `/health` must return 503 when data failed to load. In production the Docker image doesn't contain `fixtures/`, so it would 503 there, but any deploy that includes the repo (or a dev pointing at a wrong path) would quietly serve fake data and report healthy.
**Fix:** remove the implicit fallback. Tests set `DATA_DIR` to the fixtures path in `backend/tests/conftest.py` (env var or `DataLoader(data_dir=...)` injected into the app). Add a test that `/health` returns 503 for an empty `DATA_DIR` (the current failure test only checks the loader object, not the endpoint).

### R4. Dependency floors (B4) not changed
Both `pyproject.toml` files still allow `zarr>=2.14.2`, and the generator calls `to_zarr(..., zarr_format=2)`, which needs the zarr-3-era xarray API. Set `zarr>=3` and `xarray>=2024.10` (or whatever the oldest versions you actually tested are) in both packages.

## Minor
- `PROGRESS.md` says "pnpm 11 lockfile", but `frontend/pnpm-lock.yaml` wasn't changed in this commit. Fine if it was already pnpm 11-generated; R1's CI run will prove it.
- CI on GitHub still unverified by me. After R1–R4, push the branch, open a PR to `main`, and paste the green CI link into PROGRESS.md. That is the P0 exit gate.
