# PRD: Backend API — OceanScope AI

**Status:** Draft v2 (2026-10-22): live dates, `/headline`, competitor ARGO columns, periodic reload  
**Owner:** Aryan  
**Implementer:** Coding agent  
**Lives in:** `backend/` of the monorepo (see `AGENTS.md`).  
**Depends on:** the output contracts in `docs/PRD_AI_ML.md` Section 6 (manifest, prediction store, metrics files). The backend never trains or runs the model.

---

## 0. Rules for the implementer (non-negotiable)

1. **Read-only over precomputed outputs.** The API serves values from the published output tree (`PRD_AI_ML.md` 6.4). It must not import torch, load checkpoints, or compute predictions.
2. **Never fabricate data.** If a value is missing, return `null` with a coverage code, or a documented error. Do not interpolate, fill, or default numbers that are not in the files.
3. **The API contract in Section 4 is frozen** once the frontend starts using it. Changes need the owner's approval, a version note in `docs/DECISIONS.md`, and a regenerated `backend/openapi.json`.
4. **Secrets only from environment variables.** No API keys, tokens or URLs with credentials in code, tests, logs, or error messages. `.env` is git-ignored; `.env.example` lists every variable with a dummy value.
5. **The LLM never produces numbers on its own.** Every number in an assistant answer must come from a tool result in the same request (Section 6).
6. **Tests must not use the network.** Use the committed sample outputs (`fixtures/sample_outputs/`) and a mocked LLM.
7. **Do not add dependencies** outside `backend/pyproject.toml`; state why in the commit message.
8. When unsure, write the question in `docs/OPEN_QUESTIONS.md` and stop that task.

---

## 1. Goal

A small, fast, public, read-only HTTP API that lets the Next.js frontend (and the chat assistant) query model predictions, GLORYS reference values, coverage, and evaluation metrics by date, depth and location.

### Non-goals
- User accounts, auth, writes, uploads.
- Model training or live inference.
- Serving raw input datasets.

---

## 2. Stack (fixed)

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| Framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 (+ pydantic-settings for config) |
| Data access | xarray + zarr (lazy), pandas/pyarrow for CSV/Parquet |
| Data source | Hugging Face Hub dataset repo via `huggingface_hub.snapshot_download` |
| Caching | in-process LRU (`functools.lru_cache` / `cachetools`) |
| Rate limiting | `slowapi` |
| LLM client | `httpx` against an OpenAI-compatible Chat Completions endpoint (Section 6) |
| Lint/format/type | ruff, ruff format, mypy (strict on `app/`) |
| Tests | pytest, httpx `TestClient` |
| Packaging/deploy | Dockerfile → Hugging Face Spaces (Docker SDK) |

---

## 3. Configuration (env vars, via pydantic-settings)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATA_DIR` | no | `/data/outputs` | Local path of the output tree |
| `HF_DATASET_REPO` | no | – | If set and `DATA_DIR` has no `manifest.json`, download the repo there at startup |
| `HF_DATASET_REVISION` | no | `main` | Pin a commit for reproducible demos |
| `HF_TOKEN` | no | – | Only if the dataset repo is private |
| `CORS_ORIGINS` | yes (prod) | `http://localhost:3000` | Comma-separated; must include the Vercel domain(s) |
| `CHAT_ENABLED` | no | `false` | Feature flag for `/chat` |
| `LLM_BASE_URL` | if chat | – | OpenAI-compatible base URL |
| `LLM_API_KEY` | if chat | – | Never logged |
| `LLM_MODEL` | if chat | – | Model id |
| `LLM_PROVIDER_LABEL` | no | `LLM` | Display name returned to the UI (e.g. "Llama via Meta API") |
| `CHAT_RATE_LIMIT` | no | `10/minute` | Per IP |
| `CHAT_DAILY_CAP` | no | `500` | Global requests per UTC day; then 429 |
| `RELOAD_INTERVAL_MIN` | no | `60` | How often to check for a new manifest / HF revision (live mode appends a day daily). `0` disables. |
| `LOG_LEVEL` | no | `INFO` | |

Startup fails fast with a clear message if `manifest.json` is missing or invalid (validated against `app/schemas/manifest.schema.json`).

---

## 4. API contract (v1, base path `/api/v1`)

General rules:
- JSON only, UTF-8, `snake_case` keys. NaN is serialized as `null`.
- Temperatures in °C rounded to 2 decimals; coordinates to 3 decimals.
- Dates are `YYYY-MM-DD` (UTC days).
- `depth_m` must equal one of `meta.depths_m` (tolerance 0.01 m); otherwise 422. `depth_index` (0–14) is accepted as an alternative; supplying both is a 422.
- Locations snap to the nearest grid cell centre; responses return the snapped `lat`/`lon`. Points outside the region → 422 `out_of_region`.
- Dates outside `available_dates` → 404 `date_unavailable`.
- Error body (all 4xx/5xx): `{"error": {"code": "<snake_case>", "message": "<human readable>", "details": {...}|null}}`.
- Every response carries `run_id` and `synthetic` (from the manifest) at top level, except `/health`.
- Cache headers: data endpoints `Cache-Control: public, max-age=86400` plus an `ETag` derived from `run_id` + query; `/meta` `max-age=300`; `/chat` `no-store`.
- GZip middleware for responses > 1 KB.

### 4.1 `GET /health`
`200 {"status":"ok","run_id":"...","data_loaded":true}`. Returns 503 if data failed to load.

### 4.2 `GET /meta`
```json
{
  "run_id": "...", "synthetic": false, "dataset_version": "v1", "model_name": "unet",
  "region": {"lon_min":40,"lon_max":100,"lat_min":0,"lat_max":30},
  "grid_resolution_deg": 0.25,
  "grid_shape": {"lat": 120, "lon": 240},
  "depths_m": [ ...15 floats... ],
  "available_dates": {"start":"2018-01-01","end":"2021-12-31"},
  "latest_live_date": null,
  "splits": {"train":[...],"val":[...],"test":[...]},
  "test_evaluated": false,
  "color_ranges": [{"depth_m": 0.49, "vmin": 22.0, "vmax": 31.5}, ...],
  "error_range": {"vmin": -2.0, "vmax": 2.0},
  "caveats": {"glorys": "...", "argo": "...", "synthetic": "...", "live": "..." },
  "chat": {"enabled": false, "provider_label": null}
}
```
`color_ranges` = 2nd/98th percentile of `temp_glorys` per depth over published dates, computed once at startup and cached.

### 4.3 `GET /map?date=&depth_m=|depth_index=&field=pred|glorys|error`
`error` = pred − glorys.
```json
{
  "run_id":"...","synthetic":false,"date":"2020-06-01","depth_m":100.0,"field":"pred",
  "split":"val","units":"degC",
  "lat":[0.125, ...], "lon":[40.125, ...],
  "values":[[...],[...]],          // [lat][lon], null where unavailable
  "coverage":[[...],[...]],        // [lat][lon], 0 land/unsupported, 1 valid, 2 insufficient inputs
  "stats":{"min":..,"max":..,"mean":..,"n_valid":..}
}
```
Must respond in < 300 ms warm, < 2 s cold (one chunk read) on HF Spaces CPU basic.

On a `live` date (`split = "live"`), `field=glorys|error` returns 404 `reference_unavailable`; `field=pred` works as usual.

### 4.4 `GET /profile?lat=&lon=&date=`
```json
{ "run_id":"...","synthetic":false,"date":"...","lat":..,"lon":..,"split":"val","coverage":1,
  "depths_m":[...], "pred":[...], "glorys":[...], "error":[...] }
```

### 4.5 `GET /timeseries?lat=&lon=&depth_m=|depth_index=&start=&end=`
Max 366 days per request (else 422 `range_too_large`). Returns `dates[]`, `pred[]`, `glorys[]`, `split[]`.

### 4.6 `GET /region-summary?date=&depth_m=|depth_index=&lat_min=&lat_max=&lon_min=&lon_max=&field=pred|glorys|error`
Area-weighted (cos(lat)) mean, min, max, std, `n_valid`, `n_total` over cells with coverage 1. Bbox must intersect the region and contain ≥ 1 cell. Also accepts `region=arabian_sea|bay_of_bengal|equatorial` instead of a bbox (definitions from `PRD_AI_ML.md` 6.3).

### 4.7 `GET /metrics?split=val|test&group_by=depth|season|region`
Returns rows from the metrics CSVs for the active run **and** each baseline, as `{"model","split","depth_m","group","n","mae_c","rmse_c","bias_c","skill_vs_clim_pct"}`. `split=test` returns 404 `test_not_evaluated` while `test_evaluated` is false.

### 4.8 `GET /argo`
Returns the `argo_<run_id>.csv` rows (including the GREP and ARMOR3D columns, `null` where empty) plus `{"n_profiles_total","n_obs_total","caveat"}` where `caveat` is the fixed string in Section 7.

### 4.8b `GET /headline`
Returns the `<run_id>_headline.csv` rows (the pre-registered result, PRD_AI_ML §1.1) plus `{"criterion": "<fixed text from Section 7>", "met": true|false|null}` where `met` is true only if every competitor row with data has `diff_ci_high_c < 0` for the reported period (test if `test_evaluated`, else validation). 404 `headline_not_available` if the manifest has no headline file. Never computed by the backend from other files.

### 4.9 `GET /argo/profiles?date=&lat_min=&lat_max=&lon_min=&lon_max=`
Matched ARGO profiles (from the matches Parquet) for map markers: `[{platform_number, cycle, time, lat, lon, n_obs}]`, max 500 rows.

### 4.10 `POST /chat`
See Section 6. Request `{"messages":[{"role":"user"|"assistant","content":"..."}], "context": {"date":..,"depth_m":..,"lat":..,"lon":..}|null}` (max 10 messages, each ≤ 2,000 chars). Response:
```json
{ "answer": "...", "tool_calls": [{"name":"get_temperature_profile","arguments":{...},"result":{...}}],
  "provider_label": "...", "run_id": "...", "synthetic": false }
```
Returns 503 `chat_disabled` when `CHAT_ENABLED=false` or the LLM env vars are missing; 429 `rate_limited` / `daily_cap_reached`; 502 `llm_error` on upstream failure (no upstream body leaked).

OpenAPI: FastAPI's schema exported to `backend/openapi.json` by `python -m app.export_openapi`; CI fails if it is stale. The frontend generates its TypeScript types from this file.

---

## 5. Internal structure

```
backend/
├── pyproject.toml
├── Dockerfile
├── .env.example
├── openapi.json
├── app/
│   ├── main.py              # app factory, middleware, routers
│   ├── settings.py          # pydantic-settings
│   ├── data/
│   │   ├── loader.py        # download (HF), manifest validation, open zarr lazily
│   │   ├── store.py         # OutputStore: typed accessors (map, profile, timeseries, region, metrics)
│   │   └── grid.py          # snapping, bbox → index slices, area weights
│   ├── api/v1/              # one router module per endpoint group
│   ├── chat/
│   │   ├── provider.py      # OpenAI-compatible client
│   │   ├── tools.py         # tool schemas + dispatch to OutputStore
│   │   ├── prompt.py        # system prompt (Section 6.3)
│   │   └── guard.py         # number-grounding check (Section 6.4)
│   ├── schemas/             # pydantic response models + manifest.schema.json
│   └── errors.py
└── tests/
```
Routers contain no data logic; they call `OutputStore`. Chat tools call the **same** `OutputStore` methods as the HTTP endpoints.

---

## 6. Chat assistant (feature-flagged)

### 6.1 Provider
One `OpenAICompatibleProvider` using Chat Completions with `tools`. Target provider is Meta (Muse Spark / Llama API) when the owner has access; until then any OpenAI-compatible endpoint configured by env (e.g. a hosted Llama model) is acceptable. No provider-specific SDKs. If the configured provider does not support tool calling, chat stays disabled (do not attempt prompt-only number answering).

### 6.2 Tools (exact names)
| Tool | Args | Maps to |
|---|---|---|
| `get_temperature` | `lat, lon, date, depth_m` | `/profile` value at that depth |
| `get_temperature_profile` | `lat, lon, date` | `/profile` |
| `get_region_summary` | `region` or bbox, `date, depth_m, field` | `/region-summary` |
| `get_validation_metrics` | `depth_m?`, `group_by?` | `/metrics?split=val` (+ baselines) |
| `get_dataset_info` | – | `/meta` subset (dates, depths, region, caveats) |

Max 4 tool-call rounds per request; max 1,024 output tokens; 20 s total timeout. Tool errors are returned to the model as structured errors, not raised.

### 6.3 System prompt requirements (`prompt.py`, versioned constant)
Must instruct the model to: use tools for every number; say "the data doesn't cover that" when a tool returns unavailable; always distinguish *model prediction* vs *GLORYS reanalysis* vs *ARGO observation*; mention that predictions are estimates and state the validation RMSE at that depth when asked about accuracy; refuse unrelated tasks briefly; never claim real-time data (live dates are about 3 days behind real time and have no GLORYS reference; say so when asked about them). If `synthetic` is true, state that the values are synthetic demo data.

### 6.4 Grounding guard
After the final answer: extract every number with a decimal point or a unit (°C, m) from the answer and check it appears (after rounding to the shown precision) in some tool result or in the request context. If any does not, replace the answer with a fixed fallback ("I couldn't verify that answer against the data; here are the raw results:" + a compact rendering of the tool results) and log the event (without user text).

### 6.5 Abuse controls
Rate limit per IP, global daily cap, message length limits, no tool other than the five above, no URL fetching, no code execution.

---

## 7. Fixed caveat strings (`app/schemas/caveats.py`)
- `GLORYS_CAVEAT`: "Predictions are trained to reproduce the GLORYS reanalysis. GLORYS assimilates satellite altimetry and ARGO profiles, so agreement with GLORYS is not the same as agreement with the real ocean."
- `ARGO_CAVEAT`: "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent."
- `LIVE_CAVEAT`: "Live maps are made about 3 days behind real time from near-real-time inputs. There is no GLORYS reference for these dates yet, so they have not been checked the way validation dates have."
- `HEADLINE_CRITERION`: "Bay of Bengal, 50–300 m: model RMSE vs held-out ARGO floats lower than GLORYS12, the GREP ensemble mean and ARMOR3D (95% float-bootstrap interval of the difference below zero)."
- `SYNTHETIC_CAVEAT`: "This deployment is serving synthetic demo data, not model results."

Returned by `/meta` (`caveats` object) and used by the chat prompt. The frontend displays them verbatim.

---

## 8. Hosting

**Backend: Hugging Face Spaces, Docker SDK** (free CPU tier, ~16 GB RAM, public HTTPS URL). Reason: the API needs xarray/zarr, a GB-scale data store and a warm process; Vercel's Python functions have bundle-size limits, no persistent disk and cold starts that make this a poor fit.
- Dockerfile: `python:3.11-slim`, non-root user, listens on port `7860`, `uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1`.
- Data: `snapshot_download` from `HF_DATASET_REPO` into `DATA_DIR` at startup (Spaces storage is ephemeral; re-download on restart is acceptable at ≤ 3 GB).
- Free Spaces sleep after inactivity; the frontend must handle a slow first request (see frontend PRD). Before the demo, open the Space to wake it.
- Alternatives if Spaces is unsuitable (owner decides): Render/Railway/Fly with ≥ 2 GB RAM.

**Frontend:** Vercel (see `PRD_FRONTEND.md`).

---

## 9. Tests (required; `pytest -q` from `backend/`, < 30 s, no network)
1. Startup with `fixtures/sample_outputs` succeeds; with a broken manifest it fails with a clear error.
2. Every endpoint: one happy path, plus each documented error code (422 out_of_region, bad depth, both depth params; 404 date_unavailable, test_not_evaluated; 503 chat_disabled).
3. Snapping: a point exactly between cells snaps deterministically (document: round half up on index).
4. `/map` shapes equal `grid_shape`; nulls exactly where the fixture has NaN; coverage codes pass through.
5. `/region-summary` area weighting checked against a hand-computed fixture value.
6. `/profile` values equal the raw zarr values (cast to float32, rounded to 2 dp).
7. Chat with a mocked provider: tool dispatch works; a fabricated number triggers the grounding fallback; round limit enforced; API key never appears in logs or responses.
8. `openapi.json` is up to date.
9. CORS: allowed origin passes, other origins get no CORS headers.
10. Live dates: fixture `live` days return `split: "live"`, `/map?field=glorys` → 404 `reference_unavailable`, `/profile` has `glorys: null` values.
11. `/headline`: rows pass through unchanged; `met` follows the rule in 4.8b; 404 when the manifest has no headline file.
12. Reload: replacing `manifest.json` in `DATA_DIR` (or a new HF revision) is picked up within `RELOAD_INTERVAL_MIN` without restart; a broken new manifest keeps serving the old one and logs an error.

---

## 10. Milestones and acceptance

| M | Scope | Done when |
|---|---|---|
| B1 Skeleton | settings, loader, `/health`, `/meta`, Dockerfile, CI (ruff, mypy, pytest) | Runs locally on `fixtures/sample_outputs`; tests 1, 8 pass |
| B2 Data endpoints | `/map`, `/profile`, `/timeseries`, `/region-summary` | Tests 2–6 pass; `/map` warm p95 < 300 ms locally |
| B3 Metrics | `/metrics`, `/argo`, `/argo/profiles`, `/headline` | Tests pass against fixture metrics (incl. 11) |
| B4 Deploy | HF Space live serving sample outputs; then real outputs once ML M4 is done | Public URL `/api/v1/health` is 200; Vercel frontend can call it (CORS) |
| B6 Live support | `live` dates, `LIVE_CAVEAT`, periodic manifest reload (`RELOAD_INTERVAL_MIN`, default 60) | Tests 10, 12 pass |
| B5 Chat | provider, tools, prompt, guard, limits | Test 7 passes; enabled in prod only after the owner tests it |

B1–B4 must not wait for the real model: they are built and deployed against the synthetic sample outputs.

---

## 11. Open questions for the owner
- Meta API access (Muse Spark / Llama API) for chat; if none, which OpenAI-compatible provider to use, or keep chat off.
- Hugging Face account/org name for the Space and dataset repo.
