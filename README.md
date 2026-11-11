# OceanScope AI

**See the ocean below the surface.** OceanScope AI reconstructs daily sea temperature at 15 depths, from the surface down to about 1,060 m, across the North Indian Ocean. It uses only what satellites and a few other sources can see from the surface, plus recent ARGO float profiles. You can explore the result as a 2D map or a 3D cutaway block, ask questions about it in plain language (typed or spoken), and even ask it out loud through a pair of Meta smart glasses.

- **Live site:** https://ocean-embed.vercel.app
- **API:** https://oceanscope-backend.onrender.com/api/v1 (OpenAPI schema in [`backend/openapi.json`](backend/openapi.json))
- **Published outputs:** Hugging Face dataset [`zenzone3/oceanscope-outputs`](https://huggingface.co/datasets/zenzone3/oceanscope-outputs)
- **Model run shown:** `main3_aix30a` since 10 November 2026: a U-Net that takes recent ARGO floats and the ARMOR3D product as extra inputs (see [Results](#results)). Before that the site showed `ens3`, then `main2_ft`.
- **Live:** a new day is added every day at 06:00 UTC from near-real-time inputs, about 7 days behind today (see [Live mode](#live-mode-a-new-day-every-day))
- **Uptime:** an UptimeRobot monitor pings the backend every 5 minutes, backed up by a GitHub Actions ping (see [Keeping the backend awake](#keeping-the-backend-awake))

> **The result in one line.** On ARGO floats from 2021 that the model never saw, in the Bay of Bengal at 50–300 m, `main3_aix30a` scored **0.709 °C RMSE against 0.873 °C for GLORYS12** (19% lower, a significant difference), tied with the GREP ensemble (0.743 °C), and stayed behind ARMOR3D (0.570 °C), which is one of its inputs. That test was run once and is now spent.

> **Honest caveat, up front.** The model is trained to reproduce the GLORYS12 reanalysis and then fine-tuned on ARGO floats. GLORYS12 and ARMOR3D both assimilate ARGO profiles, including the floats we hold out, so our ARGO comparisons put us on equal terms with GLORYS12 but are not fully independent of those floats. Our model sees that float information through ARMOR3D, and ARMOR3D on its own is still more accurate. See [Limitations](#limitations).

## Contents

- [Why this matters](#why-this-matters)
- [What it does](#what-it-does)
- [Results](#results)
- [Architecture](#architecture)
- [Frontend (Next.js on Vercel)](#frontend-nextjs-on-vercel)
- [Backend (FastAPI on Render)](#backend-fastapi-on-render)
- [Meta AI features: "Talk to the ocean"](#meta-ai-features-talk-to-the-ocean)
- [Android: Ocean Glasses](#android-ocean-glasses)
- [Live mode: a new day every day](#live-mode-a-new-day-every-day)
- [Keeping the backend awake](#keeping-the-backend-awake)
- [ML pipeline](#ml-pipeline)
- [Reproduce it](#reproduce-it)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Data credits](#data-credits)

---

## Why this matters

Satellites measure the sea surface every day, but most of the ocean's heat sits below it, where the only direct measurements are sparse: a few thousand ARGO floats worldwide, each profiling about every 10 days. The layer between 50 and 300 m holds the thermocline, which steers cyclone strength, monsoon rainfall and fisheries in the Bay of Bengal and the Arabian Sea. OceanScope fills in that layer every day from surface observations, with a small neural network and an honest scorecard.

## What it does

| | |
|---|---|
| **Region** | 40–100°E, 0–30°N (Arabian Sea, Bay of Bengal, northern equatorial Indian Ocean) |
| **Grid** | 0.25°, daily |
| **Depths** | 15 native GLORYS12 levels: 0.5, 5, 10, 18, 29, 47, 66, 92, 131, 186, 222, 266, 380, 644 and 1,062 m (7 of them in 50–300 m) |
| **Period** | Train 1993-03-02 to 2017-12-01 · Validation 2018-01-01 to 2020-12-01 · Test 2021, sealed (the gridded test was spent on `main2_ft`, the ARGO test on `main3_aix30a`) · Live days from near-real-time inputs, about 7 days behind today |
| **Model** | Small 2D U-Net (≤ 10 M parameters) that predicts temperature anomalies from a training-period climatology. The live model also takes the last 30 days of ARGO profiles and the daily ARMOR3D field as input channels (155 in all) |
| **Headline test** | Bay of Bengal, 50–300 m: is the model's error against held-out ARGO floats lower than GLORYS12, the GREP ensemble mean and ARMOR3D? Fixed before training (see [Results](#results)) |

### At a glance

| Part | What it is | Where it runs |
|---|---|---|
| **ML** (`ml/`) | Data build, U-Net training, evaluation, prediction, publishing, daily live runs | Kaggle / Colab GPUs, GitHub Actions (live) |
| **Backend** (`backend/`) | FastAPI service: data API plus every Meta model call | Render (Docker, free tier) |
| **Frontend** (`frontend/`) | Next.js site: landing, Explorer 2D/3D, Validation, About, beach mode | Vercel |
| **Android** (`android/glasses-guide/`) | "Ocean Glasses" app for Ray-Ban/Oakley Meta glasses | Sideloaded on an Android phone |
| **Meta models** | Muse Spark, Muse Voice Transcribe, Muse Image, SAM 3.1 | Meta Model API, called only from the backend |
| **Monitoring** | UptimeRobot (every 5 min) and a GitHub Actions ping (every 10 min) | External |

---

## Results

All numbers below are RMSE in °C. "Held-out floats" are the 20% of ARGO floats that never feed the model, as input or as a training target. Every pick rule was written down before the run it judged ([`docs/DECISIONS.md`](docs/DECISIONS.md)).

### The live model: `main3_aix30a`

`main3` is a U-Net trained from scratch on 23 training years (1993–2017 without 2005 and 2006, which weren't built in time) with ARGO profiles as input channels, recency weighting and a year channel. `main3_aix30a` is `main3` fine-tuned on ARGO profiles from floats that are not held out (λ = 3), with two extra inputs: the last **30 days** of those floats, and the daily **ARMOR3D** temperature field (15 anomaly channels plus a mask, 155 input channels in all).

**One-shot 2021 ARGO test (the number to quote).** Bay of Bengal, 50–300 m, 7 held-out floats, about 16,000 measurements, run once on 10 November 2026:

| Product | RMSE vs held-out 2021 floats (°C) | Difference from ours (95% CI) |
|---|---|---|
| **OceanScope `main3_aix30a`** | **0.709** (CI 0.612–0.785) | |
| GLORYS12 | 0.873 | we are lower by 0.164 (0.018 to 0.283): **better** |
| GREP ensemble mean | 0.743 | we are lower by 0.033 (−0.021 to 0.091): **a tie** |
| ARMOR3D | 0.570 | ARMOR3D is lower by 0.139 (0.071 to 0.194): **behind** |

Depth by depth, the model beats GLORYS12 everywhere from 29 m down to 1,062 m, and is within 0.016 °C of it in the top 18 m. It is behind ARMOR3D at every depth. **Pre-registered verdict: not met**, because the headline rule asks to beat every reference, ARMOR3D included.

What this does and doesn't show: given the same ARGO information as GLORYS12 (GLORYS12 assimilates the floats; our model sees them through ARMOR3D), our model is more accurate at those floats in this band. It is not a skill independent of the floats, and ARMOR3D, one of its inputs, is more accurate still. Without the ARMOR3D channels the same recipe (`main3_aix30`) scored 0.840 on the dev half against 0.727 with them.

**Validation years, all 15 held-out floats (2018–2020), Bay of Bengal 50–300 m,** as shown on the site: ours 0.780, GLORYS12 0.786 (a tie, CI −0.028 to +0.023), GREP 0.952 (we're lower, but the CI just crosses zero), ARMOR3D 0.607. The model was picked on the 6-float dev half (0.727 there), which is why this wider score is higher.

**Against GLORYS12 on the grid, 1,036 validation days:** RMSE 0.655 °C, **34.1% skill** over the climatology baseline.

### How we got here: the ARGO rounds

The first published runs were trained only to imitate GLORYS12, and they were well behind it at real floats. From 6 to 10 November 2026 we ran a series of rounds against ARGO. To avoid fooling ourselves, the held-out floats were split into a **dev half** (6 Bay of Bengal floats, used to pick) and a **final half** (scored once). Every round had a rule written before it ran: replace the pick only if lower on the dev half.

| Run | What changed | Dev half, BoB 50–300 m |
|---|---|---|
| `ens3` | Average of `main2_ft`, `main2_ai`, `main3` (the published run until 10 Nov) | 0.867 |
| `ens3w_rm_oi` (run E) | Weighted ensemble + LightGBM residual model + guarded per-depth ARGO correction | 0.816 (final half 0.906; GLORYS12 0.794) |
| run F, `main3_aiw`, `ens4`, `ens5` | Richer residual features, warm-event weighting, more members | 0.821–0.860, no change of pick |
| `main3_aix30` (run J) | Fine-tune with a 30-day float window | 0.840, best single model without ARMOR3D |
| `main3_aix30L` (run L) | Longer training | 0.853: the fine-tune route had reached its limit |
| **`main3_aix30a` (run M)** | **ARMOR3D as 16 extra input channels** | **0.727, first run below GLORYS12 (0.756)** |
| `ens6w_rm2` (run N) | Best ensemble recipe with `main3_aix30a` added | 0.735, no change of pick |
| `main3_aix30ab` (run O) | Learned per-depth blend with ARMOR3D | 0.612, **ruled out**: about 80% of its answer below 18 m was ARMOR3D itself |

On the same dev half, GLORYS12 scored 0.756, GREP 0.855 and ARMOR3D 0.639. Run O scored best but was dropped on purpose, because its score was mostly ARMOR3D's rather than our model's. `main3_aix30a` was then the model that took the one-shot 2021 test.

### Earlier runs, for the record

- **`ens3`** (published 4–10 Nov): grid RMSE vs GLORYS12 0.644 °C (35.1% skill) on validation; held-out floats, Bay of Bengal 50–300 m, 0.957 °C against GLORYS12 0.786, GREP 0.952 and ARMOR3D 0.607. Not scored on 2021.
- **`main2_ft`** (published before `ens3`): validation 0.660 °C (33.5% skill). It took the sealed **gridded** 2021 test, run once: RMSE 0.753 °C, MAE 0.497 °C, bias −0.08 °C, 32.4% skill over climatology. Held-out floats, Bay of Bengal 50–300 m: 0.996 °C (`main2` before ARGO fine-tuning: 1.026).

The per-depth, per-season and per-region tables for the published run are in the dataset under `metrics/` and on the site's Validation page. The regime breakdown (`evaluation.regimes`) shows the ensembles do worst in warm anomalies, positive IOD and the southwest monsoon.

---

## Architecture

```mermaid
flowchart LR
  subgraph Sources["Public data sources"]
    CM["Copernicus Marine<br/>GLORYS12 · DUACS SLA/ADT · OSTIA SST<br/>MULTIOBS SSS · GREP · ARMOR3D"]
    ERA["ERA5 (Copernicus CDS / Google ARCO)<br/>10 m wind · heat flux · precipitation"]
    NASA["NASA PO.DAAC<br/>OSCAR currents"]
    STATIC["GEBCO bathymetry<br/>WOA23 climatology"]
    IDX["Climate indices<br/>Niño 3.4 · DMI · MJO RMM"]
    ARGO["Argo GDAC via ERDDAP<br/>float profiles"]
  end

  subgraph ML["ml/ (Kaggle / Colab GPU, GitHub Actions for live)"]
    BUILD["build_store<br/>stream → regrid to 0.25° → zarr"]
    TRAIN["train · ensemble<br/>U-Net on anomalies"]
    EVAL["evaluate · argo_validation<br/>baselines · headline"]
    PUB["publish"]
    LIVE["live (daily 06:00 UTC)<br/>NRT inputs + ARMOR3D NRT<br/>+ 30 days of ARGO → one new day"]
  end

  HF[("Hugging Face dataset<br/>zenzone3/oceanscope-outputs")]

  subgraph BE["backend/ (FastAPI on Render)"]
    API["Data API<br/>/map /profile /section /timeseries<br/>/metrics /argo /headline /events"]
    META["app/meta/<br/>guide · voice · postcards · lasso<br/>budget · limits · grounding guard"]
  end

  MAPI["Meta Model API<br/>Muse Spark · Voice Transcribe<br/>Muse Image · SAM 3.1"]

  subgraph FE["frontend/ (Next.js on Vercel)"]
    SITE["Landing · Explorer 2D/3D<br/>Validation · About · Beach"]
  end

  APP["android/glasses-guide<br/>Ocean Glasses app"]
  GLASSES["Ray-Ban / Oakley<br/>Meta glasses"]
  UP["UptimeRobot (5 min)<br/>+ GitHub Actions (10 min)"]

  CM --> BUILD
  ERA --> BUILD
  NASA --> BUILD
  STATIC --> BUILD
  IDX --> BUILD
  ARGO --> BUILD
  ARGO --> EVAL
  BUILD --> TRAIN --> EVAL --> PUB --> HF
  CM --> LIVE
  ERA --> LIVE
  ARGO --> LIVE
  LIVE --> HF
  HF -->|downloaded at startup,<br/>manifest re-checked hourly| API
  META <--> MAPI
  SITE <-->|HTTPS JSON| API
  SITE <-->|HTTPS JSON| META
  GLASSES <-->|Bluetooth| APP
  APP <-->|/glasses/ask · /chat| BE
  UP -->|GET /api/v1/health| API
```

The browser and the Android app only ever talk to our backend. Every Meta model call, every key and every credit check lives on the server, in `backend/app/meta/`.

---

## Frontend (Next.js on Vercel)

Next.js 15 (App Router, React 19, TypeScript), Tailwind, MapLibre GL, Three.js and WebGL2, Recharts, TanStack Query and GSAP. It deploys to Vercel from `main` at https://ocean-embed.vercel.app. Contributor setup is in [`frontend/README.md`](frontend/README.md).

### Pages

| Route | What you get |
|---|---|
| `/` | **Landing page.** A cinematic underwater scene in light and dark themes: a waving water ceiling, light shafts and caustics (lazy-loaded Three.js), schools of fish on curved routes, and GSAP micro-interactions. The viewpoint stays fixed (no pointer parallax), motion pauses in hidden tabs and honours reduced-motion, and there is a static fallback. "Dive" plays an arrival transition into the Explorer. |
| `/explore` | **Explorer.** The main tool: 2D map and 3D cutaway (below). |
| `/validation` | **Validation.** The headline card (it says "Not met" when the result isn't met), summary cards, per-depth chart, breakdown tables by depth, season and region, ARGO comparisons and the sealed-test card. |
| `/about` | **About.** What the project is, how it works, data credits and caveats, in the same underwater artwork and glass style. |
| `/beach` | **Beach mode** (phones, behind a flag). One "Point at the sea" button: it takes your location (and compass on iOS), finds the first ocean cell in front of you and shows its temperature from surface to depth as a big vertical strip with distance and direction. Location is sent only on tap and never stored. |

### Explorer

- **Arrival transition.** A shared state machine (`lib/arrival.ts`) runs the dive from the landing page: a surface sweep, a short water-colour hold while the map, tiles and `/meta` preload, then a GSAP camera move from open water to the basin to the region. Any input skips it. If the free backend is still waking up, it says so.
- **2D map, dot-matrix by default.** One dot per 0.25° cell, coloured by temperature, with hollow rings for cells with too little input and stippled land. A toggle switches to the classic continuous map (`?style=solid`). Fields: model prediction, GLORYS reference, or the error between them.
- **3D cutaway.** The region as a block hanging under the map (WebGL2), with the chosen depth as the top face, the water column on the walls, 2 °C contours and the 20 °C isotherm. It has a depth rail (every model level selectable by click, drag, wheel or keyboard), an exploded-layers mode, a slice tool and a guided tour. Desktop and tablet only; phones get the 2D map.
- **Details panel.** Click any point for its depth profile and time series, with charts themed for light and dark.
- **Floating glass HUD.** An "Ocean Controls" card (date, depth, field, play), a bottom pill toolbar, a coverage legend and round badge pins. All panels open and close with GSAP animations.
- **Date picker and play.** It covers the historical period and the live days, and skips the gap between them.
- **Banners.** A yellow banner when the data is synthetic (mocks or fixtures) and a live banner on live dates, which have no GLORYS reference yet.
- **Meta features** appear here when the backend switches them on: the guide drawer, mic button, postcards, lasso and event time machine. See [Meta AI features](#meta-ai-features-talk-to-the-ocean).

### How the frontend stays honest

- It never computes or invents numbers: every value comes from the API.
- It reads `GET /api/v1/meta` on load and shows a Meta feature only when the backend says it is on, so there is never a dead button.
- TypeScript API types are generated from `backend/openapi.json` (`pnpm generate-types`), and CI fails if they drift.
- With `NEXT_PUBLIC_USE_MOCKS=true`, Mock Service Worker answers every API call in the browser with labelled synthetic data, so the whole UI works offline.

---

## Backend (FastAPI on Render)

Python 3.11, FastAPI, xarray and zarr, pydantic-settings and httpx. It runs as a Docker web service on Render's free tier ([`render.yaml`](render.yaml), 512 MB RAM, Oregon).

### How it serves data

- **Startup.** The service binds its port first, then downloads the active run from the Hugging Face dataset in a worker thread. Only the files named in the remote `manifest.json` are fetched, so a deploy never times out on Render's port scan. Until the data is loaded, `/health` and `/meta` return 503.
- **Hourly reload.** Every `RELOAD_INTERVAL_MIN` (60) minutes it re-checks the manifest, so a new live day or a newly published run appears without a redeploy. `manifest.json` is always uploaded last, so the backend never sees a half-published run.
- **One store, one code path.** Routers hold no data logic. They call a single `OutputStore`, and the guide's data tools call the very same methods.
- **Staying under 512 MB.** Render restarted the service on 6 November 2026 for going over its 512 MB limit. Under load (20 parallel year-long time series) memory peaked at 496 MB, because every read decodes whole daily chunks and zarr fetched 10 chunks at a time per read across up to 40 request threads. Now a read gate lets only `DATA_READ_CONCURRENCY` (default 2) requests read the store at once, zarr's async concurrency is capped by `ZARR_ASYNC_CONCURRENCY` (default 2), the Docker image sets `MALLOC_ARENA_MAX=2` so freed buffers are handed back, and `HF_HUB_DISABLE_XET=1` makes the dataset download stream over plain HTTP. The same load now peaks at **233 MB** and finishes slightly faster.
- **Hardening.** NaN or infinite coordinates are refused (422), CORS is limited to the site and localhost without credentials, every response carries `nosniff`, `X-Frame-Options` and `Referrer-Policy` headers, and logs never hold user text, audio, images, keys or coordinates finer than 1°.

### API (base `/api/v1`)

| Endpoint | What it returns |
|---|---|
| `GET /health` | `200 ok` with the active `run_id` once data is loaded, `503` before. This is what the uptime monitors ping. |
| `GET /meta` | Run metadata, dates and date ranges (historical and live), depths, colour scales, caveats, and which Meta features are on |
| `GET /map` | 2D temperature slice for a date and depth (`pred`, `glorys` or `error`) |
| `GET /profile` | Temperature at all 15 depths for a point and date |
| `GET /section` | Vertical slice along a latitude or longitude |
| `GET /timeseries` | Temperature at a point and depth over a date range |
| `GET /region-summary` | Area-weighted statistics for a box |
| `GET /metrics` | Validation metrics for the model and baselines, by depth, season or region |
| `GET /argo`, `GET /argo/profiles` | Matchups against held-out ARGO floats, and float positions |
| `GET /headline` | The pre-registered headline test table |
| `GET /events` | Curated events for the time machine |
| `GET /nearest-ocean` | First ocean cell along a bearing from a location (beach mode and glasses) |
| `POST /chat` | The Ocean guide (Muse Spark) |
| `POST /transcribe` | Voice input (Muse Voice Transcribe) |
| `POST /postcard`, `GET /postcard/{name}` | Ocean postcards (Muse Image) and the cached image |
| `POST /segment` | The lasso (SAM 3.1, with a threshold fallback) |
| `POST /glasses/ask` | Spoken question from the glasses app, answered from the data |

The full schema is in [`backend/openapi.json`](backend/openapi.json). It is a frozen contract: CI fails if it is stale, and changes need the owner's approval and an entry in [`docs/DECISIONS.md`](docs/DECISIONS.md).

### Configuration

All settings come from environment variables (see [`backend/.env.example`](backend/.env.example) and `app/settings.py`). The main ones:

| Variable | Purpose |
|---|---|
| `HF_DATASET_REPO`, `HF_DATASET_REVISION`, `DATA_DIR` | Where to download the published run from, and where to keep it |
| `CORS_ORIGINS` | Allowed site origins |
| `RELOAD_INTERVAL_MIN` | How often to re-check the manifest (default 60) |
| `META_API_BASE_URL`, `META_API_KEY` | Meta Model API (`https://api.meta.ai/v1`); the key lives only in Render secrets |
| `META_CHAT_MODEL`, `META_CHAT_FALLBACK_MODEL`, `META_TRANSCRIBE_MODEL`, `META_IMAGE_MODEL`, `META_SEGMENT_MODEL` | Model ids (for example `muse-spark-1.3`, `muse-voice-transcribe-1.0`, `muse-image-1.0`, `sam-3.1`) |
| `META_CLIENT` | `http` for the real API, `mock` for an offline guide with no key and no network |
| `CHAT_ENABLED`, `VOICE_ENABLED`, `POSTCARD_ENABLED`, `SEGMENT_ENABLED`, `BEACH_ENABLED`, `GLASSES_ENABLED` | One flag per feature; all off by default |
| `SEGMENT_PROVIDER` | `sam` or `threshold` for the lasso |
| `META_BUDGET_USD`, `META_DAILY_BUDGET_USD`, `META_PRICE_*`, `META_BUDGET_HF_REPO`, `HF_BUDGET_TOKEN` | The credit budget and its ledger (below) |
| `*_RATE_LIMIT`, `*_IP_DAILY_CAP`, `*_DAILY_CAP` | Per-feature request limits |

---

## Meta AI features: "Talk to the ocean"

Meta's models are the interaction layer. They let people move through and ask about the data. They never produce numbers, never run the scientific model and never see a secret. All calls go through one `MetaModelClient` in `backend/app/meta/`, and AI-written text and images are always labelled as AI.

| Feature | Meta model | What it does |
|---|---|---|
| **Ocean guide** ("Ask the ocean") | Muse Spark 1.3 | A chat drawer in the Explorer. It answers questions by calling data tools and moves the map for you |
| **Voice input** | Muse Voice Transcribe 1.0 | A mic button in the guide. The browser records, converts to 16 kHz mono WAV and sends it; the transcript comes back for you to check before it goes to the guide |
| **Ocean postcards** | Muse Image 1.0 | A woodblock-style picture of the place and depth you are looking at, with the real numbers overlaid as text |
| **Lasso** | SAM 3.1 | Tap a feature on the map and get its outline plus area-weighted statistics from the grid |
| **Glasses guide** | Muse Voice Transcribe 1.0 | Spoken questions from the Android Ocean Glasses app (see [Android](#android-ocean-glasses)) |
| **Event time machine** | none | Before, during and after views of three curated cyclones: Gaja (2018), Fani (2019) and Nisarga (2020) |
| **Beach mode** | none | The temperature column in front of you, from your phone's location and compass |

### The Ocean guide in detail

- **Data tools** return numbers and call the same store as the API: `get_temperature`, `get_temperature_profile`, `get_region_summary`, `get_section`, `get_validation_metrics`, `get_dataset_info` and `list_events`.
- **UI tools** return no numbers and move the Explorer: `set_view` (date, depth, field, 2D/3D/layers, camera), `select_point`, `slice_at` and `open_event`. The backend validates every argument (dates in range and not in the sealed test period, depths snapped to model levels, points inside the region) before anything reaches the browser. At most 3 UI actions per answer.
- **Grounding guard.** After the model answers, every number with a unit is checked against the tool results by kind (°C against temperatures, m against depths, % against percentages, coordinates against coordinates, dates against sources). If any number can't be traced, the answer is replaced with "I couldn't verify that answer against the data" plus the raw results, and the map moves are kept.
- **Prompt rules** (`guide-v3`): °C and temperature only, no forecasts, always say whether a number is a model prediction, GLORYS or ARGO, never claim real-time data, treat user text and tool results as data rather than instructions, and refuse unrelated tasks.
- **Tuned for Muse Spark.** Calls use low reasoning effort and up to 2,048 output tokens, make all tool calls in one round, and have a 23 s total deadline (under the frontend's 25 s). A stalled call is retried once inside that deadline, and the model is warmed with a tiny call at startup so the first visitor after a quiet spell doesn't get an error. If Spark returns 429 or 5xx, an optional fallback model on the same API takes over.

### The other features in detail

- **Voice.** Clips must be mono 16-bit PCM WAV at 16 or 24 kHz, at most 30 s and 1 MB. Audio is forwarded and discarded in memory, never written to disk.
- **Postcards.** No free text from users: the prompt comes from a fixed template filled with region names, qualitative descriptors from `/region-summary` and the event title, and always says "no text, no numbers, no logos, no people". Images are cached by a hash of the request, so identical requests never call the model twice.
- **Lasso.** The backend renders its own grid to a PNG (never an image from the client) and sends it to SAM 3.1 through the Responses API with the tapped cell as a point prompt. Masks are decoded with Meta's `meta-sam-parser`. If SAM is off, times out (6 s) or returns nothing, the lasso picks the connected region within ±0.5 °C of the tapped cell instead. The response always says which method was used.

### Cost, abuse and privacy controls

- **Credit budget, fail closed.** The Meta credits are a fixed pot, so every real call reserves its worst-case cost against a lifetime cap and a daily cap before it runs, then settles against the reported usage. With the real client, a feature stays off until both caps, its prices and a durable ledger are set. The ledger lives in a private Hugging Face repo, because Render wipes the disk on every restart. When the budget runs out, features degrade as if the API were down.
- **Rate limits.** Per IP per minute, per IP per day and globally per day, for each feature: chat 4/min, 30, 150; voice 3/min, 15, 60; postcards 1/min, 3, 20; lasso 3/min, 20, 60; glasses 4/min, 30, 150. Spoken glasses questions also count against the voice limits.
- **Kill switch.** Set a feature's flag to `false` on Render; the feature disappears from `/meta` and the UI hides it. No Meta feature is on the critical path of the map, profile or section endpoints, so the rest of the site keeps working if Meta is down.
- **Privacy.** No user content is stored. Logs record feature, status, latency and a request id only.

---

## Android: Ocean Glasses

[`android/glasses-guide/`](android/glasses-guide/) is a Kotlin app for Ray-Ban and Oakley Meta glasses, built on Meta's Wearables Device Access Toolkit 1.0.1 (`mwdat-core`, `mwdat-speech`, `mwdat-inputs`, `mwdat-mockdevice`). It is a hackathon demo: Android only, sideloaded, tested on mock glasses.

**How it works.** You face the sea, tap the glasses and ask "How warm is it out there at 500 metres?". The app hears you through the glasses mic, takes the phone's GPS location and compass heading, and calls `POST /api/v1/glasses/ask`. The backend reads the depth from your question, finds the first 0.25° cell at least 5 km out along your heading with water that deep, and builds the answer from the prediction store with a fixed template, so every spoken number is exact. It always says the distance, direction and date, and it says when it blended two levels, when the water isn't that deep, or when the compass is unreliable. The app then asks the guide (`POST /chat`) to add a short, natural sentence around that answer, replies to "hi" or "help" with a greeting, and speaks the result through the glasses speakers with Android text-to-speech.

| Part | Real glasses | Mock glasses (no hardware) |
|---|---|---|
| Question audio | Glasses mic, transcribed on the glasses | Text typed into "What the mock glasses hear" |
| Start a question | Tap the touchpad or press the capture button | "Simulate touchpad tap" or Ask |
| Location and heading | Phone GPS and compass | A fake place: Puri beach facing 150° |
| Answer audio | Text-to-speech over Bluetooth to the glasses | Phone speaker |

Without any glasses connected, Ask uses the phone mic and Android's speech recogniser.

**UI.** A seafoam and mint underwater design with custom-drawn ocean artwork (`OceanArtwork.kt`).

**Build.** Open `android/glasses-guide/` in Android Studio (Ladybug or newer), with JDK 17, Android SDK 35 (min SDK 29) and Kotlin 2.2.20. Run on a phone in Developer Mode, keep "Use mock glasses" and "Fake location and heading" on, press **Connect glasses**, then **Simulate touchpad tap**. The backend needs `GLASSES_ENABLED=true`. For a local backend from the emulator, set the Backend URL to `http://10.0.2.2:7860/api/v1`. Full steps are in [`android/glasses-guide/README.md`](android/glasses-guide/README.md).

| File | Role |
|---|---|
| `MainActivity.kt` | Screen, controls and the ask flow |
| `GlassesLink.kt` | All toolkit code: connect, speech, touchpad input, mock device |
| `OceanApi.kt` | HTTP client for `/glasses/ask` and `/chat` (no extra dependencies) |
| `PhoneSensors.kt`, `Heading.kt` | GPS location and compass heading |
| `OceanArtwork.kt` | The underwater artwork |

---

## Live mode: a new day every day

Since 4 November 2026 a daily live run adds recent days to the site, not just the past. The **Live** workflow ([`.github/workflows/live.yml`](.github/workflows/live.yml)) runs daily at 06:00 UTC. It picks the newest day for which every near-real-time input exists, runs the published model from its "live kit" (a private Hugging Face repo holding the checkpoint, statistics and static fields), and uploads that day to the Hugging Face dataset, with `manifest.json` last. The backend picks it up on its next hourly reload. It keeps a rolling window of 365 live days.

**What it fetches.** Near-real-time sea level, SST, SSS, OSCAR currents and ERA5T (through Google's ARCO copy, which avoids the CDS queue). For ARGO-input models it reads the profiles of the model's own float window (30 days for `main3_aix30a`) from the Argo ERDDAP, with held-out floats left out. For ARMOR3D-input models it fetches the daily near-real-time ARMOR3D (`cmems_obs-mob_glo_phy_nrt_0.125deg_P1D-m`), on the same grid and depths as the reprocessed product the model trained on. For an ensemble kit it runs each member and averages them.

**How far behind.** The wind, heat flux and precipitation inputs trail by about 7 days and ARMOR3D by about 6, so live maps run **about 7 days behind today** (they were about 3–5 days behind with the older models, which didn't need ARMOR3D). Live days have no GLORYS reference yet, so the site marks them with a live banner.

**Checked before each switch.** Every model went through the same test before going live: run one past month twice, once on near-real-time inputs and once on the reprocessed inputs it trained on, and compare both with GLORYS12.

| Model (live from) | Month | Result |
|---|---|---|
| `main2_ft`, then `ens3` (4 Nov 2026) | Oct 2024 | Within 0.01 °C at 0–30 m and below 180 m, 0.03–0.15 °C better at 47–131 m with live inputs. Output moved by up to 0.41 (`main2_ft`) and 0.43 °C (`ens3`) at 92 m |
| `main3_aix30a` (10 Nov 2026) | Oct 2024, 31/31 days | Within 0.01 °C at 0–30 m and 186–1,062 m, 0.03–0.12 °C better at 47–131 m with live inputs. Output moved by at most 0.31 °C (at 92 m), even though near-real-time ARMOR3D differs from the reprocessed one by 0.38 °C RMSE at 50–300 m |

The switch to `main3_aix30a` was published with `scripts/kaggle_publish_single.py` (which scores its validation store on all held-out floats, publishes it and uploads its live kit in one notebook), then checked with a dry run of the Live workflow on 10 November (462 ARGO profiles from 166 floats, target day 3 November, nothing trimmed). The full runbook is [`docs/P9_LIVE_RUNBOOK.md`](docs/P9_LIVE_RUNBOOK.md).

---

## Keeping the backend awake

Render's free tier puts a web service to sleep after about 15 minutes without traffic, and waking it means re-downloading the data (a minute or more). Two monitors keep it awake:

- **UptimeRobot** sends `GET https://oceanscope-backend.onrender.com/api/v1/health` **every 5 minutes**. It is the main keep-alive and also alerts if the API goes down. It is configured in UptimeRobot's dashboard, not in this repository.
- **GitHub Actions** ([`.github/workflows/keep-render-awake.yml`](.github/workflows/keep-render-awake.yml)) pings the same endpoint every 10 minutes (minutes 7, 17, 27 and so on) as a backup, because GitHub's scheduled runs can be delayed. Any HTTP response counts as awake; only a connection failure or timeout fails the job. Set the repository variable `RENDER_URL` to point it elsewhere.

`/health` returns 503 for a few minutes after a restart while the data loads, so a short "down" blip after a deploy is expected.

---

## ML pipeline

**Data.** Each product is streamed from its provider, reduced straight away to the 0.25° grid and the 15 depths, and written to a processed zarr store, one year at a time. No raw archive is kept, which keeps the full 1993–2021 build at about 20 GB instead of about 1 TB. The build runs as parallel Kaggle notebooks, one part per range of years, and a separate step fits the training-only climatology and normalisation statistics across the parts.

**Inputs (143 channels by default).** For each day the model sees the last 7 days of sea-level anomaly and absolute dynamic topography, SST, SSS, surface currents, 10 m winds and wind-stress curl, net heat flux and precipitation. It also sees a 60-day sea-level history (which carries the slow ocean waves that move the thermocline), daily climate indices (Niño 3.4, the Indian Ocean Dipole and the MJO), recent ARGO profiles from floats that are *not* held out, bathymetry, the WOA23 climatological profile, and position and season.

**Target.** The GLORYS12 temperature anomaly at the 15 depths, relative to a training-period climatology and scaled per depth. The GREP ensemble mean is an auxiliary target. The loss counts the 50–300 m band twice as much.

**Model.** A plain 2D U-Net: 3×3 convolutions with GroupNorm and SiLU, skip connections and a 1×1 output to 15 depths. It has at most 10 M parameters and no attention. It trains with mixed precision, resumes from a checkpoint after a disconnect, and selects the best model on validation RMSE in °C only.

**Evaluation.** Baselines are climatology and per-pixel ridge regression. Metrics are computed per depth, season and region, with day-to-day flicker and per-year bias checks and physics sanity checks. Comparisons against ARGO use 20% of floats held out at the float level, which never feed the model as input. The test year (2021) is sealed: one command scores it once, on the owner's instruction, and the result is fixed after that.

**ARGO fine-tuning and inputs.** `finetune` continues training on ARGO profiles from non-held-out floats alongside the GLORYS target (weight λ), with options for the float window (`--argo-window`, saved in the checkpoint and used by predict and live), ARMOR3D input channels (`--armor-input`), a learned ARMOR3D blend (`--armor-blend`), warm-event weighting, Huber and vertical-gradient losses. ARMOR3D planes are cached once in a float16 memory map, and prediction preloads the test year into memory, which turned two multi-hour stalls into minutes.

**Ensembles and corrections.** `inference.ensemble` averages members' prediction stores, optionally with weights fitted on non-held-out floats (`ensemble_weights`). `inference.residual_model` is a LightGBM correction of the residuals at nearby floats, and `inference.argo_correct` is a guarded per-depth ARGO correction. None of these beat the single `main3_aix30a` on the dev half (see [How we got here](#how-we-got-here-the-argo-rounds)).

**Evaluation tools.** `argo_validation --half dev|final` scores either half of the held-out floats; `evaluation.argo_test` is the one-shot 2021 ARGO test (it refuses to run twice); `evaluation.regimes` breaks errors down by anomaly, ENSO, IOD, monsoon and cyclone days; `evaluation.matchup_audit` and `evaluation.spread_skill` check the matchups and ensemble spread; `evaluation.live_check` is the live-vs-reprocessed check; `evaluation.report` writes the evaluation report.

---

## Reproduce it

You need Python 3.11, Node 22 with pnpm, and free accounts for Copernicus Marine, NASA Earthdata, Copernicus CDS and Hugging Face. Credentials are read only from environment variables or notebook secrets.

### 1. Run the site locally on synthetic data (no accounts needed)

The repo ships contract-exact synthetic outputs in `fixtures/sample_outputs/` (labelled `synthetic: true`; the site shows a SYNTHETIC banner).

```bash
# backend
cd backend
pip install -e ".[dev]"
DATA_DIR=../fixtures/sample_outputs uvicorn app.main:app --port 7860

# frontend, in another terminal
cd frontend
pnpm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:7860/api/v1 pnpm dev   # http://localhost:3000
```

To run the frontend with no backend at all, set `NEXT_PUBLIC_USE_MOCKS=true`. To see every Meta feature without a key, either add `NEXT_PUBLIC_MOCK_META_FEATURES=true` to the mocks, or run the backend with `META_CLIENT=mock` and the `*_ENABLED` flags set to `true`.

Working on the frontend? Start with [`frontend/README.md`](frontend/README.md): Node and pnpm versions, `.env.local` setup (deployed backend, mocks or local fixtures), checks, the Meta feature flags, and the branch and PR flow.

### 2. Build the data store (Kaggle or Colab)

Set the secrets `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` and `CDSAPI_KEY`, then from `ml/`:

```bash
pip install -e .
# one part per range of years; run parts in parallel notebooks
python -u -m oceanscope_ml.data.build_store --config configs/experiment.yaml \
    --years 1993-1999 --part --out-dir <out> --scratch-dir /tmp/scratch --stream --static-file <static_fields.nc>
# fit train-only climatology and normalisation stats across all parts, then check every year
python -m oceanscope_ml.data.parts stats --parts <A.zarr> <B.zarr> ... --out stats.zarr
python -m oceanscope_ml.data.parts check --parts <A.zarr> <B.zarr> ... --stats stats.zarr
```

`build_store --dry-run --years 2019` prints each remote request and its estimated size without downloading anything. A year that is cut off is redone on the next run; finished years are skipped. Expect about 1 to 4 hours per year, mostly GLORYS and (for 2018–2021) ARMOR3D.

### 3. Baselines, training, prediction and evaluation (GPU)

On Kaggle, with the part stores attached as inputs, GPU and Internet on and an `HF_TOKEN` secret, one script runs stats, training, prediction, evaluation, baselines and (with `OCEANSCOPE_PUBLISH=1`) publishing: `%run ml/scripts/kaggle_train_and_publish.py`. The other notebook scripts in [`ml/scripts/`](ml/scripts/):

| Script | What it does |
|---|---|
| `kaggle_finetune.py` | ARGO fine-tune; `OCEANSCOPE_FT_ARGO_WINDOW`, `OCEANSCOPE_FT_ARMOR=1`, `OCEANSCOPE_FT_ARMOR_BLEND=1`, `OCEANSCOPE_FT_WARM` pick the options |
| `kaggle_armor3d_fetch.py` | Builds the training-period ARMOR3D side stores (2004–2017) |
| `kaggle_argo_check.py`, `kaggle_ensemble_correct.py`, `kaggle_run_e.py`, `kaggle_run_f.py` | Score members, ensembles and corrections on the dev half |
| `kaggle_final_half.py`, `kaggle_argo_test.py` | The one-time final-half score and the one-shot 2021 ARGO test (`OCEANSCOPE_SINGLE` for one model) |
| `kaggle_publish_single.py`, `kaggle_publish_ensemble.py` | Publish a model or an ensemble as the site's run and upload its live kit |
| `kaggle_live_kit.py`, `kaggle_armor_nrt_check.py` | Build a live kit and run the live-vs-reprocessed check; inspect the near-real-time ARMOR3D |

The core steps are:

```bash
STORES="<A.zarr> <B.zarr> <C.zarr> <D.zarr>"
python -m oceanscope_ml.models.baselines --config configs/experiment.yaml --store $STORES --stats stats.zarr
python -m oceanscope_ml.train --config configs/experiment.yaml --store $STORES --stats stats.zarr --run-id <run_id>
python -m oceanscope_ml.inference.predict --config configs/experiment.yaml --store $STORES --stats stats.zarr \
    --checkpoint checkpoints/<run_id>/best.pt
python -m oceanscope_ml.evaluation.evaluate --config configs/experiment.yaml \
    --predictions outputs/predictions/predictions_<run_id>.zarr --store $STORES --stats stats.zarr
python -m oceanscope_ml.data.download --config configs/experiment.yaml --products argo \
    --start 2018-01-01 --end 2020-12-31 --out-dir raw
python -m oceanscope_ml.evaluation.argo_validation --config configs/experiment.yaml \
    --predictions outputs/predictions/predictions_<run_id>.zarr --store $STORES --stats stats.zarr \
    --argo raw/argo/*.nc --holdout argo_eval_floats.json
```

Training resumes automatically if you run the same command with the same `--run-id`. All settings (region, depths, dates, splits, hyperparameters) live in [`ml/configs/experiment.yaml`](ml/configs/experiment.yaml).

The final test on 2021 runs **once**, only on the owner's go: `python -m oceanscope_ml.evaluation.final_test ... --owner-go`.

### 4. Publish and deploy

```bash
python -m oceanscope_ml.publish --config configs/experiment.yaml --dry-run   # lists what would upload
HF_TOKEN=... python -m oceanscope_ml.publish --config configs/experiment.yaml
```

`manifest.json` is uploaded last, so the backend never sees a half-published run. The backend is a Docker web service on Render ([`render.yaml`](render.yaml)) that reads `HF_DATASET_REPO` and `HF_DATASET_REVISION`. The frontend deploys to Vercel with `NEXT_PUBLIC_API_BASE_URL` set to the backend. For the Meta features, add the `META_*` variables and feature flags as Render secrets.

### Checks

```bash
cd ml       && ruff check . && ruff format --check . && mypy src && pytest -q
cd backend  && ruff check . && ruff format --check . && mypy app && pytest -q && python -m app.export_openapi --check
cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the checks for whichever areas a change touches, plus the frontend's Playwright end-to-end tests. Tests never read the test year and never call the real Meta API.

---

## Repository layout

```
ml/                    Python package oceanscope_ml: data build, models, training, evaluation, inference, live, publish
backend/               FastAPI app (data API + app/meta/ for every Meta model call), Dockerfile, openapi.json, tests
frontend/              Next.js app (landing, Explorer 2D/3D, validation, about, beach), unit and e2e tests
android/glasses-guide/ Ocean Glasses Android app for Meta glasses (Kotlin, Wearables Device Access Toolkit)
fixtures/              synthetic, contract-exact sample outputs for development (< 5 MB)
docs/                  PRDs, DECISIONS.md, PHASES.md, OPEN_QUESTIONS.md, PROGRESS.md, runbooks, reviews
.github/workflows/     ci.yml, live.yml (daily live day), keep-render-awake.yml (backend ping)
render.yaml            Render blueprint for the backend
```

Rules for contributors and coding agents are in [`AGENTS.md`](AGENTS.md). Design decisions, with dates and who approved them, are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Limitations

- **Trained to imitate a reanalysis.** The base target is GLORYS12, which itself assimilates altimetry and ARGO. Skill against GLORYS is not skill against the real ocean. The ARGO comparison is the closer check, but ARGO is assimilated into GLORYS12 and ARMOR3D too, so it is not fully independent.
- **The pre-registered headline was not met.** On 2021 the model beats GLORYS12 and ties GREP at held-out floats in the Bay of Bengal at 50–300 m, but ARMOR3D is more accurate. On the validation years it only ties GLORYS12.
- **It leans on ARMOR3D.** ARMOR3D is an input and is still more accurate on its own. Without it, our best model (0.840 on the dev half) was behind GLORYS12 (0.756). The live site depends on Copernicus publishing near-real-time ARMOR3D.
- **Small samples.** The Bay of Bengal headline rests on 6 to 15 floats per period (7 in 2021), so the confidence intervals are wide.
- **Training years.** 2005 and 2006 were not built in time and are not in the training set.
- **Before 2004**, there are few ARGO floats, so the ARGO input channels are empty by design for those years. ARMOR3D side stores cover 2004–2017, one day in three.
- **Not tuned exhaustively.** There was no hyperparameter search, and the per-input-group ablation study planned in the PRD hasn't been run beyond the ARMOR3D on/off comparison, so we make no claim about which other inputs matter most.
- **The test year is spent.** The gridded 2021 test was used on `main2_ft` and the ARGO 2021 test on `main3_aix30a`. Nothing else can be scored on 2021.
- **Resolution.** 0.25° and daily. Fronts, eddies and coastal features smaller than about 25 km are smoothed out, and cells less than half ocean are masked.
- **Depth.** 15 levels down to about 1,060 m. Values between levels in the profile chart (and in glasses answers) are interpolated linearly.
- **Region.** The North Indian Ocean box only (40–100°E, 0–30°N). The model has not been tested elsewhere.
- **Not a forecast.** Each map is a reconstruction of a past day from that day's and earlier observations.
- **Live days are unchecked.** They run about 7 days behind real time and have no GLORYS reference yet. Near-real-time inputs, ARMOR3D especially, differ from the reprocessed ones the model trained on; the one-month check above bounds the effect but can't rule it out for every day.
- **AI features are an interface, not a source.** The guide's wording, postcards and lasso shapes come from Meta's models; the numbers always come from the model outputs. The grounding guard still has known gaps (for example, a number spelled out in words).
- **Glasses app is a demo.** It is tested on Meta's mock glasses only, not on real hardware, and it uses dev/beta toolkit APIs, so it can't be published as it is. The answer is for a cell about 27 km wide, and the phone's compass sets the direction.
- **Hosting.** The backend runs on Render's free tier (512 MB). After a restart it re-downloads the data, which takes a few minutes; the uptime monitors keep restarts rare but can't prevent them.

---

## Data credits

OceanScope AI is built entirely on open data. Thank you to everyone who collects, processes and shares it.

- **E.U. Copernicus Marine Service:** GLORYS12 reanalysis (`GLOBAL_MULTIYEAR_PHY_001_030`), DUACS sea level (`SEALEVEL_GLO_PHY_L4_MY_008_047`), OSTIA SST (`SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`), multi-observation SSS (`MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013`), GREP ensemble (`GLOBAL_MULTIYEAR_PHY_ENS_001_031`) and ARMOR3D (`MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`). *"This study has been conducted using E.U. Copernicus Marine Service Information."*
- **ERA5**, Copernicus Climate Change Service (C3S) at ECMWF, accessed through the Copernicus Climate Data Store and Google's public ARCO-ERA5 copy. *"Contains modified Copernicus Climate Change Service information."*
- **OSCAR** ocean surface currents, Earth & Space Research, distributed by NASA PO.DAAC.
- **Argo** float data. *"These data were collected and made freely available by the International Argo Program and the national programs that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the Global Ocean Observing System."* Argo (2000), Argo float data and metadata from Global Data Assembly Centre (Argo GDAC), SEANOE, https://doi.org/10.17882/42182. Accessed through ERDDAP.
- **GEBCO** Compilation Group, GEBCO 2023 Grid.
- **World Ocean Atlas 2023**, NOAA National Centers for Environmental Information.
- **Niño 3.4 index**, NOAA Climate Prediction Center. **MJO RMM index** (Wheeler and Hendon, 2004), Australian Bureau of Meteorology. The Indian Ocean Dipole index is computed by us from OSTIA.
- **Basemap:** OpenFreeMap, with map data © OpenStreetMap contributors.

## Built by

Aryan, with Claude (planning, review and code), Gemini and Codex (implementation), on Meta's models for the AI features.
