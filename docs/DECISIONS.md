# Decisions log

Format: `## YYYY-MM-DD — title` then decision, reason, who approved.

## 2026-10-22 — Stack and hosting
- Backend: FastAPI (Python 3.11) on Hugging Face Spaces (Docker). Frontend: Next.js on Vercel. Approved by Aryan (stack); hosting split recommended by Claude because Vercel's Python functions don't suit a GB-scale xarray/zarr data store.
- Training on Google Colab / Kaggle.
- Model predicts anomalies relative to a training-period GLORYS climatology; small U-Net; climatology and ridge/LightGBM baselines.
- Streamlit dashboard from the handoff is superseded.

## 2026-10-22 — Richer inputs, ablations, ARGO fine-tuning
- Core inputs expanded: SST (OSTIA), SSS, ADT, wind-stress curl, ERA5 net heat flux and precipitation, GEBCO bathymetry, WOA23 climatology profile, and a 7-day input history. Reason: SST/SSS are the most informative inputs in published work, and SSS matters especially for the Bay of Bengal barrier layer.
- Required ablation study per input group; ARMOR3D added as an external baseline at ARGO points.
- ARGO fine-tuning is the first stretch goal, with a 20% float-level hold-out for evaluation.
- Approved by Aryan.

## 2026-10-22 — Long sea-level history, climate indices, flicker and trend checks
- Added a 60-day sea-level (SLA, ADT) history at 5-day steps, so the model sees the slow ocean waves that move the thermocline.
- Added daily climate-mode indices as inputs: Niño 3.4 (ENSO), DMI (Indian Ocean Dipole, computed from OSTIA) and MJO RMM1/2. Niño 3.4 carries information from outside the model's box.
- Rejected: CO₂ / "global warming" scalars and a year/trend channel (no new information; risks extrapolating a trend into the test year). GloFAS river discharge is optional, kept only if its ablation helps.
- Required evaluation: day-to-day flicker metric and per-year bias (warming check) (PRD_AI_ML §5.8). A temporal model is triggered only if flicker is flagged; feeding back the previous day's prediction is not used in v1.
- Approved by Aryan (long history + flicker metric); climate indices proposed by Claude in answer to Aryan's request for extra environmental variables.

## 2026-10-22 — Round 4: win condition, ARGO input, GREP, 1993 start, live mode
- **Primary success criterion (pre-registered):** in the Bay of Bengal at 50–300 m, lower RMSE than GLORYS12, the GREP mean and ARMOR3D against held-out ARGO floats, with float-bootstrap CIs (PRD_AI_ML §1.1). Fixed before training; changes only with a new dated entry made before the affected run.
- Recent ARGO profiles (non-held-out floats, last 10 days) become input channels; ARGO fine-tuning moves from stretch to core (§5.9).
- GREP ensemble mean added as auxiliary training target and as a competitor. Catalogue says 1993 to 15 Dec 2020, docs say "regularly updated"; confirm at M1. GLORYS12 stays the primary target.
- Training period starts 1993 (splits 1993–2017 / 2018–2020 / 2021 sealed, [CONFIRM] at M1).
- Depth-weighted loss (2× at 50–300 m) and physics sanity checks in evaluation.
- Live mode added as phase P9 (daily scheduled run on GitHub Actions, ~3 days behind real time, no online learning). Requires a later contract note for the `live` split label.
- Stretch order: ensemble + uncertainty → hyper-local 1/12° Bay of Bengal model → temporal model.
- Approved by Aryan ("go" on items 1–8; 9–11 as stretch).

## 2026-10-22 — Contract additions for Round 4 (PRD_AI_ML v5, PRD_BACKEND v2, PRD_FRONTEND v2)
- ARGO metrics and matches gain GREP and ARMOR3D columns; new `<run_id>_headline.csv` with optional `files.headline` in the manifest (schema_version stays 1, older manifests remain valid).
- `split_of_each_date` may contain `live`; live dates have all-NaN `temp_glorys`.
- Backend: `/headline`, `latest_live_date` in `/meta`, `LIVE_CAVEAT` and `HEADLINE_CRITERION` strings, `reference_unavailable` error, periodic manifest reload. Frontend: headline card (shows "Not met" honestly), Live badge, updated data sources.
- Fixtures and schema copies to be updated in P1-B0.
- Approved by Aryan as part of "go" on Round 4 and live mode.

## 2026-10-24 — Phase 2 hosting (Render), RAM limits, and tokenless basemap
- **Backend hosting**: Hugging Face Spaces now requires a paid Pro subscription to run Docker or Python spaces. Per PRD_BACKEND §8, backend is deployed to Render (Docker Web Service, free tier).
- **Resource limit note for P6**: Render's free tier provides 512 MB of RAM. For the P2 synthetic dataset, memory consumption is ~150 MB and fits comfortably. For the real multi-year full-resolution predictions in P6, memory pressure may occur; a bigger plan (e.g. Starter 2 GB RAM) or another host may be required when real outputs are published.
- **CORS configuration**: `CORS_ORIGINS` in `render.yaml` set to `https://ocean-embed-xi.vercel.app,http://localhost:3000` with regex matching for Vercel preview domains in backend middleware.
- **Basemap**: Replaced CARTO basemap (which burns "API KEY REQUIRED" watermarks on public non-localhost origins) with OpenFreeMap (tokenless, open-source Positron and Dark Matter vector styles with no API key or watermarks).
- **Synthetic banner**: Embedded `<Banner type="synthetic" />` directly inside the sticky `<Header />` component to ensure it cannot be scrolled out of view and remains visible on every page.
- Approved by Aryan.

## 2026-10-23 — Full-period data: stream and reduce, no raw archive (PRD_AI_ML v6)
- The one-month sample showed raw downloads scale to about 1 TB for 1993–2020 with the current downloader (OSCAR, CCMP and WOA23 only come as global files; GLORYS at 1/12° is 640 MB per month). Even with cropping it would be 150–200 GB.
- Decision: the full period is built by a new `build_store` stage that streams each product (Copernicus ARCO via `open_dataset`, ERA5 daily statistics, server-side or crop-then-delete for OSCAR/CCMP, OPeNDAP for WOA23), reduces it to the 0.25° grid and 15 depths at once, and writes only the processed store. It runs one year at a time on Colab/Kaggle and is resumable. Targets: store ≤ 20 GB (float16 ≤ 12 GB), scratch ≤ 20 GB.
- `download.py` stays for the one-month sample and live mode only.
- Dataset IDs corrected to the live catalogue (checked by Aryan with `copernicusmarine describe`): DUACS `…allsat-l4-duacs-0.125deg_P1D`, OSTIA product `…_010_011`, SSS `cmems_obs-mob_glo_phy-sss_my_multi_P1D`, GREP `cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m` (variable `thetao_mean`), ARMOR3D `MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012` / `cmems_obs-mob_glo_phy_my_0.125deg_P1D-m` (now daily).
- No contract change: the processed store schema (§6.1) is unchanged.
- Approved by Aryan ("let's do that then").

## 2026-10-24 — 3D cutaway explorer (PRD_FRONTEND v3, PRD_BACKEND v3)
- The Explorer gets a *3D* view: the region drawn as a rotatable block hanging below the basemap, the top face at the selected depth, side walls showing the column below, contours every 2 °C and the 20 °C isotherm, and an animated "dive" when the depth changes. An "exploded layers" mode and an optional slice tool come after. The 2D map stays as the default on phones and the fallback everywhere.
- Built as one MapLibre custom layer in raw WebGL2, lazily loaded. No three.js, deck.gl or other 3D library, because the `/` bundle is already near its 350 KB budget.
- **Contract addition (additive, nothing existing changes):** `GET /section` returns one row or column of the prediction store across all 15 depths (PRD_BACKEND §4.11). Reason: building the walls from 15 `/map` calls would move about 15× the data from a free-tier backend. `manifest.json` and the prediction store are unchanged.
- Review of the current map code found two breaches of PRD_FRONTEND §0 (hardcoded region bounds and a hand-drawn land mask in `lib/oceanMask.ts`); both are to be fixed first (PHASES P2b-F0).
- Proposed by Claude after Aryan liked the concept mockup; Aryan asked for it to be written into the PRDs, rules and phases ("how do you reckon we can implement it… put it into the respective PRDs"), 2026-10-24.

## 2026-10-24 — Meta features become core: "Talk to the ocean" (PRD_BACKEND v4, PRD_FRONTEND v4, PHASES P7)
- P7 moves from optional to core, because the hackathon is built around Meta's models. Included, in this order: **Ocean guide** (Muse Spark 1.3 with data tools plus UI tools that move the map and the 3D view), **voice dive** (Muse Voice Transcribe), **event time machine** (curated, sourced events in 2018–2020), **ocean postcards** (Muse Image, fixed prompt template, numbers only as HTML captions), **lasso** (SAM 3.1 through the Meta Model API, with a deterministic ±0.5 °C connected-region fallback), and **beach mode** on phones (location and compass → nearest ocean cell → profile).
- **Contract additions (additive):** `/chat` gains `ui_actions` and `grounded`; `/meta` gains `features`; new `/transcribe`, `/events`, `/postcard` (+ `/postcard/<hash>.webp`), `/segment`, `/nearest-ocean`; new caveat strings `AI_CAVEAT`, `POSTCARD_CAVEAT`, `SEGMENT_CAVEAT`. Nothing existing changes; the ML output contract is untouched.
- Guardrails: every Meta call goes through the backend (key only in Render secrets); Meta models never produce numbers (grounding guard; images carry no numbers); UI actions are validated by the backend; no user content stored; per-feature flags, rate limits and daily caps; every feature degrades cleanly and the site works with all flags off.
- Scope guardrail: P7 never delays the ML critical path; if time runs short, cut beach mode, then SAM (keep the threshold lasso), then postcards, then the time machine.
- Muse Code is the recommended implementer for the guide (P7-B1, P7-F1); Muse Image is used to make the landing-page art layers (PLAN_LANDING §4).
- Rejected: Ray-Ban glasses integration (no device to test on; a mocked demo would cost trust); wearables, Presence Platform and Horizon tools (no fit). Muse Glimmer only as a chat fallback model.
- Approved by Aryan ("let's include the spark as navigator idea, voice dive, event time machine, ocean postcards, SAM lasso"; SAM expected on the Meta API; "add these … changes to our plans"), 2026-10-24.

## 2026-10-24 — 15 model depths (P1-M3)
- `depths_m` = 0.494, 5.078, 9.573, 18.496, 29.445, 47.374, 65.807, 92.326, 130.666, 186.126, 222.475, 266.040, 380.213, 643.567, 1062.440 m: native GLORYS12 levels, taken from the data inspection of 2026-10-24 (`outputs/reports/data_inspection.md`).
- Chosen to put 7 levels in 50–300 m, where the primary criterion is measured (PRD_AI_ML §1.1) and the thermocline sits; the inspect script's proposal had 7 levels in the top 20 m and only 4 in 50–300 m.
- Changing this later means rebuilding the processed store and the prediction store.
- Proposed by Claude; approved by Aryan on the decision card ("Thermocline-focused set"), 2026-10-24.

## 2026-10-24 — Winds from ERA5 instead of CCMP (PRD_AI_ML §3, §4)
- `u_wind`, `v_wind` and `wind_stress_curl` now come from ERA5 10 m winds (`u10`, `v10`, Copernicus CDS), requested together with the other ERA5 variables (area subset; daily statistics for the full build). The wind-stress bulk formula and curl method are unchanged.
- Reason: CCMP V3.0 from Remote Sensing Systems is one 66 MB global file per day and the server ran at ~30 KB/s from Aryan's connection: over two days for the 91-day sample and roughly 600+ GB for 1993–2021. ERA5 is subset server-side and uses an account we already have.
- Trade-off: ERA5 winds are model reanalysis rather than satellite-blended (CCMP); acceptable for a 0.25° daily input. CCMP stays in `download.py` as an optional product (only fetched when named), not part of the default download.
- Proposed by Claude; approved by Aryan on the decision card ("ERA5 10 m winds"), 2026-10-24.

## 2026-10-25 — Remaining M1 configuration (P1-M3)
- **Region:** 40–100°E, 0–30°N (unchanged).
- **Period:** 1993-01-01 to 2021-12-31; start stays 1993 (not 2004). ARGO input channels are empty before 2004 by design (`argo_input.min_year`).
- **Splits:** train 1993-03-02 → 2017-12-01, val 2018-01-01 → 2020-12-01, test 2021-01-01 → 2021-12-31 (sealed), 30-day gaps (unchanged).
- **GREP end date:** taken as 2020-12-15 (catalogue) unless `copernicusmarine describe` shows later data. The 2021 headline then compares the model with GLORYS12 and ARMOR3D only; GREP is reported as "n/a" for the test year (PRD_AI_ML §1.1).
- **GREP top level:** GREP has no level at 0.494 m (its first is ~0.506 m). Its first level is used for the 0.494 m level (≈1 cm offset). GREP is also requested down to the first level below 1062.44 m so the deepest level can be interpolated (REVIEW_P3 A6).
- **ARGO source:** Argo GDAC via ERDDAP (already used for the January sample).
- `publish.hf_dataset_repo` = `zenzone3/oceanscope-outputs` (in use since P2).
- Still `[CONFIRM]`, to be settled by validation experiments in P4: `history_days`, `long_history.days`, `aux_weight`, `depth_weights.in_band`, `finetune_argo.lambda`.
- Proposed by Claude (REVIEW_P3 A8); approved by Aryan ("approved"), 2026-10-25.

## 2026-10-27 — GREP end date and SSS version (from `copernicusmarine describe`)
- **GREP end date:** `cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m` covers 1993-01-01 to 2025-12-31, so the 2020-12-15 cutoff no longer applies. Under the 2026-10-25 rule ("unless `copernicusmarine describe` shows later data"), GREP is built for 2021 and the 2021 headline compares the model with GLORYS12, GREP and ARMOR3D (PRD_AI_ML §1.1).
- **SSS version:** `cmems_obs-mob_glo_phy-sss_my_multi_P1D` has only version 202311 (released 2023-11-30, retired 2026-11-24 13:00 UTC). The version is pinned in `PRODUCT_SPECS` so every year uses 202311, and all years must be built before 24 Nov.
- Recorded by Claude from the `describe` output Aryan ran on 2026-10-27.

## 2026-10-27 — Smoke run before the full build, and a multi-part store
- Before any long build, one short Kaggle run (`ml/notebooks/smoke_cell.py`) builds the first 14 days of 1993, 2000, 2007, 2014, 2020 and 2021, each as its own part, then runs the stats step, the OK/FAIL check and the baselines. The full build starts only after it is clean. Its metrics are not results (mean-only climatology on 14 days per year).
- The full build runs as part stores built with `build_store --part`, which skips the per-store finalize. `python -m oceanscope_ml.data.parts stats` then fits the train-only climatology and norm stats across all parts into a separate stats store, and baselines read `--store <parts…> --stats <stats>`. Parts are not merged.
- Asked for by Aryan ("test whatever we have to in the smaller builds… before starting the large ones"), 2026-10-27.

## 2026-10-27 — Training plan decisions (OceanScope training plan doc)
1. **Build:** the full 1993–2021 store is built as 4 parallel Kaggle parts (A 1993–1999, B 2000–2006, C 2007–2013, D 2014–2021, reusing the built 2017–2018), plus one separate stats step across the parts. Parts are not merged.
2. **Training code** is written and tested on the existing 2017–2018 store while the full build runs.
3. **Ablations** all use the same reduced budget (about 25 epochs) so they compare fairly; only the final model gets the full budget.
4. **GPU:** Colab (Google AI Pro, 200 compute units per month) is the main GPU; Kaggle (about 30 GPU h) is the backup and runs the builds.
- Proposed by Claude in the training plan; approved by Aryan ("I approve of the 4 decisions in the training plan"), 2026-10-27.

## 2026-10-28 — ARMOR3D only for validation and test years
- Part C's first round took 3.7 h per year, and ARMOR3D alone was 170 min of 2007 and 80 min of 2008 (GLORYS about 51 min, GREP about 25 min, ERA5 about 14 min, the rest under 5 min).
- ARMOR3D is only an external competitor at ARGO points (PRD_AI_ML §3, PHASES P3-4: "for the validation and test periods"), never an input or a target. `build_store` now fetches it only for years with a val or test date (2018–2021) and writes it as NaN for train-only years, listed in the store attr `armor3d_skipped_years`. The check reports those rows as "skipped (train-only year)". Train years already built with ARMOR3D keep it.
- Proposed by Claude from the part C log, 2026-10-28.

## 2026-11-02 — Temporal checks: train-year sample and Ridge (PRD_AI_ML §5.8)
- The §5.8.2 train-year bias rows come from a day-sampled train store (`predict --split train`, every 30th day, `TRAIN_SAMPLE_STRIDE_DAYS`). It is ML-internal: it is not published, not in the manifest and not in the §6 contract. The store is written as `predictions_<run_id>_train.zarr`, so it cannot overwrite the validation store.
- Flicker (§5.8.1) is not computed on the day-sampled store: its days are not consecutive. Physics (§5.8.3) uses only the full validation store.
- The Ridge baseline enters the flicker and bias checks, rebuilt day by day from `baseline_ridge.npz` with the same features as `baselines.py`. Its validity mask is the U-Net's mask intersected with its finite predictions. With zero weights and zero intercept it reproduces climatology exactly (test).
- `experiment.yaml` is unchanged, so the config hash of stores and checkpoints is unchanged.
- Proposed by Claude, 2026-11-02. Owner review at the P5-4 review.

## 2026-11-02 — Fine-tune λ and the published run
- The ARGO fine-tune sweep picked λ = 0.3 (`finetune_argo.lambda`, [CONFIRM] in PRD_AI_ML §5.9). `experiment.yaml` still holds the default; the value used for `main2_ft` is 0.3 and is recorded here until the owner confirms it.
- `main2_ft` is the fine-tuned run, published to zenzone3/oceanscope-outputs. The served run is `active_run_id` in manifest.json. Owner to confirm it is `main2_ft`.

## 2026-11-02 — Backend startup on Render (P6)
- The first data load (which downloads the dataset from HF) runs in a worker thread from the lifespan, so uvicorn binds its port at once. Render's deploy of 479dbf1 failed with "Port scan timeout reached, no open ports detected" while `snapshot_download` was 28% through 9,462 files. Until the load finishes, `/health` and `/meta` return 503.
- The download is limited to the files the remote `manifest.json` names (`allow_patterns`), so only the active run's prediction store is fetched. If the remote manifest cannot be read, everything is downloaded as before.
- The module-level `loader` no longer loads at import (`DataLoader(autoload=False)`); `DataLoader(data_dir=...)` still loads at construction, as the tests expect.
- Proposed by Claude, 2026-11-02.

## 2026-11-03 — Synthetic caveat hidden on real data; Meta model ids and limits
- `/meta.caveats.synthetic` is `null` when the manifest has `synthetic: false`, and `SYNTHETIC_CAVEAT` otherwise. Reason: PRD_BACKEND §7 says the frontend shows caveats verbatim, so the synthetic-demo text must not reach real data. This changes a frozen contract field (`string` becomes `string | null`), so `openapi.json` and the frontend types are regenerated in the same change.
- Model ids, env vars only: guide `muse-spark-1.3`; voice `muse-voice-transcribe-1.0`; postcards `muse-image-1.0`. The contributor id `muse-spark-1.3-contributor` is not used, because Meta trains on inputs sent to it.
- Limits, per team and per model: muse-spark-1.3 3,000 requests/min and 4,000,000 tokens/min; muse-image-1.0 150 requests/min; muse-voice-transcribe-1.0 no request or token limit, but 256 concurrent streams and 16,000 streams/hour. A 429 from Meta is not waited on for interactive requests: chat falls back to `META_CHAT_FALLBACK_MODEL` if set, otherwise the caller gets 502 at once.
- The API key stays in Render secrets: not committed, not in fixtures, not logged. The real client stays behind the existing flags (`META_CLIENT=http`, `CHAT_ENABLED`, `VOICE_ENABLED`), off until the owner confirms the key is in Render and approves a first live call.
- Approved by Aryan, 2026-11-03.

## 2026-11-03 — Time-machine events: three, ranked on validation data
- `app/data/events.json` ships three events: Cyclone Gaja (2018), Cyclone Fani (2019), Cyclone Nisarga (2020). Titli, Amphan and the 2019 positive Indian Ocean Dipole were dropped.
- Ranking: a validation-only check, `event_skill_check.py`, run on Kaggle against main2_ft, reading validation dates only. For each event, the area-weighted RMS of pred minus GLORYS in the event box and days, divided by the same measure on a sample of validation days. Gaja 0.77 (15 days), Nisarga 0.85 (12), Fani 0.93 (21). Titli 1.12, Amphan 1.16, the IOD 1.17 (275-day window, not comparable). Lower is better; a ratio below 1 means a better match than a typical validation day.
- Caveats the time machine must show: the match is with GLORYS, not with observations; the events were chosen by model fit on validation years, which the test year does not affect; the windows are 12 to 21 days, so the scores are noisy.
- The test split is never read: all three events lie in 2018 to 2020, and the startup validator checks the file against the real grid.
- Approved by Aryan, 2026-11-03.

## 2026-11-04 — Meta credit budget, tighter limits, and P7-R guide hardening
- **Credit budget (fail closed).** Every real Meta call goes through `app/meta/budget.py`: before the call the worst-case cost is reserved against a lifetime cap (`META_BUDGET_USD`) and a UTC-day cap (`META_DAILY_BUDGET_USD`); after it the reservation is settled with the reported `usage`, or kept in full when the cost is unknown or the call failed. With `META_CLIENT=http`, a feature stays off (and `/meta.features` says so) until both caps, that feature's prices and a durable ledger are set. When the budget is spent, calls stop and features degrade as if the API were down (guide 502, lasso threshold method). Reason: the Meta credits are a fixed $150 with no top-up.
- **Ledger.** On Render the disk is wiped on every deploy and restart, so the ledger lives in a private HF dataset repo (`META_BUDGET_HF_REPO`, token `HF_BUDGET_TOKEN`), synced every $0.25 or 5 min of spend and at shutdown; a failed save pauses all Meta calls. An unreadable ledger disables calls instead of counting as zero.
- **Prices** come only from env vars read off the Meta dashboard; none are in code.
- **Request limits** tightened and a per-IP daily cap added: chat 4/min, 30 per IP per day, 150 per day; voice 3/min, 15, 60; postcards 1/min, 3, 20; lasso 3/min, 20, 60.
- **Sealed test period in the guide's data tools.** `get_temperature`, `get_temperature_profile`, `get_region_summary` and `get_section` refuse dates in the sealed test period while `test_evaluated` is false (`sealed_test_period`), as the UI tools already did. Reason: P7-R found the guide could quote test-period error against GLORYS.
- **Contract change:** `ChatContext.date` now has the pattern `YYYY-MM-DD` (`openapi.json` regenerated; the frontend types do not change). Context fields are also validated against the store (date available and not sealed, depth snapped to `depths_m`, point in the region); invalid fields are dropped before the prompt and the guard.
- **Grounding guard v2** (PRD_BACKEND §6.4 tightened): numbers match by kind (°C to temperature fields, m to depth fields, % to percentages, °N/°E to coordinates), an explicit sign must match, bare "°" and "degrees" count as °C, °F, kelvin and units the data does not have (PSU, feet, miles, knots) never ground, dates must appear in a source, and an answer that repeats the system prompt falls back. `get_section` returns `isotherm_c` so "20 °C" is grounded.
- **Prompt `guide-v2`:** °C only; temperature only (no salinity, currents, forecasts); coordinates only for places known to be in the region, none for fictional ones; only dates in range, none in the sealed test period.
- Known gaps left (P7-R harness strict xfails): a made-up place given real in-region coordinates, a number spelled in words, and a real temperature quoted at the wrong depth.
- Approved by Aryan ("do the recommended"), 2026-11-04.

## 2026-11-04 — Guide tuned on the first real Muse Spark run
- Base URL `https://api.meta.ai/v1` (Meta API reference); the dashboard page URL returns HTTP 307.
- Muse Spark always reasons and its reasoning tokens count as output (about 600 per turn at the default effort; 7–8 s per call). Chat calls now send `reasoning_effort` (`META_CHAT_REASONING_EFFORT`, default `low`) and `max_completion_tokens` (the API's name; `max_tokens` is a deprecated alias), raised from 1,024 to 2,048 because several final answers came back empty. The guide's total timeout is 23 s, under the frontend's 25 s.
- Prompt `guide-v3`: the dataset facts (model depths, date range, region bounds) are in the system prompt, so the model needs no `get_dataset_info` round, and it is asked to make all tool calls in one round. The same facts, and dates the user wrote (never numbers), count as guard sources.
- P7-R harness: an upstream error now fails the quality group, and a real run stops after 2 in a row.
- Proposed by Claude from the 2026-11-04 real run ($0.41 of a $2 local cap).

## 2026-11-04 — Voice, postcard and SAM calls checked against the Meta docs
- **Voice (contract change, `/transcribe`).** Muse Voice Transcribe's file endpoint is `POST {META_API_BASE_URL}/asr/transcribe` with two multipart parts, `request` (JSON: `model`, `mode: PUSH_TO_TALK`, `audioEncoding: WAV`, optional `languageBias` such as `["English"]`) and `audio`, and answers `{sessionId, transcript, audioDurationMs, turns}` (dev.meta.ai/docs/speech-to-text). It takes only mono 16-bit PCM WAV at 16 or 24 kHz. So the browser now decodes its MediaRecorder clip and uploads 16 kHz mono WAV (`frontend/src/lib/wav.ts`; 30 s is 960 kB, inside the 1 MB cap), and `/transcribe` accepts only that format (`audio/wav`, `audio/x-wav`, `audio/wave`, header checked; anything else is 415 `unsupported_audio`). WebM, MP4 and Ogg are no longer accepted. No new dependency on either side.
- **Postcards.** `/images/generations` with `model`, `prompt`, `n`, `response_format: b64_json`, `output_format: webp` matches Meta's `CreateImageRequest`; the reply is `data[0].b64_json` (WebP is the default). Requests now add `reasoning_strength: low` (default `high`), because Muse Image reasons before it renders. The backend timeout drops from 45 s to 40 s, under the frontend's 45 s abort, like the guide (23 s under 25 s).
- **SAM.** SAM 3.1 has no segmentation endpoint: it runs through `POST /responses` (streamed), takes a point as an `<|point;…|>` token in the text, and returns masks as base85 tokens in the output text, which Meta decodes with its own parser library. Our `segment()` call is wrong and stays unused: run the lasso with `SEGMENT_PROVIDER=threshold` (no Meta call, no cost). Rewriting it is a separate task.
- Prices from dev.meta.ai/docs/pricing-rate-limits (2026-11-04): Muse Voice Transcribe $0.18 per hour of audio (`META_PRICE_PER_AUDIO_MIN=0.003`), Muse Image $0.01 per image (`META_PRICE_PER_IMAGE=0.01`), SAM 3.1 $2.50 per 1,000 images.
- Proposed by Claude; waits on Aryan's approval.

## 2026-11-04 — Lasso calls SAM 3.1 through the Responses API
- SAM 3.1 has no segmentation route. `HttpMetaModelClient.segment()` now posts to `POST {META_API_BASE_URL}/responses` (`model`, `stream: true`, one user message with an `input_text` object prompt and an `input_image` data URL of our rendered PNG). The tapped cell's centre pixel is sent as one positive point for object 1 on frame 0, `<0f>1<|point;x=…;y=…;w=…;h=…|>` (dev.meta.ai/docs/sam/segmenting).
- The reply's output text holds box and mask records, the masks as base85 rasters local to each box. They are parsed and decoded with Meta's `meta-sam-parser` (new backend dependency: pure Python, no dependencies of its own, SAM License; it is the decoder Meta's docs and cookbook use, and the codec is not documented well enough to reimplement safely). Each raster is scaled to its box and placed there, and all returned objects count, as before (OPEN_QUESTIONS 2026-11-03).
- A reply without a usable mask, a stream that ends before `response.completed`, a `response.failed` or `error` event, or a timeout (10 s for the whole call) falls back to the threshold method, as before. The old guessed `/images/segmentations` route and the COCO RLE helpers are removed.
- To turn it on: `SEGMENT_PROVIDER=sam`, `META_SEGMENT_MODEL=sam-3.1`, `META_PRICE_PER_SEGMENT=0.0025` ($2.50 per 1,000 images).
- Asked for by Aryan ("begin the SAM for the lasso as planned"), 2026-11-04.

## 2026-11-04 — Glasses ocean guide (Android app on the Wearables toolkit, hackathon demo only)
- Revives the Ray-Ban/Oakley Meta glasses idea dropped on 2026-10-24, because Meta's Mock Device Kit now lets the app be built and tested without glasses. Scope: hackathon demo only, Android only, sideloaded; Aryan cannot borrow glasses, so the mic, speakers and touchpad are untested on real hardware and the demo is labelled "tested on mock glasses" (plan: claude.ai/artifact/St11MrD4J3wZbVNZKZWqmC).
- Contract change (approved by Aryan, "get to it", 2026-11-04): new `POST /glasses/ask` (PRD_BACKEND §4.18) behind `GLASSES_ENABLED`, and optional `min_depth_m` on `/nearest-ocean` (§4.17). No change to existing responses.
- The answer is built from the prediction store by a fixed template, not by the guide. Reason: the guide may not compute numbers (prompt rule), and blending between levels is a computed number; a template also makes every spoken number exact and costs nothing. The Meta model in the loop is Muse Voice Transcribe. Deviation from the plan doc, which proposed the guide.
- Physical limits stated in every answer: the nearest answer is the first 0.25° cell (about 27 km wide) at least 5 km out along the heading that has water at the asked depth, with its real distance; heading comes from the phone's compass (the glasses expose no location or heading).
- Proposed by Claude; asked for by Aryan.

## 2026-11-05 — Explorer dot-matrix default and floating glass HUD (PRD_FRONTEND §3.1, §3.1b, §4)
- **Dot-matrix map default with Classic toggle:** `/explore` now defaults to a 1-dot-per-0.25°-cell dot-matrix renderer (`gridStyle = "dots"`) in both 2D Map and 3D Cutaway views, with a `Dots` | `Classic` toggle in `ViewSwitch` that switches back to the continuous block renderer (`gridStyle = "solid"`, persisted as `?style=solid` while keeping default URLs clean). Zero points are interpolated or invented (`PRD_FRONTEND` §0 rule 1).
- **Scientific colormaps preserved:** keeps `cmocean thermal` for temperature (`pred`, `glorys`) and `balance` for difference (`error`) across 2D and 3D, rather than replacing them with a decorative monochrome palette.
- **Accessible coverage shapes in dot mode:** colour is never the only signal (`PRD_FRONTEND` §4): in `dots` mode, `coverage = 1` draws a filled colormap dot, `coverage = 2` (insufficient inputs) draws a hollow ring, and `coverage = 0` (land/unmodelled) draws a smaller muted stipple dot over a dark obsidian (or light paper) backing; `CoverageLegend` switches between dot/ring symbols and classic block/hatch swatches accordingly.
- **Functional-only floating glass HUD:** `/explore` uses a full-bleed map with a floating glass `Ocean Controls` card on the left, a bottom-center floating glass pill toolbar (`ViewSwitch`, `ARGO Floats`, and flag-gated `Events` / `Make a postcard` / `Lasso` / `Ask the ocean`), and circular HUD badge pins for selected cells and ARGO profiles. No non-functional widgets, gauges or invented numbers are added (`AGENTS.md` §5.1).
- Approved by Aryan, 2026-11-05.

## 2026-11-05 — Cinematic underwater landing page (PRD_FRONTEND §4; replaces PLAN_LANDING visual direction)
- Aryan approved replacing the woodblock landing with the two supplied underwater references: warm sunlit teal in light mode, deep cyan-black in dark mode; centered Manrope typography, glass navigation, rounded CTA, and matching lower sections. Existing destinations, scientific copy, credits and the non-dismissible synthetic banner remain.
- Explicit exception to AGENTS §5.11: decorative AI-edited landing backgrounds and transparent fish may be committed. They are labelled **AI-generated illustration; not observational data**, and their asset metadata records `synthetic: true` and the generation prompts. They never supply scientific values. No Meta endpoint or model call is added to the frontend.
- Three.js (`three`, with development-only `@types/three`) renders the layered orthographic scene, localized water-refraction shader, moving highlights/shafts/caustics and flexing fish sprites. GSAP handles timing, swimming paths, entrances and microinteractions. Dependencies are necessary for the requested shader-based scene and choreography; no paid runtime service. Shader programs compile asynchronously to avoid blocking the first frame.
- Motion is calm and cinematic. A pause control freezes ambient rendering; reduced motion skips the canvas entirely. Missing textures, unavailable WebGL or context loss retain the HTML/poster fallback. Scene timing pauses offscreen and in hidden tabs; mobile draws at 30 fps with fewer fish and capped pixel ratio. All renderer resources and animation/listener registrations are disposed.
- The theme is now shared between landing and Explorer: stored `theme` first, otherwise system preference; a local bootstrap runs before hydration, and changes persist across navigation. Navigation stays ordinary HTML without JS. The old woodblock dive is replaced by a short underwater fade with route-ready focus transfer.
- Landing links do not automatically prefetch Explorer/validation bundles; the CTA still warms those resources on hover/focus/touch. Budgets: initial landing-specific JS ≤15 KB gzipped, deferred scene/choreography ≤180 KB, route JS ≤350 KB excluding opt-in development MSW fixtures; initial image requests ≤350 KB desktop / ≤180 KB mobile. These replace the old always-paper styling and static-plate-only direction, not the API or ML contracts.
- Approved by Aryan in this chat (full implementation plan; Three.js and generated decorative assets explicitly permitted), 2026-11-05. Work remains on `frontend/dot-matrix-explore`, local commit only.

## 2026-11-05 — Landing → Explorer arrival transition (PRD_FRONTEND §3.1, §4)
- The CTA no longer fades to a full-screen colour and navigates. It plays one continuous GSAP transition that also hides the Explorer's load: the hero copy lifts away and the underwater scene sinks while the water surface sweeps down the screen (rising out of the sea); the screen holds the basemap's water colour across the route change; when the Explorer's map has drawn, the overlay fades into it, the real MapLibre camera zooms out from open water to the basin and flies into the region, the grid fades in, and the header, controls, toolbar and map controls slide in from the edges; desktop then tilts into the 3D view the CTA asked for.
- No invented imagery: the "zoomed out" and "zoomed in" frames are the live basemap and model grid. The overlay never holds the screen for more than 15 s; after 4 s it explains that the backend may be waking up.
- Runs in any browser (no View Transitions API needed); reduced motion keeps the plain navigation. GSAP is loaded when the transition starts, so the landing's initial-JS budget (DECISIONS above) is unchanged: `/` first load 143 kB before and after.
- Asked for by Aryan in this chat, 2026-11-05.

## 2026-11-05 — Immersive landing motion with a fixed viewpoint (PRD_FRONTEND §4)
- Aryan approved the stronger motion direction: rolling water, traveling sunbeams/reef caustics, fish at multiple apparent depths with individual tail cycles and curved routes, occasional closer passes, suspended particles, and GSAP glass glints/hover/focus feedback.
- **No pointer-bound parallax or floating viewpoint.** The camera remains fixed. The old scene-wide `pointermove` handler and pointer uniform are removed. Pointer interactions affect only the actual controls; ordinary autonomous water/fish/light motion continues without cursor input.
- The ceiling is now a subdivided, tilted Three.js plane with animated vertex displacement and wave normals. Its projected image coordinates blend into the existing labelled art. Mirrored edge sampling prevents the photograph's edge pixels from stretching into stripes when a wave crosses the frame.
- Fish use continuous curved routes with headings aligned to travel, independent tail speeds, and invisible offscreen loop endpoints; they no longer swim backwards on a yoyo tween. Mobile uses fewer segments, fish, and particles and retains its frame-rate cap. Pause, reduced motion, fallback, offscreen suspension, cancellation and GPU cleanup still apply.
- Hero posters load eagerly, then their decoding and an idle slot gate graphics initialization, so the richer scene does not compete with the first screen for bandwidth/CPU.
- The separately approved landing → Explorer arrival sequence remains connected; this change does not replace it. No new assets, dependencies, APIs, scientific content, or data contracts.
- Approved by Aryan ("I like all the ideas except ... the parallax effect bound by the pointer ... implement everything else"), 2026-11-05.

## 2026-11-06 — Next model round: dev/final float halves, correction guards, loss options (PRD_AI_ML §1.1, §5.4, §5.9, §7, §10.1)
- **Held-out floats split into a dev half and a final half**, stratified by region (a float's region is its median position over the validation profiles; within each region the floats are shuffled with seed 2026 and the first ceil(n/2) go to dev; `evaluation.argo_validation.holdout_halves`). From now on, runs are compared and picked on the **dev half only**. The **final half is scored once**, on the run picked with dev. Reason: ens3 was already chosen by comparing runs on all held-out floats, so its 0.957 °C is selection-biased (P5-4 review). Limitation: the Bay of Bengal has 15 held-out floats, so each half has 7–8 there and the CIs get wider. The pre-registered criterion (region, depths, metric, references) is unchanged.
- **ARGO correction guards:** `--min-depth` (levels shallower than this are never corrected; default 1.0 m in the Kaggle script, so the 0.5 m level is left alone) and `--outlier-sigma` (residuals beyond K robust sigmas of the per-depth median are dropped; default 4). Reason: on main2_ai the tuning floats said 0.5 m improved and the held-out floats said it got worse (0.34 → 0.77 °C).
- **Loss options for fine-tuning:** Huber loss (`--huber-delta`, normalized units) and a vertical-gradient term on GLORYS (`--vgrad-weight`, MSE of the level-to-level change in °C scaled by the levels' mean sigma). Both off by default; existing runs are unchanged. They are CLI flags rather than `experiment.yaml` keys because that file feeds `compute_config_hash` for the stores already built (same reason as the headline constants in `argo_validation.py`). Experiment runs are tagged (`main2_aiv`) and never overwrite existing runs.
- **Uncertainty check before any uncertainty map:** `evaluation.spread_skill` compares the ensemble spread with its error at the held-out floats (ratio, ±1σ/±2σ coverage, spread–error correlation, a scale fitted on half the floats and checked on the other half). A `temp_pred_std` field on the site stays a contract change that needs a separate approval.
- Not done, and why: MLD or thermocline depth from GLORYS as an input (GLORYS is the target and a competitor, so it would leak); anything that reads 2021 (already scored once on main2_ft).
- Approved by Aryan ("okay get to work, do the code part and I'll go run the kaggle"), 2026-11-06, after the shortlist in the thread "Plan next model improvements against ARGO".

## 2026-11-06 — Run E: weighted ensemble, tree-model residual correction, regimes, matchup audit
- **Weighted ensemble** (`inference.ensemble_weights`, `ensemble --weights`): one set of weights (≥ 0, sum 1, grid of 0.05) for the whole domain and every depth, fitted on **non-held-out** floats in the validation period. No per-depth or per-region weights: too few floats, they would overfit.
- **Tree-model residual correction** (`inference.residual_model`): LightGBM on ARGO − model at non-held-out floats, with depth, predicted temperature, position, day of year, SST, SLA, wind speed and distance/age of the nearest other float. Five-fold CV by float picks the tree count. Every input exists in live mode. New dependency `lightgbm` (free, CPU, preinstalled on Kaggle). Fitted on non-held-out floats, never on the dev half (8 Bay of Bengal floats is too few to fit on).
- **Regime table** (`evaluation.regimes`) and **matchup audit** (`evaluation.matchup_audit`): reading aids only; they never pick a run or a setting. The primary matchup stays the same day; ±1 day and the 3-day mean are sensitivity rows.
- Known limitation: in run E the per-depth correction is tuned on top of the tree-corrected store with floats the tree was fitted on, so its tuning scores are optimistic. The dev-half check is unaffected.
- Approved by Aryan ("did you add the new suggestions too?", after the feasibility reply listing 1, 8, 2 and 9 for run E), 2026-11-06.

## 2026-11-06 — Dev-half pick for the final-half score: ens3w_rm_oi
- Bay of Bengal 50–300 m, dev half (6 floats), validation years: ens3 0.867, ens3_oi 0.829, ens3w 0.862, ens3w_rm 0.836, **ens3w_rm_oi 0.816** (CI 0.700–0.881), main2_aiv 0.878, main2_ai 0.885; GLORYS12 0.756, GREP 0.855, ARMOR3D 0.639. Picked by the rule set before the runs: lowest dev-half RMSE. The lead over ens3_oi (0.013 °C) is within the noise.
- Caveat found: ensemble weights and the tree model are fitted on non-held-out floats that the ARGO-input members (main2_ai, main3) also see as inputs, so their fit scores are optimistic (weights went 0.1 / 0.0 / 0.9 but only moved the dev score from 0.867 to 0.862). The dev and final halves are never inputs, so their scores are unaffected.
- The final half is scored once on ens3w_rm_oi (`scripts/kaggle_final_half.py`), with ens3 on the same floats for reference only.

## 2026-11-06 — Final-half score of ens3w_rm_oi; not published; next round (run F)
- **Final half (9 Bay of Bengal floats, scored once):** ens3w_rm_oi 0.906 °C (CI 0.868–0.967); ens3 0.981 on the same floats (reference only); GLORYS12 0.794 (difference CI 0.073–0.154: behind), GREP 0.977 (difference CI −0.30–0.19: tie), ARMOR3D 0.598. The warm bias at 65–186 m is gone (ens3 +0.22 to +0.32 °C, now within ±0.06). The correction made 643 m and 1062 m slightly worse (0.406 → 0.426, 0.377 → 0.426). Pre-registered headline: still **not met**.
- **Not published.** Aryan: "I don't want to publish it yet, let's keep improving the model until it beats glorys12" (2026-11-06). ens3 stays live; the live-mode code for the correction steps waits.
- **Run F** (`scripts/kaggle_run_f.py`, `residual_model --features v2`, `--max-depth` in `residual_model` and `argo_correct`): the tree correction also sees the nearby non-held-out floats' residuals at the same level (weighted mean, L = 1.5°, τ = 5 days, own float left out in training) and their weight, the predicted anomaly against the train-only climatology, the predicted profile's vertical gradient and mixed-layer depth (from the prediction, never GLORYS), the 14-day SST change and the 28-day SLA change. Aimed at the weak regimes in run E: warm anomalies, SW monsoon, positive IOD. Neither correction touches levels below 600 m. Defaults (`v1`, no max depth) leave earlier runs reproducible.
- **How run F is judged.** Both halves of the held-out floats have now been looked at, so a dev-half gain alone is weak evidence (6 floats; it has already picked one run). Run F replaces ens3w_rm_oi only if it is lower on the dev half **and** the tree's out-of-fold CV on non-held-out floats (Bay of Bengal 50–300 m) also improves. Any claim of beating GLORYS12 needs an untouched test, scored once (see the open decision on the 2021 ARGO headline).

## 2026-11-06 — The 2021 ARGO headline is the untouched test (scored once)
- **Aryan's choice** (decision card "Pick the untouched test that decides if we beat GLORYS12": "2021 ARGO floats, scored once"). No run has been scored against the 2021 held-out floats. Caveat stated on the card: main2_ft's 2021 score against GLORYS (P8-1) has been seen; the ARGO score has not.
- **What is fixed before the run, from the validation years only:** member checkpoints, ensemble weights, the tree model (saved by `residual_model` as `models/residual_model_<run>.txt` + `.json`), the per-depth correction plan and its guards (saved by `argo_correct` as `models/argo_correct_<run>.json`). Nothing is fitted or tuned on 2021.
- **How 2021 is used:** non-held-out 2021 profiles are used as live mode would use them (the members' ARGO inputs, and the residuals the corrections spread around). All held-out floats are scored (the dev/final halves were a split of the validation years). Bay of Bengal 50–300 m, float bootstrap, as `argo_validation`. GREP ends in 2020, so a reference with values at fewer than 95% of the matchups is n/a and left out of the verdict (as decided 2026-10-25 for the 2021 headline). The plain ensemble (live recipe) is scored on the same floats for reference.
- **Code:** `evaluation.argo_test` (refuses without `--owner-go` and when `<run>_test_headline.csv` exists), `scripts/kaggle_argo_test.py` (stops unless `OCEANSCOPE_OWNER_GO=1`). To allow it: `predict_store` builds ARGO input channels for the test split (only reachable with `allow_test`), `ensemble(allow_test=True)` takes test-year members, `data.argo.fetch_month(allow_test=True)` downloads 2021 months. Every default still refuses 2021.
- **When:** only after the run picked on the validation years is final (run F or ens3w_rm_oi), and only on Aryan's go. Run once; the first result is the one quoted. If ens3w_rm_oi (run E) is the pick, its tree is refitted first with the saving code (same settings), because run E did not save the model.

## 2026-11-06 — Run F result: no change of pick; 2021 test held
- Dev half, Bay of Bengal 50–300 m: ens3w_rm2 0.821 °C (CI 0.711–0.892), ens3w_rm2_oi 0.821 (the per-depth tuning found no level where the extra correction helped, so the plan is empty). Run E's ens3w_rm_oi: 0.816. GLORYS12 0.756, GREP 0.855, ARMOR3D 0.639.
- Tree CV on non-held-out floats, Bay of Bengal 50–300 m: v2 0.722 → 0.677 °C, v1 (run E) 0.722 → 0.687. The nearby-float residual is by far the most used feature, so the v2 tree does in one step what tree + OI did in run E.
- By the rule set before the run (lower on the dev half and better CV), run F does **not** replace ens3w_rm_oi; the two are a tie within noise.
- Regimes (dev half): warm anomalies still the gap (1.27 vs GLORYS12 1.01), then SW monsoon (0.91 vs 0.78) and positive IOD (0.89 vs 0.70).
- The one-shot 2021 test (step G) is held while the dev half still shows the model behind GLORYS12, so it is not spent on a run expected to miss.

## 2026-11-06 — Warm-event fine-tune of main3 (main3_aiw), then a re-weighted ensemble
- **Aryan's choice** (decision card "Pick what to do next about the gap to GLORYS12": "Retrain base model for warm events").
- `finetune --warm-weight W`: training days are drawn with replacement, weighted from 1 (no warm anomaly) to W (the warmest 1% of days). Warmth is the mean positive GLORYS anomaly (in sigma units, against the train-only climatology) over the Bay of Bengal 50–300 m band, so it uses training targets only, never ARGO and never validation days. Needs the ARGO input channels; stored with the loss settings so a resume with other settings is refused.
- Run: main3 fine-tuned with ARGO inputs, λ = 1.0, W = 4 → `main3_aiw` (dev-half ARGO check printed by the fine-tune script). Then `kaggle_run_f.py` with members main2_ai, main2_ft, main3, main3_aiw (`ens4`): weights fitted on non-held-out floats, v2 tree, dev half.
- **Rules set before the run:** ens4w_rm2(_oi) replaces ens3w_rm_oi only if it is lower on the dev half and its tree CV (Bay of Bengal 50–300 m, after correction) is lower too. The 2021 test (step G) is spent only once a run is below GLORYS12 (0.756 °C) on the dev half.

## 2026-11-07 — main3_aiw result; detrended warm weighting
- H (main3, ARGO inputs, λ = 1.0, W = 4): best epoch 1, val ARGO RMSE on non-held-out floats 0.706 → 0.696 °C (Bay of Bengal 50–300 m 0.892 → 0.878). Dev half, Bay of Bengal 50–300 m: **main3_aiw 0.860** (CI 0.780–0.924) vs main3 0.867. A small gain.
- Flaw found: the weighting counted 3142 of 4264 ARGO-era training days as warm (weight above 2), because the 2004–2017 ocean is warmer than the 1993–2017 climatology. It mostly up-weighted recent years, not warm events.
- `finetune --warm-detrend` (`OCEANSCOPE_FT_WARM_DETREND=1`): warmth is how far the day's mean band anomaly sits above that year's mean, so only warm events within a year get the extra weight. Off by default; main3_aiw is unchanged. A second run (main3_aiwd) waits for the ens4 result (step I).

## 2026-11-07 — ens4 result (step I): no change of pick
- Weights on non-held-out floats: main2_ai 0.0, main2_ft 0.0, main3 0.3, main3_aiw 0.7 (fit RMSE 0.681 → 0.647 °C). Tree CV, Bay of Bengal 50–300 m: 0.717 → 0.669 (run F 0.677). The per-depth correction again found nothing to add.
- Dev half, Bay of Bengal 50–300 m: **ens4w_rm2 0.827** (CI 0.705–0.894) vs ens3w_rm_oi 0.816 and ens3w_rm2 0.821. GLORYS12 0.756. Not lower on the dev half, so by the rule set before the run it does not replace ens3w_rm_oi.
- Warm anomalies did not move (1.30 vs 1.27 in run F; GLORYS12 1.01), nor did SW monsoon (0.92) or positive IOD (0.90). Five rounds of changes (corrections, weights, tree features, warm-day fine-tune) have all landed at 0.82 ± 0.01 on the dev half.
- Likely reason: GLORYS12 assimilates the held-out floats' own profiles, and its advantage is largest exactly where those profiles carry information the surface fields do not (warm subsurface events). Our model never sees them.

## 2026-11-07 — ARGO-heavy fine-tune with a 30-day float window (main3_aix30, ens5)
- **Asked by Aryan** ("let's try the top 2 methods you mentioned"): (1) weigh the ARGO loss much more than GLORYS in the fine-tune, (2) let the model and the tree correction see floats from the last 30 days instead of 10.
- `finetune --argo-window N` sets the ARGO input window in the loaded config only (not `experiment.yaml`, which feeds the store hash). The window is saved in the checkpoint (`argo_window_days`), and `predict.load_checkpoint` applies it, so prediction and the 2021 test always use the window a model was trained with. A fine-tune without `--argo-window` keeps its base model's window. `live.kit` refuses a checkpoint whose window differs from the live fetch (10 days) until live mode fetches the longer window.
- `residual_model --window-days N` (default 10) for the nearby-float features; saved with the model and applied unchanged by `evaluation.argo_test`, which now reads residuals from 31 days before the test year.
- **Bug fixed on the way:** `argo_inputs_for` refused any end date in the test year, so the 2021 test would have stopped at its first ARGO-input member. It now reaches the test year only with `allow_test` (passed by `predict_store` only when `allow_test` is set), and reads the test year separately from December 2020.
- Run J: main3, ARGO inputs, 30-day window, λ ∈ {3, 10} (the one with the lowest validation ARGO RMSE on non-held-out floats is kept) → `main3_aix30`. Run K: `kaggle_run_f.py` with members main2_ai, main2_ft, main3, main3_aiw, main3_aix30 (`ens5`), tree window 30 days.
- **Rules set before the run:** as for ens4. ens5w_rm2 replaces ens3w_rm_oi only if lower on the dev half and its tree CV is lower; the 2021 test is spent only once a run is below GLORYS12 (0.756 °C) on the dev half.

## 2026-11-07 — Run J result: main3_aix30
- main3 fine-tuned with ARGO inputs over a 30-day window: λ 3 best epoch 5, λ 10 best epoch 6 (the cap), both 0.687 °C validation ARGO RMSE on non-held-out floats (base 0.709); λ 3 kept (first of the tie). Bay of Bengal 50–300 m on those floats 0.888 → 0.866.
- Dev half, Bay of Bengal 50–300 m: **main3_aix30 0.840** (CI 0.735–0.933), against main3 0.867 and main3_aiw 0.860; GLORYS12 0.756. The best single member so far. Both λ were still improving at 6 epochs.

## 2026-11-07 — Run K result (ens5): no change of pick
- Out of memory on the first try (5-member weight grid, 10,626 points); fixed by computing the grid RMSE from the members' error Gram matrix (exact), then rerun unchanged.
- Weights on non-held-out floats: main2_ai 0.0, main2_ft 0.0, main3 0.25, main3_aiw 0.6, main3_aix30 0.15. Tree (30-day window) CV, Bay of Bengal 50–300 m: 0.716 → 0.671 (ens4 0.669). The per-depth correction again found nothing to add, so ens5w_rm2_oi equals ens5w_rm2.
- Dev half, Bay of Bengal 50–300 m: **ens5w_rm2 0.823** (CI 0.700–0.890) vs ens3w_rm_oi 0.816; GLORYS12 0.756. Not lower on the dev half and CV not lower than ens4, so by the rule set before the run it does not replace ens3w_rm_oi. The 2021 test stays held.
- Warm anomalies 1.28 (GLORYS12 1.01), SW monsoon 0.92, positive IOD 0.89. The best single member (main3_aix30, 0.840) got only 0.15 of the weight; the ensemble lands at 0.82 again, the sixth round in a row.

## 2026-11-08 — Keep chasing GLORYS12 after run L
- **Aryan's choice** (decision card "Decide what to do if run L still doesn't beat GLORYS12": "Keep chasing GLORYS12"). Run L: main3 fine-tuned with ARGO inputs, 30-day window, λ 3, up to 14 epochs (`main3_aix30L`).
- If L misses, the next round is a bigger change (larger model, new inputs) rather than more ensembling or correction, since six rounds of those have all landed at about 0.82 on the dev half.

## 2026-11-08 — Run L result: longer training does not help
- main3 fine-tuned with ARGO inputs, 30-day window, λ 3, up to 14 epochs: best epoch 7, val ARGO RMSE on non-held-out floats 0.686 °C (J: 0.687; base 0.709). Early stopping ended it at epoch 9; the curve is flat from epoch 3.
- Dev half, Bay of Bengal 50–300 m: **main3_aix30L 0.853** (CI 0.748–0.948) vs main3_aix30 0.840; GLORYS12 0.756 (gap 0.097, CI 0.050–0.148). No change of pick; the fine-tune route has reached its limit.

## 2026-11-08 — ARMOR3D as a model input (round M)
- **Aryan's choice** (decision card "Pick the bigger change to try next for beating GLORYS12": "Use ARMOR3D as an input"). This overrides PRD_AI_ML §3 ("ARMOR3D: external baseline only … never an input") for runs that turn it on; the default config and every earlier run are unchanged.
- **What it changes about the comparison:** ARMOR3D assimilates every ARGO float, the held-out ones included. A model with ARMOR3D inputs is compared with GLORYS12 on equal terms (both have seen the floats), but its §1.1 comparison with ARMOR3D itself is no longer independent, and its held-out-float scores no longer measure skill without those floats. Results of such runs are reported with that note.
- `inputs.armor3d_input` (set in the loaded config only, not in `experiment.yaml`): 16 channels after the ARGO ones, `armor3d_anom@<d>m` = (ARMOR3D − training climatology) / sigma_d for the 15 depths, and `armor3d_mask`. Same-day ARMOR3D (the daily MY product). Fine-tuned in with zero initial weights, like the ARGO channels, so epoch 0 is the base model.
- Data: the part stores have ARMOR3D for 2018–2021 only (DECISIONS 2026-10-28). Training years come from side stores `armor3d_<year>.zarr` written by `python -m oceanscope_ml.data.armor_inputs build` / `scripts/kaggle_armor3d_fetch.py`: every 3rd day (counted from 1993-01-01), training dates only (refused past the training split), reduced exactly as the part stores. 2004–2017 (the ARGO-input era).
- The test year reaches ARMOR3D inputs only with `allow_test` (the one-shot 2021 test). `live.kit` refuses ARMOR3D-input checkpoints: live mode has no ARMOR3D feed yet.
- Run M: main3 + ARGO inputs (30-day window, λ 3, as run J) + ARMOR3D inputs → `main3_aix30a`. Compared with main3_aix30 (0.840 dev half), which differs only by the ARMOR3D channels.
- **Rules set before the run:** main3_aix30a (or an ensemble/tree built on it) replaces ens3w_rm_oi only if lower on the dev half (and, for a corrected run, its tree CV is lower). The 2021 test is spent only once a run is below GLORYS12 (0.756 °C) on the dev half, and only on Aryan's go.

## 2026-11-09 — Run M stalled: ARMOR3D planes are now cached once
- Run M (first attempt) found all 14 side stores and selected 1421 train / 1036 val days, then ran for 11.6 h on a T4 x2 without finishing even the epoch-0 evaluation. Nothing was scored, so there is no result to record and the rules above are unchanged.
- Cause: `ArmorInputs.planes` read one day at a time from Zarr: `clim` from the stats store, `armor3d` from the part stores and the side stores. Each read decompresses whole chunks (the reason the training cache exists, PRD §5.1), and the DataLoader workers also inherited open dask handles.
- Fix: `ArmorInputs` now writes every day's planes once at start-up into a float16 memory map (reading each side store and each 32-day span of the part stores once), and each worker reads one slice per sample. Same values as before up to float16 rounding, the same rounding the training cache uses. Rerun run M unchanged.

## 2026-11-09 — Run M result: main3_aix30a is below GLORYS12 on the dev half
- Rerun with the cached planes: 1421 train / 1036 val days, 4.1–4.4 min per epoch, best epoch 5 of 6 (λ 3). Validation ARGO RMSE (non-held-out floats, whole region) 0.572 °C vs 0.709 °C for main3 at epoch 0; Bay of Bengal 50–300 m on those floats 0.695 vs 0.888.
- Dev half, Bay of Bengal 50–300 m (6 floats, 20 021 obs): **main3_aix30a 0.727** (CI 0.641–0.752); GLORYS12 0.756 (difference −0.028, CI −0.077 to −0.021); GREP 0.855; ARMOR3D 0.639 (ARMOR3D still lower by 0.088, CI 0.064–0.139). Compared with main3_aix30 (0.840), the only change is the ARMOR3D channels: −0.113.
- By the rules set before the run it replaces ens3w_rm_oi (0.816) as the pick on the dev half. It is the first run below GLORYS12 on the dev half, so the one-shot 2021 test may now be spent, on Aryan's go only.
- Caveat (as recorded for round M): ARMOR3D assimilates the held-out floats, so this score compares with GLORYS12 on equal terms but is not skill independent of those floats. The model is still behind ARMOR3D itself, which it takes as input. The dev half has only 6 Bay of Bengal floats.


## 2026-11-09 — Run N: our best recipe on top of the ARMOR3D model
- **Aryan's ask** (thread, after run M): give the ARMOR3D model the same improvements as our best, then test. Read as: the ens3w_rm_oi recipe (weighted ensemble → tree correction → guarded per-depth ARGO correction) with main3_aix30a added as a member.
- Run N: `kaggle_run_f.py` with members main2_ai, main2_ft, main3, main3_aiw, main3_aix30, main3_aix30a (`ens6`), tree window 30 days, dev half. Nothing reads 2021.
- **Rules set before the run:** ens6 (or ens6w_rm2 / ens6w_rm2_oi) replaces main3_aix30a only if lower than 0.727 °C on the dev half. The winner is the one that takes the single 2021 test, on Aryan's go. `evaluation.argo_test --single` (and `OCEANSCOPE_SINGLE` in `kaggle_argo_test.py`) can score a single model if main3_aix30a stays the pick.
- Expectation, stated in advance: the tree and per-depth corrections learn from the same non-held-out floats that ARMOR3D already assimilates, so the gain may be small; in run K the weights also favoured the strongest member.

## 2026-11-09 — Run N result (ens6): main3_aix30a stays the pick
- Weights on non-held-out floats: main3_aix30a 0.7, main3_aiw 0.25, main3 0.05, the rest 0. Tree (30-day window) CV, Bay of Bengal 50–300 m: 0.664 → 0.642. The per-depth correction again found nothing to add (every depth's plan empty), so ens6w_rm2_oi equals ens6w_rm2.
- Dev half, Bay of Bengal 50–300 m (6 floats): **ens6w_rm2 0.735** (CI 0.638–0.771) vs main3_aix30a 0.727; GLORYS12 0.756, GREP 0.855, ARMOR3D 0.639. Not lower than 0.727, so by the rule set before the run main3_aix30a stays the pick. It is the model for the single 2021 test (`kaggle_argo_test.py` with `OCEANSCOPE_SINGLE=main3_aix30a`), on Aryan's go.
- Regimes (ens6w_rm2 vs GLORYS12): better in cold anomalies, normal conditions, cyclones (0.531 vs 0.691), La Niña, neutral IOD, inter- and NE monsoon; worse in warm anomalies (1.140 vs 1.012), positive IOD and the SW monsoon.

## 2026-11-09 — Run O: a learned blend with ARMOR3D (Aryan: "let's try it")
- Why: main3_aix30a (0.727) is still behind ARMOR3D alone (0.639) on the dev half, though ARMOR3D is one of its inputs. Its ARMOR3D channels start at zero weight and have 6 epochs to be picked up; the validation curve had flattened by epoch 5–6.
- `--armor-blend` (`ArmorBlend` in `data/armor_inputs.py`): output = w_d · ARMOR3D + (1 − w_d) · U-Net wherever ARMOR3D has a value, with one learned weight w_d per depth (sigmoid, starts at 0.5, learning rate 100× the base, no weight decay). Both terms are anomalies scaled by sigma_d, so no rescaling. Where ARMOR3D is missing, the U-Net's output is used alone. The U-Net still sees the ARMOR3D channels, so it can learn the correction that the blend needs. The weights are logged every epoch, and predict restores the blend from the checkpoint (`armor_blend`).
- Run O: as run M (main3, ARGO inputs, 30-day window, λ 3, ARMOR3D inputs) + blend, up to 12 epochs (patience 2) → `main3_aix30ab`.
- **Rules set before the run:** main3_aix30ab replaces main3_aix30a only if it is lower than 0.727 °C on the dev half. The 2021 test is spent once, on whichever is the pick when Aryan says go.

## 2026-11-09 — Run O result: main3_aix30ab is the new pick
- Log checks: `ARMOR3D side stores: 14` (2004–2017); ARMOR3D inputs 1066 days from the part stores + 1695 from the side stores. Best epoch 7 of 10 (early stop, patience 2); validation ARGO RMSE on non-held-out floats 0.515 (base) → 0.420 °C (run M's best was higher; Bay of Bengal 50–300 m on those floats 0.644 → 0.508). Epoch 0's `train loss nan` is the untrained base row, not a fault.
- Learned ARMOR3D weight per depth at the end: about 0.3 at the surface, 0.6 at 5 m, 0.7 at 10 m, 0.76–0.84 from 18 m down. The model is now mostly ARMOR3D plus a U-Net correction.
- Dev half, Bay of Bengal 50–300 m (6 floats, 20 021 obs): **main3_aix30ab 0.612** (CI 0.503–0.652); GLORYS12 0.756 (difference −0.144, CI −0.205 to −0.134); GREP 0.855 (−0.243); ARMOR3D 0.639 (−0.027, CI −0.035 to −0.001, so only just below ARMOR3D). Whole-domain RMSE vs GLORYS on the 1036 validation days 0.680 °C (it drifts away from GLORYS as it leans on ARMOR3D, as expected).
- Lower than 0.727, so by the rule set before the run **main3_aix30ab replaces main3_aix30a** as the pick. It is the model for the single 2021 test (`kaggle_argo_test.py` with `OCEANSCOPE_SINGLE=main3_aix30ab`, run O notebook notebook51fb30cf78 attached instead of run M's), on Aryan's go.
- Caveat, stronger than before: ARMOR3D assimilates every ARGO float, the held-out ones included, and the model now takes about 80% of its answer from ARMOR3D below 18 m. The win over GLORYS12 is on equal terms (both have seen the floats) but isn't skill independent of the floats, and the margin over ARMOR3D itself is small with only 6 Bay of Bengal floats. It can't run live (no live ARMOR3D feed).

## 2026-11-09 — main3_aix30ab ruled out; main3_aix30a is the pick; 2021 test go
- **Aryan's choice** (thread, after the run O result): main3_aix30ab is dropped ("let's ditch this one"). Its learned weight puts about 80% of the 50–300 m answer on ARMOR3D, so its score is mostly ARMOR3D's, not our model's. This overrides the pick rule for run O.
- **main3_aix30a** (0.727 on the dev half; ARMOR3D only as 16 of 155 input channels, no output blend) is the pick again.
- Claim to be tested: given the same ARGO float information as GLORYS12 (which assimilates the held-out floats; our model sees it through ARMOR3D), our model scores lower at the held-out floats. Always stated with: ARMOR3D is an input, and ARMOR3D alone scores lower (0.639 on the dev half); main3_aix30 without ARMOR3D is 0.840.
- **Go for the one-shot 2021 test on main3_aix30a** (Aryan: "let's actually test it once for all and see if our claim holds up in actuality", 2026-11-09). GREP ends in 2020, so the 2021 verdict is against GLORYS12 (and ARMOR3D for reference); the GREP comparison stays a validation-year result.

## 2026-11-09 — 2021 test attempt 1 stalled before any scoring; preload fix
- The first 2021 test run (notebook97c612307d) fetched the 13 ARGO months and started predicting 2021, then ran 85+ min without finishing the first 32 of 335 days. It was stopped before any prediction store or score existed, so the test is still unspent (no `main3_aix30a_test_headline.csv`).
- Cause: the training cache excludes the test year by design, so prediction read every sample straight from the five part stores and the stats store (history windows and clim decompressed per sample), the same slow path that stalled run M.
- Fix: `OceanScopeDataset.preload()` loads the days the split reads (history included) and `clim` into memory once; `predict_store` calls it when there is no cache. Samples are identical (test: `test_preloaded_samples_match_the_store`). Rerun the same cell unchanged.

## 2026-11-10 — 2021 ARGO test result (main3_aix30a, scored once)
- Rerun after the preload fix (notebook97c612307d): 335 test days predicted in about 2 min; 23 held-out floats in 2021, 269 303 matches.
- **Bay of Bengal 50–300 m, all held-out floats, 2021 (7 floats, 15 901 obs):** main3_aix30a **0.709 °C** (CI 0.612–0.785); GLORYS12 0.873 (difference −0.164, CI −0.283 to −0.018: **lower than GLORYS12**); GREP 0.743 (difference −0.033, CI −0.091 to +0.021: **a tie**); ARMOR3D 0.570 (ARMOR3D lower by 0.139, CI 0.071–0.194). Headline verdict as pre-registered: **Not met** (not below every reference).
- GREP had values for 2021 at these matchups, so it was scored (an earlier note in this thread that GREP ends in 2020 was wrong for this product version).
- Per depth (all held-out floats): lower RMSE than GLORYS12 at 29–1062 m, slightly higher at 0–18 m (within 0.016 °C); behind ARMOR3D at every depth.
- What can be said: on the untouched 2021 floats, given the same ARGO information (through ARMOR3D as an input), our model is more accurate than GLORYS12 in the Bay of Bengal 50–300 m band and ties GREP; ARMOR3D, one of its inputs, is more accurate still. The test is spent; this is the number quoted. Nothing published.

## 2026-11-10 — Running main3_aix30a live (Aryan: "let's do it")
- Plan: (1) Kaggle check of the near-real-time ARMOR3D datasets (`scripts/kaggle_armor_nrt_check.py`: datasets, daily or weekly, delay, NRT − MY over the Bay of Bengal at 50–300 m); (2) live ARMOR3D fetch and lifting the live-kit ARMOR3D lock, shaped by that result; (3) the P9-1 style live-vs-reprocessed check on a past month; (4) dry run; the site switches only on Aryan's OK.
- Done now: live mode keeps each member's ARGO window. `KitMember.argo_window_days` (in kit.json; old kits default to the config's 10 days), `Kit.argo_fetch_days` = the widest window of the ARGO members, used by `live.run` and `evaluation.live_check`. `live.kit` no longer refuses a 30-day member. The ARMOR3D refusal stays until step 2.
- **ARMOR3D NRT check (Kaggle, 2026-11-10):** `cmems_obs-mob_glo_phy_nrt_0.125deg_P1D-m` is daily, 0.125°, same depths as the MY product, 2024-09-08 to 2026-11-03 (7 days behind). NRT − MY over the Bay of Bengal, 12 days of April–June 2025: 50–300 m RMSE 0.381 °C, NRT warmer by up to +0.30 °C at 100–175 m (largest RMSE 0.64 °C at 100 m). How much of that reaches the model output is measured by the live-vs-reprocessed check below.
- **Live ARMOR3D (code):** `live.yaml` `sources.armor3d` (NRT) and `live_check.reprocessed.armor3d` (MY); `live.inputs.load_armor3d` fetches and reduces it with `reduce_armor3d` (as the training stores), `with_armor` adds it to the input check as needed on the target day (no history), so a live run picks the newest day that has ARMOR3D; the window store carries `armor3d`; `predict_days` attaches `ArmorInputs(live=True)` (days after the test period only). `live.kit` accepts ARMOR3D members (`member_config` turns the channels on); `kaggle_live_kit.py` builds a single-model kit when one member is named.
- Next: live-vs-reprocessed check of main3_aix30a on 2024-10 (`kaggle_live_kit.py`, no upload). The site switches only on Aryan's OK, and switching means publishing main3_aix30a's validation store as the active run first (the live run refuses a kit that is not the published run).
- **Live-vs-reprocessed check, main3_aix30a, 2024-10 (Kaggle notebook3c0508305d):** 31/31 days complete in both sets; 626 ARGO profiles from 94 non-held-out floats in both. RMSE against GLORYS12, NRT − reprocessed inputs: within ±0.01 °C at 0–30 m and 186–1062 m; NRT lower (better) by 0.025–0.123 °C at 47–131 m. Output difference NRT vs reprocessed: 0.11–0.12 °C at 0–30 m, 0.14–0.31 °C at 47–131 m, ≤0.11 °C below. Smaller than for the accepted ens3 (0.43 °C at 92 m) and main2_ft (0.41) checks. By the P9-1 standard used for main2_ft, live inputs do not degrade the model. Report: project files `live-armor/live_mode_check_main3_aix30a_2024-10.md`. Enabling it on the site is Aryan's decision.

## 2026-11-10 — Switch the site to main3_aix30a (Aryan's choice on the card)
- **Aryan's choice** (decision card "Switch the live site to main3_aix30a?": "Switch to main3_aix30a"), after the 2021 test and the live-vs-reprocessed check.
- Steps: merge this branch into main (the daily live workflow runs from main); `scripts/kaggle_publish_single.py` scores main3_aix30a's validation store against all held-out floats of 2018–2020 (the fine-tune scored only the dev half), publishes it as the active run, and builds and uploads its live kit in the same notebook; then one live dry run by hand, then the daily schedule picks it up.
- What visitors see changes: the active model, its validation metrics, and live maps about 7–8 days behind today (ARMOR3D NRT lag) instead of about 5.

## 2026-11-13 — Error bars: a Bayesian last layer (Batch-and-Match), checked on 2018–2020 and 2021
- **Aryan's request** (project thread, 2026-11-13): "Build a basic uncertainty version of OceanScope and check whether its error bars are honest on the 2021 Argo floats. Even a rough result is enough." This is the "ensemble + uncertainty" stretch goal (DECISIONS 2026-10-22, Round 4).
- Method (`evaluation.uncertainty`): main3_aix30a stays frozen. Per depth level, its error at held-out ARGO observations gets a heteroscedastic Bayesian linear model on the 32 features that feed the U-Net's 1x1 head (mean correction and log error size, 66 parameters per level), fitted with Batch-and-Match VI (Cai, Modi et al., ICML 2024). Baselines: one constant error size per level, and the last layer at its posterior mean only. Coverage of the 68% and 95% intervals per depth band with float-bootstrap CIs, and reliability curves.
- What is fitted on what: only held-out floats (never model inputs). Everything is fitted on 2018–2020. Inside those years: fit on the dev half of the floats, score the final half. Then fit on all 2018–2020 held-out floats and score 2021.
- **The 2021 part is a second look at the sealed test year.** The mean prediction is the model scored once on 2021-11-10 and its 2021 headline does not change; nothing is fitted on 2021. Calibration numbers on 2021 are reported as a second look, never as an untouched result, and are not used to tune anything afterwards.
- Kaggle: `scripts/kaggle_uncertainty.py` (one run, `OCEANSCOPE_OWNER_GO=1` for 2021). Nothing is published.
- **Result (2026-11-13, notebook480bb00de4 + local refit with 8e882f5):** on the 2021 floats (second look) the per-level Bayesian last layer covers 0.71 / 0.94 overall (constant bar 0.83 / 0.98) and 0.74 / 0.96 in the Bay of Bengal 50–300 m (constant 0.79 / 0.97). Dev→final half check in 2018–2020: 0.71 / 0.93, but about 3.5% of rows get blown-up error bars (feature extrapolation); any fix is judged on that check only. Full posterior vs posterior-mean plug-in barely differ (parameter uncertainty is small at this data size). The last layer's mean correction (BoB 0.709 → 0.694 on 2021) is not a test result.
- **Fix and final result (2026-11-13, Aryan's review before emailing):**
  - **Settings, picked on the 2018–2020 dev→final float split only** (rule written before each round: [docs/uncertainty/SELECTION.md](uncertainty/SELECTION.md)): 8 whitened feature directions, features clipped to the fit range, predicted error size capped at 3× the constant bar, mean correction on.
  - **New scores.** CRPS and Gaussian NLL are added next to coverage, with sharpness.
  - **2021, scored once with those settings:**
    - Coverage 0.74 / 0.95 (constant bar 0.83 / 0.98), mean sd 0.40 vs 0.49 °C.
    - CRPS 5.7% (4.3–7.3) below the constant bar.
    - Bay of Bengal 50–300 m: 0.79 / 0.97 for both methods, CRPS 4.2% (2.6–5.5) below the constant bar.
  - **Blow-ups.** No observation now gets a bar above 5× the constant one. The cap binds on 3.0% of observations in the split and 0.04% in 2021.
  - **Public page:** `docs/uncertainty/README.md`.
- **Disclosure and checks (2026-11-13, Aryan's review):**
  - **2021 views.** 2021 has been viewed four times: the one-shot accuracy test, the unconverged first error-bar run, the first converged version, and the final run. The 2021 error-bar numbers are therefore a confirmation, not a clean test.
  - **NLL.** BaM's NLL gain over the posterior-mean plug-in is concentrated where features are clipped or the bar is capped: 4.0 vs 24.1 on those 7.1% of split rows, against 0.478 vs 0.484 everywhere else.
  - **Bay of Bengal.** The near-constant bars there are not caused by clipping or the cap, which hit its 50–300 m rows less often (1.2% vs 10.5% elsewhere). Every setting tried gives the Bay of Bengal near-constant bars.
  - **Permanent link.** Tag `uncertainty-v1`.
- **Hierarchical error bars (2026-11-13, Aryan's plan; 2018–2020 only, 2021 not read):** the rule was written before any fit ([docs/uncertainty/HIERARCHY.md](uncertainty/HIERARCHY.md)). The split is 5-fold CV over the 51 floats, stratified by region. There are three regions, because no held-out float lies south of 0°.
  - **Signal in the Bay of Bengal 50–300 m:** small. Monsoon phase has none. The U-Net's predicted stratification and the global layer's log sd do: CV R² on log r² rises from 0.010 (depth only) to 0.013 and 0.018. Sea-surface salinity is untested (it needs one Kaggle collect).
  - **NUTS reference (numpyro, 92/186/380 m):** τ is well identified (log τ about −2.3 to −3.1) and the posterior is not a funnel. Centred NUTS breaks at 380 m (252 divergences).
  - **BaM centred:** right location for log τ, sd 0.53–0.68× NUTS, which is the funnel verdict by the rule.
  - **BaM non-centred:** sd about 0.1× NUTS even when started at the answer, and unstable in CV.
  - **Laplace:** wrong location for log τ (non-centred joint mode).
  - **Scores:** the centred-BaM hierarchy meets the "sharpens" rule in the Bay of Bengal 50–300 m. Mean sd 0.746 vs 0.823 (global) and 0.834 (constant), 95% coverage 0.95, CRPS skill 3.7% vs 3.0%; the paired difference CI is −0.0045 to −0.00005 °C, just excluding 0.
  - **Email question:** parameterisation sensitivity of BaM in hierarchical models (non-centring made it worse here).
  - **Cross-check at 380 m (same day):** the official GSM-VI BaM shows the same non-centred collapse (sd of log τ 0.06–0.25× NUTS, from three starts). The score-based divergence is about 100× lower at the collapsed solutions (2.6–7.3) than at the NUTS-matched Gaussian (491). So the cause is the objective, not the optimiser.
