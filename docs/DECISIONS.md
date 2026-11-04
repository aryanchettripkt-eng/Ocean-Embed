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
- **CORS configuration**: `CORS_ORIGINS` in `render.yaml` set to `https://ocean-embed.vercel.app,http://localhost:3000` with regex matching for Vercel preview domains in backend middleware.
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
