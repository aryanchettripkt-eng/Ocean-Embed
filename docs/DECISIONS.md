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
