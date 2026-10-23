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

