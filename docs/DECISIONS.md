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
