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
