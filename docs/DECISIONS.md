# Decisions log

Format: `## YYYY-MM-DD — title` then decision, reason, who approved.

## 2026-10-22 — Stack and hosting
- Backend: FastAPI (Python 3.11) on Hugging Face Spaces (Docker). Frontend: Next.js on Vercel. Approved by Aryan (stack); hosting split recommended by Claude because Vercel's Python functions don't suit a GB-scale xarray/zarr data store.
- Training on Google Colab / Kaggle.
- Model predicts anomalies relative to a training-period GLORYS climatology; small U-Net; climatology and ridge/LightGBM baselines.
- Streamlit dashboard from the handoff is superseded.
