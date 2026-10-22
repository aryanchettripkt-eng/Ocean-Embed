# PRD: AI / ML Pipeline — OceanScope AI

**Status:** Draft v4 (2026-10-22): v3 (richer inputs, ablations, ARMOR3D baseline, ARGO fine-tuning) + long sea-level history, climate-mode indices, temporal-consistency and per-year bias metrics  
**Lives in:** `ml/` of the monorepo (see `AGENTS.md`). Python package `oceanscope_ml` at `ml/src/oceanscope_ml/`.  
**Owner:** Aryan  
**Implementer:** Coding agent  
**Source of truth for science decisions:** `OceanScope_AI_Project_Handoff.md` (v2). Where this PRD and the handoff disagree, **this PRD wins** for implementation details. Raise a question instead of resolving a conflict yourself.

---

## 0. Rules for the implementer (read first, non-negotiable)

1. **Do not invent facts about the data.** Variable names, units, coordinate names, depth values, time ranges and fill values must come from inspecting the actual files (`oceanscope_ml.data.inspect` output). If a file is missing, stop and report; do not generate synthetic data unless a task explicitly says "synthetic fixture for tests".
2. **Do not invent results.** Never write a metric, score, or plot caption that was not produced by running code. Do not hardcode numbers into README, dashboard, or assistant prompts.
3. **Test split is sealed.** No code path may read test-period data except `oceanscope_ml.evaluation.final_test`, which runs once, at the end, when the owner asks. Hyperparameter tuning, early stopping and model selection use the validation split only.
4. **Training-only statistics.** Normalization stats and the climatology are computed from the training period only. A unit test enforces this.
5. **No silent NaN filling.** Missing values stay NaN in stored arrays and are handled by explicit masks. Filling with 0 is only allowed on *normalized* model inputs, and only together with a mask channel.
6. **Every tunable lives in `ml/configs/experiment.yaml`.** No magic numbers for region, depths, dates, paths or hyperparameters inside the package.
7. **Code lives in `ml/src/oceanscope_ml/`; notebooks (`ml/notebooks/`) are thin.** Notebooks may only: install the package, set environment variables/paths, call entry points, display outputs. No model or preprocessing logic in notebooks.
8. **Do not change a contract in Section 6** (file layouts, variable names, dims, units) without the owner's approval. The backend and frontend depend on them.
9. **Do not add dependencies** outside `ml/pyproject.toml` without stating why in the PR/commit message.
10. **Small, verifiable steps.** Each task in Section 9 ends with its acceptance check passing. Do not start the next milestone with the previous one's checks failing.
11. When unsure, **stop and ask**. Write the question in `docs/OPEN_QUESTIONS.md` and pause that task.
12. **Credentials never touch git.** Copernicus Marine / NASA Earthdata / Hugging Face tokens come from environment variables or Colab/Kaggle secrets only.

---

## 1. Goal

Produce daily 3D temperature fields (15 depths, 0.25° grid) over the North Indian Ocean from surface/dynamics inputs, with honest, reproducible evaluation, saved in a fixed format the backend serves.

### In scope
- Scripted data download, inspection, preprocessing, alignment, climatology.
- Baselines: climatology; per-pixel ridge or LightGBM.
- Main model: small U-Net predicting normalized anomalies.
- Training on Google Colab / Kaggle with resumable checkpoints.
- Evaluation vs. GLORYS (validation, then test once) and vs. ARGO.
- Batch inference to the prediction store (Section 6) and publishing it to the Hugging Face Hub dataset the backend reads.

### Out of scope (unless the owner adds it)
- Transformers, diffusion models, large foundation models.
- Uncertainty estimation, temporal models, SST input (stretch, Section 10).
- Any live/online inference in the web app.
- LLM/Muse Spark integration (belongs to the backend PRD).

---

## 2. Fixed decisions (`ml/configs/experiment.yaml`)

Values marked **[CONFIRM]** are defaults the owner must confirm after Milestone 1 inspection. The agent must not change them on its own.

```yaml
project: oceanscope-ai
dataset_version: v1                 # bump when preprocessing output changes

region:
  lon_min: 40.0   # [CONFIRM]
  lon_max: 100.0  # [CONFIRM]
  lat_min: 0.0    # [CONFIRM]
  lat_max: 30.0   # [CONFIRM]
  lon_convention: "-180_180"
  lat_order: ascending

grid:
  resolution_deg: 0.25
  # cell centres: lon_min + 0.125, ..., lat_min + 0.125, ...

time:
  start: "2010-01-01"   # [CONFIRM] = intersection of all product coverages
  end:   "2021-12-31"   # [CONFIRM]
  day_boundary: UTC
splits:
  train: ["2010-01-01", "2018-12-01"]   # [CONFIRM]
  val:   ["2019-01-01", "2020-12-01"]   # [CONFIRM]
  test:  ["2021-01-01", "2021-12-31"]   # [CONFIRM] sealed
  gap_days: 30                           # dropped at the end of train and val

depths_m: []   # [CONFIRM] exactly 15 values copied from GLORYS native depth coordinate, ascending, last ~1000 m

inputs:
  core: [sla, adt, sst, sss, u_current, v_current, u_wind, v_wind, wind_stress_curl]
  forcing: [net_heat_flux, precip]          # ERA5 daily means
  static: [bathymetry, woa_clim_profile]    # bathymetry (GEBCO, log-depth); WOA23 temperature at the 15 depths for that month
  extra: [doy_sin, doy_cos, lat, lon, ocean_mask]
  history_days: 7          # stack the dynamic channels (core + forcing) for days t-6..t; [CONFIRM] try 7, 14, 30 in ablations
  long_history:            # slow ocean memory (Rossby/Kelvin waves move the thermocline over weeks-months)
    vars: [sla, adt]
    days: 60               # [CONFIRM] 30 or 60 after ablation
    stride: 5              # frames at t-5, t-10, ..., t-60 (12 frames; days t..t-6 already in history_days)
    reduce: mean           # each frame = mean over its 5-day window, so frames are not noisy single days
  indices: [nino34, dmi, mjo_rmm1, mjo_rmm2]   # daily scalars, broadcast to the grid as constant channels (day t only)
  ablation_groups:         # each run removes one group; see §5.7
    sst: [sst]
    sss: [sss]
    dynamics: [u_current, v_current]
    winds: [u_wind, v_wind, wind_stress_curl]
    forcing: [net_heat_flux, precip]
    static: [bathymetry, woa_clim_profile]
    history: history_days=1
    long_history: long_history.days=0
    indices: [nino34, dmi, mjo_rmm1, mjo_rmm2]

climatology:
  method: harmonic        # mean + first 3 annual harmonics, per cell and depth
  n_harmonics: 3

paths:                    # overridden by env vars OCEANSCOPE_DATA_DIR, OCEANSCOPE_OUT_DIR, OCEANSCOPE_CKPT_DIR
  raw: data/raw
  processed: data/processed
  outputs: outputs
  checkpoints: checkpoints

train:
  seed: 42
  model: unet
  base_channels: 32
  depth_levels: 4
  batch_size: 8
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs: 80
  early_stopping_patience: 10
  amp: true
  num_workers: 2

inference:
  publish_splits: [val, test]   # dates written to the published prediction store (keeps it ~1-2 GB)
  publish_dtype: float16

publish:
  hf_dataset_repo: ""          # [CONFIRM] e.g. <hf-user>/oceanscope-outputs; token from env HF_TOKEN
```

---

## 3. Data sources (inputs vs. targets)

| Key | Product | Role | Native res. (verify) | Notes |
|---|---|---|---|---|
| `sla` | DUACS L4 SLA | input | 0.25° daily | Use `sla` variable; record reference period. |
| `u_current`, `v_current` | OSCAR | input | 0.25° or 1/3° (version-dependent) | Bilinear to grid if coarser. |
| `u_wind`, `v_wind` | CCMP | input | 0.25°, 6-hourly | Daily mean of the 4 timesteps; require ≥3 valid. |
| `thetao` | GLORYS12 reanalysis | target | 1/12°, 50 depths | Area-mean of valid ocean sub-cells to 0.25°; cell valid if ≥50% sub-cells valid. |
| `sst` | OSTIA L4 SST (Copernicus) | input | 0.05° daily | Area-mean to 0.25°; convert K → °C. |
| `sss` | Copernicus MULTIOBS SSS (or SMAP L4) | input | 0.125°–0.25°, daily/weekly | If weekly, linear-interpolate in time and document it. |
| `adt` | DUACS L4 ADT | input | 0.25° daily | Same product as SLA. |
| `wind_stress_curl` | derived from CCMP | input | 0.25° | Compute stress with a bulk formula (ρ_air=1.22, Cd per Large & Pond), then curl with centred differences, NaN at coast. |
| `net_heat_flux`, `precip` | ERA5 single levels (Copernicus CDS) | input | 0.25° hourly | Daily mean/sum. Needs CDS account. |
| `bathymetry` | GEBCO | static input | 15″ | Area-mean to 0.25°; use log(1+depth). |
| `woa_clim_profile` | World Ocean Atlas 2023 temperature | static input (by month) | 0.25° monthly | Interpolate to the 15 depths; 15 channels. |
| `nino34` | NOAA CPC/PSL Niño 3.4 SST anomaly index | input (scalar, broadcast) | daily (or weekly, linearly interpolated) | ENSO state. Its source region is in the Pacific, **outside our box**, so this is information the model can't otherwise see (atmospheric bridge + Indonesian Throughflow). |
| `dmi` | Dipole Mode Index (IOD), computed by us from OSTIA | input (scalar, broadcast) | daily | West box 50–70°E 10°S–10°N minus east box 90–110°E 10°S–0°, SST anomaly vs the **train-period** OSTIA climatology. Needs a small extra OSTIA download covering 10°S–0° (outside the main bbox). The IOD strongly tilts the Indian Ocean thermocline. |
| `mjo_rmm1`, `mjo_rmm2` | BoM Real-time Multivariate MJO index (Wheeler & Hendon) | input (scalar, broadcast) | daily | Intraseasonal convection/wind bursts that drive equatorial Kelvin waves. Public text file, no account. |
| ARMOR3D | Copernicus multi-observation 3D T | external baseline only | 0.25° weekly/daily | Compared at ARGO points; never an input. |
| ARGO | QC'd profiles (e.g. Argo GDAC / Copernicus INSITU) | evaluation; optional fine-tuning (§10) | point profiles | Use QC flags 1–2 only. Floats used for fine-tuning are never used for evaluation. |

**Rejected inputs (do not add):** atmospheric CO₂ or any other "global warming" scalar, and a calendar-year/trend channel. They are near-linear in time, carry no information that SST/SLA don't already contain, and let the model extrapolate a trend into the sealed test year instead of reading it from observations. Warming is handled by the observed inputs plus the per-year bias check in §5.8.

**Optional (ablation only, if time allows):** daily river discharge for the Ganges–Brahmaputra, Irrawaddy and Godavari mouths from GloFAS (Copernicus CDS, same account as ERA5), as scalar channels. SSS already captures most of this signal; keep it only if the ablation shows a gain in the Bay of Bengal upper 100 m.

**Leakage caveat (must appear in every results report):** GLORYS assimilates altimetry and ARGO; skill vs. GLORYS is not skill vs. the real ocean.

---

## 4. Pipeline stages and entry points

Each stage is a CLI module run from `ml/`: `python -m <module> --config configs/experiment.yaml`. Each stage is idempotent (re-running overwrites its own output only) and logs to `outputs/logs/<stage>_<timestamp>.log`.

| # | Module | Reads | Writes |
|---|---|---|---|
| 0 | `oceanscope_ml.data.download` | credentials (env), config | `data/raw/<product>/...` + `data/raw/MANIFEST.json` (file list, sizes, checksums, product IDs/versions) |
| 1 | `oceanscope_ml.data.inspect` | `data/raw/**` | `outputs/reports/data_inspection.md` + `.json` |
| 2 | `oceanscope_ml.data.preprocess` | raw files, config | `data/processed/oceanscope_<dataset_version>.zarr` (Section 6.1) |
| 3 | `oceanscope_ml.models.baselines` | processed store (train+val) | `outputs/metrics/baseline_<name>_val.csv`, baseline model files |
| 4 | `oceanscope_ml.training.train` | processed store (train+val) | checkpoints, `outputs/metrics/runs.csv`, `outputs/metrics/<run_id>_val.csv` |
| 5 | `oceanscope_ml.evaluation.evaluate` | processed store (val), checkpoint | per-depth/season/region metrics CSVs (Section 6.3) |
| 6 | `oceanscope_ml.evaluation.argo_validation` | ARGO profiles, prediction store | `outputs/metrics/argo_<run_id>.csv`, `argo_matches.parquet` |
| 7 | `oceanscope_ml.inference.predict` | processed store, checkpoint | `outputs/predictions/predictions_<run_id>.zarr` (Section 6.2) |
| 8 | `oceanscope_ml.evaluation.final_test` | test split (only here) | `outputs/metrics/<run_id>_test.csv` — run once, on owner's request |
| 9 | `oceanscope_ml.publish` | prediction store, metrics, manifest | uploads to `publish.hf_dataset_repo` (Section 6.4) |

### 4.1 Download stage requirements
- Products and access (verify current product IDs/endpoints during implementation, record them in `data/raw/MANIFEST.json`):
  - GLORYS12, DUACS SLA + ADT, OSTIA SST, MULTIOBS SSS, ARMOR3D: Copernicus Marine Toolbox (`copernicusmarine` Python package), subset server-side to the region bbox (+1° margin), date range and (GLORYS) the 15 depths ± neighbours. OSTIA also needs the extra 10°S–0°, 50–110°E strip for the DMI. Needs `COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD`.
  - The history windows need inputs from **60 days before** `time.start` (and before each split start); download that lead-in too.
  - OSCAR: NASA PO.DAAC (Earthdata login, `earthaccess` package).
  - CCMP: Remote Sensing Systems.
  - ERA5 (and optional GloFAS): Copernicus CDS (`cdsapi`, key in env `CDSAPI_URL`/`CDSAPI_KEY`), area-subset requests, one month per request.
  - GEBCO (bbox subset), WOA23 monthly 0.25° temperature: one-off files.
  - Climate indices: Niño 3.4 from NOAA CPC/PSL, MJO RMM from the Australian Bureau of Meteorology; plain text files, no account. DMI is computed from OSTIA (§3). Record the exact URLs in `MANIFEST.json`.
  - ARGO: Argo GDAC (or Copernicus INSITU), profiles inside the bbox and date range.
- Server-side subsetting is mandatory where the service supports it; never download global files when a subset is possible.
- Resumable: skip files already present with matching size/checksum.
- `--dry-run` prints planned requests and estimated sizes without downloading.
- Start with **one month** (`--start 2019-01-01 --end 2019-01-31`) for M1 development before the full period.

---

## 5. Model specification

### 5.1 Input tensor
`X: float32 [B, C, H_pad, W_pad]`. Channel order is generated by one function, `oceanscope_ml.data.channels.channel_list(config)`, and saved in every checkpoint; nothing else may hardcode it. Order: dynamic channels (core + forcing) for each history day oldest→newest, then static channels, then extra channels. Long-history frames (oldest→newest) follow the dynamic channels, then index channels, then static, then extra. With defaults: 11 dynamic × 7 days + 2 × 12 long-history frames + 4 indices + 1 bathymetry + 15 WOA + 5 extra = 126 channels.  
- Core/forcing/static channels: z-scored with train-period per-channel mean/std (scalar per channel, ocean cells only), then NaN → 0.  
- Index channels: z-scored with train-period mean/std of the index, constant over the grid; NaN not allowed (fail loudly if an index has gaps after interpolation).
- `lat`, `lon`: min-max scaled to [-1, 1] over the region.  
- `ocean_mask`: 1 where the GLORYS surface cell is valid, else 0.  
- Pad H and W with zeros (mask 0) to the next multiple of 2^depth_levels; crop outputs back.

### 5.2 Target tensor
`Y: float32 [B, 15, H, W]` = `(thetao − clim) / sigma_d`, where `clim` is the harmonic climatology for that day-of-year, cell, depth, and `sigma_d` is the train-period std of the anomaly at depth d (ocean cells). NaN where the target is invalid.  
`M: bool [B, 15, H, W]` = target valid.

### 5.3 Architecture
Plain 2D U-Net: `depth_levels` down/up stages, each stage = 2 × (Conv3×3 → GroupNorm(8) → SiLU), MaxPool down, bilinear upsample + conv up, skip connections by concatenation, final Conv1×1 → 15 channels. Parameters must be ≤ 10 M (assert in tests). No attention blocks in v1.

### 5.4 Loss
Masked MSE over normalized anomalies:  
`L = sum(M * (Ŷ − Y)²) / sum(M)` (Y's NaNs replaced by 0 under M=0). No depth weighting in v1.

### 5.5 Training loop requirements
- AMP when CUDA is available; deterministic seeding (`torch`, `numpy`, `random`).
- Checkpoint every epoch to `<ckpt_dir>/<run_id>/last.pt` (model, optimizer, scheduler, scaler, epoch, best_val, RNG states, config hash) and best to `best.pt`.
- On start, if `last.pt` exists for the given `--run-id`, resume from it. `--run-id` defaults to `<date>_<model>_<short_git_hash>`.
- Scheduler: cosine or ReduceLROnPlateau (pick one, document it).
- Early stopping on validation masked RMSE in °C (not normalized loss).
- Append one row per finished run to `outputs/metrics/runs.csv`: `run_id, git_commit, dataset_version, config_hash, epochs, best_epoch, val_rmse_c_mean, val_mae_c_mean, wall_minutes, gpu_name, torch_version`.

### 5.6 Baselines (must exist before U-Net results are reported)
1. **Climatology:** prediction = `clim`.
2. **Per-pixel regression:** features = core + forcing inputs for day t + static + doy_sin/cos + lat/lon at the cell; targets = 15 anomalies. Ridge (required) and LightGBM (optional). Train on a random subsample of train-period ocean cells (≤ 5 M rows) to fit in Colab RAM.
3. **ARMOR3D (external):** evaluated only at ARGO matchups (§7), alongside GLORYS.


### 5.7 Ablations (required for the evaluation report)
Train the full model, then one run per `inputs.ablation_groups` entry with that group removed (same seed, same epochs budget). Report validation RMSE per depth for each, as `outputs/metrics/ablation_<run_id>.csv` with columns `removed_group, depth_m, rmse_c, delta_rmse_c`. Also report a history-length sweep (1, 7, 14, 30 days) if Colab/Kaggle time allows. This table is the main scientific evidence of which inputs matter.

### 5.8 Temporal consistency and trend checks (required for the evaluation report)
Written by `evaluate.py` on validation dates only (test only in `final_test.py`). These files are ML-internal, not part of the §6 contract or the manifest.
1. **Day-to-day flicker** → `outputs/metrics/<run_id>_val_temporal.csv`, columns `model, depth_m, n, tend_rmse_pred_c, tend_rmse_glorys_c, tend_err_rmse_c, lag1_autocorr_pred, lag1_autocorr_glorys`, one row per depth plus `all`. Tendency = T(t) − T(t−1) per valid cell and depth, using only consecutive valid days. `tend_err_rmse_c` = RMS of (Δpred − ΔGLORYS). Compute for the U-Net and every baseline. **Flag in the report** any depth where `tend_rmse_pred_c > 1.5 × tend_rmse_glorys_c` (the model jumps around more than the ocean does). If flagged at ≥ 3 depths, that is the trigger for stretch item §10.2 (temporal model).
2. **Per-year bias (warming check)** → `outputs/metrics/<run_id>_bias_by_year.csv`, columns `model, split, year, depth_m, n, bias_c, rmse_c`, for train years (from a held-in sample, labelled `split = train`) and val years. Because the target is an anomaly from a **fixed** 2010–2018 climatology, a warming ocean shows up as growing positive anomalies. A bias trend across years means the model is damping the trend; report it with the slope (°C/yr) per depth.
3. Plot one 90-day time series at three fixed points (Arabian Sea, Bay of Bengal, equatorial; coordinates in config) at 0, 100 and 300 m: pred vs GLORYS → `outputs/figures/timeseries_<run_id>.png`.

---

## 6. Data contracts (frozen; backend depends on these)

### 6.1 Processed store `data/processed/oceanscope_<dataset_version>.zarr`
| Variable | Dims | dtype | Units | Notes |
|---|---|---|---|---|
| `sla`, `u_current`, `v_current`, `u_wind`, `v_wind` | (time, lat, lon) | float32 | m, m s⁻¹, m s⁻¹, m s⁻¹, m s⁻¹ | raw physical units, NaN = missing |
| `thetao` | (time, depth, lat, lon) | float32 | °C | GLORYS on common grid |
| `clim` | (dayofyear, depth, lat, lon) | float32 | °C | dayofyear 1..366, train only |
| `ocean_mask` | (lat, lon) | bool | – | |
| `target_valid` | (time, depth, lat, lon) | bool | – | |
| coords `time`, `lat`, `lon`, `depth` | | datetime64[ns], float64 | –, degrees_north, degrees_east, m | |

Attributes: `dataset_version`, `config_hash`, `created_utc`, `source_files` (JSON list), `norm_stats` (JSON: per-channel mean/std, per-depth sigma), `split_dates` (JSON). float16 copy allowed for upload, as `oceanscope_<v>_f16.zarr`, same schema.

### 6.2 Prediction store `outputs/predictions/predictions_<run_id>.zarr`
Dates = those in `inference.publish_splits` (default val + test). Stored dtype for `temp_pred`/`temp_glorys` = `inference.publish_dtype` (float16 default); consumers must cast to float32. Chunking: `(time=1, depth=15, lat=all, lon=all)` so one day is one chunk.
| Variable | Dims | dtype | Units |
|---|---|---|---|
| `temp_pred` | (time, depth, lat, lon) | float16/float32 | °C (absolute, climatology added back) |
| `temp_glorys` | (time, depth, lat, lon) | float16/float32 | °C (reference; present for all published dates) |
| `coverage` | (time, lat, lon) | int8 | 0 = land/unsupported, 1 = valid prediction, 2 = insufficient inputs (prediction masked) |

Attributes: `run_id`, `dataset_version`, `model_name`, `created_utc`, `split_of_each_date` (JSON mapping date ranges → train/val/test), `depths_m`.  
A prediction cell is "insufficient inputs" if any of `sla`, `sst`, `u_wind`, `v_wind` is invalid on day t, or fewer than 7 of the 9 core inputs are valid at that cell; `temp_pred` is NaN there.

### 6.3 Metrics files `outputs/metrics/`
- `<run_id>_val.csv`, `<run_id>_test.csv`, `baseline_<name>_val.csv`: columns `model, split, depth_m, n, mae_c, rmse_c, bias_c, skill_vs_clim_pct`. One row per depth plus a row with `depth_m = all`.
- `<run_id>_val_by_season.csv`: same + `season` (DJF, MAM, JJA, SON).
- `<run_id>_val_by_region.csv`: same + `region` (`arabian_sea`: lon<78; `bay_of_bengal`: lon≥78 and lat≥5; `equatorial`: lat<5).
- `argo_<run_id>.csv`: columns `depth_m, n_profiles, n_obs, mae_c, rmse_c, bias_c, glorys_mae_c, glorys_rmse_c` (GLORYS vs. ARGO at the same points, for context).
- `manifest.json` (exact schema, the backend validates it):
```json
{
  "schema_version": 1,
  "synthetic": false,
  "active_run_id": "2026-10-30_unet_ab12cd3",
  "dataset_version": "v1",
  "model_name": "unet",
  "created_utc": "2026-10-30T12:00:00Z",
  "region": {"lon_min": 40.0, "lon_max": 100.0, "lat_min": 0.0, "lat_max": 30.0},
  "grid_resolution_deg": 0.25,
  "depths_m": [0.49, "... 15 values ..."],
  "available_dates": {"start": "2019-01-01", "end": "2021-12-31"},
  "splits": {"train": ["2010-01-01","2018-12-01"], "val": ["2019-01-01","2020-12-01"], "test": ["2021-01-01","2021-12-31"]},
  "test_evaluated": false,
  "files": {
    "predictions": "predictions/predictions_<run_id>.zarr",
    "metrics_val": "metrics/<run_id>_val.csv",
    "metrics_val_by_season": "metrics/<run_id>_val_by_season.csv",
    "metrics_val_by_region": "metrics/<run_id>_val_by_region.csv",
    "metrics_test": null,
    "baselines_val": ["metrics/baseline_climatology_val.csv", "metrics/baseline_ridge_val.csv"],
    "argo": "metrics/argo_<run_id>.csv",
    "argo_matches": "metrics/argo_matches_<run_id>.parquet"
  }
}
```
All paths are relative to the manifest's directory. `null` means not produced yet.

### 6.4 Published layout (Hugging Face Hub dataset repo)
```
manifest.json
predictions/predictions_<run_id>.zarr/...
metrics/*.csv, metrics/*.parquet
reports/evaluation_<run_id>.md
```
The backend downloads exactly this tree. Publishing a new run = upload new files, then overwrite `manifest.json` last.

Bias = mean(pred − reference). skill_vs_clim_pct = 100 × (1 − RMSE_model / RMSE_clim).

---

## 7. ARGO validation spec
- Profiles: QC flags 1–2 for TEMP and PRES/DEPTH; position flag 1–2; within region and the prediction date range.
- Matching: nearest grid cell (cell centre within 0.125°) and same UTC day. Interpolate the model profile (15 levels) linearly in depth to observed depths within [min(depths_m), max(depths_m)]; no extrapolation. Then bin errors to the nearest of the 15 depths for reporting.
- Report n profiles, n observations, MAE/RMSE/bias per depth, and the same metrics for GLORYS vs. ARGO and ARMOR3D vs. ARGO.
- **Float-level hold-out:** randomly assign 20% of ARGO floats (by `platform_number`, fixed seed) to `argo_eval_floats.json`. These are the only floats used for ARGO metrics. The remaining floats may be used for fine-tuning (§10).
- Output `argo_matches_<run_id>.parquet`: `platform_number, cycle, time, lat, lon, depth_m, temp_obs_c, temp_pred_c, temp_glorys_c`.
- The report must state that ARGO data is assimilated in GLORYS and therefore is not fully independent.

---

## 8. Required tests (`ml/tests/`, run with `pytest -q` from `ml/`, CPU only, < 60 s)
Use small synthetic fixtures (e.g. 3 days × 16 × 16 grid) generated in `tests/conftest.py`.
1. `test_shapes`: dataset item shapes match Section 5.1/5.2; U-Net output shape equals target shape for odd H, W (padding/crop works).
2. `test_mask_loss`: loss ignores masked cells (changing masked target values does not change the loss).
3. `test_no_test_leakage`: dataset/stat functions raise if asked for test dates outside `final_test`; norm stats computed only from train dates.
4. `test_climatology`: harmonic fit on a pure sinusoid recovers it within 1e-3.
5. `test_inverse_transform`: normalize → denormalize round-trip within 1e-5.
6. `test_param_count`: U-Net with config defaults ≤ 10 M params.
7. `test_prediction_contract`: `predict` on the fixture writes a store matching Section 6.2 exactly (names, dims, dtypes, attrs).
8. `test_resume`: train 1 epoch, kill, resume → epoch counter and optimizer state continue.
9. `test_manifest_schema`: generated `manifest.json` validates against the Section 6.3 schema (`ml/src/oceanscope_ml/schemas/manifest.schema.json`, JSON Schema, shared with the backend by copy at `backend/app/schemas/manifest.schema.json`; a CI check diffs the two).

### 8.1 Shared sample outputs (for backend and frontend development)
`python -m oceanscope_ml.fixtures.make_sample_outputs --out ../fixtures/sample_outputs` writes a tiny but contract-exact output tree (Section 6.4 layout: 10 days, real region bbox at 1° spacing, 15 plausible depths, synthetic but physically plausible temperatures decreasing with depth, a land mask, some `coverage = 2` cells, and metrics CSVs with an obvious `SYNTHETIC` model name). It must not import torch. The output is committed to git (< 5 MB) so the backend and frontend can be built before any real data exists. **Synthetic values must never be presented as real results:** `manifest.json` carries `"synthetic": true`, and the backend/frontend show a banner when it is set.

---

## 9. Milestones and acceptance criteria

| M | Tasks | Done when |
|---|---|---|
| M1 Data | `download.py` (one month first), `inspect.py`, `preprocess.py`, climatology, config confirmed | Inspection report exists; owner has confirmed all [CONFIRM] values; processed store passes a schema check; one day plotted for every variable in `outputs/figures/m1_sample_day.png`; tests 3–5 pass |
| M2 Baselines + U-Net | `baselines.py`, `dataset.py`, `unet.py`, `train.py`, Colab + Kaggle notebooks | Baseline val CSVs exist; U-Net trains on Colab and Kaggle; resume works across a session restart; U-Net val RMSE < climatology val RMSE at ≥ 10 of 15 depths (if not, report and stop — do not tune on test) |
| M3 Evaluation | `evaluate.py`, `argo_validation.py` | All Section 6.3 and §5.8 files exist for the best run; evaluation report `outputs/reports/evaluation_<run_id>.md` with tables generated by code and the leakage caveat |
| M4 Inference + publish | `predict.py`, `manifest.json`, `publish.py` | Prediction store for published dates passes the contract test; total published size ≤ 3 GB; manifest validates against 6.3 schema; files visible in the HF dataset repo; backend loads them (backend PRD acceptance) |
| M5 Final test | `final_test.py` | Run once on owner's instruction; results added to the evaluation report, clearly labelled |

---

## 10. Stretch (only on owner request, in this order)
1. **ARGO fine-tuning:** after GLORYS training, fine-tune with loss = masked MSE vs GLORYS + λ × MSE vs ARGO profiles (from training-period, non-held-out floats, interpolated to the 15 depths at matched cells/days). Tune λ on validation-period ARGO from non-held-out floats only. Report the held-out-float ARGO metrics before and after. This is the only way to beat GLORYS against real observations.
2. Temporal model (ConvLSTM over the history frames, or a recurrent U-Net) **only if** §5.8 flags flicker at ≥ 3 depths or the history ablation shows longer history keeps helping. Do **not** feed the model's own previous-day prediction back as an input in v1: errors compound over long runs and training needs unrolled sequences, which is costly on Colab.
3. MC-dropout or ensemble uncertainty → `temp_pred_std` variable added to 6.2 (contract change, needs approval).

---

## 11. Colab / Kaggle runbook (the notebooks must implement exactly this)
1. Set env vars: `OCEANSCOPE_DATA_DIR`, `OCEANSCOPE_OUT_DIR`, `OCEANSCOPE_CKPT_DIR` (Drive path on Colab; `/kaggle/input/...` read-only data and `/kaggle/working/...` outputs on Kaggle).
2. `pip install -q "git+<repo_url>@<branch>#subdirectory=ml"` (or `pip install -e` from Drive).
3. Print `torch.__version__`, GPU name, git commit.
4. `python -m oceanscope_ml.training.train --config configs/experiment.yaml --run-id <id>` (resumes automatically).
5. Copy `outputs/metrics/*` and `best.pt` to persistent storage at the end of each epoch (done by train.py, not the notebook).

---

## 12. Open questions for the owner
- Create accounts: Copernicus Marine, NASA Earthdata, Hugging Face (for the published outputs).
- Exact region and 15 depths (after M1 inspection).
- Which ARGO source to use (GDAC vs. Copernicus INSITU).
- Is the processed dataset small enough for a Kaggle Dataset (limit ~100 GB per dataset at time of writing; verify)?
