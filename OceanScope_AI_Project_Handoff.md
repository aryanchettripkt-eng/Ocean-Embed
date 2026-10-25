# Project Handoff: AI-Powered Subsurface Ocean Temperature Reconstruction

**Project name:** OceanScope AI  
**Project type:** AI / Climate Tech / Oceanography  
**Target region:** North Indian Ocean  
**Primary objective:** Reconstruct daily 3D ocean temperature fields from surface and ocean-dynamics data using deep learning.  
**Hackathon priority:** Build a working, scientifically defensible prototype with measurable performance and an interactive dashboard.  
**Training compute:** Google Colab and Kaggle notebooks (free/limited GPU sessions). See Section 4.1.

> **Revision v2 (2026-10-22), engineering review.** Changes from v1: planned for Colab/Kaggle compute (4.1); the model predicts temperature anomalies relative to a GLORYS climatology (5, 6); the patch-centre CNN was dropped in favour of going straight to a small U-Net, with climatology and LightGBM/linear baselines (6, 7); the open decisions (region, depths, split years, regridding) are listed explicitly (5.0); the SLA→GLORYS assimilation leakage caveat was added (3D, 7); the dashboard and assistant read precomputed predictions (8, 9); scope was trimmed and experiment tracking added (2, 11).

---

## 1. Project Overview

### The problem

Satellites provide valuable information about the ocean surface, sea level, and related conditions, but they do not directly measure temperature throughout the water column.

Subsurface temperature is important for understanding:

- Ocean heat content and climate variability.
- Marine heatwaves and subsurface warming.
- Ocean circulation and water-mass structure.
- Changes in the upper 1,000 metres of the ocean.

Direct underwater observations are spatially and temporally sparse. This makes it challenging to produce complete daily maps of subsurface temperature.

### The proposed solution

Build an AI-powered system that learns relationships between sea-level anomalies, ocean currents, winds, and subsurface temperatures from ocean reanalysis data.

The system will generate temperature estimates at 15 depth levels across the North Indian Ocean on a daily 0.25° grid, subject to the coverage and quality of the input data.

The output will be a 3D temperature dataset and interactive website that lets users explore ocean conditions by date, location, and depth.

**Scientific qualification:** The model initially learns to reproduce GLORYS reanalysis temperatures. Independent ARGO evaluation is needed to determine how well those predictions agree with observations.

---

## 2. Project Goals

### Minimum viable product (MVP)

1. Load and align the four available data sources.
2. Prepare daily gridded training examples.
3. Train a small neural network to predict 15 subsurface temperature levels.
4. Evaluate the model on held-out data.
5. Compare predictions with independent quality-controlled ARGO observations where possible.
6. Display temperature maps and prediction errors in an interactive dashboard.

### Stretch goals

- Add satellite sea-surface temperature as an additional input.
- Add recent days of observations for temporal context.
- Implement a U-Net or other fully convolutional architecture.
- Estimate prediction uncertainty.
- Detect potential subsurface warm anomalies.
- Integrate Meta's conversational AI.
- Add optional map-region selection with SAM 3.1.

Do not make all stretch goals prerequisites for the initial demo. Priority order if time remains: (1) Meta conversational assistant over precomputed outputs, (2) SST input, (3) temporal context, (4) uncertainty, (5) SAM. SAM and temporal models are explicitly optional and should be dropped first.

---

## 3. Available Datasets

### A. DUACS SSH / SLA

**Role:** Model input.

Use daily sea-surface height or sea-level anomaly fields on a 0.25° grid, subject to product availability.

Expected feature:

- `sla`: sea-level anomaly.

Verify the selected product's variables, reference conventions, coverage, units, timestamps, and quality flags.

### B. OSCAR ocean currents

**Role:** Model input.

Use horizontal current components:

- `u_current`: zonal current.
- `v_current`: meridional current.

Confirm units, temporal aggregation, spatial coverage, and product version. Depending on the version, OSCAR may be coarser than 0.25° (older products are 1/3°) or not daily; document the regridding and any temporal interpolation used.

### C. CCMP winds

**Role:** Model input.

Use:

- `u_wind`: zonal wind component.
- `v_wind`: meridional wind component.

Check whether the selected product represents instantaneous or averaged winds, and document the time convention and units. CCMP is typically distributed 6-hourly; average to daily means (document the day boundary, e.g. 00–24 UTC) before alignment.

### D. GLORYS temperature reanalysis

**Role:** Supervised training target.

Use temperature fields at the 15 selected depth levels.

The target tensor should contain the temperature at each depth for every grid cell and training date.

Check that the selected depth levels cover the desired surface-to-1,000 m range. Record the actual depth coordinates rather than assuming they are evenly spaced.

GLORYS (GLORYS12) is distributed at 1/12° on 50 depth levels. Choose the 15 levels explicitly (see Section 5.0) and coarsen to 0.25° by averaging the valid ocean cells, not by point sampling.

GLORYS is a reanalysis, not independent ground truth. Training against it teaches the model to approximate GLORYS.

**Leakage caveat:** GLORYS assimilates along-track satellite altimetry (the same signal behind DUACS SLA) as well as ARGO profiles. The inputs and the target are therefore not independent, and part of the model's apparent skill reflects the assimilation system. State this in the limitations and in any results shown to judges.

### E. ARGO observations

**Role:** Independent evaluation wherever feasible.

Prefer individual quality-controlled ARGO profiles over gridded products for pointwise validation.

For each profile:

- Extract its location and observation time.
- Obtain the observed temperatures and their actual depth coordinates.
- Interpolate predictions to the observed depths, or interpolate the observed profile to target depths when the measurements support it.
- Apply quality control and document missing observations.

Check whether observations used for evaluation may have been assimilated into GLORYS. Where possible, use withheld profiles or floats and separate time periods.

---

## 4. Proposed Technology Stack

| Component | Technology |
|---|---|
| Coding and debugging | Muse Code |
| Training compute | Google Colab / Kaggle notebooks (GPU) |
| Scientific model | Python + PyTorch (mixed precision) |
| Tabular baseline | LightGBM or ridge regression |
| Experiment tracking | Weights & Biases, or a CSV/JSON log in `outputs/` |
| Gridded data processing | Xarray, NumPy |
| Tabular operations | Pandas |
| Numerical preprocessing | Scikit-learn |
| Model evaluation | Scikit-learn, NumPy |
| Visualizations | Plotly |
| Dashboard | Streamlit |
| Conversational AI | Muse Spark through the available Meta Model API |
| Optional visual segmentation | SAM 3.1 |
| Storage | NetCDF or Zarr for gridded data; Parquet for profile tables |

Muse Glimmer may be explored if the hackathon platform provides the required access and its capabilities suit a specific task. Verify the current API, model names, access requirements, and rate limits before implementation.

Meta's tools should complement the scientific model, not replace it.

### 4.1 Compute environment: Colab and Kaggle

Training runs in hosted notebooks, which shapes the design:

- **Sessions end unexpectedly.** Colab sessions last at most about 12 hours (often less), and Kaggle caps GPU time at roughly 30 hours a week with sessions of about 12 hours. Check the current limits. Save a checkpoint every epoch (model, optimizer, scheduler, epoch, best metric, RNG state) to Google Drive or Kaggle output, and make `train.py` resume from the latest checkpoint automatically.
- **Do not download raw data in each session.** Preprocess once (locally or in one dedicated notebook), crop to the study region, and save the aligned inputs, targets, masks and climatology as one compact Zarr/NetCDF store (float16 for model arrays, chunked by time). Upload it as a private Kaggle Dataset and/or a Google Drive folder. Record its version and the config used to build it.
- **Keep logic out of notebooks.** All code lives in the `src/` package. Notebooks only install it (`pip install git+https://...` or from a mounted Drive), set paths, and call entry points such as `python -m src.training.train --config configs/experiment.yaml`.
- **Make the code environment-agnostic.** Read data, output and checkpoint paths from the config or environment variables, so the same code runs on Colab (`/content/drive/...`), Kaggle (`/kaggle/input/...`, `/kaggle/working/...`) and locally.
- **Fit the hardware.** Assume a single T4/P100-class GPU with about 16 GB. Use mixed precision (`torch.autocast`) and a small U-Net (a few million parameters) on full-region daily inputs; this fits comfortably. Load the processed dataset into RAM where possible instead of reading from Drive on each step, because Drive I/O is slow.
- **Pin dependencies.** Keep a `requirements.txt` with versions compatible with the preinstalled Colab/Kaggle PyTorch, and log library versions at the start of every run.

---

## 5. Data Preparation Pipeline

The first major engineering task is to build a reproducible preprocessing pipeline.

### Step 0: Decisions to fix before writing preprocessing code

Record each of these in `configs/experiment.yaml` and do not change them silently afterwards:

| Decision | Proposed default (confirm against the data) |
|---|---|
| Study region bounding box | e.g. 40°E–100°E, 0°–30°N (Arabian Sea + Bay of Bengal); adjust to data coverage |
| Longitude convention | −180…180, latitude ascending |
| 15 GLORYS depth levels | Chosen from the 50 native levels, denser near the surface and thermocline, ending close to 1,000 m; record the exact depths in metres |
| Time period | Intersection of all four input/target products, e.g. 2010–2021 |
| Chronological splits | e.g. train 2010–2018, validation 2019–2020, test 2021 (test is untouched until the final evaluation) |
| Regridding | Conservative/area-mean coarsening for finer products (GLORYS); bilinear interpolation for coarser products (OSCAR if needed), with masks carried through |
| Daily aggregation | Daily means on a stated UTC day boundary |

### Step 1: Inspect the source files

For each dataset, record:

- Variable names and dimensions.
- Coordinate names and units.
- Timestamp conventions.
- Spatial resolution and coverage.
- Missing-value conventions.
- Quality flags.
- Depth coordinates, where applicable.

Do not assume all files use the same coordinate names or calendar conventions.

### Step 2: Define the study region

Restrict all datasets to the selected North Indian Ocean bounding box.

Use a consistent longitude convention and latitude ordering. Account for land cells, coastlines, and missing observations.

### Step 3: Align time and space

Create a common daily timeline and a common 0.25° grid.

For continuous fields, select an appropriate interpolation or regridding method. Preserve masks and avoid silently replacing missing values with zero.

Be particularly careful when regridding near coastlines and when changing resolution from a coarser source product.

### Step 4: Construct the input channels

Initial feature set:

`[SLA, U_current, V_current, U_wind, V_wind]`

Optional features:

- Satellite sea-surface temperature.
- Latitude and longitude.
- Day-of-year sine and cosine.
- Recent observations from previous days.
- A land/ocean mask channel (recommended; cheap and helps near coasts).

Include day-of-year sine/cosine and latitude/longitude in the first U-Net run; they are essentially free.

Latitude, longitude, and seasonal features can help the model learn regional and seasonal patterns.

SLA is not equivalent to sea-surface temperature. Because the current dataset list lacks SST, treat adding SST as a potentially valuable experiment rather than claiming it is already available.

### Step 5: Construct the target

The target contains 15 GLORYS temperature levels.

Conceptually:

- Input: `[C, H, W]` per day (C = 5 core channels plus any optional ones).
- Target: `[15, H, W]` per day.

Here, `H` and `W` are the latitude and longitude dimensions of the selected region.

**Predict anomalies, not absolute temperatures.** Compute a daily GLORYS climatology per grid cell and depth from the training years only (day-of-year mean, smoothed, e.g. with a 31-day window or the first few harmonics). The model's target is `T − T_clim`; at inference, add the climatology back. Most of the variance in absolute temperature is the mean vertical structure and the seasonal cycle, which the climatology already captures. Learning the anomaly gives a much easier problem, and the same climatology doubles as the first baseline.

### Step 6: Normalize

Calculate feature means and standard deviations using training data only.

Apply the same transformations to validation, test, and inference data.

Normalize target anomalies per depth (anomaly variance differs strongly between the surface and 1,000 m). Keep track of the target scaling and the climatology, so model outputs can be converted back to degrees Celsius.

### Step 7: Save prepared data

Use NetCDF or Zarr for large gridded arrays. Store configuration, coordinate definitions, units, and data-quality information alongside the processed data.

The processed store should contain: inputs, targets, masks, the climatology, normalization statistics (training-only), coordinates (including the exact depths), and a version tag. This single artifact is what gets uploaded to Kaggle/Drive (see Section 4.1).

---

## 6. Model Development Strategy

### Phase A: Non-neural baselines

Before any neural network, build two cheap baselines that run on a CPU in minutes:

1. **Climatology:** predict `T_clim` (anomaly = 0).
2. **Per-pixel regression:** a ridge regression or LightGBM model mapping the input features at a grid cell (plus lat, lon and day of year) to the 15 anomaly values.

These set the bar the neural model must clear, and they surface data problems early.

### Phase B: Small U-Net (main model)

Go directly to a small fully convolutional U-Net on whole-region daily maps. A patch-centre CNN (predicting only the centre pixel of each 32 × 32 patch) is not recommended: it recomputes overlapping context for every pixel, is far slower to train and run, and is no simpler to implement than a U-Net with a masked loss.

Input:

`[batch, C, H, W]` (pad H and W to a multiple of 2^(number of downsampling levels))

Output:

`[batch, 15, H, W]`, the normalized temperature anomaly at each target depth.

Keep it small: 3–4 levels, about 32–64 base channels, a few million parameters. A U-Net is a practical starting point because it combines broad spatial context with fine-scale features.

### Phase C: Temporal context

If time permits, provide several consecutive days of input fields rather than just one day.

For example, use a seven-day input window to predict the current day's temperature field, provided the input timing is appropriate for the intended use.

A temporal CNN or compact recurrent architecture can be tested later. Do not begin with a large transformer.

### Loss function

Use masked mean squared error:

\[
L =
\frac{\sum_{i,d}m_{i,d}(\hat T_{i,d}-T_{i,d})^2}
{\sum_{i,d}m_{i,d}}
\]

Where:

- `T` is the GLORYS temperature target.
- `T_hat` is the model prediction.
- `m` indicates valid target values.

Use appropriate masks and reject examples with insufficient valid input coverage. Consider depth weighting only after establishing a baseline.

### Training configuration

A reasonable starting point:

- Optimizer: AdamW.
- Learning rate: 0.001.
- Batch size: 32–256, depending on the architecture and hardware.
- Maximum epochs: 50–100.
- Early stopping patience: 8–10 epochs.
- Weight decay: `1e-4`.
- Random seed: fixed for reproducibility.
- Checkpoint: best validation model, plus a `last` checkpoint every epoch for resuming (Section 4.1).
- Mixed precision (AMP) on GPU.
- Light augmentation only if it is physically valid (no flips: currents and winds have direction).

These are starting values, not guaranteed optimal settings. Tune them against validation performance.

### Experiment tracking

Every run logs its config, dataset version, git commit, per-depth validation metrics and wall-clock time to W&B or to an append-only `outputs/metrics/runs.csv`. Without this, results from different Colab/Kaggle sessions cannot be compared reliably.

---

## 7. Evaluation and Scientific Validation

### Dataset splitting

Do not randomly split neighbouring grid cells into training and test sets. Ocean observations are spatially and temporally correlated.

Use chronological splits, with the years fixed in Section 5.0, for example:

- Training: 2010–2018.
- Validation: 2019–2020.
- Test: 2021, touched only once for the final reported numbers.

Leave a short gap (e.g. 30 days) between splits so the autocorrelation of ocean anomalies does not leak across the boundary.

Also evaluate spatial generalization by holding out geographic blocks or entire ARGO floats when feasible.

Fit all preprocessing statistics on training data only.

### Metrics

Calculate metrics independently at all 15 depths:

- Mean absolute error (MAE), in °C.
- Root mean squared error (RMSE), in °C.
- Mean bias, in °C.
- Number of valid observations.

Evaluate by depth, season, region, and data availability where sample sizes permit.

### Baselines

Compare the model against:

1. The training-period GLORYS daily climatology (Section 5, Step 5).
2. Per-pixel ridge regression or LightGBM on the same inputs.
3. Persistence, where an appropriate previous-day reference exists.

Report skill relative to climatology (e.g. RMSE reduction %) per depth; this is the most informative single number for judges.

The neural network should demonstrate measurable improvement over reasonable baselines.

### ARGO validation

Compare model predictions with held-out ARGO observations at matched locations, times, and depths.

Report:

- The number of profiles and observations.
- Depth-dependent errors.
- Spatial and seasonal coverage.
- Matching tolerances.
- Any known overlap between ARGO observations and GLORYS assimilation.

Do not describe agreement with GLORYS alone as proof of accuracy against the real ocean. Also note that because GLORYS assimilates altimetry (Section 3D), skill against GLORYS partly reflects shared information between the inputs and the target.

---

## 8. Dashboard Specification

Build the first interface with Streamlit and Plotly.

The dashboard does not run the model. It reads precomputed prediction fields, per-depth metrics and coverage masks from `outputs/` (Zarr/NetCDF, plus CSV/Parquet for metrics). Predictions are generated in batch on Colab/Kaggle and downloaded. This keeps the app fast, needs no GPU, and makes it deployable on Streamlit Community Cloud or similar.

### Main screen

**Title:** OceanScope AI — North Indian Ocean Explorer

The interface should include:

1. **Interactive map:** display a selected temperature level.
2. **Date selector:** select a date for which predictions exist.
3. **Depth selector:** select one of the 15 depth levels.
4. **Color scale:** display temperature in °C.
5. **Location inspector:** select a grid cell and display the predicted temperature profile.
6. **Evaluation panel:** show MAE, RMSE, and bias on the held-out evaluation set.
7. **Data coverage indicator:** distinguish valid predictions, missing input regions, and unsupported areas.
8. **Uncertainty display:** add only when a defensible uncertainty estimate has been implemented.

The dashboard must distinguish predicted temperatures from GLORYS training targets and ARGO observations.

Do not imply that the system can produce validated predictions for every day or every grid cell when data coverage is incomplete.

---

## 9. Meta AI Integration

> **Updated 2026-10-24:** the current plan for Meta's tools (Ocean guide with Muse Spark as navigator, voice, event time machine, Muse Image postcards, SAM lasso, beach mode) is in `docs/PRD_BACKEND.md` §6, `docs/PRD_FRONTEND.md` §3.4 and `docs/PHASES.md` P7. This section is background only.

Meta's tools add a conversational and visual interaction layer.

### Muse Code

Use as the coding assistant for:

- Implementing preprocessing.
- Writing PyTorch models.
- Debugging data alignment.
- Building evaluation scripts.
- Creating the Streamlit dashboard.
- Writing tests and documentation.

### Muse Spark / Meta Model API

Connect the conversational assistant to controlled backend functions.

Suggested functions:

- `get_temperature(latitude, longitude, date, depth)`
- `get_temperature_profile(latitude, longitude, date)`
- `get_region_summary(region, date, depth)`
- `get_validation_metrics(depth, region)`

The assistant should retrieve real model outputs through these functions and explain them in plain language. These functions read the same precomputed prediction and metric files as the dashboard; they never run the model live and never call the LLM to produce numbers.

It should not invent temperature values, accuracy scores, or explanations unsupported by the data.

Check current API documentation, model availability, authentication, rate limits, and hackathon access requirements before integration.

### SAM 3.1

Treat SAM as an optional feature.

Possible workflow:

- Render a temperature map.
- Let users select or outline a visual region.
- Convert that selection to a geographic mask.
- Calculate regional temperature statistics using the underlying numerical grid.

SAM produces visual segmentation; it does not independently establish oceanographic boundaries or detect marine heatwaves scientifically. If it is too costly or unreliable for the map type, use a simple geographic selection tool instead.

---

## 10. Repository Structure

Suggested project layout:

```text
oceanscope-ai/
├── README.md
├── requirements.txt
├── pyproject.toml          # makes src/ pip-installable in Colab/Kaggle
├── configs/
│   └── experiment.yaml     # region, depths, splits, paths, hyperparameters
├── notebooks/
│   ├── colab_train.ipynb   # thin wrappers: install package, mount data, call entry points
│   └── kaggle_train.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── inspect.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── baselines.py    # climatology, ridge/LightGBM
│   │   └── unet.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── argo_validation.py
│   └── inference/
│       └── predict.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
└── outputs/
    ├── metrics/
    └── figures/
```

Keep raw datasets out of version control. Store large model checkpoints and the processed dataset on Google Drive or as Kaggle Datasets, not in git.

---

## 11. Recommended Implementation Order

### Milestone 1: Data readiness

- Inspect all dataset sources.
- Fix the Section 5.0 decisions: region, 15 depth levels, period, splits, regridding.
- Align daily timestamps and grid coordinates.
- Verify quality masks and missing values.
- Compute the training-period climatology.
- Produce one complete example input and target tensor.
- Upload the processed store to Kaggle/Drive and verify it loads in a notebook.

**Deliverable:** a documented, reproducible dataset pipeline and a versioned processed dataset.

### Milestone 2: Baselines and the first U-Net

- Evaluate the climatology baseline.
- Train the ridge/LightGBM per-pixel baseline.
- Train the small U-Net on anomalies on Colab/Kaggle, with resumable checkpoints and experiment logging.
- Evaluate against held-out GLORYS validation data.
- Save per-depth metrics and skill relative to climatology.

**Deliverable:** baselines plus a trained U-Net with a comparison table.

### Milestone 3: Independent evaluation

- Match predictions to quality-controlled ARGO profiles.
- Check potential GLORYS assimilation overlap.
- Calculate error metrics by depth.
- Document the evaluation limitations.

**Deliverable:** an honest scientific validation report.

### Milestone 4: Full-grid inference

- Run the trained U-Net in batch over all supported dates.
- Add the climatology back and convert to °C.
- Save output arrays with depth, coordinate and coverage metadata.

**Deliverable:** daily 3D temperature fields.

### Milestone 5: Dashboard

- Add map, date and depth controls.
- Add location-specific profiles.
- Display evaluation metrics and data coverage.
- Make missing data and prediction limitations visible.

**Deliverable:** a working demo that judges can explore.

### Milestone 6: Meta integration

- Connect Muse Spark through the available API.
- Add question-answering over model outputs.
- Add SAM-based visual selection only if time remains.

**Deliverable:** an interactive ocean-science assistant.

---

## 12. Definition of Done

- [ ] All source datasets are documented and aligned.
- [ ] The input and target tensors are reproducible.
- [ ] The model trains without data leakage.
- [ ] Held-out evaluation metrics are reported for all depths.
- [ ] Baseline comparisons are included.
- [ ] ARGO comparisons are conducted and their independence limitations documented.
- [ ] At least one supported daily 3D prediction field can be generated.
- [ ] The model beats the climatology and per-pixel regression baselines on the validation set at most depths.
- [ ] Training resumes cleanly after a Colab/Kaggle session ends.
- [ ] The processed dataset and the best checkpoint are versioned and stored outside git.
- [ ] The dashboard displays temperature maps, depth profiles, and evaluation information.
- [ ] The conversational assistant uses real outputs rather than fabricated scientific values.
- [ ] Setup instructions and known limitations, including the GLORYS assimilation caveat, are documented.

---

## 13. Instructions for the Coding Agent

Build this incrementally. Start by inspecting the actual dataset files and writing a data-validation script. Do not assume variable names, units, file formats, exact depth levels, or time coverage until verified.

Implement the smallest reproducible training experiment first. Assume training happens on Colab/Kaggle (Section 4.1): keep code in `src/`, notebooks thin, paths configurable, and checkpoints resumable. Keep the model, preprocessing, configuration, and evaluation code separate. Write tests for tensor shapes, missing-value masks, normalization, and prediction outputs.

Do not invent evaluation results. Do not use test data for tuning. Do not treat GLORYS as independent ground truth. Prioritize a correct, demonstrable scientific pipeline over a complex architecture or unnecessary agent integrations.

**First task:** inspect the available datasets and generate a short report listing their variables, dimensions, time ranges, coordinate systems, depth levels, missing-value patterns, and proposed common grid. Use that report to implement preprocessing and train the first baseline model.

---

## Final Project Summary

**OceanScope AI** is an AI-powered ocean exploration platform that estimates subsurface temperature profiles across the North Indian Ocean. It uses DUACS sea-level anomalies, OSCAR currents, and CCMP winds as model inputs; GLORYS temperatures as supervised training targets; and ARGO observations for independent evaluation where feasible. A compact U-Net, trained on Colab/Kaggle to predict anomalies from a GLORYS climatology, generates 15-depth temperature maps, while a Streamlit dashboard and Meta AI integration make the predictions accessible to researchers and hackathon judges.

The project's success depends on reliable data engineering, demonstrable predictive skill, transparent evaluation, and a useful interactive interface—not simply on using a sophisticated neural network.
