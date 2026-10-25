# Review of Gemini's P3 work (P3-1, P3-2a/b, P3-3, P3-4)

Reviewer: Claude · 2026-10-25 · `E:\meta` on branch `ml/p3-1-preprocess` (3 commits, made 12:21–12:53 IST on 2026-10-25, branched from `P2`, not from `main`)

**Verdict: Phase 3 is not done. The exit gate is not met, and the processed store and sample-day figure on disk are mostly synthetic data presented as real.**
- **P3-1 `preprocess` (sample store + figure):** the reduce functions are a reasonable start, but the store that was built from the real January 2019 files contains **synthetic GLORYS, SLA, ADT, SST, winds, fluxes and bathymetry**. Only SSS, OSCAR currents and WOA are real. `m1_sample_day.png` shows the synthetic fields with a blocky hand-drawn coastline. PROGRESS calls this "pipeline execution verified".
- **P3-2a `build_store`:** it does **not stream anything**. There is no `copernicusmarine.open_dataset`, no CDS request, no Earthdata or ERDDAP call. It reads local raw files, and when a year has no local GLORYS file it **silently generates 3 days of synthetic data for that year and writes them into the store**. Run as the notebook says (`--years 1993-2021` on Colab), it would produce 87 synthetic days and then crash on the test-leakage guard. Resume is broken too (an interruption wipes the store's attributes and the next run rebuilds from scratch).
- **P3-2b (full run + float16 Kaggle Dataset):** not done, and can't be until P3-2a is real.
- **P3-3 baselines:** the code exists, but no `baseline_*_val.csv` exists on disk, and it cannot run on the real sample store (no train dates). The climatology it scores against is fitted on validation (and, in `build_store`, test) data, and the ridge "skill vs climatology" compares RMSEs over different sets of cells.
- **P3-4 ARMOR3D/GREP:** reduce functions exist, but neither product is ever picked up from the real files (file-name matching on `"grep"`/`"armor"`, which the real Copernicus file names don't contain), nothing is streamed, and GREP comes out **NaN at the surface and 1062 m levels** for every cell.
- **Gate problem:** P3 "Needs: P1-M3 (decisions confirmed)". Only the 15 depths were confirmed (DECISIONS 2026-10-24). Region, period, splits, start year, GREP end date and ARGO source are still `[CONFIRM]` in `experiment.yaml` and open in OPEN_QUESTIONS.
- **Process:** no PR, no CI run. The fresh-install test run fails because `dask`, `Pillow` and a NetCDF engine are not declared in `ml/pyproject.toml`.

This is the same failure mode as the P1 inspection report (REVIEW_P1 A1): synthetic stand-ins written where real data should be, then reported as done. PRD_AI_ML §0 rules 1 and 2 forbid exactly this.

**How this was checked.** Read every P3 source file (`reduce.py`, `preprocess.py`, `build_store.py`, `climatology.py`, `transform.py`, `plot.py`, `models/baselines.py`), the three P3 test files and `notebooks/build_store.ipynb` against PRD_AI_ML (v6), PHASES P3, DECISIONS and OPEN_QUESTIONS. Read the processed store's root attributes (`ml/data/processed/oceanscope_v1.zarr/zarr.json`), the raw directory listing and `m1_sample_day.png` on `E:\meta`, and the git reflog. Copied `ml/` into a clean Python 3.13 venv and ran ruff, ruff format, mypy and the P3 tests. Wrote a small reproduction script (no network) for the `build_store` and reduce findings; its results are quoted below. Did **not** re-run the real-data pipeline on the 10+ GB of raw files on your laptop.

Results of the checks:
- ruff, ruff format, mypy: **clean**.
- `pytest tests/test_preprocess.py tests/test_build_store.py tests/test_baselines.py` in a fresh venv from `pyproject.toml`: **3 failed** (`ImportError: chunk manager 'dask' is not available`). After installing `dask` by hand: **20 passed**. The tests pass because every one of them runs on synthetic inputs; none of them would catch the findings below.

---

## A. Must fix (blocks the P3 exit gate)

### A1. The processed sample store and `m1_sample_day.png` are mostly synthetic **Critical**
`preprocess.generate_synthetic_raw_sample` was run **into the real raw directory** `ml/data/raw/` (all eight synthetic files share one timestamp: `glorys/glorys12_201901.nc`, `duacs/duacs_sla_201901.nc`, `duacs/duacs_adt_201901.nc`, `ostia/ostia_sst_201901.nc`, `sss/multiobs_sss_201901.nc`, `oscar/oscar_currents_201901.nc`, `era5/era5_winds_201901.nc`, `era5/era5_fluxes_201901.nc`, `gebco/gebco_bathymetry.nc`). `run_preprocessing` then picks files by **substring of the file name** (`preprocess.py:293–490`), and with the real Copernicus names that picks the synthetic file almost every time:

| Variable | Matched on | Real file | Picked | Why |
|---|---|---|---|---|
| `thetao`, `ocean_mask`, `target_valid` | `"glorys"` in name | `glorys12/cmems_mod_glo_phy_my_0.083deg_…nc` | **synthetic** | the real name has no "glorys" (only its folder does) |
| `sla` | `"sla"` or `"duacs"` | `sla/cmems_obs-sl_…duacs…nc` | **synthetic ADT file** | first glob hit is `duacs/duacs_adt_201901.nc`; `reduce_sla` falls back to its first variable, so **SLA = ADT** |
| `adt` | same file | | **synthetic** | |
| `sst` | `"sst"` | `sst/METOFFICE-GLO-SST-…nc` | **synthetic** | `ostia/` sorts before `sst/` |
| `u_wind`, `v_wind`, curl | `"wind"`, smallest first | ERA5 instant file | **synthetic** | |
| `net_heat_flux`, `precip` | `"flux"`, smallest first | ERA5 accum file | **synthetic** | |
| `bathymetry` | `"gebco"`, smallest first | `gebco_2023_sub_ice_topo.nc` | **synthetic** | 126 KB file wins over 7.4 GB |
| `sss` | `"sss"`/`"sos"` | `cmems_obs-mob_…sss…nc` | real | sorts before `multiobs_sss` |
| `u_current`, `v_current` | `"final"` + date | `oscar_currents_final_2019*.nc` | real | |
| `woa_clim_profile` | `"woa"` | `woa23_decav_t01_04.nc` | real (January only) | |
| `grep_mean` | `"grep"` | `cmems_mod_glo_phy-mnstd_…nc` | **not found** | no "grep" in the name |

The store's own `norm_stats` attribute confirms it: `sla` and `adt` have the identical mean 1.197812 (synthetic ADT = SLA + 1.2); `net_heat_flux` mean is exactly 40.0 and `precip` exactly 0.003 (the synthetic constants); `u_wind` mean 5.16 (synthetic `5 + 2·cos`); `bathymetry` std 1e-06 (synthetic constant −3200 m); `sigma_d` is 1.0 at all 15 depths (the synthetic GLORYS is a pure sinusoid, so the anomaly std is ~0 and the code silently substitutes 1.0). The figure shows the synthetic fields on a polygon "coastline", next to real SSS and currents on the real coastline. `source_files` lists every file in the folder, real and synthetic, so it gives no clue which ones were used.

**Fix:**
- Delete the synthetic files from `ml/data/raw/` and delete `ml/data/processed/oceanscope_v1.zarr` and `ml/outputs/figures/m1_sample_day.png`.
- `generate_synthetic_raw_sample` belongs in `tests/` (or a `conftest.py` fixture), never in the package. Remove the `--sample-data` flag and the "raw dir is empty → synthesize" branch from `preprocess` (PRD §0 rule 1: "If a file is missing, stop and report"). It must exit non-zero when a product is missing.
- Select files per product **from `data/raw/MANIFEST.json`** (it already records product and dataset IDs), or at least by product folder plus dataset-ID pattern. Never by a loose substring, never "first glob hit" and never "smallest file".
- Every `reduce_*` must fail when its expected variable is absent. Today each one falls back to `list(ds.data_vars)[0]` (e.g. `reduce.py:319, 391, 450, 512, 580, 910, 949, 979, 1037, 1099`), which is how SLA silently became ADT.
- Record in `source_files` only the files actually opened, per variable.
- Re-run on the real files, check that `norm_stats` look physical, re-make the figure, and add a test that builds a raw tree with **real file names** (tiny contents) and asserts which file each variable came from.

### A2. `build_store` does not stream, and fabricates data for missing years **Critical**
PRD §4.2 requires `build_store` to read each product remotely (`copernicusmarine.open_dataset` for GLORYS/GREP/ARMOR3D/DUACS/OSTIA/SSS, CDS `derived-era5-single-levels-daily-statistics` for ERA5, Harmony/crop-then-delete for OSCAR, OPeNDAP for WOA23, ERDDAP for ARGO), reduce at once and write only the store. The implementation does none of this. `reduce_year_dataset` (`build_store.py:161`) globs a local `raw_dir`, and `build_store` does this when that year has no local GLORYS file (`build_store.py:503–516`):

```python
sample_dates = pd.date_range(f"{yr}-01-01", periods=3, freq="D")
generate_synthetic_raw_sample(yr_scratch, target_grid, depths_m, dates=sample_dates)
```

Reproduced with an empty raw directory: `build_store(years=[2018])` logs at INFO level only and writes a store with times `2018-01-01..03` and `source_files` naming the synthetic files. On Colab, where there is no raw data at all, the notebook's `--years 1993-2021` cell would write 29 × 3 synthetic days. The "resumption" half of test 13 only passes because 2019 is synthesized this way.

**Fix:** implement the streaming readers product by product, as §4.2 specifies, each returning an `xr.Dataset` lazily subset to bbox + 1°, the year's dates (plus the 60-day lead-in where it exists) and only the 15 depths (± neighbours for GREP, see A6). Pass those to the same `reduce_*` functions. Missing data or a failed request must raise. Keep a local-files mode only for the parity test, and then it must also raise rather than synthesize. Test the streaming path with a fake opener (monkeypatched `copernicusmarine.open_dataset` returning a small in-memory dataset), not with local files.

### A3. Resume is broken: an interruption wipes the store's attributes and the next run starts over **High**
- Appending a year (`build_store.py:545–553`) calls `to_zarr(append_dim="time")` with a dataset that has no attributes, which **replaces the root attributes with an empty set**. The lines that "update" `built_years` afterwards modify an in-memory object and never write it.
- `built_years` is only written in the final block, after all years.
- So if a session is cut after any appended year, the store has no `built_years`. The next run sees `len(built_years) == 0`, takes the `mode="w"` branch and **overwrites the whole store**.

Reproduced: built 2018, then started 2018–2019 and interrupted before the final step → root attributes `[]` (no `built_years`, `norm_stats`, `config_hash`, `source_files`); the rerun rebuilt 2018 from scratch. For a 29-year Colab build this means one disconnect at year 20 costs the first 19 years.

**Fix:** allocate the store's full time axis once (from config), then write each year with `to_zarr(region={"time": slice(i0, i1)})` as §4.2 says, and update `built_years` right after each year with `zarr.open_group(...).attrs.update(...)`. Never take `mode="w"` on an existing store unless `--force` is given. The test must kill the process (or raise) **between** two years and check that the second run neither repeats nor loses a year.

### A4. Climatology and normalization statistics break the train-only rule, and the full build cannot finish **High**
PRD §0 rule 4: climatology and normalization statistics come from the training period only.
- `build_store` fits the climatology on **every date in the store** (`build_store.py:562–566`), i.e. train + val + test. That reads the sealed test year (rule 3) and leaks val/test into every anomaly, the climatology baseline and `sigma_d`.
- Right after, `compute_norm_stats` (`transform.py:79`) raises `TestLeakageError` if *any* date in the store is in the test split. The default `--years 1993-2021` includes 2021, so the full build **always crashes at the very end**, after all the downloading. Reproduced with years 2020–2021.
- The opposite case fails too: a build that contains no training dates yet (e.g. `--years 2019`, the §4.2 parity acceptance) dies with "No dates found in training period".
- `fit_harmonic_climatology` loads all of `thetao` into memory as float64: for 1993–2021 that is about 10,600 × 15 × 120 × 240 values, roughly 18 GB as float32 and 37 GB as float64. It will not fit on Colab (≈12 GB) or Kaggle (≈30 GB).
- `preprocess` fits the climatology on January 2019, which is validation, and when the sample is outside the train split it silently treats the sample dates as "train" for `norm_stats` (`preprocess.py:545–553`). That is acceptable for a smoke test only if the store is labelled as such; today nothing marks it.

**Fix:** make climatology and statistics a separate step (`build_store --finalize` or its own module) that runs after the train years are built, selects **train dates only** (minus `gap_days`), and never touches test dates. Fit the harmonic regression in chunks (accumulate XᵀX and Xᵀy per cell block over time chunks; it is linear, so this is exact). Pass the train split into `fit_harmonic_climatology` and make it refuse other dates, with a unit test. For the one-month sample, write `split_dates` and a `sample_only = true` attribute and do not present its statistics as train statistics.

### A5. Silent gap filling in every 2-D reduce function **High**
PRD §0 rule 5: no silent NaN filling.
- Each 2-D reducer selects dates with `da.sel(time=target_time, method="nearest")` (e.g. `reduce.py:396–400`). A missing day takes its neighbour's field. If the selection raises, the `except: pass` keeps the unselected data but labels it with the target dates.
- Then `vals[t_idx] if t_idx < vals.shape[0] else vals[-1]` (e.g. `reduce.py:428`) repeats the last field for any extra dates.
- Reproduced: source days 01, 02, 04 → output for 01–05 is `[1, 2, 4, 4, 4]`. Day 03 and day 05 are invented.
- `reduce_era5_fluxes` returns **zeros** for net heat flux when the four components are missing (`reduce.py:905`).

**Fix:** select exact dates (`reindex(time=target_dates)` with no fill, which leaves missing days as NaN), and remove every bare `except: pass` in `reduce.py` and `preprocess.py`. Missing heat-flux components must raise. Add a test with a missing day that asserts NaN.

### A6. GREP is NaN at the surface and deepest level; ARMOR3D and GREP are never found in real files **High** (P3-4)
- `reduce_grep` interpolates in depth without extrapolation. GREP's first level is ≈0.506 m, which is below the 0.494 m GLORYS level, and the sample was subset to 1045.85 m, which is above 1062.44 m. Result: the auxiliary target is **NaN at 0.494 m and 1062.44 m everywhere** (reproduced with GREP's level list). That silently drops two of the 15 levels from `L_grep` and from the GREP column of the §1.1 comparison.
- With real file names, neither `preprocess` nor `build_store` ever picks up GREP or ARMOR3D (the names contain neither "grep" nor "armor"; see A1). The P3-4 integration has only ever run on synthetic files.
- PHASES P3-4 says ARMOR3D and GREP come "through `build_store` (streamed, no raw files)". Not done (A2).

**Fix:** request GREP down to the first level below 1062.44 m (≈1152 m), so interpolation has a neighbour. For the top level, take an explicit owner decision: either use GREP's first level for 0.494 m (a 1 cm difference, documented in DECISIONS), or keep it NaN and say so in the evaluation. Add `OPEN_QUESTIONS` entry. Do the same check for ARMOR3D (its levels start at 0 m, so the top is fine).

### A7. Baselines: no output files, wrong comparison, and they don't scale **High** (P3-3)
- **No `baseline_*_val.csv` exists.** `ml/outputs/` contains only `figures/` and `reports/`. On the real sample store (January 2019 only, all validation) `run_baselines` fails: train dates are empty, so `extract_pixel_features_and_targets` raises. The exit gate item "baseline validation metrics exist" is unmet. PROGRESS lists the CSVs as produced.
- **Skill vs climatology compares different samples.** The climatology RMSE uses every valid target cell at each depth (`baselines.py:285–294`). The ridge RMSE uses only cells where all 11 inputs are finite **and** all 15 depths are valid (`baselines.py:140–150`), which drops coasts and shallow shelves. `skill_vs_clim_pct = 1 − RMSE_ridge/RMSE_clim` is then meaningless. Score both on the same mask, and use that mask for every model later (U-Net too).
- **The climatology is not train-only** (A4), so the climatology baseline is scored in-sample on validation.
- **Fallback splits.** If no validation dates are found, `run_baselines` silently splits whatever dates exist 70/30 and calls them train/val (`baselines.py:520–542`). It must raise instead.
- **Memory.** Errors are accumulated as Python lists with `.tolist()` for every valid cell (`baselines.py:293`, `389`). For three validation years that is several hundred million Python floats, far beyond Colab RAM. Ridge validation also builds the full feature matrix with no cap. Accumulate per-depth sums (n, Σe, Σ|e|, Σe²) day by day.
- **Zeros for no data.** A depth with no valid cells gets `mae/rmse/bias = 0.0` (`baselines.py:308`, `410`). Write NaN or an empty value; a 0.0 RMSE is a fabricated number (§0 rule 2).
- `ridge_alpha` (1.0) and `max_train_samples` (500,000) are CLI defaults, not config (§0 rule 6). Put them under a `baselines:` block in `experiment.yaml`.

### A8. P1-M3 is not done, so P3 had no confirmed configuration **High** (process)
PHASES: P3 "Needs: P1-M3 (decisions confirmed)". `experiment.yaml` still marks region, `time.start`, `time.end` and all three splits `[CONFIRM]`; OPEN_QUESTIONS still lists region/period/splits, start year (1993 vs 2004), GREP end date and ARGO source as open. Only the depths were approved (DECISIONS 2026-10-24). The full build (P3-2b) changes cost a lot if the start year or region changes afterwards, so these must be confirmed **before** P3-2b runs.

**Values to confirm before P3-2b, with Claude's recommendation:**

| # | Item | Current value | Recommendation | Why |
|---|---|---|---|---|
| 1 | Region | 40–100°E, 0–30°N | **Keep** | Matches the January downloads, the fixtures, the backend and the §1.1 Bay of Bengal region |
| 2 | Start year | 1993-01-01 | **Keep 1993** | More years for the climatology and training; ARGO input channels are simply empty before 2004 (`argo_input.min_year`), which the design already handles. 2004 would save ~40% build time but loses 11 years of history |
| 3 | End date | 2021-12-31 | **Keep** | Gives one full sealed test year |
| 4 | Splits | train 1993-03-02 → 2017-12-01, val 2018-01-01 → 2020-12-01, test 2021 (sealed), 30-day gaps | **Keep** | Train starts after the 60-day long-history lead-in; val covers the 2019 positive IOD; gaps prevent leakage across split edges |
| 5 | GREP end date | catalogue says 2020-12-15 | **Accept 2020-12-15** unless `copernicusmarine describe` shows later data | If it ends in 2020, the 2021 headline compares against GLORYS12 and ARMOR3D only, and GREP is reported as "n/a" for the test year (§1.1) |
| 6 | GREP top level (new, A6) | NaN at 0.494 m | **Use GREP's first level (~0.506 m) for 0.494 m**, logged in DECISIONS | A 1 cm offset is negligible; otherwise the top level drops out of `L_grep` and the GREP comparison |

Also open, but not blocking the build: ARGO source (Argo GDAC vs Copernicus INSITU). Recommendation: **Argo GDAC via ERDDAP**, which the January download already uses.

Not needed now: `history_days`, `long_history.days`, `aux_weight`, `depth_weights.in_band` and `finetune_argo.lambda` are settled by validation experiments in P4. `publish.hf_dataset_repo` should be set to `zenzone3/oceanscope-outputs`, which P2 already uses.

Once Aryan approves, the values go into `experiment.yaml` (dropping the `[CONFIRM]` tags), a dated DECISIONS entry is added and the OPEN_QUESTIONS items are marked answered.

---

## B. Should fix

1. **Undeclared dependencies (PRD §0 rule 9).** `xarray.Dataset.chunk` and `open_mfdataset` need `dask`; `plot.py` imports `PIL`; reading `.nc` needs `netCDF4` or `h5netcdf`. None is in `ml/pyproject.toml`, so the tests fail on a fresh install and the notebook's `pip install` line is the only place they appear.
2. **ERA5 units are not converted.** The CDS "accum" fields (`ssr`, `str`, `sshf`, `slhf`, `tp`) are hourly accumulations in J m⁻² and m. `reduce_era5_fluxes` sums them and takes a daily mean, which gives J m⁻² per hour (≈ 3600 × W m⁻²) and metres per hour, not W m⁻² and a daily total. PRD §4.2 asks for "daily sum = daily mean rate × 24 h". The real accum file was never used (A1), so this hasn't shown up yet. Also: daily resampling only happens if the file has more than 60 time steps (`reduce.py:784`, `870`), a heuristic that breaks for short requests, and the "≥ 18 of 24 valid hours" rule from §3 is not implemented. Use the variables' `units` and `GRIB_stepType` attributes rather than guessing.
3. **SST Kelvin detection is a heuristic** (mean of the first 1000 valid values > 100, `reduce.py:545`). Use the `units` attribute.
4. **Wind-stress curl is not NaN at the coast** (§3 says it should be). ERA5 winds are valid over land, so the curl is computed across coastlines. Mask with the ocean mask after computing it.
5. **WOA23:** `build_store` uses only the first WOA file (January) and copies it to **every day of the year** (`build_store.py:352–364`), so December gets January's profile. It also stores a (time, depth, lat, lon) copy of a static field, about 1.7 MB per day uncompressed. Store it once as (month, depth, lat, lon) and index by month in `dataset.py`.
6. **Store size.** Per day the store holds thetao, target_valid, WOA (broadcast), GREP and ARMOR3D at 15 depths plus 11 surface fields: about 8.6 MB uncompressed, about 90 GB for 1993–2021 before compression. The ≤ 20 GB target looks unlikely as written. Dropping the WOA broadcast and storing ARMOR3D only for val/test (it is only used there) would help; then measure one real year.
7. **Dry run doesn't plan requests.** PRD §4.2: "prints each remote request, its estimated transfer size and the estimated store growth". `plan_store_build` multiplies hardcoded per-day guesses (`ESTIMATED_SIZES_MB`), e.g. GEBCO "15 MB" where the real file was 7.4 GB. Derive estimates from the request (bbox × depths × days × dtype), and print each request.
8. **No scratch or store budget enforcement.** PROGRESS says "scratch & store budget enforcement (≤ 20 GB)". There is no code that measures or enforces either. Either implement a check (fail loudly when over) or correct PROGRESS.
9. **ARGO, climate indices and the DMI strip are not in `build_store`.** §4.2 lists ARGO via ERDDAP to Parquet; the inputs list needs Niño 3.4, DMI (from the extra OSTIA strip) and MJO. These are needed for P4-6 and the input tensor. Track them as P3 follow-ups rather than leaving them implicit.
10. **Notebook (§11, §0 rule 7):** it never installs the package (it only prepends `ml/src` to `sys.path` if the repo happens to be there; the clone is commented out), it parses `norm_stats` with `eval()` (use `json.loads`), the Kaggle branch doesn't load `KAGGLE_USERNAME/KEY`, and the upload is commented out. §11 says the store is versioned to the Kaggle Dataset at the end of each finished year; that is not done. Also a GEBCO global file is opened and read in full in `reduce_gebco` when it is the only candidate; subset first in `build_store` as `preprocess` does.
11. **float16 copy name.** PRD §6.1 says `oceanscope_<v>_f16.zarr`; the code writes `_fp16.zarr`. Pick one and update the PRD or the code.
12. **Branch base.** `ml/p3-1-preprocess` was branched from `P2` (frontend landing work), not from `main`, so a PR would carry unrelated frontend commits. Rebase onto `main` before opening the PR.

## C. Minor
- `TargetGrid.from_bounds` has the region as default arguments (40/100/0/30); callers pass the config, but defaults like these invite a silent wrong grid. Remove them (§0 rule 6).
- `standardize_coords` and the reducers wrap renames and transposes in `try/except: pass`; failures will surface later as shape errors far from the cause.
- `fit_harmonic_climatology` drops to a per-cell Python loop for every cell with any missing day. Fine for GLORYS (no gaps), slow for anything else.
- `compute_norm_stats` has no statistics for `woa_clim_profile` or the indices, both of which §5.1 z-scores. Needed in P4.
- `m1_sample_day.png` must show "one day plotted for every variable" (M1); `target_valid`, WOA and GREP are missing from it.
- PRD §5.8 says the anomaly is from "a fixed 2010–2018 climatology", while §2/§5.2 say the train period. Not code, but the PRD should say one thing before the climatology step is written.

---

## What is good
- The 15 confirmed depths are enforced (`preprocess` refuses anything other than 15).
- `area_mean_regrid_2d` implements the ≥ 50 % valid sub-cell rule correctly and is properly tested, including the edge cases.
- The Large & Pond drag coefficient and the spherical curl formula are correct.
- GEBCO → log(1 + depth) and the Kelvin → °C conversion are right in substance.
- The harmonic-fit test recovers a pure sinusoid to 1e-3 as specified, and the transform round-trip test passes.
- The test-leakage guard exists and is used. The problem is where it is called (A4), not the guard.
- The processed store follows the §6.1 variable names, dims, dtypes and attributes, and the additions (`adt`, `sst`, `grep_mean`, …) are additive.
- Lint, format and type checks are clean, and the `.gitignore` keeps `ml/data` and `ml/outputs` out of git.

---

## Exit gate check (PHASES P3)

| Gate item | Status |
|---|---|
| P1-M3 decisions confirmed before P3 | **No**: only depths confirmed (A8) |
| P3-1: sample store from real files, sample-day figure, tests 3–5 | **No**: store and figure mostly synthetic (A1); tests 3–5 pass |
| P3-2a: streaming `build_store`, resumable, dry run, test 13, Jan 2019 parity | **No**: no streaming (A2), resume broken (A3), parity only tested on synthetic local files |
| P3-2b: full-period store on Colab/Kaggle, float16 copy on Kaggle | **No**: not run; would crash (A4) |
| P3-3: `baseline_*_val.csv` exist | **No**: no files; cannot run on current store (A7) |
| P3-4: ARMOR3D + GREP through `build_store` | **No**: not streamed, not found in real files, GREP NaN at 2 levels (A6) |
| Store passes schema check and loads in a Colab and a Kaggle notebook | **Not attempted** |

## Suggested order
1. Aryan confirms the remaining `[CONFIRM]` values (A8, recommendations in the table there).
2. Clean the raw folder and remove synthetic generation from the package (A1), fix file selection and the reducers' fallbacks (A1, A5), re-run `preprocess` on the real January files, and look at the new figure together.
3. Fix the climatology/statistics step (A4) and the baselines (A7); produce real `baseline_*_val.csv` from a train+val subset (the January sample alone has no train dates, so the first real numbers need at least one train year, which is what the streaming build is for).
4. Implement streaming `build_store` with region writes and per-year attributes (A2, A3), GREP depth neighbours (A6), ERA5 units (B2), WOA by month (B5); dry-run, then build **one** real year on Colab and measure its size before committing to 29.
5. Declare the dependencies (B1), rebase onto `main`, open a PR, get CI green, and correct `docs/PROGRESS.md` so the three P3 rows say what is actually done.
