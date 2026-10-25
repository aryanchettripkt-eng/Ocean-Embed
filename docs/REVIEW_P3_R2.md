# Re-review of Gemini's P3 fixes (REVIEW_P3 round 2)

Reviewer: Claude · 2026-10-25 · `E:\meta` on branch `ml/p3-1-preprocess`, new commit "ml: fix P3 data preprocessing, streaming build_store and baselines (… REVIEW_P3)" on top of the three P3 commits. Still branched from `P2`, still no PR or CI run.

**Verdict: real progress, but P3 is still not done.**
- **Fixed:** the synthetic data is gone. The January 2019 sample store and the figure are now built from the real files. Resume works. The climatology and statistics no longer read test or validation data in `build_store`. Gaps stay NaN. The baselines score both models on the same cells. The dependencies are declared.
- **Still not done:** `build_store` **still does not stream**. A `stream_copernicus()` function was added but nothing calls it, yet PROGRESS calls the build "streaming". The full build would also run out of memory before writing a single year.
- **New bugs in the fixed code:** the WOA field in the store is **entirely empty** (and has the wrong shape), the GLORYS **1062 m level is empty** (a float32 rounding bug), the wind-stress-curl normalization is wrong, and the `build_store` tests run on a box that is all land, so they check nothing.
- **Exit gate:** still not met. There is no full store, no float16 Kaggle copy and no `baseline_*_val.csv`. The baseline files can't exist yet, because the January sample has no train dates; that part is expected.

**How this was checked.** Read every changed file (`preprocess.py`, `reduce.py`, `build_store.py`, `climatology.py`, `transform.py`, `baselines.py`, `plot.py`, `pyproject.toml`, `experiment.yaml`, the new `tests/synthetic_data.py` and the three P3 test files). Read the new store's root attributes and per-array metadata on `E:\meta`, listed its chunk files, and looked at the new `m1_sample_day.png`. Installed `ml/` into a fresh Python 3.13 venv from `pyproject.toml`: ruff, ruff format and mypy are **clean**, and the P3 tests give **26 passed** with no manual installs. Re-ran the round-1 reproduction script (offline, no network) against the new code; its results are quoted below. I did not re-run the pipeline on the 10+ GB of raw files.

---

## Status of round-1 findings

| # | Finding | Status | Evidence |
|---|---|---|---|
| A1 | Sample store and figure mostly synthetic | **Fixed** (with new bugs R1, R2) | Synthetic files deleted from `ml/data/raw`; generator moved to `tests/synthetic_data.py`; `--sample-data` removed; files picked from MANIFEST / product folder; `source_files_by_var` records what was opened. `norm_stats` are now physical (SLA mean 0.052 m, ADT 0.82 m, SST 27.2 °C, SSS 34.8, net heat flux −19 ± 78 W m⁻², precip 2.4 mm/day). Figure shows real fields on the real coastline. Reducers raise `KeyError` on a missing variable. |
| A2 | `build_store` doesn't stream; fabricates missing years | **Half fixed** | No more synthetic years: an empty raw dir now raises `FileNotFoundError` (reproduced). But `stream_copernicus()` (`build_store.py:274`) is **never called**; there is no CDS, Earthdata or ERDDAP reader. `build_store` still only reads local raw files, so on Colab it can't build any year. See R3. |
| A3 | Interruption wipes attributes; resume restarts | **Fixed** | Store is preallocated, years go in with region writes, and `built_years` is written to disk after each year. Reproduced: build 2018, interrupt 2018–2019 at the finalize step → `built_years = [2018, 2019]`, all attributes kept; the rerun skipped both years and the data survived. |
| A4 | Climatology/stats use val+test; full build crashes; memory | **Mostly fixed** | Climatology and stats now use the train slice only, with guards; the test-year crash is gone; a store with no train dates is marked `sample_only`. **Memory not fixed:** `fit_harmonic_climatology` still loads the whole train period as float64 (~31 GB for 1993–2017). See R3. |
| A5 | Silent gap filling | **Fixed** | Exact-day matching; missing days stay NaN. Reproduced: source days 01, 02, 04 → output `[1, 2, NaN, 4, NaN]`. Heat flux no longer falls back to zeros. |
| A6 | GREP NaN at 0.494 m and 1062 m; GREP/ARMOR3D never found | **Partly fixed** | Both products are found and in the store; the 0.494 m level is filled from GREP's first level as decided. **1062.44 m is still NaN** (reproduced): the raw GREP file stops at 1045.85 m and was never re-requested deeper. |
| A7 | Baselines: no CSVs, unequal masks, memory, fallback split, zeros | **Fixed in code** | One shared mask for both models; streaming sums instead of lists; the 70/30 fallback now raises; NaN instead of 0.0; `ridge_alpha` and `max_train_samples` moved to `experiment.yaml`. No CSVs yet (expected: no train dates in the sample). But see R2: with the current mask the baselines would score **zero cells**. |
| A8 | P1-M3 not confirmed | **Fixed** | Confirmed 2026-10-25 (DECISIONS). |
| B1 | Undeclared dependencies | **Fixed** | `dask`, `netCDF4` and `Pillow` are in `pyproject.toml`; a fresh install passes. |
| B2 | ERA5 units | **Fixed** | J m⁻² ÷ 3600 → W m⁻² (based on the `units` attribute), daily mean of the flux, daily sum of precipitation, ≥ 18 hours required. Small leftovers in C. |
| B3 | SST Kelvin heuristic | **Fixed** | Uses the `units` attribute. |
| B4 | Curl not NaN at coast | **Fixed** | Masked unless the cell and its 4 neighbours are ocean. |
| B5 | WOA January copied to every day | **Fixed in `build_store`, broken in `preprocess`** | See R1. |
| B6 | Store size | **Open** | No real year has been measured; ARMOR3D is still stored for every day. |
| B7 | Dry run doesn't plan requests | **Partly fixed** | Now lists each product, service and dataset ID with sizes from the grid and depths; source pixel counts and the 75 MB/25 MB static sizes are still hardcoded. |
| B8 | No budget enforcement | **Fixed** | `check_budget()` raises above 20 GB scratch or store. |
| B9 | ARGO, indices, DMI strip not in `build_store` | **Open** | |
| B10 | Notebook | **Not touched** | `build_store.ipynb` is unchanged: no package install, `eval()` on `norm_stats`, upload commented out. |
| B11 | float16 name | **Fixed** | `oceanscope_<v>_f16.zarr`, `--fp16` kept as an alias. |
| B12 | Branch base | **Open** | Still on top of `P2`. |

---

## New findings (must fix)

### R1. WOA in the sample store is entirely empty and has the wrong shape **High**
The store's `woa_clim_profile` has shape **(12, 31, 15, 120, 240)** with dims `month, time, depth, lat, lon`, and **no chunk files at all**, so every value reads back as NaN. The WOA panel in the new figure is blank.

Cause: WOA is opened with `decode_times=False`, so each monthly file keeps its own numeric `time` coordinate. `reduce_woa` returns `(time, depth, lat, lon)`. `preprocess` concatenates the 12 months on a new `month` dim, and when the store is assembled with the daily `time` coordinate, xarray aligns WOA's numeric times against the 31 dates, finds no match, and fills everything with NaN (`preprocess.py`, the WOA step and dataset assembly). `build_store` takes `.values[0]` and gets `(12, depth, lat, lon)` correctly, so **the two paths now disagree**, which is exactly what test 13 is supposed to catch.

**Fix:** in `reduce_woa`, drop or squeeze WOA's own `time` and always return `(month, depth, lat, lon)`; use the same code in both paths. Add a test that loads two WOA-like files with a numeric `time` coordinate and asserts the shape and that values are finite.

### R2. GLORYS 1062.44 m level is empty (float32 rounding), which also empties the baselines **High**
The raw GLORYS depth coordinate is float32. Its deepest level is stored as 1062.43994…, which is **less than** the configured 1062.440, so `interpolate_depths_1d` treats 1062.44 as below the data and writes NaN. Reproduced with a float32 depth coordinate: NaN at 1062.44 only. That is why `sigma_d` in the store is 1.0 for the last level (the "no data" fallback).

Consequences:
- The deepest target level is missing from the processed store, and would be missing from the full build too.
- `compute_valid_pixel_mask` in the baselines requires **all 15 levels** to be valid, so with level 15 empty everywhere **no cell passes**, and both baselines would score nothing once train data exists.

**Fix:**
- Match configured depths to source levels with a tolerance (e.g. `np.isclose(…, atol=0.01)`) and select those levels directly instead of interpolating. The 15 depths are native GLORYS levels by decision.
- In the baselines, use a **per-depth** mask (inputs valid + target valid at that depth). Requiring all 15 levels also drops every shelf cell shallower than ~1000 m from every metric, which biases the scores towards open ocean.
- Add a test with float32 depth coordinates.

### R3. The full build still cannot run **High**
1. **No streaming.** `stream_copernicus()` is dead code. `reduce_year_dataset` still only globs local files; there is no ERA5 daily-statistics request, no OSCAR reader, no WOA OPeNDAP, no ERDDAP. On Colab every year raises `FileNotFoundError`. PROGRESS's "streaming build_store" is not true yet.
2. **Preallocation is eager and needs ~73 GB of RAM.** The template store is built with `np.full(...)` for every variable over the full period before `.chunk().to_zarr()`. For 1993–2021 that is 10,592 days × (11 surface fields + `thetao`, `grep_mean`, `armor3d` at 15 levels + `target_valid`), about **72.9 GB** in memory (computed from the config). Use `dask.array.full(..., chunks=…)` with `to_zarr(..., compute=False)` so only metadata is written.
3. **Climatology memory.** `fit_harmonic_climatology` still does `.values` on the whole train slice and casts to float64: about 31 GB for 1993–2017. Accumulate XᵀX and Xᵀy over time chunks (exact, since the fit is linear).
4. `filter_files_by_year` returns **all** files when none has the year in its name, so a year can silently be built from another year's file if the dates happen to overlap. It should return nothing and let the caller raise.

### R4. Wind-stress-curl normalization is wrong **Medium**
The store's `norm_stats` give `wind_stress_curl` mean 0.0 and std **1.0**. Real curl values are ~10⁻⁷ N m⁻³, so `compute_norm_stats` rounds the mean to 6 decimals (→ 0.0) and treats the std as "zero" because it is below 1e-6 (→ 1.0). Reproduced: a field with true std 2e-7 gets `{"mean": 0.0, "std": 1.0}`. Normalized, the channel becomes ~10⁻⁷ everywhere: useless as an input. Fix: don't round the stats (store full floats), and use a relative guard (e.g. std == 0 or std < 1e-6 × |typical value|), or store the curl in 10⁻⁷ N m⁻³ units.

### R5. The `build_store` tests run on an all-land box, so they check nothing **Medium**
`tests/test_build_store.py`'s `MockConfig` uses 40–42°E, 0–2°N. The synthetic generator marks everything at lat ≤ 12°N and lon ≤ 51°E as land, so **every value in that box is NaN**. Verified: the synthetic GLORYS file there has 0 finite values. The test 13 parity check compares NaN with NaN, and the resume test passes without any data being written. Move the box to open ocean (e.g. 60–62°E, 10–12°N, which I used to check resume), and assert that each variable has finite values before comparing. With real data in the parity test, R1 would have failed it.

---

## Minor
- ERA5 precipitation uses `nansum`, so a day with 18–23 hours is under-counted and an all-NaN cell becomes 0 rather than NaN. Scale by 24/n or require 24 hours. The ≥ 18-hour check counts timestamps, not valid values per cell.
- ERA5 hourly accumulations are labelled by the end of the hour, so the 00 UTC value belongs to the previous day; the day grouping is off by one hour.
- `build_store` stores `norm_stats` and `split_dates` as JSON objects while `preprocess` stores JSON strings; consumers have to handle both. Pick one (§6.1 says JSON).
- `TargetGrid.from_bounds` still has the region as default arguments (round-1 C).
- PROGRESS's new P3 row says "streaming build_store" and "complete"; please correct it after R1–R5.

---

## Exit gate check (PHASES P3)

| Gate item | Round 1 | Now |
|---|---|---|
| P1-M3 decisions confirmed | No | **Yes** |
| P3-1: sample store from real files, figure, tests 3–5 | No | **Mostly**: real data, but WOA empty (R1) and 1062 m empty (R2) |
| P3-2a: streaming `build_store`, resumable, dry run, test 13 | No | **Partly**: resumable and dry run yes; no streaming (R3); test 13 vacuous (R5) |
| P3-2b: full store + float16 on Kaggle | No | **No** (blocked by R3) |
| P3-3: `baseline_*_val.csv` exist | No | **No**: code fixed, needs a train year and R2 |
| P3-4: ARMOR3D + GREP through `build_store` | No | **Partly**: found and stored; not streamed; GREP 1062 m still NaN |
| Store loads in a Colab and a Kaggle notebook | Not attempted | Not attempted |

## Suggested order
1. R2 and R1 (both small), then re-run `preprocess` on the January files and check the WOA panel and `sigma_d[14]`.
2. R5 (move the test box) and R4.
3. R3: implement the streaming readers (start with Copernicus, since `stream_copernicus` already exists, then the ERA5 daily-statistics request), lazy preallocation and chunked climatology. Request GREP down to ~1152 m.
4. Dry-run on Colab, build **one** real train year (e.g. 2010), measure its size, run `--finalize` on it and produce the first real `baseline_*_val.csv` with one val year.
5. Fix the notebook (B10), rebase onto `main`, open a PR, get CI green, and correct PROGRESS.
