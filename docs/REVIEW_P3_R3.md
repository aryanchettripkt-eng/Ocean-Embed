# Re-review of the P3 round-2 fixes (REVIEW_P3 round 3)

Reviewer: Claude · 2026-10-25 · `E:\meta` on branch `ml/p3-1-preprocess`, new commit "ml: fix round-2 review findings R1-R5 and minor items". The branch is still based on `P2`, and there is still no PR or CI run.

**Verdict: four of the five round-2 bugs are fixed, and the sample store now looks right. The full build still can't run, and PROGRESS overstates what was done.**
- **Fixed:**
  - WOA has the right shape and is filled in (R1).
  - The GLORYS 1062 m level has data (R2).
  - The curl normalization is correct (R4).
  - The tests run on open ocean and check for real values (R5).
  - The baselines use a per-depth mask, so shelf cells are kept.
  - The template store is lazy now, which removes the 73 GB preallocation.
- **Still blocking the full build:**
  - `--stream` only streams GLORYS. Every other product still needs a local file, so the notebook fails on the first year (S1).
  - The finalize step still loads the whole training period into RAM: about 16 GB for the climatology and roughly 47 GB for the normalization stats (S2).
  - A streamed GLORYS year is also loaded fully into memory, about 15 GB (S3).
- **PROGRESS is inaccurate.** It claims "multi-source streaming readers (Copernicus, CDS, OSCAR, WOA OPeNDAP)" and "0 GB RAM" for the climatology. Neither is true (S1, S2).
- **Exit gate:** still not met. There is no full store, no float16 copy and no `baseline_*_val.csv`.

**How this was checked.**
- Diffed every changed file against the round-2 snapshot: `build_store.py`, `reduce.py`, `climatology.py`, `transform.py`, `preprocess.py`, `plot.py`, `baselines.py`, the tests and the notebook.
- Read the rebuilt store's metadata and looked at the new figure.
- Ran ruff, ruff format and mypy: all **clean**.
- pytest: **39 passed, 2 failed**. The 2 failures are `test_gitignore` and `test_sample_outputs`, which need the repo root (`.gitignore`, `fixtures/`). My copy only had `ml/`, so these are not regressions.
- Re-ran the round-2 reproduction script against the new code, plus two new checks (the `--stream` path and `filter_files_by_year`). I did not run the pipeline on the raw files.

---

## Status of round-2 findings

| # | Finding | Status | Evidence |
|---|---|---|---|
| R1 | WOA empty, wrong shape | **Fixed** | `reduce_woa` drops WOA's own `time` and always returns `(month, depth, lat, lon)`, and both paths use it. The store's `woa_clim_profile` is `(12, 15, 120, 240)` with chunks. In the figure, the WOA panel at 92 m shows real values (14–27 °C). Test 13 now checks WOA shape, finite values and parity. |
| R2 | GLORYS 1062 m empty; baselines would score zero cells | **Fixed** | Depths match native levels within 0.05 m and are sliced directly. Reproduced: with a float32 depth coordinate, no level is NaN. `sigma_d[14]` = 0.031 °C (was the 1.0 fallback). The baselines now mask per depth with `target_valid`, and Ridge is fitted per depth on valid samples, so the climatology and Ridge baselines are still scored on the same cells. |
| R3.1 | No streaming | **Mostly open** | See S1. Only GLORYS goes through `stream_copernicus`. |
| R3.2 | Eager 73 GB preallocation | **Fixed** | `dask.array.full` + `to_zarr(compute=False)`. Unwritten days read back as NaN (fill value is NaN). Resume still works: reproduced with 2018 then 2018–2019 interrupted at finalize, giving `built_years = [2018, 2019]` and data on the 6 written days only. |
| R3.3 | Climatology memory | **Not fixed in practice** | `fit_harmonic_climatology` now accumulates over time chunks, but `build_store` calls it with `t_3d.values`, which loads all of it first. See S2. |
| R3.4 | `filter_files_by_year` falls back to all files | **Partly fixed** | See S4. |
| R4 | Curl normalization | **Fixed** | Stats aren't rounded any more, and the guard is now < 1e-12. Store: curl mean 9.8e-9, std 2.02e-7. Reproduced: a field with true std 2e-7 gives std 1.93e-7. |
| R5 | Tests on an all-land box | **Fixed** | `MockConfig` box is now 60–62°E, 10–12°N, and the parity test asserts finite values for each variable before comparing. |
| A6 | GREP 1062 m NaN | **Open** | Still NaN (reproduced). The raw GREP file ends at 1045.85 m. `download.py` asks for `max_depth_m: 1100`, but GREP's next level is ~1245 m, so the request needs to go to at least ~1250 m. |
| B6 | Store size not measured | **Open** | |
| B7 | Dry run partly hardcoded | **Open** (unchanged) | |
| B9 | ARGO, indices, DMI strip not in `build_store` | **Open** | |
| B10 | Notebook | **Partly fixed** | Now installs the package, uses `json.loads` instead of `eval()`, and passes `--stream`. See S5. |
| B12 | Branch base | **Open** | Still on `P2`. |
| Minor | `TargetGrid.from_bounds` default region | **Fixed** | |
| Minor | ERA5 precip `nansum` | **Fixed** | Valid hours are counted per cell, ≥ 18 is required, and the sum is scaled by 24/n. |
| Minor | ERA5 hour-ending label | **Open** | The 00 UTC accumulation still counts toward the wrong day. |
| Minor | JSON strings vs objects | **Fixed** | `preprocess` and `build_store` both store objects now. |

---

## New and remaining findings

### S1. `--stream` only streams GLORYS; the notebook build fails on year one **High**
In `reduce_year_dataset`, only the GLORYS step has an `elif stream:` branch. SLA/ADT, SST, SSS, OSCAR, ERA5, GREP, ARMOR3D and WOA still glob local files and raise `FileNotFoundError` when they're missing. **Reproduced:** with `stream=True` and an empty raw dir, GLORYS was streamed (mock), then it raised `FileNotFoundError: Missing SLA/ADT raw data for year 2018`. The notebook's step 5 runs exactly this on Colab with no raw files, so it would stop on 1993. `scratch_dir` is passed into `reduce_year_dataset` but never used.

The commit message says "wired streaming readers", and PROGRESS says "multi-source streaming readers (Copernicus, CDS, OSCAR, WOA OPeNDAP)". There is no CDS, OSCAR (Earthdata) or WOA reader in the code. **Fix:**
- Add stream branches for the other Copernicus products (SLA/ADT, SST, SSS, GREP, ARMOR3D) using `stream_copernicus`, which is a small change.
- Add an ERA5 daily-statistics CDS request and an OSCAR Earthdata download into `scratch_dir`.
- Add a test that runs a whole year with `stream=True` and mocked readers.
- Correct PROGRESS.

### S2. Finalize still loads the full training period (~16 GB + ~47 GB) **High**
- **Climatology:** `build_store.py` calls `fit_harmonic_climatology(t_3d.values, …)`, so the whole 1993-03-02 to 2017-12-01 `thetao` (9,041 days × 15 × 120 × 240, float32) is loaded first. That is **15.6 GB**, before the chunked loop even starts. Pass the lazy `t_3d` instead.
- **Partial cells:** inside `fit_harmonic_climatology`, the partial-cell branch does `thetao.values.reshape(...)` on the whole array whenever **any** cell has a gap. So even with a lazy input, a single gappy cell brings back the full load. Read only those cells, chunk by chunk.
- **Normalization stats:** `compute_norm_stats` (called with the whole store) does `ds_train["thetao"].values`, then builds `clim_on_dates` and `anomaly_train` at the same size. That is about **3 × 15.6 ≈ 47 GB**. Every 2D channel is also loaded in full (about 1 GB each, which is fine). Compute `sigma_d` with running sums over time chunks.
- PROGRESS's "chunked O(1) climatology accumulators (0 GB RAM vs 31 GB)" is not true end-to-end.

### S3. A streamed GLORYS year is loaded whole (~15 GB) **Medium**
`reduce_glorys12` does `da.values` on the full input. For one streamed year at 1/12° over the padded box (~744 × 384 cells, the ~35 native levels between 0.49 and 1062 m, 365 days), that is about **14.6 GB**, more than a Colab session has. `stream_copernicus` also requests the full depth range, not just the 15 levels. **Fix:** loop over time inside `reduce_glorys12`, a month at a time or with `.isel(time=…)`, and select the 15 matched levels before loading. The same applies to the local-file path once yearly GLORYS files are used.

### S4. `filter_files_by_year` still falls back to all files for most file names **Low**
The new "other year" check uses `\b(19\d\d|20\d\d)\b`. `\b` doesn't fire between `_` and a digit, or inside `20181102`. **Reproduced:**
- OSCAR files `…final_20181102.nc` / `…20190101.nc` asked for 2020 → **both returned**.
- `argo_profiles_2019-01-01_….nc` asked for 2020 → returned.
- Only the DUACS-style `…_2018-11-02-2019-01-31.nc` is rejected.

Because days are matched exactly, the wrong files give NaN channels rather than wrong numbers, but a whole year of currents could come out empty with no error. Use `(?<!\d)(19|20)\d\d(?!\d)`, or just return `[]` when nothing matches. Also, `yr_str in str(f)` matches the year anywhere in the full path.

### S5. Notebook upload step uploads an empty folder **Low**
Step 7 writes `dataset-metadata.json` to `data/processed/kaggle_upload` and runs `kaggle datasets create` on that folder, but it never copies `oceanscope_v1_f16.zarr` into it. Also, DECISIONS sets `hf_dataset_repo: zenzone3/oceanscope-outputs` for outputs. That is fine for run outputs, but it's worth confirming the Kaggle dataset is still where the store is meant to go.

---

## Exit gate check (PHASES P3)

| Gate item | Round 2 | Now |
|---|---|---|
| P1-M3 decisions confirmed | Yes | Yes |
| P3-1: sample store from real files, figure, tests 3–5 | Mostly | **Yes** (WOA and 1062 m fixed; GREP 1062 m is a data request, tracked under P3-4) |
| P3-2a: streaming `build_store`, resumable, dry run, test 13 | Partly | **Partly**: resumable, lazy, real test 13; streaming is GLORYS-only (S1) |
| P3-2b: full store + float16 on Kaggle | No | **No** (blocked by S1–S3) |
| P3-3: `baseline_*_val.csv` exist | No | **No**: code is ready, needs one real train year and one val year |
| P3-4: ARMOR3D + GREP through `build_store` | Partly | **Partly**: not streamed; GREP 1062 m NaN |
| Store loads in a Colab and a Kaggle notebook | Not attempted | Not attempted |

## Suggested order
1. **S1:** add the Copernicus stream branches (five products, same function), then CDS ERA5 and OSCAR into `scratch_dir`. Add a mocked full-year stream test.
2. **S2:** pass the lazy `thetao` to the climatology, fix the partial-cell load, and chunk `sigma_d` in `compute_norm_stats`.
3. **S3:** reduce GLORYS month by month, loading only the 15 levels.
4. Re-request GREP with max depth ≥ 1250 m. Fix S4 and S5.
5. On Colab, build **one** real train year plus one val year, measure the size (B6), run `--finalize`, and produce the first real `baseline_*_val.csv`.
6. Correct PROGRESS, rebase onto `main`, open a PR and get CI green.
