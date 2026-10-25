# Check of the round-3 fix list (REVIEW_P3 round 4)

Reviewer: Claude · 2026-10-25 · `E:\meta`, branch `ml/p3-1-preprocess`, commit "ml: complete streaming readers, memory chunking, and day boundary".

**How this was checked**
- Diffed every changed file against the round-3 snapshot.
- Ruff, ruff format and mypy are clean.
- pytest: 41 passed, plus the 2 tests that need the repo root.
- Ran the year filter against the real raw file names from `E:\meta\ml\data\raw` together with the real MANIFEST.

## Status of the 9 fixes

| # | Fix | Status |
|---|---|---|
| 1 | Stream every product | **Done, with bugs.** Every product now has a stream branch, and a mocked full-year test builds from an empty raw folder. See N1–N4. |
| 2 | Climatology memory | **Done.** The lazy `t_3d` is passed in, and partial cells are read in chunks. |
| 3 | Norm-stats memory | **Done.** `sigma_d` is computed from running sums over 365-day chunks. |
| 4 | GLORYS memory | **Done.** It processes 31-day chunks and selects the 15 levels before loading. |
| 5 | GREP depth | **Done.** `download.py` and the stream request both go to 1250 m. |
| 6 | Year filter | **Done, but it broke local builds.** See N5. |
| 7 | Notebook upload | **Done.** It copies `oceanscope_v1_f16.zarr` into the upload folder. |
| 8 | ERA5 day boundary | **Done** for the hourly path: times are shifted by −1 h before grouping by day. |
| 9 | PROGRESS | **Done.** The round-2 row is corrected and there is a new round-3 row. It will need updating again after N1–N6. |

## New findings

**N1. Three streaming dataset IDs are wrong (High).** The IDs in `build_store.py` don't match the IDs in `download.py`, which match the DECISIONS entry checked with `copernicusmarine describe`:
- SLA uses `cmems_obs-sl_glo_phy-ssh_my_0.25deg_P1D-m`. It should be `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`.
- SSS uses `MULTIOBS_GLO_PHY_SSS_L4_MY_015_015`, which is a product ID, not a dataset ID. It should be `cmems_obs-mob_glo_phy-sss_my_multi_P1D`.
- ARMOR3D uses the old weekly `dataset-armor-3d-rep-weekly`. It should be `cmems_obs-mob_glo_phy_my_0.125deg_P1D-m` (daily).

These would fail on Colab. The mocked test misses this because it only matches substrings. **Fix:** read the IDs from the `download.py` product table instead of hardcoding them, and make the mock assert the exact IDs.

**N2. Streamed ERA5 precipitation is 24× too small (High).** The CDS daily-statistics request asks for `daily_mean`. For `total_precipitation`, that is the mean of hourly accumulations, in m per hour. The daily path of `reduce_era5_fluxes` stores it as-is, while the local hourly path stores the daily sum in m/day, so the two paths disagree by 24×. **Fix:** multiply `tp` by 24 in the daily path. Add a test giving both paths the same hourly data and asserting they agree. The heat flux is fine: J m⁻² ÷ 3600.

**N3. Stream failures for GEBCO, WOA, GREP and ARMOR3D are only logged (Medium).** Each one is wrapped in `try/except` with `logger.warning`, so a year can be built **without bathymetry or WOA** and no error is raised. **Fix:** raise for GEBCO and WOA. For GREP, only allow skipping dates after its end date (2020-12-15); otherwise raise.

**N4. Static fields are re-downloaded every year, and ARMOR3D is loaded whole (Medium).**
- `stream_gebco` downloads the **global 7.5 GB** GEBCO file into the per-year scratch folder, and that folder is deleted after each year. So it is fetched 29 times.
- `stream_woa` opens 12 global WOA files over OPeNDAP every year.

**Fix:** build bathymetry and WOA once (on the first year, or in a separate step) and skip them when they are already in the store. Also check the CEDA GEBCO URL.

`reduce_armor3d` (and `reduce_grep`) call `.values` on a whole streamed year. For ARMOR3D at 1/8° and ~33 levels, that is about **6 GB**, which is risky on Colab. Loop over time in chunks, as was done for GLORYS.

**N5. The year filter now rejects the real ERA5 files (Medium).** The local ERA5 files are `data_stream-oper_stepType-instant.nc` and `…-accum.nc`, with no year in the name. Checked against the real raw names and MANIFEST: `filter_files_by_year(..., 2019)` returns `[]` for both, so a local `build_store --years 2019` (without `--stream`) now raises "Missing ERA5". **Fix:** for files without a year in the name, open them and check their time range, or rename the ERA5 downloads to include the year.

**N6. Minor.** The year filter's extra pattern `2019(\d{2}|\d{4})?` also matches strings like `201999`. It's harmless with the current file names.

## Fix list for Gemini
1. N1: use the dataset IDs from `download.py`; make the mock check exact IDs.
2. N2: multiply streamed ERA5 `tp` by 24; add a test that the local and streamed paths agree.
3. N3: raise instead of warning for GEBCO and WOA; allow GREP to be missing only after 2020-12-15.
4. N4: stream GEBCO and WOA once, not every year; chunk `reduce_armor3d` and `reduce_grep` over time.
5. N5: make local ERA5 files without a year in the name work again (check their time range).
6. Update the PROGRESS row.
