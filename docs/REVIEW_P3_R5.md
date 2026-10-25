# Check of the round-4 fix list (REVIEW_P3 round 5)

Reviewer: Claude · 2026-10-25 · `E:\meta`, branch `ml/p3-1-preprocess`, commit "ml: fix dataset IDs, precip scaling, error raising, static caching and ERA5 year filtering".

**How this was checked:** diffed against round 4. Ruff, ruff format and mypy are clean. pytest gives 44 passed, plus 2 failures in tests that need the repo root. I also ran two small reproductions (below).

## Status of the 6 fixes

| # | Fix | Status |
|---|---|---|
| 1 | Dataset IDs | **Done.** All stream calls read the IDs from `PRODUCT_SPECS` in `download.py`, and the mock checks the exact IDs. |
| 2 | ERA5 precip ×24 | **Done.** There is a parity test showing the hourly and daily paths agree. One minor point is in the list below. |
| 3 | Raise on stream failures | **Done.** GEBCO, WOA and ARMOR3D raise errors. GREP only warns after 2020-12-15. |
| 4 | Static caching and chunked ARMOR3D/GREP | **Done.** GEBCO and WOA are cached in memory and in the base scratch folder, and both reducers work in 31-day chunks. |
| 5 | ERA5 files without a year in the name | **Done, but not for the real files.** See F2. |
| 6 | PROGRESS | **Done.** |

## Remaining findings

**F1. Building 2020 will crash on GREP (High).** GREP ends on 2020-12-15. `reduce_year_dataset` does `ds_red_grep.sel(time=target_dates)`, and `.sel` raises `KeyError` when any date is missing. **Reproduced:** GREP data for 1–15 Dec 2020, then `.sel` over all of December gave `KeyError: not all values found in index 'time'`. Because 2020 is in the validation split, the full build would stop there. ARMOR3D uses the same `.sel` pattern and would crash the same way if a day is missing. **Fix:** use `.reindex(time=target_dates)` for GREP and ARMOR3D, so missing days become NaN as they do for every other product. Add a test where GREP ends partway through the year.

**F2. The time-range check looks for `time`, but new CDS files use `valid_time` (Medium).** `filter_files_by_year` opens a file without a year in its name and checks `ds.coords["time"]`. ERA5 NetCDF from the new CDS names this coordinate `valid_time`; the codebase already renames `valid_time` in `standardize_coords` because of that. **Reproduced:** a `data_stream-oper_stepType-instant.nc` with a `valid_time` coordinate covering 2019 is still rejected for 2019. The new test only uses a `time` coordinate, so it passes anyway. I couldn't copy your 315 MB ERA5 file to confirm its coordinate name (the transfer timed out), but it came from the new CDS. **Fix:** run `standardize_coords` (or check `valid_time` too) before reading the time range, and add `valid_time` to the test.

**F3. Minor: ×24 applies to any input that isn't hourly.** The ×24 depends only on the input not being hourly. A file that is already a daily total, or an hourly file with a single timestamp, would be scaled wrongly. **Fix:** decide from the variable's `units`, or from a flag set by `stream_era5_daily`, instead of the time spacing.

## Fix list for Gemini
1. F1: use `.reindex(time=target_dates)` instead of `.sel` for GREP and ARMOR3D; add a test where GREP ends on 2020-12-15.
2. F2: standardize coordinates (`valid_time` → `time`) inside `filter_files_by_year`; add `valid_time` to the test.
3. F3: apply the ×24 only when the data is a daily mean of hourly accumulations (flag or units).
