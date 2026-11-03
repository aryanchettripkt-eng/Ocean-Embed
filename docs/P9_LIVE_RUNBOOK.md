# P9 live mode runbook

Code: `ml/src/oceanscope_ml/live/` (P9-3) and `ml/src/oceanscope_ml/evaluation/live_check.py` (P9-1).
Settings: `ml/configs/live.yaml`. It is kept out of `experiment.yaml` because `train.py` refuses to resume a run whose config hash changed.
Workflow: `.github/workflows/live.yml` (manual trigger only until the schedule is uncommented).

## Steps for Aryan, in order

1. **Confirm the NRT dataset IDs** in `configs/live.yaml` (`live.sources`) with `copernicusmarine describe --dataset-id <id>`. The OSCAR NRT short name is `OSCAR_L4_OC_NRT_V2.0`. ERA5T comes from the CDS daily-statistics dataset.
2. **Build the live kit** from the run that is published now (`active_run_id` in the HF `manifest.json`), then upload it to a private HF model repo. Set `live.kit_repo` to that repo first. The live run refuses a kit that is not for the active run.
   - As of 2026-11-03 the active run is `main2_ft`. Its checkpoint is `main2_ft_l0.3/best_ft.pt` from the fine-tune notebook, and the stats store comes from the stats notebook.
   - The checkpoint stores the name `main2_ft_l0.3`, so pass `--run-id main2_ft` (the published name):
   ```
   cd ml
   python -m oceanscope_ml.live kit --checkpoint <checkpoints>/main2_ft_l0.3/best_ft.pt \
       --stats <stats.zarr> --static-store <any part.zarr> --out kit/ \
       --run-id main2_ft --upload
   ```
   - Rebuild the kit whenever a new run is published.
3. **P9-1 check**: run the model on one past month with NRT inputs and with reprocessed inputs:
   ```
   python -m oceanscope_ml.evaluation.live_check --kit-dir kit/ --month YYYY-MM \
       [--glorys-dataset cmems_mod_glo_phy_myint_0.083deg_P1D-m]
   ```
   - The month cannot be in 2021 (sealed test year). Pick a month where both the NRT archive and GLORYS exist.
   - The script needs the Copernicus, Earthdata and CDS credentials in the environment.
   - Read `outputs/reports/live_mode_check.md` and decide whether to enable live mode. The script does not set a pass mark.
4. **Add GitHub secrets**: `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `CDSAPI_KEY`, `HF_TOKEN`. The HF token needs write access to the outputs repo and read access to the kit repo.
5. **Run once by hand**: Actions › Live › Run workflow. First run with `dry_run` on, then with it off. Manual runs need the workflow on the default branch.
6. **Enable the schedule**: uncomment `schedule:` in `live.yml` (daily 06:00 UTC).
7. **P9-4**: let it run 7 days unattended. Then read the run logs (step summary and the `live-run-log` artifact).

## What one run does
- **Picks the day.** It chooses the newest day, within `max_lag_days`, for which every input exists for the day and for its whole history window. Newer days are skipped, and the log says why. If no day qualifies, the run fails (red).
- **Stops early when there is nothing new.** If that day is already published, the run ends green with status `up_to_date`.
- **Appends the day to the published store.** It downloads only the store's metadata and coordinates, not the data. The day goes in with the `live` label in `split_of_each_date` and NaN `temp_glorys`.
- **Keeps a rolling window.** Live days are capped at 365. Trimming only starts once there are 30 days past the cap, so it runs about monthly.
- **Uploads data before the manifest.** The changed store files go up in one HF commit, then `manifest.json` (`available_dates.end`) in its own commit. If the manifest commit is lost, the next run repairs it first.
- **Expected lag.** ERA5T on CDS is about 5 days behind, so expect live days about 5–6 days behind real time, not 3. The log records each input's latency.
