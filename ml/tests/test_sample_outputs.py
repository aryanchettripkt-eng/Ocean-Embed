"""Tests proving contract validity of sample outputs fixtures (Review A2 & P0 exit gate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from oceanscope_ml.fixtures.make_sample_outputs import generate_sample_outputs


def _get_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "oceanscope_ml"
        / "schemas"
        / "manifest.schema.json"
    )
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def _validate_output_tree(out_dir: Path) -> None:
    # 1. Validate manifest against JSON schema
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    schema = _get_schema()
    validator = jsonschema.Draft7Validator(
        schema,
        format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER,
    )
    validator.validate(manifest)
    assert manifest["synthetic"] is True
    assert len(manifest["depths_m"]) == 15

    # 2. Validate Zarr store can be opened by xarray and satisfies Section 6.2
    zarr_rel_path = manifest["files"]["predictions"]
    zarr_full_path = out_dir / zarr_rel_path
    assert zarr_full_path.exists(), f"Missing predictions zarr at {zarr_full_path}"

    ds = xr.open_zarr(str(zarr_full_path))
    assert set(ds.data_vars) == {"temp_pred", "temp_glorys", "coverage"}
    assert set(ds.coords) == {"time", "depth", "lat", "lon"}

    # Coordinate assertions
    assert len(ds.depth) == 15
    assert len(ds.lat) == 30  # 0.5 to 29.5
    assert len(ds.lon) == 60  # 40.5 to 99.5
    assert np.allclose(ds.lat.values[0], 0.5)
    assert np.allclose(ds.lon.values[0], 40.5)

    # Chunking: one day per chunk (time chunk == 1)
    for var_name in ["temp_pred", "temp_glorys", "coverage"]:
        chunks = ds[var_name].encoding.get("chunks")
        assert chunks is not None, f"Variable {var_name} missing chunk encoding"
        assert chunks[0] == 1, f"Variable {var_name} chunk[0] is {chunks[0]}, expected 1 (daily)"

    # Coverage values subset of {0, 1, 2}
    cov_vals = set(np.unique(ds["coverage"].values))
    assert cov_vals.issubset({0, 1, 2}), f"Unexpected coverage values: {cov_vals}"

    # Masking: temp_pred must be NaN wherever coverage != 1
    cov_arr = ds["coverage"].values
    pred_arr = ds["temp_pred"].values
    invalid_mask = cov_arr != 1
    # Broadcast across depth
    invalid_broadcast = np.broadcast_to(
        invalid_mask[:, np.newaxis, :, :],
        pred_arr.shape,
    )
    assert np.all(np.isnan(pred_arr[invalid_broadcast])), "temp_pred not NaN where coverage != 1"

    # Live dates (§12): last 2 days (index 8, 9) have all-NaN temp_glorys
    glorys_arr = ds["temp_glorys"].values
    assert np.all(np.isnan(glorys_arr[8:])), "temp_glorys not all-NaN on live dates"
    # Predictions on live dates must exist where coverage == 1
    assert np.any(~np.isnan(pred_arr[8:])), "temp_pred has no valid predictions on live dates"

    # Root attributes from PRD_AI_ML Section 6.2
    for attr in [
        "run_id",
        "dataset_version",
        "model_name",
        "created_utc",
        "split_of_each_date",
        "depths_m",
    ]:
        assert attr in ds.attrs, f"Missing root attribute '{attr}' in prediction store"

    splits_map = json.loads(ds.attrs["split_of_each_date"])
    assert any("live" in v for v in splits_map.values()), (
        "Missing 'live' split in split_of_each_date"
    )

    # Splits from PRD_AI_ML Section 2 / 6.3
    assert manifest["splits"]["train"] == ["1993-03-02", "2017-12-01"]
    assert manifest["splits"]["val"] == ["2018-01-01", "2020-12-01"]
    assert manifest["splits"]["test"] == ["2021-01-01", "2021-12-31"]

    # 3. Validate metrics CSVs
    val_csv_path = out_dir / manifest["files"]["metrics_val"]
    val_df = pd.read_csv(val_csv_path)
    expected_cols = {
        "model",
        "split",
        "depth_m",
        "n",
        "mae_c",
        "rmse_c",
        "bias_c",
        "skill_vs_clim_pct",
    }
    assert expected_cols.issubset(val_df.columns)
    # Check depth_m = all exists and n equals sum of per-depth n
    assert "all" in val_df["depth_m"].astype(str).values
    depth_rows = val_df[val_df["depth_m"].astype(str) != "all"]
    all_row = val_df[val_df["depth_m"].astype(str) == "all"].iloc[0]
    assert int(depth_rows["n"].sum()) == int(all_row["n"])

    # Season CSV
    season_path = out_dir / manifest["files"]["metrics_val_by_season"]
    season_df = pd.read_csv(season_path)
    for season in ["DJF", "MAM", "JJA", "SON"]:
        s_df = season_df[season_df["season"] == season]
        s_depths = s_df[s_df["depth_m"].astype(str) != "all"]
        s_all = s_df[s_df["depth_m"].astype(str) == "all"].iloc[0]
        assert int(s_depths["n"].sum()) == int(s_all["n"])

    # Region CSV
    region_path = out_dir / manifest["files"]["metrics_val_by_region"]
    region_df = pd.read_csv(region_path)
    for reg in ["arabian_sea", "bay_of_bengal", "equatorial"]:
        r_df = region_df[region_df["region"] == reg]
        r_depths = r_df[r_df["depth_m"].astype(str) != "all"]
        r_all = r_df[r_df["depth_m"].astype(str) == "all"].iloc[0]
        assert int(r_depths["n"].sum()) == int(r_all["n"])

    # ARGO parquet and CSV (including GREP/ARMOR3D competitor columns)
    argo_csv = out_dir / manifest["files"]["argo"]
    argo_df = pd.read_csv(argo_csv)
    expected_argo_cols = {
        "depth_m",
        "n_profiles",
        "n_obs",
        "mae_c",
        "rmse_c",
        "bias_c",
        "glorys_mae_c",
        "glorys_rmse_c",
        "grep_mae_c",
        "grep_rmse_c",
        "armor3d_mae_c",
        "armor3d_rmse_c",
    }
    assert expected_argo_cols.issubset(argo_df.columns)

    argo_parquet = out_dir / manifest["files"]["argo_matches"]
    pq_table = pq.read_table(argo_parquet)
    assert len(pq_table) > 0
    expected_parquet_cols = {
        "platform_number",
        "cycle",
        "time",
        "lat",
        "lon",
        "depth_m",
        "temp_obs_c",
        "temp_pred_c",
        "temp_glorys_c",
        "temp_grep_c",
        "temp_armor3d_c",
    }
    assert expected_parquet_cols.issubset(set(pq_table.column_names))

    # Headline table (Section 1.1 / 7)
    headline_rel = manifest["files"].get("headline")
    assert headline_rel is not None, "Missing files.headline in manifest"
    headline_path = out_dir / headline_rel
    assert headline_path.exists(), f"Missing headline file at {headline_path}"
    headline_df = pd.read_csv(headline_path)
    expected_headline_cols = {
        "period",
        "product",
        "n_floats",
        "n_obs",
        "rmse_c",
        "ci_low_c",
        "ci_high_c",
        "diff_vs_model_c",
        "diff_ci_low_c",
        "diff_ci_high_c",
    }
    assert expected_headline_cols.issubset(headline_df.columns)
    assert len(headline_df) > 0


def test_sample_outputs_generation_in_temp_dir(tmp_path: Path) -> None:
    """Verify that running the generator produces contract-exact valid outputs."""
    generate_sample_outputs(tmp_path)
    _validate_output_tree(tmp_path)


def test_committed_sample_outputs() -> None:
    """Verify that the committed fixtures/sample_outputs directory is valid."""
    committed_dir = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_outputs"
    assert committed_dir.exists(), f"Committed fixtures not found at {committed_dir}"
    _validate_output_tree(committed_dir)
