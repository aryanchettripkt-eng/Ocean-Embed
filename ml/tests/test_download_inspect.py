"""Tests for download and inspection modules (PRD_AI_ML §4, §4.1, §7)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from oceanscope_ml.config import load_config
from oceanscope_ml.data.download import plan_download
from oceanscope_ml.data.inspect import (
    generate_markdown_report,
    inspect_raw_data,
    propose_15_depths_from_native,
)
from oceanscope_ml.data.manifest import load_manifest, record_downloaded_file, save_manifest


def test_plan_download_lead_in_and_bbox(tmp_path: Path) -> None:
    """Verify download planning adds lead-in, bounding box buffer, and depth ranges."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text(
        """
project: test
region:
  lon_min: 40.0
  lon_max: 100.0
  lat_min: 0.0
  lat_max: 30.0
inputs:
  long_history:
    days: 60
"""
    )
    cfg = load_config(config_file)

    plan, total_mb = plan_download(
        config=cfg,
        start_date="2019-01-01",
        end_date="2019-01-31",
        selected_products=["glorys12", "sla", "sst", "grep", "armor3d", "ccmp"],
        include_lead_in=True,
    )

    assert total_mb > 0
    assert "glorys12" in plan
    assert "sla" in plan
    assert "grep" in plan
    assert "armor3d" in plan
    assert "ccmp" in plan

    # GLORYS has depth range subsetting
    assert "depth_range" in plan["glorys12"]
    assert plan["glorys12"]["depth_range"]["min_depth_m"] == 0.0

    # SLA needs 60-day lead in: 2019-01-01 minus 60 days is 2018-11-02
    assert plan["sla"]["start_date"] == "2018-11-02"

    # Region buffer: lat_min - 1.0 = -1.0, lat_max + 1.0 = 31.0
    assert plan["glorys12"]["bbox"]["lat_min"] == -1.0
    assert plan["glorys12"]["bbox"]["lat_max"] == 31.0
    assert plan["glorys12"]["bbox"]["lon_min"] == 39.0
    assert plan["glorys12"]["bbox"]["lon_max"] == 101.0

    # SST has extended south boundary for DMI
    assert plan["sst"]["bbox"]["lat_min"] == -10.0

    # GREP uses ensemble-mean dataset_id
    assert "mnstd" in plan["grep"]["dataset_id"]

    # CCMP provider is Remote Sensing Systems
    assert "Remote Sensing Systems" in plan["ccmp"]["provider"]


def test_manifest_lifecycle(tmp_path: Path) -> None:
    """Verify MANIFEST.json creation, file recording, checksum computation, and persistence."""
    manifest = load_manifest(tmp_path)
    assert manifest["manifest_version"] == "1.0"
    assert "products" in manifest

    test_file = tmp_path / "glorys" / "test_slice.nc"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_bytes(b"OCEANSCOPE_DUMMY_DATA_12345")

    record_downloaded_file(
        manifest=manifest,
        product_id="glorys12",
        file_path=test_file,
        raw_dir=tmp_path,
        source_id="GLOBAL_MULTIYEAR_PHY_001_030",
    )

    save_manifest(tmp_path, manifest)
    reloaded = load_manifest(tmp_path)

    assert "glorys12" in reloaded["products"]
    files = reloaded["products"]["glorys12"]["files"]
    assert len(files) == 1
    assert files[0]["rel_path"] == "glorys/test_slice.nc"
    assert files[0]["size_bytes"] == len(b"OCEANSCOPE_DUMMY_DATA_12345")
    assert len(files[0]["sha256"]) == 64


def test_inspect_empty_raw_dir_fails(tmp_path: Path) -> None:
    """Verify inspect_raw_data fails with FileNotFoundError if raw directory has no files."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text("project: test\n")
    cfg = load_config(config_file)

    empty_raw = tmp_path / "raw"
    empty_raw.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="no raw data found"):
        inspect_raw_data(config=cfg, raw_dir=empty_raw)


def test_inspect_real_netcdf_dataset(tmp_path: Path) -> None:
    """Verify inspect_raw_data inspects real NetCDF files and extracts coordinates and depths."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text("project: test\n")
    cfg = load_config(config_file)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic GLORYS NetCDF with 20 native depth levels
    native_depths = [
        0.494,
        1.5,
        2.646,
        5.078,
        9.573,
        14.3,
        21.599,
        34.434,
        55.765,
        92.326,
        155.851,
        266.040,
        350.0,
        453.938,
        600.0,
        763.333,
        900.0,
        1000.0,
        1200.0,
        1452.251,
    ]
    ds = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "lat", "lon"),
                np.ones((2, len(native_depths), 4, 4), dtype=np.float32) * 25.0,
                {"units": "degrees_C", "standard_name": "sea_water_potential_temperature"},
            )
        },
        coords={
            "time": [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")],
            "depth": native_depths,
            "lat": [10.0, 10.25, 10.5, 10.75],
            "lon": [60.0, 60.25, 60.5, 60.75],
        },
    )
    glorys_path = raw_dir / "glorys" / "glorys_test.nc"
    glorys_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(glorys_path)

    # Inspect the raw directory
    inspection = inspect_raw_data(config=cfg, raw_dir=raw_dir)

    assert inspection["file_count"] == 1
    assert len(inspection["files"]) == 1
    file_info = inspection["files"][0]
    assert file_info["file"] == "glorys_test.nc"
    assert "thetao" in file_info["variables"]
    assert file_info["variables"]["thetao"]["units"] == "degrees_C"
    assert len(file_info["native_depths"]) == len(native_depths)

    # 15 depths proposed from native
    proposed = inspection["proposed_15_depths"]
    assert len(proposed) == 15
    assert proposed[0] == native_depths[0]
    assert 1452.251 in proposed

    # Test markdown report generation
    report_md = generate_markdown_report(inspection, cfg)
    assert "[CONFIRM]" in report_md
    assert "glorys_test.nc" in report_md
    assert "Proposed 15 depth tiers" in report_md


def test_propose_15_depths() -> None:
    """Verify propose_15_depths_from_native selects exactly 15 sorted native depths."""
    native = [float(x) for x in range(0, 2000, 50)]  # 40 levels
    proposed = propose_15_depths_from_native(native)
    assert len(proposed) == 15
    assert proposed == sorted(proposed)


def test_inspect_rejects_non_glorys_depths(tmp_path: Path) -> None:
    """Verify inspect_raw_data does NOT propose depths if only GREP/ARMOR3D is present.

    (PRD_AI_ML §2).
    """
    config_file = tmp_path / "exp.yaml"
    config_file.write_text("project: test\n")
    cfg = load_config(config_file)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Write a GREP file with depth coordinates
    ds = xr.Dataset(
        data_vars={
            "thetao": (
                ("depth", "lat", "lon"),
                np.ones((5, 2, 2), dtype=np.float32) * 20.0,
            )
        },
        coords={
            "depth": [0.5, 10.0, 50.0, 100.0, 500.0],
            "lat": [10.0, 12.0],
            "lon": [60.0, 62.0],
        },
    )
    grep_path = raw_dir / "grep" / "grep_ensemble.nc"
    grep_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(grep_path)

    inspection = inspect_raw_data(config=cfg, raw_dir=raw_dir)
    assert inspection["glorys_status"] == "missing_glorys_file"
    assert inspection["proposed_15_depths"] == []

    report_md = generate_markdown_report(inspection, cfg)
    assert "No GLORYS reanalysis file found" in report_md
    assert "cannot be derived from GREP or ARMOR3D" in report_md


def test_argo_stratified_regional_holdout(tmp_path: Path) -> None:
    """Verify ARGO holdout stratifies by region and covers Bay of Bengal years (PRD_AI_ML §7)."""
    import pandas as pd

    from oceanscope_ml.data.inspect import build_real_argo_holdout

    csv_path = tmp_path / "argo_profiles.csv"
    # Create 15 floats: 5 BoB (years 2018, 2019, 2020), 5 Arabian Sea, 5 Equatorial
    rows = []
    for i in range(1, 6):
        # BoB (lat >= 5.0, lon >= 78.0)
        yr = 2018 + (i % 3)
        rows.append(
            {
                "platform_number": f"bob_{i}",
                "latitude": 15.0,
                "longitude": 85.0,
                "date": f"{yr}-06-15",
            }
        )
    for i in range(1, 6):
        # Arabian Sea (lat >= 5.0, lon < 78.0)
        rows.append(
            {
                "platform_number": f"as_{i}",
                "latitude": 15.0,
                "longitude": 65.0,
                "date": "2019-06-15",
            }
        )
    for i in range(1, 6):
        # Equatorial (lat < 5.0)
        rows.append(
            {"platform_number": f"eq_{i}", "latitude": 2.0, "longitude": 75.0, "date": "2019-06-15"}
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    holdout = build_real_argo_holdout([csv_path], seed=42, holdout_fraction=0.20)

    assert holdout["total_floats"] == 15
    assert holdout["counts_by_region"]["bay_of_bengal"] == 5
    assert holdout["counts_by_region"]["arabian_sea"] == 5
    assert holdout["counts_by_region"]["equatorial"] == 5

    # 20% of 5 is 1 in each region
    assert holdout["held_out_by_region"]["bay_of_bengal"] >= 1
    assert holdout["held_out_by_region"]["arabian_sea"] >= 1
    assert holdout["held_out_by_region"]["equatorial"] >= 1
    assert len(holdout["held_out_floats"]) >= 3


def test_is_glorys_file() -> None:
    """Verify is_glorys_file recognizes Copernicus GLORYS names and folders."""
    from oceanscope_ml.data.inspect import is_glorys_file

    # 1. Copernicus official filename without 'glorys' in name
    p1 = Path("/data/raw/cmems_mod_glo_phy_my_0.083deg_P1D-m_2019-01-01.nc")
    assert is_glorys_file(p1) is True

    # 2. Generic filename inside a glorys12 folder
    p2 = Path("/data/raw/glorys12/data_20190101.nc")
    assert is_glorys_file(p2) is True

    # 3. Product code 001_030
    p3 = Path("/data/raw/GLOBAL_MULTIYEAR_PHY_001_030_daily.nc")
    assert is_glorys_file(p3) is True

    # 4. Competitor GREP (001_031) must be rejected
    p4 = Path("/data/raw/grep/cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m.nc")
    assert is_glorys_file(p4) is False

    # 5. Competitor ARMOR3D (015_012) must be rejected
    p5 = Path("/data/raw/armor3d/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m.nc")
    assert is_glorys_file(p5) is False


def test_download_specs_and_formatting() -> None:
    """Verify CCMP daily path, OSCAR v2 dataset name, and GEBCO sub-ice topo grid."""
    from oceanscope_ml.data.download import PRODUCT_SPECS

    # 1. CCMP has daily/ source_url
    assert PRODUCT_SPECS["ccmp"]["source_url"].endswith("daily/")

    # 2. OSCAR uses active v2 dataset
    assert PRODUCT_SPECS["oscar"]["source_id"] == "OSCAR_L4_OC_FINAL_V2.0"
    assert PRODUCT_SPECS["oscar"]["dataset_id"] == "OSCAR_L4_OC_FINAL_V2.0"

    # 3. GEBCO uses sub-ice topography grid
    assert PRODUCT_SPECS["gebco"]["source_id"] == "GEBCO_2023_sub_ice_topo"
    assert PRODUCT_SPECS["gebco"]["dataset_id"] == "gebco_2023_sub_ice_topo"
    assert "sub-ice" in PRODUCT_SPECS["gebco"]["notes"]
