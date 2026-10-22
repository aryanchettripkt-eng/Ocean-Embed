"""Tests for download and inspection modules (PRD_AI_ML §4, §4.1, §7)."""

from pathlib import Path

from oceanscope_ml.config import load_config
from oceanscope_ml.data.download import plan_download
from oceanscope_ml.data.inspect import (
    DEFAULT_GLORYS_15_DEPTHS,
    generate_markdown_report,
    generate_stratified_argo_holdout,
    inspect_raw_data,
)
from oceanscope_ml.data.manifest import load_manifest, record_downloaded_file, save_manifest


def test_plan_download_lead_in_and_bbox(tmp_path: Path) -> None:
    """Verify download planning correctly adds 60-day lead in and 1-deg bounding box buffer."""
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
        selected_products=["glorys12", "sla", "sst"],
        include_lead_in=True,
    )

    assert total_mb > 0
    assert "glorys12" in plan
    assert "sla" in plan

    # GLORYS does not need lead in
    assert plan["glorys12"]["start_date"] == "2019-01-01"

    # SLA needs 60-day lead in: 2019-01-01 minus 60 days is 2018-11-02
    assert plan["sla"]["start_date"] == "2018-11-02"

    # Region buffer: lat_min - 1.0 = -1.0, lat_max + 1.0 = 31.0
    assert plan["glorys12"]["bbox"]["lat_min"] == -1.0
    assert plan["glorys12"]["bbox"]["lat_max"] == 31.0
    assert plan["glorys12"]["bbox"]["lon_min"] == 39.0
    assert plan["glorys12"]["bbox"]["lon_max"] == 101.0

    # SST has extended south boundary for DMI
    assert plan["sst"]["bbox"]["lat_min"] == -10.0


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


def test_stratified_argo_holdout() -> None:
    """Verify 20% stratified float holdout covers all regional basins reproducibility."""
    holdout1 = generate_stratified_argo_holdout(seed=42)
    holdout2 = generate_stratified_argo_holdout(seed=42)

    assert holdout1["total_held_out"] == holdout2["total_held_out"]
    assert holdout1["held_out_floats"] == holdout2["held_out_floats"]

    # Verify each subregion has held-out floats
    strat = holdout1["stratification"]
    assert "bay_of_bengal" in strat
    assert "arabian_sea" in strat
    assert "equatorial" in strat

    assert strat["bay_of_bengal"]["held_out_floats"] > 0
    assert strat["arabian_sea"]["held_out_floats"] > 0
    assert strat["equatorial"]["held_out_floats"] > 0


def test_inspect_and_report(tmp_path: Path) -> None:
    """Verify inspection report contains the 15 depths and confirmation table."""
    config_file = tmp_path / "exp.yaml"
    config_file.write_text("project: test\n")
    cfg = load_config(config_file)

    inspection = inspect_raw_data(config=cfg, raw_dir=tmp_path)
    holdout = generate_stratified_argo_holdout(seed=42)
    report_md = generate_markdown_report(inspection, holdout, cfg)

    assert "[CONFIRM]" in report_md
    assert "GLORYS12" in report_md
    assert str(DEFAULT_GLORYS_15_DEPTHS[0]) in report_md
    assert "ARGO Float Stratified Hold-Out" in report_md
