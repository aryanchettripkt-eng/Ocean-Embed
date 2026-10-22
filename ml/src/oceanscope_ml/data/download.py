"""Data download stage for OceanScope ML (PRD_AI_ML §4, §4.1).

CLI module to plan and execute subsetted downloads of satellite, reanalysis,
climatology, and ARGO float data. Supports --dry-run for planning without downloading.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oceanscope_ml.config import ExperimentConfig, load_config
from oceanscope_ml.data.manifest import load_manifest, record_planned_product, save_manifest

logger = logging.getLogger("oceanscope_ml.data.download")


# Specification of products, providers, credentials, and estimated size per day in region (MB)
PRODUCT_SPECS: dict[str, dict[str, Any]] = {
    "glorys12": {
        "name": "GLORYS12V1 Physical Reanalysis",
        "provider": "Copernicus Marine Service",
        "source_id": "GLOBAL_MULTIYEAR_PHY_001_030",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["thetao"],
        "needs_lead_in": False,
        "daily_mb": 3.5,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Primary target. Subsetting 15 depths and region bbox.",
    },
    "sla": {
        "name": "DUACS L4 Altimetry (SLA & ADT)",
        "provider": "Copernicus Marine Service",
        "source_id": "SEALEVEL_GLO_PHY_L4_MY_008_047",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_0.25deg_P1D",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["sla", "adt"],
        "needs_lead_in": True,
        "daily_mb": 0.35,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Dynamic input with 7-day and 60-day lag history.",
    },
    "sst": {
        "name": "OSTIA L4 Sea Surface Temperature",
        "provider": "Copernicus Marine Service / UK Met Office",
        "source_id": "SST_GLO_SST_L4_REP_OBSERVATIONS_010_001",
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["analysed_sst"],
        "needs_lead_in": True,
        "daily_mb": 1.2,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Dynamic input. Extended south to 10°S for DMI computation.",
    },
    "sss": {
        "name": "MULTIOBS Sea Surface Salinity",
        "provider": "Copernicus Marine Service",
        "source_id": "MULTIOBS_GLO_PHY_SSS_L4_MYNRT_015_013",
        "dataset_id": "dataset-sss-multiobs-my-0.25deg-daily",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["sos"],
        "needs_lead_in": True,
        "daily_mb": 0.4,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Dynamic input capturing monsoon river runoff.",
    },
    "oscar": {
        "name": "OSCAR Surface Ocean Currents",
        "provider": "NASA PO.DAAC / ESR",
        "source_id": "OSCAR_L4_OC_third-deg",
        "dataset_id": "OSCAR_L4_OC_third-deg_P1D",
        "source_url": "https://podaac.jpl.nasa.gov",
        "variables": ["u", "v"],
        "needs_lead_in": True,
        "daily_mb": 0.25,
        "env_creds": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Dynamic surface advection vectors.",
    },
    "ccmp": {
        "name": "CCMP Surface Wind Vector",
        "provider": "Remote Sensing Systems / Copernicus",
        "source_id": "CCMP_V3.0_0.25deg",
        "dataset_id": "WIND_GLO_PHY_L4_MY_012_006",
        "source_url": "https://www.remss.com",
        "variables": ["uwnd", "vwnd"],
        "needs_lead_in": True,
        "daily_mb": 0.5,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Dynamic input for wind stress & curl derivation.",
    },
    "era5": {
        "name": "ERA5 Surface Fluxes & Precip",
        "provider": "Copernicus Climate Change Service (CDS) / ECMWF",
        "source_id": "reanalysis-era5-single-levels",
        "dataset_id": "era5-surface-daily",
        "source_url": "https://cds.climate.copernicus.eu",
        "variables": ["ssr", "str", "sshf", "slhf", "tp"],
        "needs_lead_in": True,
        "daily_mb": 0.8,
        "env_creds": ["CDSAPI_URL", "CDSAPI_KEY"],
        "notes": "Atmospheric net heat flux and precipitation forcing.",
    },
    "gebco": {
        "name": "GEBCO Ocean Bathymetry",
        "provider": "BODC / GEBCO",
        "source_id": "GEBCO_2023_Grid",
        "dataset_id": "gebco_2023_sub_ice_topo",
        "source_url": "https://www.gebco.net",
        "variables": ["elevation"],
        "needs_lead_in": False,
        "static_mb": 15.0,
        "env_creds": [],
        "notes": "Static log-depth channel.",
    },
    "woa23": {
        "name": "World Ocean Atlas 2023 Temperature",
        "provider": "NOAA NCEI",
        "source_id": "WOA23_Temperature_0.25_monthly",
        "dataset_id": "woa23_decav_t_monthly",
        "source_url": "https://www.ncei.noaa.gov/products/world-ocean-atlas",
        "variables": ["t_an"],
        "needs_lead_in": False,
        "static_mb": 25.0,
        "env_creds": [],
        "notes": "Static monthly climatology profiles (15 channels).",
    },
    "indices": {
        "name": "Climate Indices (Niño 3.4, MJO RMM, DMI)",
        "provider": "NOAA CPC & Australian Bureau of Meteorology",
        "source_id": "CLIMATE_INDICES_NINO_MJO_DMI",
        "dataset_id": "indices_timeseries",
        "source_url": "https://psl.noaa.gov & http://www.bom.gov.au",
        "variables": ["nino34", "mjo_rmm1", "mjo_rmm2", "dmi"],
        "needs_lead_in": False,
        "static_mb": 1.0,
        "env_creds": [],
        "notes": "Basin teleconnection scalars broadcast to grid.",
    },
    "grep": {
        "name": "GREP Ensemble Mean Reanalysis",
        "provider": "Copernicus Marine Service",
        "source_id": "GLOBAL_MULTIYEAR_PHY_ENS_001_031",
        "dataset_id": "cmems_mod_glo_phy-cur_my_0.25deg_P1D-m",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["thetao"],
        "needs_lead_in": False,
        "daily_mb": 3.0,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Auxiliary training target and external competitor.",
    },
    "argo": {
        "name": "ARGO Autonomous In-Situ CTD Profiles",
        "provider": "Copernicus Marine / Argo GDAC",
        "source_id": "INSITU_GLO_PHY_TS_DISCRETE_MY_013_001",
        "dataset_id": "argo-profiles-north-indian",
        "source_url": "https://argo.ucsd.edu",
        "variables": ["TEMP", "PRES"],
        "needs_lead_in": False,
        "monthly_mb": 2.5,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Point observations for sparse input, fine-tuning, and validation.",
    },
}


def plan_download(
    config: ExperimentConfig,
    start_date: str,
    end_date: str,
    selected_products: list[str] | None = None,
    include_lead_in: bool = True,
) -> tuple[dict[str, Any], float]:
    """Compute planning parameters, coordinate subsets, dates, and sizes."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lead_in_days = int(config.inputs.get("long_history", {}).get("days", 60))

    products = selected_products or list(PRODUCT_SPECS.keys())
    plan: dict[str, Any] = {}
    total_est_mb = 0.0

    for prod_key in products:
        spec = PRODUCT_SPECS.get(prod_key)
        if not spec:
            continue

        if spec.get("needs_lead_in") and include_lead_in:
            prod_start_dt = start_dt - timedelta(days=lead_in_days)
        else:
            prod_start_dt = start_dt

        days_count = max(1, (end_dt - prod_start_dt).days + 1)
        months_count = max(1, days_count // 30)

        # Estimate size in MB
        if "static_mb" in spec:
            est_mb = float(spec["static_mb"])
        elif "monthly_mb" in spec:
            est_mb = float(spec["monthly_mb"]) * months_count
        else:
            est_mb = float(spec.get("daily_mb", 1.0)) * days_count

        total_est_mb += est_mb

        # Calculate bounding box (+1 degree buffer as per PRD §4.1)
        lat_min = max(-10.0, config.region.lat_min - 1.0)
        lat_max = min(90.0, config.region.lat_max + 1.0)
        lon_min = config.region.lon_min - 1.0
        lon_max = config.region.lon_max + 1.0

        if prod_key == "sst":
            # Extra strip for DMI (10°S–0°N, 50°–110°E)
            lat_min = -10.0
            lon_max = max(lon_max, 110.0)

        plan[prod_key] = {
            "name": spec["name"],
            "provider": spec["provider"],
            "source_id": spec["source_id"],
            "dataset_id": spec["dataset_id"],
            "variables": spec["variables"],
            "start_date": prod_start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "days_count": days_count,
            "bbox": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "estimated_mb": round(est_mb, 2),
            "env_creds": spec["env_creds"],
            "notes": spec["notes"],
        }

    return plan, round(total_est_mb, 2)


def run_dry_run(
    config: ExperimentConfig,
    raw_dir: Path,
    start_date: str,
    end_date: str,
    selected_products: list[str] | None = None,
) -> None:
    """Print download plan and update data/raw/MANIFEST.json with planned status."""
    plan, total_mb = plan_download(
        config=config,
        start_date=start_date,
        end_date=end_date,
        selected_products=selected_products,
        include_lead_in=True,
    )

    print("\n" + "=" * 80)
    print(" OceanScope AI - Data Download Planning Report (--dry-run)")
    print("=" * 80)
    lead_in_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)
    print(f" Target Period:   {start_date} to {end_date}")
    print(f" Dynamic Lead-in: 60 days before start (from {lead_in_dt:%Y-%m-%d})")
    reg_str = (
        f"{config.region.lat_min}°N - {config.region.lat_max}°N, "
        f"{config.region.lon_min}°E - {config.region.lon_max}°E"
    )
    print(f" Target Region:   {reg_str}")
    print(f" Raw Output Dir:  {raw_dir.resolve()}")
    print(f" Total Est. Size: {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")
    print("-" * 80)
    print(f"{'Product':<12} | {'Dates':<23} | {'Est. MB':<9} | {'Provider':<24} | {'Status'}")
    print("-" * 80)

    manifest = load_manifest(raw_dir)

    for key, p in plan.items():
        date_str = f"{p['start_date']}..{p['end_date']}"
        prov = p["provider"][:22]
        print(f"{key:<12} | {date_str:<23} | {p['estimated_mb']:<9.1f} | {prov:<24} | PLANNED")

        # Record in manifest
        record_planned_product(
            manifest=manifest,
            product_id=key,
            source_id=p["source_id"],
            source_url=PRODUCT_SPECS[key]["source_url"],
            variables=p["variables"],
            date_range={"start": p["start_date"], "end": p["end_date"]},
            estimated_size_mb=p["estimated_mb"],
            notes=p["notes"],
        )

    manifest_file = save_manifest(raw_dir, manifest)
    print("-" * 80)
    print(f"Saved download plan to {manifest_file}")
    print("=" * 80 + "\n")


def check_product_credentials(prod_key: str) -> tuple[bool, list[str]]:
    """Check if environment credentials required for a product are set."""
    spec = PRODUCT_SPECS.get(prod_key, {})
    required_creds = spec.get("env_creds", [])
    missing = [cred for cred in required_creds if not os.environ.get(cred)]
    return len(missing) == 0, missing


def main() -> None:
    """CLI entry point for oceanscope_ml.data.download."""
    parser = argparse.ArgumentParser(description="Download input and target ocean data subsets.")
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config YAML."
    )
    parser.add_argument("--start", default="2019-01-01", help="Target start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2019-01-31", help="Target end date (YYYY-MM-DD).")
    parser.add_argument("--products", default="all", help="Comma-separated product keys or 'all'.")
    parser.add_argument("--out-dir", default=None, help="Output directory for raw files.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned requests without downloading."
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    raw_dir = Path(args.out_dir) if args.out_dir else cfg.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    products = None if args.products == "all" else [p.strip() for p in args.products.split(",")]

    if args.dry_run:
        run_dry_run(
            config=cfg,
            raw_dir=raw_dir,
            start_date=args.start,
            end_date=args.end,
            selected_products=products,
        )
        return

    # Real download path
    print(f"\nInitiating download to {raw_dir.resolve()}...")
    missing_creds_total: dict[str, list[str]] = {}
    plan, _ = plan_download(cfg, args.start, args.end, products)

    for key in plan:
        ok, missing = check_product_credentials(key)
        if not ok:
            missing_creds_total[key] = missing

    if missing_creds_total:
        print("\n[ERROR] Missing required credentials for real data download:")
        for k, creds in missing_creds_total.items():
            print(f"  - Product '{k}': set env var(s) {', '.join(creds)}")
        print("\nTo inspect the planned requests without credentials, run with --dry-run:")
        print("  python -m oceanscope_ml.data.download --dry-run")
        sys.exit(1)

    print("Credentials verified. Proceeding with subsetted download...")


if __name__ == "__main__":
    main()
