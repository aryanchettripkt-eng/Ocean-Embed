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
from oceanscope_ml.data.indices import MJO_URL, NINO34_URL, fetch_mjo, fetch_nino34, save_indices
from oceanscope_ml.data.manifest import (
    load_manifest,
    record_downloaded_file,
    record_planned_product,
    save_manifest,
)

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
        "depth_subset": True,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Primary target. Subsetting 15 depths (+- neighbours) and region bbox.",
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
        "depth_subset": False,
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
        "depth_subset": False,
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
        "depth_subset": False,
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
        "depth_subset": False,
        "env_creds": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Dynamic surface advection vectors.",
    },
    "ccmp": {
        "name": "CCMP Surface Wind Vector",
        "provider": "Remote Sensing Systems",
        "source_id": "CCMP_V03.0_0.25deg",
        "dataset_id": "ccmp_v03.0_daily",
        "source_url": "https://data.remss.com/ccmp/v03.0/",
        "variables": ["uwnd", "vwnd"],
        "needs_lead_in": True,
        "daily_mb": 0.5,
        "depth_subset": False,
        "env_creds": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Dynamic input for wind stress & curl derivation from Remote Sensing Systems.",
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
        "depth_subset": False,
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
        "depth_subset": False,
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
        "depth_subset": False,
        "env_creds": [],
        "notes": "Static monthly climatology profiles (15 channels).",
    },
    "indices": {
        "name": "Climate Indices (Niño 3.4 Anomaly, MJO RMM)",
        "provider": "NOAA CPC/PSL & Australian BoM",
        "source_id": "CLIMATE_INDICES_NINO_MJO",
        "dataset_id": "indices_nino34_anom_mjo",
        "source_url": f"{NINO34_URL} & {MJO_URL}",
        "variables": ["nino34", "mjo_rmm1", "mjo_rmm2"],
        "needs_lead_in": False,
        "static_mb": 1.0,
        "depth_subset": False,
        "env_creds": [],
        "notes": (
            "Niño 3.4 anomaly from NOAA PSL; MJO RMM from BoM (HTTPS). DMI computed from OSTIA."
        ),
    },
    "grep": {
        "name": "GREP Ensemble Mean Reanalysis",
        "provider": "Copernicus Marine Service",
        "source_id": "GLOBAL_MULTIYEAR_PHY_ENS_001_031",
        "dataset_id": "cmems_mod_glo_phy-tem_phymnstd_my_0.25deg_P1D-m",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["thetao"],
        "needs_lead_in": False,
        "daily_mb": 3.0,
        "depth_subset": True,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "Auxiliary training target and external competitor. Subsetting 15 depths.",
    },
    "armor3d": {
        "name": "ARMOR3D L4 Observation-based 3D Temperature",
        "provider": "Copernicus Marine Service",
        "source_id": "MULTIOBS_GLO_PHY_TS_REP_015_004",
        "dataset_id": "dataset-armor-3d-rep-weekly",
        "source_url": "https://marine.copernicus.eu",
        "variables": ["to"],
        "needs_lead_in": False,
        "daily_mb": 1.5,
        "depth_subset": True,
        "env_creds": ["COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"],
        "notes": "External baseline competitor compared at ARGO points; never an input.",
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
        "depth_subset": False,
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
    """Compute planning parameters, coordinate subsets, depths, dates, and sizes."""
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

        plan_item: dict[str, Any] = {
            "name": spec["name"],
            "provider": spec["provider"],
            "source_id": spec["source_id"],
            "dataset_id": spec["dataset_id"],
            "source_url": spec["source_url"],
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

        if spec.get("depth_subset"):
            # 15 depths +- neighbours (0 to 1050 m)
            plan_item["depth_range"] = {"min_depth_m": 0.0, "max_depth_m": 1050.0}

        plan[prod_key] = plan_item

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
    print(" OceanScope AI — Data Download Planning Report (--dry-run)")
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

        record_planned_product(
            manifest=manifest,
            product_id=key,
            source_id=p["source_id"],
            source_url=p["source_url"],
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


def download_indices_product(raw_dir: Path, manifest: dict[str, Any]) -> None:
    """Download Niño 3.4 and MJO indices."""
    print("  Fetching climate indices (Niño 3.4 anomaly and BoM MJO)...")
    indices_dir = raw_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    nino_df = fetch_nino34(NINO34_URL)
    mjo_df = fetch_mjo(MJO_URL)
    saved = save_indices(indices_dir, nino_df=nino_df, mjo_df=mjo_df)

    for name, p_str in saved.items():
        record_downloaded_file(
            manifest=manifest,
            product_id="indices",
            file_path=Path(p_str),
            raw_dir=raw_dir,
            source_id="CLIMATE_INDICES_NINO_MJO",
        )
    print(f"  Saved indices to {indices_dir}")


def download_copernicus(
    key: str, item: dict[str, Any], raw_dir: Path, manifest: dict[str, Any]
) -> None:
    """Download Copernicus Marine product via copernicusmarine toolbox."""
    import copernicusmarine

    out_subdir = raw_dir / key
    out_subdir.mkdir(parents=True, exist_ok=True)
    bbox = item["bbox"]

    kwargs: dict[str, Any] = {
        "dataset_id": item["dataset_id"],
        "variables": item["variables"],
        "minimum_longitude": bbox["lon_min"],
        "maximum_longitude": bbox["lon_max"],
        "minimum_latitude": bbox["lat_min"],
        "maximum_latitude": bbox["lat_max"],
        "start_datetime": f"{item['start_date']}T00:00:00",
        "end_datetime": f"{item['end_date']}T23:59:59",
        "output_directory": str(out_subdir),
        "force_download": False,
    }
    if item.get("depth_range"):
        dr = item["depth_range"]
        kwargs["minimum_depth"] = dr["min_depth_m"]
        kwargs["maximum_depth"] = dr["max_depth_m"]

    print(f"  Downloading {key} ({item['name']}) via copernicusmarine subset...")
    result = copernicusmarine.subset(**kwargs)
    if isinstance(result, list):
        for p in result:
            record_downloaded_file(manifest, key, Path(p), raw_dir, item["source_id"])
    elif result:
        record_downloaded_file(manifest, key, Path(result), raw_dir, item["source_id"])


def download_era5(item: dict[str, Any], raw_dir: Path, manifest: dict[str, Any]) -> None:
    """Download ERA5 surface fluxes via ECMWF CDS API."""
    import cdsapi  # type: ignore[import-not-found]

    out_subdir = raw_dir / "era5"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_file = out_subdir / f"era5_{item['start_date']}_{item['end_date']}.nc"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"  ERA5 file {out_file.name} already exists. Skipping.")
        record_downloaded_file(manifest, "era5", out_file, raw_dir, item["source_id"])
        return

    print("  Downloading ERA5 surface fluxes via cdsapi...")
    c = cdsapi.Client()
    bbox = item["bbox"]
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": item["variables"],
            "area": [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]],
            "date": f"{item['start_date']}/{item['end_date']}",
            "time": [f"{h:02d}:00" for h in range(24)],
        },
        str(out_file),
    )
    record_downloaded_file(manifest, "era5", out_file, raw_dir, item["source_id"])


def download_url_file(
    url: str,
    out_file: Path,
    key: str,
    source_id: str,
    raw_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Download a file from an HTTP/HTTPS URL with size verification."""
    import urllib.request

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"  File {out_file.name} already exists. Skipping.")
        record_downloaded_file(manifest, key, out_file, raw_dir, source_id)
        return

    print(f"  Fetching {key} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "OceanScope-AI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "wb") as out:
        out.write(resp.read())

    record_downloaded_file(manifest, key, out_file, raw_dir, source_id)


def execute_download(
    config: ExperimentConfig,
    raw_dir: Path,
    start_date: str,
    end_date: str,
    products: list[str] | None = None,
) -> None:
    """Execute subsetted download of raw data."""
    plan, _ = plan_download(config, start_date, end_date, products)
    manifest = load_manifest(raw_dir)

    # 1. Check credentials
    missing_creds_total: dict[str, list[str]] = {}
    for key in plan:
        ok, missing = check_product_credentials(key)
        if not ok:
            missing_creds_total[key] = missing

    if missing_creds_total:
        print("\n[ERROR] Missing required credentials for real data download:")
        for k, creds in missing_creds_total.items():
            print(f"  - Product '{k}': missing env var(s) {', '.join(creds)}")
        print("\nTo inspect the planned requests without credentials, run with --dry-run:")
        print("  python -m oceanscope_ml.data.download --dry-run")
        sys.exit(1)

    # 2. Check for copernicus products
    copernicus_prods = [
        k for k in plan if k in ["glorys12", "sla", "sst", "sss", "grep", "armor3d", "argo"]
    ]
    if copernicus_prods:
        try:
            import copernicusmarine  # noqa: F401
        except ImportError:
            print(
                "\n[ERROR] Copernicus Marine toolbox is not installed. "
                "Install with: pip install copernicusmarine\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # 3. Execute downloads product-by-product
    for key, item in plan.items():
        try:
            if key in ["glorys12", "sla", "sst", "sss", "grep", "armor3d"]:
                download_copernicus(key, item, raw_dir, manifest)
            elif key == "era5":
                download_era5(item, raw_dir, manifest)
            elif key == "ccmp":
                # Remote Sensing Systems CCMP daily V03.0 across all dates in range
                import urllib.request

                ccmp_dir = raw_dir / "ccmp"
                ccmp_dir.mkdir(parents=True, exist_ok=True)
                cur_dt = datetime.strptime(item["start_date"], "%Y-%m-%d")
                end_dt = datetime.strptime(item["end_date"], "%Y-%m-%d")

                d_range_ccmp = f"{item['start_date']} to {item['end_date']}"
                print(f"  Downloading CCMP daily wind vectors ({d_range_ccmp})...")
                while cur_dt <= end_dt:
                    d_str = cur_dt.strftime("%Y%m%d")
                    year = cur_dt.strftime("%Y")
                    month = cur_dt.strftime("%m")
                    day_file = ccmp_dir / f"CCMP_Wind_Analysis_{d_str}_V03.0_L4.0.nc"

                    if not (day_file.exists() and day_file.stat().st_size > 0):
                        # Primary: Remote Sensing Systems monthly directory
                        ccmp_url = (
                            f"{item['source_url']}Y{year}/M{month}/"
                            f"CCMP_Wind_Analysis_{d_str}_V03.0_L4.0.nc"
                        )
                        try:
                            req = urllib.request.Request(
                                ccmp_url, headers={"User-Agent": "OceanScope-AI/1.0"}
                            )
                            with urllib.request.urlopen(req, timeout=30) as resp:
                                day_file.write_bytes(resp.read())
                        except Exception:
                            # Fallback: Earthdata PO.DAAC / direct annual path
                            alt_url = (
                                f"{item['source_url']}{year}/"
                                f"CCMP_Wind_Analysis_{d_str}_V03.0_L4.0.nc"
                            )
                            req = urllib.request.Request(
                                alt_url, headers={"User-Agent": "OceanScope-AI/1.0"}
                            )
                            with urllib.request.urlopen(req, timeout=30) as resp:
                                day_file.write_bytes(resp.read())

                    record_downloaded_file(manifest, "ccmp", day_file, raw_dir, item["source_id"])
                    cur_dt += timedelta(days=1)

            elif key == "oscar":
                # NASA PO.DAAC OSCAR ocean currents with Earthdata login
                import gzip
                import urllib.request

                oscar_dir = raw_dir / "oscar"
                oscar_dir.mkdir(parents=True, exist_ok=True)
                out_path = oscar_dir / f"oscar_{item['start_date']}_{item['end_date']}.nc"

                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"  OSCAR file {out_path.name} already exists. Skipping.")
                    record_downloaded_file(manifest, "oscar", out_path, raw_dir, item["source_id"])
                else:
                    print("  Downloading OSCAR ocean currents with Earthdata credentials...")
                    user = os.environ.get("EARTHDATA_USERNAME", "")
                    passwd = os.environ.get("EARTHDATA_PASSWORD", "")

                    downloaded = False
                    # Attempt earthaccess search & download if available
                    try:
                        import earthaccess  # type: ignore[import-not-found]

                        earthaccess.login(strategy="environment")
                        results = earthaccess.search_data(
                            short_name="OSCAR_L4_OC_third-deg",
                            temporal=(item["start_date"], item["end_date"]),
                            bounding_box=(
                                item["bbox"]["lon_min"],
                                item["bbox"]["lat_min"],
                                item["bbox"]["lon_max"],
                                item["bbox"]["lat_max"],
                            ),
                        )
                        if results:
                            downloaded_files = earthaccess.download(results, str(oscar_dir))
                            if downloaded_files:
                                downloaded = True
                                for f_str in downloaded_files:
                                    p = Path(f_str)
                                    if p.name.endswith(".gz"):
                                        decomp = p.with_suffix("")
                                        decomp.write_bytes(gzip.decompress(p.read_bytes()))
                                        p.unlink()
                                        record_downloaded_file(
                                            manifest, "oscar", decomp, raw_dir, item["source_id"]
                                        )
                                    else:
                                        record_downloaded_file(
                                            manifest, "oscar", p, raw_dir, item["source_id"]
                                        )
                    except Exception as e:
                        logger.info("earthaccess fallback: %s", e)

                    if not downloaded:
                        # Authenticated PO.DAAC Cloud archive download
                        year = item["start_date"][:4]
                        oscar_url = (
                            "https://archive.podaac.earthdata.nasa.gov/"
                            f"podaac-ops-cumulus-protected/OSCAR_L4_OC_third-deg/oscar_vel{year}.nc.gz"
                        )
                        req = urllib.request.Request(
                            oscar_url, headers={"User-Agent": "OceanScope-AI/1.0"}
                        )
                        import base64

                        creds_b64 = base64.b64encode(f"{user}:{passwd}".encode()).decode()
                        req.add_header("Authorization", f"Basic {creds_b64}")

                        with urllib.request.urlopen(req, timeout=60) as resp:
                            raw_bytes = resp.read()
                            if raw_bytes[:2] == b"\x1f\x8b":  # gzip magic bytes
                                out_path.write_bytes(gzip.decompress(raw_bytes))
                            else:
                                out_path.write_bytes(raw_bytes)
                        record_downloaded_file(
                            manifest, "oscar", out_path, raw_dir, item["source_id"]
                        )

            elif key == "gebco":
                # GEBCO bathymetry with zip extraction to ensure genuine NetCDF
                import urllib.request
                import zipfile

                gebco_dir = raw_dir / "gebco"
                gebco_dir.mkdir(parents=True, exist_ok=True)
                out_path = gebco_dir / "gebco_2023_sub_ice_topo.nc"

                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"  GEBCO file {out_path.name} already exists. Skipping.")
                    record_downloaded_file(manifest, "gebco", out_path, raw_dir, item["source_id"])
                else:
                    print("  Downloading GEBCO sub-ice bathymetry from BODC...")
                    # BODC NetCDF zip package
                    gebco_zip_url = (
                        "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/"
                    )
                    temp_zip = gebco_dir / "gebco_download.zip"
                    req = urllib.request.Request(
                        gebco_zip_url, headers={"User-Agent": "OceanScope-AI/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        content_bytes = resp.read()

                    if zipfile.is_zipfile(temp_zip) or content_bytes[:4] == b"PK\x03\x04":
                        temp_zip.write_bytes(content_bytes)
                        with zipfile.ZipFile(temp_zip, "r") as z:
                            nc_files = [
                                m for m in z.namelist() if m.endswith(".nc") or m.endswith(".nc4")
                            ]
                            if nc_files:
                                target_nc = nc_files[0]
                                with z.open(target_nc) as src, open(out_path, "wb") as dst:
                                    dst.write(src.read())
                        temp_zip.unlink(missing_ok=True)
                    else:
                        out_path.write_bytes(content_bytes)

                    record_downloaded_file(manifest, "gebco", out_path, raw_dir, item["source_id"])

            elif key == "woa23":
                # World Ocean Atlas 2023 12 monthly climatology files (t01_04 to t12_04)
                woa_dir = raw_dir / "woa23"
                woa_dir.mkdir(parents=True, exist_ok=True)
                print("  Downloading WOA23 12 monthly climatology files (PRD §2, §3)...")
                base_woa = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/0.25/"

                for m_idx in range(1, 13):
                    m_str = f"{m_idx:02d}"
                    out_file = woa_dir / f"woa23_decav_t{m_str}_04.nc"
                    url = f"{base_woa}woa23_decav_t{m_str}_04.nc"
                    download_url_file(
                        url=url,
                        out_file=out_file,
                        key="woa23",
                        source_id=item["source_id"],
                        raw_dir=raw_dir,
                        manifest=manifest,
                    )

            elif key == "argo":
                # Argo GDAC ERDDAP tabledap subsetted NetCDF download (PRD §4.1, §7)
                import urllib.parse
                import urllib.request

                argo_dir = raw_dir / "argo"
                argo_dir.mkdir(parents=True, exist_ok=True)
                out_path = argo_dir / f"argo_profiles_{item['start_date']}_{item['end_date']}.nc"

                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"  ARGO profiles file {out_path.name} already exists. Skipping.")
                    record_downloaded_file(manifest, "argo", out_path, raw_dir, item["source_id"])
                else:
                    d_range = f"{item['start_date']}..{item['end_date']}"
                    print(f"  Querying Argo GDAC ERDDAP for North Indian Ocean ({d_range})...")
                    bbox = item["bbox"]
                    query_vars = "platform_number,cycle_number,time,latitude,longitude,pres,temp"
                    constraints = (
                        f"&time>={item['start_date']}T00:00:00Z"
                        f"&time<={item['end_date']}T23:59:59Z"
                        f"&latitude>={bbox['lat_min']}"
                        f"&latitude<={bbox['lat_max']}"
                        f"&longitude>={bbox['lon_min']}"
                        f"&longitude<={bbox['lon_max']}"
                        f"&pres<=1050"
                    )
                    encoded_query = urllib.parse.quote(query_vars + constraints, safe="&,=,")
                    erddap_url = (
                        f"https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?{encoded_query}"
                    )

                    req = urllib.request.Request(
                        erddap_url, headers={"User-Agent": "OceanScope-AI/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        content_bytes = resp.read()

                    # Verify downloaded file is genuine NetCDF (magic CDF or HDF5)
                    if not (content_bytes[:3] == b"CDF" or content_bytes[:4] == b"\x89HDF"):
                        hdr = content_bytes[:30]
                        raise RuntimeError(f"Argo ERDDAP response was not valid NetCDF: {hdr!r}")

                    out_path.write_bytes(content_bytes)
                    record_downloaded_file(manifest, "argo", out_path, raw_dir, item["source_id"])

            elif key == "indices":
                download_indices_product(raw_dir, manifest)
            else:
                raise RuntimeError(f"No downloader implemented for product '{key}'")
        except Exception as exc:
            print(f"\n[ERROR] Download failed for product '{key}': {exc}", file=sys.stderr)
            sys.exit(1)

    save_manifest(raw_dir, manifest)
    print("\nDownload execution completed successfully.")


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

    print(f"\nInitiating subsetted download to {raw_dir.resolve()}...")
    execute_download(
        config=cfg,
        raw_dir=raw_dir,
        start_date=args.start,
        end_date=args.end,
        products=products,
    )


if __name__ == "__main__":
    main()
