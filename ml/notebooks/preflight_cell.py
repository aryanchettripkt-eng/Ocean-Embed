# Pre-flight: tests every remote source with a tiny request (a few minutes total).
import urllib.request
from pathlib import Path

import xarray as xr

from oceanscope_ml.data import build_store as bs
from oceanscope_ml.data.download import PRODUCT_SPECS

bbox = (40.125, 99.875, 0.125, 29.875)  # lon_min, lon_max, lat_min, lat_max
dates = ("2017-01-01", "2017-01-02")
tmp = Path("/tmp/preflight")
tmp.mkdir(exist_ok=True)


def chk(name, fn):
    try:
        print("OK  ", name, "->", fn())
    except Exception as e:
        print("FAIL", name, "->", repr(e)[:400])


def oscar():
    import earthaccess

    earthaccess.login(strategy="environment")
    r = earthaccess.search_data(
        short_name="OSCAR_L4_OC_FINAL_V2.0",
        temporal=dates,
        bounding_box=(bbox[0], bbox[2], bbox[1], bbox[3]),
    )
    return f"{len(r)} granules found"


def era5():
    import cdsapi

    out = tmp / "era5_test.nc"
    cdsapi.Client().retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_net_solar_radiation",
                "surface_net_thermal_radiation",
                "surface_sensible_heat_flux",
                "surface_latent_heat_flux",
                "total_precipitation",
            ],
            "year": "2017",
            "month": ["01"],
            "day": ["01"],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [bbox[3] + 1, bbox[0] - 1, bbox[2] - 1, bbox[1] + 1],
            "format": "netcdf",
        },
        str(out),
    )
    ds = xr.open_dataset(out)
    return f"vars={list(ds.data_vars)} dims={dict(ds.sizes)}"


def gebco():
    url = (
        "https://data.ceda.ac.uk/bodc/gebco/global/gebco_2023/sub_ice_topo/netcdf/"
        "gebco_2023_sub_ice_topo.nc"
    )
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "OceanScope-AI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return f"HTTP {r.status}, {int(r.headers.get('Content-Length', 0)) / 1e9:.1f} GB"


def woa():
    url = (
        "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/decav/0.25/"
        "woa23_decav_t01_04.nc"
    )
    ds = xr.open_dataset(url, decode_times=False)
    return f"vars include t_an={'t_an' in ds.data_vars} dims={dict(ds.sizes)}"


def cm(key, var, depths=None):
    def f():
        ds = bs.stream_copernicus(
            dataset_id=PRODUCT_SPECS[key]["dataset_id"],
            bbox=bbox,
            dates=dates,
            variables=[var],
            depths=depths,
        )
        return f"dims={dict(ds.sizes)}"

    return f


chk("OSCAR (Earthdata)", oscar)
chk("GEBCO (CEDA)", gebco)
chk("WOA23 (OPeNDAP)", woa)
chk("GREP (Copernicus)", cm("grep", "thetao_mean", (0.0, 1250.0)))
chk("ARMOR3D (Copernicus)", cm("armor3d", "to", (0.0, 1100.0)))
chk("ERA5 (CDS, may queue a few minutes)", era5)
