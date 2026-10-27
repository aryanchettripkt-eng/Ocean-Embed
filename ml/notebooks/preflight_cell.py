# Pre-flight: tests every remote source with a tiny request (a few minutes total).
from pathlib import Path

import xarray as xr

from oceanscope_ml.data import build_store as bs
from oceanscope_ml.data.download import PRODUCT_SPECS

bbox = (40.125, 99.875, 0.125, 29.875)  # lon_min, lon_max, lat_min, lat_max
dates = ("2017-01-01", "2017-01-02")
tmp = Path("/tmp/preflight")
tmp.mkdir(exist_ok=True)
STATIC = None  # set by the static-fields check; cell 4 passes it to --static-file


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
    out.unlink(missing_ok=True)
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
    kind = "zip" if out.read_bytes()[:2] == b"PK" else "netcdf/other"
    ds = bs._open_cds_download(out)
    return f"file={kind} vars={list(ds.data_vars)} dims={dict(ds.sizes)}"


def era5_arco():
    ds = xr.open_zarr(bs.ARCO_ERA5_URL, chunks=None, consolidated=True, zarr_format=2)
    sub = ds["total_precipitation"].sel(
        time=slice("2017-01-01T00", "2017-01-01T03"),
        latitude=slice(bbox[3] + 1, bbox[2] - 1),
        longitude=slice(bbox[0] - 1, bbox[1] + 1),
    )
    return f"dims={dict(sub.load().sizes)}"


def find_static():
    hits = (
        sorted(Path("/kaggle/input").rglob("static_fields.nc"))
        if Path("/kaggle/input").exists()
        else []
    )
    if not hits:
        raise FileNotFoundError("static_fields.nc not found under /kaggle/input (add the dataset)")
    return hits[0]


def static():
    global STATIC
    STATIC = str(find_static())
    ds = xr.open_dataset(STATIC)
    return f"{STATIC} vars={list(ds.data_vars)} dims={dict(ds.sizes)}"


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
chk("Static fields (GEBCO + WOA)", static)
chk("GREP (Copernicus)", cm("grep", "thetao_mean", (0.0, 1250.0)))
chk("ARMOR3D (Copernicus)", cm("armor3d", "to", (0.0, 1100.0)))
chk("ERA5 (Google Cloud ARCO)", era5_arco)
