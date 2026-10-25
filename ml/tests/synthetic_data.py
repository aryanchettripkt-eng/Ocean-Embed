"""Synthetic data generator for ML tests only (PRD_AI_ML Section 0 Rule 3).

Synthetic data stays labelled and must never be used in production or committed
as real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.data.reduce import TargetGrid


def generate_synthetic_raw_sample(
    raw_dir: Path,
    target_grid: TargetGrid,
    depths_m: list[float],
    dates: pd.DatetimeIndex,
) -> list[str]:
    """Generate realistic synthetic NetCDF files in raw_dir for offline testing.

    Returns list of generated file relative paths.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    n_time = len(dates)
    n_lat = len(target_grid.lats)
    n_lon = len(target_grid.lons)
    lats = target_grid.lats
    lons = target_grid.lons

    # Create land mask for realistic geography
    lat_2d, lon_2d = np.meshgrid(lats, lons, indexing="ij")
    is_arabia = (lat_2d >= 13.0) & (lon_2d <= 56.0)
    is_india = (
        (lat_2d >= 8.0) & (lon_2d >= 69.0) & (lon_2d <= 87.0) & (lat_2d + 0.5 * lon_2d > 48.0)
    )
    is_africa = (lat_2d <= 12.0) & (lon_2d <= 51.0)
    is_burma = (lat_2d >= 16.0) & (lon_2d >= 92.0)
    ocean_mask_2d = ~(is_arabia | is_india | is_africa | is_burma)

    yr_str = str(dates[0].year)

    # 1. GLORYS12: (time, depth, lat, lon)
    glorys_dir = raw_dir / "glorys12"
    glorys_dir.mkdir(parents=True, exist_ok=True)
    glorys_path = glorys_dir / (
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_thetao_"
        f"39.00E-101.00E_1.00S-31.00N_0.49-1062.44m_{yr_str}01.nc"
    )

    n_depth = len(depths_m)
    t_3d = np.full((n_time, n_depth, n_lat, n_lon), np.nan, dtype=np.float32)
    depth_arr = np.array(depths_m, dtype=np.float32)

    for t_idx in range(n_time):
        doy = (
            dates[t_idx].day_of_year
            if hasattr(dates[t_idx], "day_of_year")
            else dates[t_idx].dayofyear
        )
        annual_cycle = 1.8 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)
        for i_lat, lat in enumerate(lats):
            surf_t = 28.5 - 0.15 * lat + annual_cycle
            for i_lon, lon in enumerate(lons):
                if ocean_mask_2d[i_lat, i_lon]:
                    t_profile = 4.0 + (surf_t - 4.0) * np.exp(-depth_arr / 220.0)
                    t_3d[t_idx, :, i_lat, i_lon] = t_profile.astype(np.float32)

    ds_glorys = xr.Dataset(
        data_vars={"thetao": (("time", "depth", "latitude", "longitude"), t_3d)},
        coords={"time": dates, "depth": depth_arr, "latitude": lats, "longitude": lons},
        attrs={"synthetic": "true"},
    )
    ds_glorys.to_netcdf(glorys_path)
    generated_files.append(str(glorys_path.relative_to(raw_dir)))

    # 2. SLA & ADT
    duacs_dir = raw_dir / "sla"
    duacs_dir.mkdir(parents=True, exist_ok=True)
    sla_path = (
        duacs_dir
        / f"cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_sla-adt_{yr_str}01.nc"
    )

    sla_vals = np.where(
        ocean_mask_2d, 0.05 * np.sin(lon_2d / 5.0) * np.cos(lat_2d / 4.0), np.nan
    ).astype(np.float32)
    sla_3d = np.tile(sla_vals, (n_time, 1, 1))
    adt_3d = sla_3d + 1.2

    ds_sla = xr.Dataset(
        data_vars={
            "sla": (("time", "lat", "lon"), sla_3d),
            "adt": (("time", "lat", "lon"), adt_3d),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_sla.to_netcdf(sla_path)
    generated_files.append(str(sla_path.relative_to(raw_dir)))

    # 3. OSTIA SST (stored in Kelvin)
    ostia_dir = raw_dir / "sst"
    ostia_dir.mkdir(parents=True, exist_ok=True)
    sst_path = ostia_dir / f"METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_{yr_str}01.nc"

    sst_celsius = t_3d[:, 0, :, :]
    sst_kelvin = np.where(~np.isnan(sst_celsius), sst_celsius + 273.15, np.nan).astype(np.float32)
    ds_sst = xr.Dataset(
        data_vars={"analysed_sst": (("time", "lat", "lon"), sst_kelvin)},
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_sst["analysed_sst"].attrs["units"] = "Kelvin"
    ds_sst.to_netcdf(sst_path)
    generated_files.append(str(sst_path.relative_to(raw_dir)))

    # 4. SSS
    sss_dir = raw_dir / "sss"
    sss_dir.mkdir(parents=True, exist_ok=True)
    sss_path = sss_dir / f"cmems_obs-mob_glo_phy-sss_my_multi_P1D_sos_{yr_str}01.nc"

    sss_vals = np.where(ocean_mask_2d, 35.0 - 0.05 * lat_2d + 0.02 * lon_2d, np.nan).astype(
        np.float32
    )
    sss_3d = np.tile(sss_vals, (n_time, 1, 1))
    ds_sss = xr.Dataset(
        data_vars={"sos": (("time", "lat", "lon"), sss_3d)},
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_sss.to_netcdf(sss_path)
    generated_files.append(str(sss_path.relative_to(raw_dir)))

    # 5. OSCAR Currents
    oscar_dir = raw_dir / "oscar"
    oscar_dir.mkdir(parents=True, exist_ok=True)
    for t_idx, d in enumerate(dates):
        d_str = pd.Timestamp(d).strftime("%Y%m%d")
        oscar_p = oscar_dir / f"oscar_currents_final_{d_str}.nc"
        u_curr = np.where(ocean_mask_2d, 0.2 * np.cos(lat_2d / 3.0), np.nan).astype(np.float32)
        v_curr = np.where(ocean_mask_2d, 0.1 * np.sin(lon_2d / 4.0), np.nan).astype(np.float32)
        ds_osc_day = xr.Dataset(
            data_vars={
                "u": (("time", "lat", "lon"), u_curr[np.newaxis, ...]),
                "v": (("time", "lat", "lon"), v_curr[np.newaxis, ...]),
            },
            coords={"time": [d], "lat": lats, "lon": lons},
            attrs={"synthetic": "true"},
        )
        ds_osc_day.to_netcdf(oscar_p)
        if t_idx == 0:
            generated_files.append(str(oscar_p.relative_to(raw_dir)))

    # 6. ERA5 Winds & Fluxes
    era5_dir = raw_dir / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    instant_path = era5_dir / "data_stream-oper_stepType-instant.nc"
    accum_path = era5_dir / "data_stream-oper_stepType-accum.nc"

    u_wind = (5.0 + 2.0 * np.cos(lat_2d / 4.0)).astype(np.float32)
    v_wind = (-3.0 + 1.5 * np.sin(lon_2d / 6.0)).astype(np.float32)
    ds_inst = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), np.tile(u_wind, (n_time, 1, 1))),
            "v10": (("time", "lat", "lon"), np.tile(v_wind, (n_time, 1, 1))),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_inst["u10"].attrs["units"] = "m s**-1"
    ds_inst["v10"].attrs["units"] = "m s**-1"
    ds_inst.to_netcdf(instant_path)
    generated_files.append(str(instant_path.relative_to(raw_dir)))

    # In ERA5, fluxes are in W m**-2 (or J m**-2 per hour)
    net_heat = np.full((n_time, n_lat, n_lon), 40.0, dtype=np.float32)
    precip = np.full((n_time, n_lat, n_lon), 0.003, dtype=np.float32)
    ds_acc = xr.Dataset(
        data_vars={
            "ssr": (("time", "lat", "lon"), net_heat * 0.4),
            "str": (("time", "lat", "lon"), net_heat * 0.2),
            "sshf": (("time", "lat", "lon"), net_heat * 0.2),
            "slhf": (("time", "lat", "lon"), net_heat * 0.2),
            "tp": (("time", "lat", "lon"), precip),
        },
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    for v in ["ssr", "str", "sshf", "slhf"]:
        ds_acc[v].attrs["units"] = "W m**-2"
    ds_acc["tp"].attrs["units"] = "m"
    ds_acc.to_netcdf(accum_path)
    generated_files.append(str(accum_path.relative_to(raw_dir)))

    # 7. GEBCO Bathymetry
    gebco_dir = raw_dir / "gebco"
    gebco_dir.mkdir(parents=True, exist_ok=True)
    gebco_path = gebco_dir / "gebco_2023_sub_ice_topo.nc"

    elev = np.where(ocean_mask_2d, -3200.0, 500.0).astype(np.float32)
    ds_gebco = xr.Dataset(
        data_vars={"elevation": (("lat", "lon"), elev)},
        coords={"lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_gebco.to_netcdf(gebco_path)
    generated_files.append(str(gebco_path.relative_to(raw_dir)))

    # 8. WOA23 Monthly
    woa_dir = raw_dir / "woa23"
    woa_dir.mkdir(parents=True, exist_ok=True)
    for m in range(1, 13):
        m_str = f"{m:02d}"
        woa_p = woa_dir / f"woa23_decav_t{m_str}_04.nc"
        woa_profile = 25.0 * np.exp(-depth_arr / 200.0)
        woa_3d = np.tile(woa_profile[:, np.newaxis, np.newaxis], (1, n_lat, n_lon))
        ds_woa = xr.Dataset(
            data_vars={"t_an": (("depth", "lat", "lon"), woa_3d.astype(np.float32))},
            coords={"depth": depth_arr, "lat": lats, "lon": lons},
            attrs={"synthetic": "true"},
        )
        ds_woa.to_netcdf(woa_p)
        if m == 1:
            generated_files.append(str(woa_p.relative_to(raw_dir)))

    # 9. GREP Auxiliary Target
    grep_dir = raw_dir / "grep"
    grep_dir.mkdir(parents=True, exist_ok=True)
    grep_path = grep_dir / f"cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m_thetao_mean_{yr_str}01.nc"
    ds_grep = xr.Dataset(
        data_vars={"thetao_mean": (("time", "depth", "lat", "lon"), t_3d)},
        coords={"time": dates, "depth": depth_arr, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_grep.to_netcdf(grep_path)
    generated_files.append(str(grep_path.relative_to(raw_dir)))

    # 10. ARMOR3D Baseline
    armor_dir = raw_dir / "armor3d"
    armor_dir.mkdir(parents=True, exist_ok=True)
    armor_path = armor_dir / f"cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_to_{yr_str}01.nc"
    ds_armor = xr.Dataset(
        data_vars={"to": (("time", "depth", "lat", "lon"), t_3d)},
        coords={"time": dates, "depth": depth_arr, "lat": lats, "lon": lons},
        attrs={"synthetic": "true"},
    )
    ds_armor.to_netcdf(armor_path)
    generated_files.append(str(armor_path.relative_to(raw_dir)))

    return generated_files
