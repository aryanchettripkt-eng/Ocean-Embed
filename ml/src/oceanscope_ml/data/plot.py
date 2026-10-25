"""Plotting module for OceanScope AI inspection and validation figures.

Generates outputs/figures/m1_sample_day.png for PRD_AI_ML M1 acceptance.
Uses pure numpy and Pillow (no heavy matplotlib dependency needed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw

logger = logging.getLogger("oceanscope_ml.data.plot")


def colormap_lut(name: str = "coolwarm", n: int = 256) -> np.ndarray:
    """Generate RGB lookup table (n, 3) for standard colormaps."""
    t = np.linspace(0.0, 1.0, n)
    lut = np.zeros((n, 3), dtype=np.uint8)

    if name == "coolwarm":
        # Smooth transition: blue (59, 76, 192) -> white (220, 220, 220) -> red (180, 4, 38)
        r = np.where(t < 0.5, 59 + (220 - 59) * (t / 0.5), 220 + (180 - 220) * ((t - 0.5) / 0.5))
        g = np.where(t < 0.5, 76 + (220 - 76) * (t / 0.5), 220 + (4 - 220) * ((t - 0.5) / 0.5))
        b = np.where(t < 0.5, 192 + (220 - 192) * (t / 0.5), 220 + (38 - 220) * ((t - 0.5) / 0.5))
    elif name == "viridis":
        # Purple (68, 1, 84) -> Teal (33, 145, 140) -> Yellow (253, 231, 37)
        r = np.where(t < 0.5, 68 + (33 - 68) * (t / 0.5), 33 + (253 - 33) * ((t - 0.5) / 0.5))
        g = np.where(t < 0.5, 1 + (145 - 1) * (t / 0.5), 145 + (231 - 145) * ((t - 0.5) / 0.5))
        b = np.where(t < 0.5, 84 + (140 - 84) * (t / 0.5), 140 + (37 - 140) * ((t - 0.5) / 0.5))
    elif name == "turbo":
        # Blue -> Cyan -> Green -> Yellow -> Red
        r = np.clip(np.sin(np.pi * (t - 0.2)) * 255.0 + 50.0, 0, 255)
        g = np.clip(np.sin(np.pi * t) * 230.0 + 20.0, 0, 255)
        b = np.clip(np.cos(np.pi * t * 0.9) * 220.0 + 35.0, 0, 255)
    else:  # Grayscale
        r = t * 255.0
        g = t * 255.0
        b = t * 255.0

    lut[:, 0] = np.clip(r, 0, 255).astype(np.uint8)
    lut[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
    lut[:, 2] = np.clip(b, 0, 255).astype(np.uint8)
    return lut


def render_panel(
    data_2d: np.ndarray,
    title: str,
    units: str,
    width: int = 440,
    height: int = 280,
    cmap_name: str = "coolwarm",
    land_mask: np.ndarray | None = None,
) -> Image.Image:
    """Render a single 2D variable panel with colormap, title, and mini-colorbar."""
    img = Image.new("RGB", (width, height), color=(24, 27, 34))
    draw = ImageDraw.Draw(img)

    # Margins and plot dimensions
    top_margin = 32
    bottom_margin = 38
    side_margin = 16

    plot_w = width - 2 * side_margin
    plot_h = height - top_margin - bottom_margin

    # Compute valid data range for colormapping
    valid_data = data_2d[~np.isnan(data_2d)]
    if len(valid_data) > 0:
        v_min, v_max = float(np.min(valid_data)), float(np.max(valid_data))
        if abs(v_max - v_min) < 1e-6:
            v_max = v_min + 1.0
    else:
        v_min, v_max = 0.0, 1.0

    # Normalize to [0, 1] for colormap lookup
    lut = colormap_lut(cmap_name)
    n_lut = len(lut)

    with np.errstate(invalid="ignore"):
        norm_data = np.clip((data_2d - v_min) / (v_max - v_min), 0.0, 1.0)
        idx_data = (norm_data * (n_lut - 1)).astype(np.int32)

    # 3D RGB array for data
    rgb_arr = np.zeros((data_2d.shape[0], data_2d.shape[1], 3), dtype=np.uint8)
    nan_mask = np.isnan(data_2d)
    if land_mask is not None:
        nan_mask = nan_mask | (~land_mask)

    valid_cells = ~nan_mask
    rgb_arr[valid_cells] = lut[idx_data[valid_cells]]
    # Land / invalid color: dark slate gray
    rgb_arr[nan_mask] = [45, 52, 64]

    # Flip vertically so latitude 0 is at the bottom, latitude 30 is at the top
    rgb_flipped = np.flipud(rgb_arr)
    data_img = Image.fromarray(rgb_flipped, mode="RGB")
    data_img_resized = data_img.resize((plot_w, plot_h), resample=Image.Resampling.NEAREST)

    img.paste(data_img_resized, (side_margin, top_margin))

    # Border around plot
    draw.rectangle(
        [side_margin, top_margin, side_margin + plot_w, top_margin + plot_h],
        outline=(80, 90, 105),
    )

    # Title text
    title_text = f"{title} [{units}]"
    draw.text((side_margin + 2, 8), title_text, fill=(230, 237, 243))

    # Colorbar at the bottom
    cbar_w = 180
    cbar_h = 10
    cbar_x = side_margin + plot_w - cbar_w
    cbar_y = height - 24

    cbar_gradient = np.tile(np.linspace(0, n_lut - 1, cbar_w, dtype=np.int32), (cbar_h, 1))
    cbar_rgb = lut[cbar_gradient]
    cbar_img = Image.fromarray(cbar_rgb, mode="RGB")
    img.paste(cbar_img, (cbar_x, cbar_y))
    draw.rectangle([cbar_x, cbar_y, cbar_x + cbar_w, cbar_y + cbar_h], outline=(100, 110, 125))

    # Colorbar min/max labels
    lbl_min = f"{v_min:.2f}"
    lbl_max = f"{v_max:.2f}"
    draw.text((cbar_x - 45, cbar_y - 1), lbl_min, fill=(175, 185, 200))
    draw.text((cbar_x + cbar_w + 6, cbar_y - 1), lbl_max, fill=(175, 185, 200))

    return img


def plot_sample_day(
    ds: xr.Dataset,
    sample_date: str,
    out_path: Path,
) -> Path:
    """Generate 16-panel figure for all variables on sample_date (PRD_AI_ML M1).

    Parameters
    ----------
    ds : xr.Dataset containing all reduced variables
    sample_date : date string, e.g. "2019-01-15"
    out_path : destination path for outputs/figures/m1_sample_day.png

    Returns
    -------
    Path to saved figure
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select day slice
    if "time" in ds.coords:
        try:
            day_slice = ds.sel(time=sample_date)
        except Exception:
            day_slice = ds.isel(time=0)
    else:
        day_slice = ds

    actual_date = (
        str(day_slice.coords["time"].values)[:10] if "time" in day_slice.coords else sample_date
    )

    # Ocean mask
    ocean_mask = (
        day_slice["ocean_mask"].values.astype(bool) if "ocean_mask" in day_slice.data_vars else None
    )

    panel_defs: list[dict[str, Any]] = [
        {"var": "sla", "title": "Sea Level Anomaly (SLA)", "units": "m", "cmap": "coolwarm"},
        {
            "var": "adt",
            "title": "Absolute Dynamic Topography (ADT)",
            "units": "m",
            "cmap": "coolwarm",
        },
        {
            "var": "sst",
            "title": "Sea Surface Temperature (OSTIA)",
            "units": "°C",
            "cmap": "coolwarm",
        },
        {"var": "sss", "title": "Sea Surface Salinity (MULTIOBS)", "units": "psu", "cmap": "turbo"},
        {
            "var": "u_current",
            "title": "Zonal Surface Current (OSCAR)",
            "units": "m/s",
            "cmap": "coolwarm",
        },
        {
            "var": "v_current",
            "title": "Meridional Current (OSCAR)",
            "units": "m/s",
            "cmap": "coolwarm",
        },
        {"var": "u_wind", "title": "10 m Zonal Wind (ERA5)", "units": "m/s", "cmap": "coolwarm"},
        {
            "var": "v_wind",
            "title": "10 m Meridional Wind (ERA5)",
            "units": "m/s",
            "cmap": "coolwarm",
        },
        {
            "var": "wind_stress_curl",
            "title": "Wind Stress Curl (Large & Pond)",
            "units": "N/m³",
            "cmap": "coolwarm",
        },
        {
            "var": "net_heat_flux",
            "title": "Net Heat Flux (ERA5)",
            "units": "W/m²",
            "cmap": "coolwarm",
        },
        {"var": "precip", "title": "Total Precipitation (ERA5)", "units": "m/day", "cmap": "turbo"},
        {
            "var": "bathymetry",
            "title": "Bathymetry log(1+depth) (GEBCO)",
            "units": "log m",
            "cmap": "viridis",
        },
        {
            "var": "thetao",
            "title": "GLORYS Temperature (Depth 0.49 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 0,
        },
        {
            "var": "thetao",
            "title": "GLORYS Temperature (Depth 92.3 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 7,
        },
        {
            "var": "clim",
            "title": "Harmonic Climatology (Depth 92.3 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 7,
        },
        {
            "var": "target_valid",
            "title": "Target Valid Mask (Depth 92.3 m)",
            "units": "bool",
            "cmap": "gray",
            "d": 7,
        },
        {
            "var": "woa_clim_profile",
            "title": "WOA23 Profile (Depth 92.3 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 7,
        },
        {
            "var": "grep_mean",
            "title": "GREP Ensemble Mean (Depth 0.49 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 0,
        },
        {
            "var": "armor3d",
            "title": "ARMOR3D Temperature (Depth 0.49 m)",
            "units": "°C",
            "cmap": "coolwarm",
            "d": 0,
        },
        {
            "var": "ocean_mask",
            "title": "Ocean Mask (1=Ocean, 0=Land)",
            "units": "bool",
            "cmap": "gray",
        },
    ]

    panel_w, panel_h = 440, 275
    cols, rows = 5, 4

    header_h = 60
    total_w = cols * panel_w
    total_h = header_h + rows * panel_h

    canvas = Image.new("RGB", (total_w, total_h), color=(13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Main title
    header_title = f"OceanScope AI — Milestone 1 Sample Day Inspection ({actual_date})"
    draw.text((24, 18), header_title, fill=(240, 246, 252))
    draw.text(
        (total_w - 380, 22),
        "0.25° Grid · North Indian Ocean (40-100°E, 0-30°N)",
        fill=(139, 148, 158),
    )
    draw.line([(0, header_h - 2), (total_w, header_h - 2)], fill=(48, 54, 61), width=2)

    for i, pdef in enumerate(panel_defs):
        row = i // cols
        col = i % cols
        pos_x = col * panel_w
        pos_y = header_h + row * panel_h

        var_name = pdef["var"]
        if var_name in day_slice.data_vars or var_name in ds.data_vars:
            arr = (
                day_slice[var_name].values
                if var_name in day_slice.data_vars
                else ds[var_name].values
            )
            # Handle depth and time indexing so data_2d is strictly (lat, lon)
            if "d" in pdef and arr.ndim == 4:
                # e.g. thetao (time, depth, lat, lon) or clim (dayofyear, depth, lat, lon)
                d_idx = min(pdef["d"], arr.shape[1] - 1)
                if var_name == "clim":
                    ts = pd.Timestamp(actual_date)
                    doy = ts.day_of_year if hasattr(ts, "day_of_year") else ts.dayofyear
                    data_2d = arr[min(doy - 1, arr.shape[0] - 1), d_idx, :, :]
                elif var_name == "woa_clim_profile":
                    ts = pd.Timestamp(actual_date)
                    month_idx = min(ts.month - 1, arr.shape[0] - 1)
                    data_2d = arr[month_idx, d_idx, :, :]
                else:
                    data_2d = arr[0, d_idx, :, :]
            elif "d" in pdef and arr.ndim == 3:
                d_idx = min(pdef["d"], arr.shape[0] - 1)
                data_2d = arr[d_idx, :, :]
            elif arr.ndim == 3:
                data_2d = arr[0, :, :]
            else:
                data_2d = arr

            while data_2d.ndim > 2:
                data_2d = data_2d[0]

            if data_2d.dtype == bool:
                data_2d = data_2d.astype(np.float32)
        elif var_name == "ocean_mask" and ocean_mask is not None:
            data_2d = ocean_mask.astype(np.float32)
        else:
            data_2d = np.full((120, 240), np.nan, dtype=np.float32)

        panel_img = render_panel(
            data_2d=data_2d,
            title=pdef["title"],
            units=pdef["units"],
            width=panel_w - 8,
            height=panel_h - 8,
            cmap_name=pdef["cmap"],
            land_mask=ocean_mask,
        )
        canvas.paste(panel_img, (pos_x + 4, pos_y + 4))

    canvas.save(out_path, format="PNG")
    logger.info("Saved sample day figure to %s", out_path)
    return out_path
