"""Output store: typed accessors over OceanScope predictions and metrics."""

from __future__ import annotations

import json
import logging

import numpy as np
import xarray as xr

from app.data.loader import DataLoader
from app.data.loader import loader as default_loader
from app.errors import DataNotLoadedError
from app.schemas.caveats import (
    ARGO_CAVEAT,
    GLORYS_CAVEAT,
    LIVE_CAVEAT,
    SYNTHETIC_CAVEAT,
)
from app.schemas.models import (
    AvailableDates,
    Caveats,
    ChatInfo,
    ColorRange,
    ErrorRange,
    GridShape,
    MetaResponse,
    Region,
    Splits,
)
from app.settings import settings

logger = logging.getLogger(__name__)


class OutputStore:
    """Read-only access to published prediction store and evaluation metrics."""

    def __init__(self, loader: DataLoader | None = None) -> None:
        self.loader = loader or default_loader
        self._ds: xr.Dataset | None = None
        self._cached_meta: MetaResponse | None = None
        self._cached_color_ranges: list[ColorRange] | None = None

    def _ensure_loaded(self) -> None:
        if not self.loader.is_loaded or self.loader.manifest is None:
            raise DataNotLoadedError(self.loader.load_error or "Data store is not loaded")

    def get_predictions_dataset(self) -> xr.Dataset:
        """Open predictions zarr store lazily."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        if self._ds is None:
            pred_rel_path = self.loader.manifest["files"]["predictions"]
            pred_full_path = self.loader.data_dir / pred_rel_path
            self._ds = xr.open_zarr(str(pred_full_path), consolidated=True)
        return self._ds

    def _compute_color_ranges(self, ds: xr.Dataset, depths: list[float]) -> list[ColorRange]:
        """Compute 2nd/98th percentile of temp_glorys per depth over published dates."""
        if self._cached_color_ranges is not None:
            return self._cached_color_ranges

        ranges: list[ColorRange] = []
        glorys_var = ds["temp_glorys"]

        for d_idx, d_m in enumerate(depths):
            vals = glorys_var.isel(depth=d_idx).values
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                vmin = float(np.percentile(valid_vals, 2))
                vmax = float(np.percentile(valid_vals, 98))
            else:
                vmin, vmax = 0.0, 35.0
            ranges.append(
                ColorRange(
                    depth_m=float(d_m),
                    vmin=round(vmin, 2),
                    vmax=round(vmax, 2),
                )
            )

        self._cached_color_ranges = ranges
        return ranges

    def _get_latest_live_date(self, ds: xr.Dataset) -> str | None:
        """Extract the latest live date from split_of_each_date attribute if present."""
        raw_splits = ds.attrs.get("split_of_each_date")
        if not raw_splits:
            return None

        if isinstance(raw_splits, str):
            try:
                split_map = json.loads(raw_splits)
            except Exception:
                return None
        elif isinstance(raw_splits, dict):
            split_map = raw_splits
        else:
            return None

        live_end_dates: list[str] = []
        for date_range, split_name in split_map.items():
            if str(split_name).lower() == "live":
                parts = str(date_range).split("/")
                end_date = parts[-1].strip()
                live_end_dates.append(end_date)

        if not live_end_dates:
            return None

        return max(live_end_dates)

    def get_meta(self) -> MetaResponse:
        """Return dataset metadata, coordinates, caveats, and cached color ranges."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        manifest = self.loader.manifest

        ds = self.get_predictions_dataset()

        # Compute grid shape from region and resolution
        reg_dict = manifest["region"]
        res = float(manifest["grid_resolution_deg"])
        lat_count = int(round((reg_dict["lat_max"] - reg_dict["lat_min"]) / res))
        lon_count = int(round((reg_dict["lon_max"] - reg_dict["lon_min"]) / res))
        grid_shape = GridShape(lat=lat_count, lon=lon_count)

        depths = [float(d) for d in manifest["depths_m"]]
        color_ranges = self._compute_color_ranges(ds, depths)
        latest_live_date = self._get_latest_live_date(ds)

        chat_info = ChatInfo(
            enabled=settings.CHAT_ENABLED,
            provider_label=settings.LLM_PROVIDER_LABEL if settings.CHAT_ENABLED else None,
        )

        caveats = Caveats(
            glorys=GLORYS_CAVEAT,
            argo=ARGO_CAVEAT,
            synthetic=SYNTHETIC_CAVEAT,
            live=LIVE_CAVEAT,
        )

        return MetaResponse(
            run_id=manifest["active_run_id"],
            synthetic=manifest["synthetic"],
            dataset_version=manifest["dataset_version"],
            model_name=manifest["model_name"],
            region=Region(**manifest["region"]),
            grid_resolution_deg=manifest["grid_resolution_deg"],
            grid_shape=grid_shape,
            depths_m=depths,
            available_dates=AvailableDates(**manifest["available_dates"]),
            latest_live_date=latest_live_date,
            splits=Splits(**manifest["splits"]),
            test_evaluated=manifest["test_evaluated"],
            color_ranges=color_ranges,
            error_range=ErrorRange(vmin=-2.0, vmax=2.0),
            caveats=caveats,
            chat=chat_info,
        )


store = OutputStore(default_loader)
