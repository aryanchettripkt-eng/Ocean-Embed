"""Output store: typed accessors over OceanScope predictions and metrics."""

from __future__ import annotations

import json
import logging
import re
from datetime import date as dt_date
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from app.data.grid import compute_area_weights, get_region_mask, snap_coordinate
from app.data.loader import DataLoader
from app.errors import (
    APIError,
    BadDepthError,
    BothDepthParamsError,
    DataNotLoadedError,
    DateUnavailableError,
    HeadlineNotAvailableError,
    InvalidDateFormatError,
    OutOfRegionError,
    RangeTooLargeError,
    ReferenceUnavailableError,
    TestNotEvaluatedError,
)
from app.schemas.caveats import (
    ARGO_CAVEAT,
    GLORYS_CAVEAT,
    HEADLINE_CRITERION,
    LIVE_CAVEAT,
    SYNTHETIC_CAVEAT,
)
from app.schemas.models import (
    ArgoMetricRow,
    ArgoProfileMatch,
    ArgoProfilesResponse,
    ArgoResponse,
    AvailableDates,
    Caveats,
    ChatInfo,
    ColorRange,
    ErrorRange,
    GridShape,
    HeadlineResponse,
    HeadlineRow,
    MapResponse,
    MapStats,
    MetaResponse,
    MetricRow,
    MetricsResponse,
    ProfileResponse,
    Region,
    RegionSummaryResponse,
    Splits,
    TimeseriesResponse,
)
from app.settings import settings

logger = logging.getLogger(__name__)


DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date(date_str: str) -> dt_date:
    """Validate date format as strict YYYY-MM-DD and return parsed date object."""
    if not isinstance(date_str, str) or not DATE_REGEX.match(date_str):
        raise InvalidDateFormatError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
    try:
        return dt_date.fromisoformat(date_str)
    except ValueError:
        raise InvalidDateFormatError(f"Invalid date '{date_str}'. Expected valid calendar date.")


class OutputStore:
    """Read-only access to published prediction store and evaluation metrics."""

    def __init__(self, loader: DataLoader | None = None) -> None:
        self._custom_loader = loader
        self._ds: xr.Dataset | None = None
        self._cached_meta: MetaResponse | None = None
        self._cached_color_ranges: list[ColorRange] | None = None
        self._cached_metrics: dict[str, pd.DataFrame] = {}
        self._cached_argo: pd.DataFrame | None = None
        self._cached_argo_matches: pd.DataFrame | None = None
        self._cached_headline: pd.DataFrame | None = None
        self._last_loader_id: int | None = None
        self._last_manifest_id: int | None = None

    def clear_caches(self) -> None:
        """Explicitly clear cached metadata, metrics, and dataset handles."""
        self._ds = None
        self._cached_meta = None
        self._cached_color_ranges = None
        self._cached_metrics.clear()
        self._cached_argo = None
        self._cached_argo_matches = None
        self._cached_headline = None
        self._last_manifest_id = None

    @property
    def loader(self) -> DataLoader:
        """Active DataLoader instance."""
        if self._custom_loader is not None:
            return self._custom_loader
        import app.data.loader as loader_module

        return loader_module.loader

    def _ensure_loaded(self) -> None:
        active_loader = self.loader
        manifest_id = id(active_loader.manifest) if active_loader.manifest is not None else None
        if self._last_loader_id != id(active_loader) or self._last_manifest_id != manifest_id:
            self._ds = None
            self._cached_meta = None
            self._cached_color_ranges = None
            self._cached_metrics.clear()
            self._cached_argo = None
            self._cached_argo_matches = None
            self._cached_headline = None
            self._last_loader_id = id(active_loader)
            self._last_manifest_id = manifest_id

        if not active_loader.is_loaded or active_loader.manifest is None:
            raise DataNotLoadedError(active_loader.load_error or "Data store is not loaded")

    def get_predictions_dataset(self) -> xr.Dataset:
        """Open predictions zarr store lazily."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        if self._ds is None:
            pred_rel_path = self.loader.manifest["files"]["predictions"]
            pred_full_path = self.loader.data_dir / pred_rel_path
            self._ds = xr.open_zarr(str(pred_full_path), consolidated=True)
        return self._ds

    def resolve_depth(
        self,
        depth_m: float | None = None,
        depth_index: int | None = None,
    ) -> tuple[int, float]:
        """Resolve depth parameter to (depth_index, depth_m).

        Raises BothDepthParamsError if both are supplied.
        Raises BadDepthError if neither or invalid.
        """
        self._ensure_loaded()
        assert self.loader.manifest is not None
        depths = [float(d) for d in self.loader.manifest["depths_m"]]

        if depth_m is not None and depth_index is not None:
            raise BothDepthParamsError()

        if depth_m is None and depth_index is None:
            raise BadDepthError("Provide either depth_m or depth_index")

        if depth_index is not None:
            if depth_index < 0 or depth_index >= len(depths):
                raise BadDepthError(
                    f"depth_index {depth_index} must be between 0 and {len(depths) - 1}"
                )
            return depth_index, depths[depth_index]

        assert depth_m is not None
        for idx, d in enumerate(depths):
            if abs(d - depth_m) <= 0.01:
                return idx, d

        raise BadDepthError(f"depth_m {depth_m} does not match any valid model depth")

    def get_date_split(self, date_str: str) -> str | None:
        """Verify date availability and return its split label (train, val, test, live) or None."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        validate_date(date_str)
        available = self.loader.manifest["available_dates"]
        start_date = available["start"]
        end_date = available["end"]

        if date_str < start_date or date_str > end_date:
            raise DateUnavailableError(
                f"Date {date_str} is outside available dates ({start_date} to {end_date})"
            )

        ds = self.get_predictions_dataset()
        raw_splits = ds.attrs.get("split_of_each_date")
        if raw_splits:
            try:
                split_map = json.loads(raw_splits) if isinstance(raw_splits, str) else raw_splits
                for date_range, split_name in split_map.items():
                    parts = str(date_range).split("/")
                    if len(parts) == 2 and parts[0] <= date_str <= parts[1]:
                        return str(split_name)
            except Exception:
                pass

        # Check manifest splits
        splits_dict = self.loader.manifest.get("splits", {})
        for s_name, s_range in splits_dict.items():
            if len(s_range) == 2 and s_range[0] <= date_str <= s_range[1]:
                return str(s_name)

        return None

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

    def get_map(
        self,
        date: str,
        depth_m: float | None = None,
        depth_index: int | None = None,
        field: Literal["pred", "glorys", "error"] = "pred",
    ) -> MapResponse:
        """Return 2D grid slice for a given date, depth, and field (PRD_BACKEND §4.3)."""
        d_idx, target_depth = self.resolve_depth(depth_m, depth_index)
        split = self.get_date_split(date)

        if split == "live" and field in ("glorys", "error"):
            raise ReferenceUnavailableError()

        ds = self.get_predictions_dataset()
        try:
            ds_date = ds.sel(time=date)
        except KeyError:
            raise DateUnavailableError(f"Date {date} is unavailable in dataset")

        cov_2d = ds_date["coverage"].values.astype(int)

        if field == "pred":
            vals_2d = ds_date["temp_pred"].isel(depth=d_idx).values.astype(np.float32)
        elif field == "glorys":
            vals_2d = ds_date["temp_glorys"].isel(depth=d_idx).values.astype(np.float32)
        elif field == "error":
            p = ds_date["temp_pred"].isel(depth=d_idx).values.astype(np.float32)
            g = ds_date["temp_glorys"].isel(depth=d_idx).values.astype(np.float32)
            vals_2d = p - g
        else:
            raise APIError(422, "invalid_field", f"Field '{field}' must be pred, glorys, or error")

        # Stats over valid ocean cells
        valid_mask = (cov_2d == 1) & ~np.isnan(vals_2d)
        valid_vals = vals_2d[valid_mask]
        n_valid = int(len(valid_vals))

        if n_valid > 0:
            stats = MapStats(
                min=round(float(np.min(valid_vals)), 2),
                max=round(float(np.max(valid_vals)), 2),
                mean=round(float(np.mean(valid_vals)), 2),
                n_valid=n_valid,
            )
        else:
            stats = MapStats(min=None, max=None, mean=None, n_valid=0)

        # Coordinate arrays rounded to 3 decimals
        lats = [round(float(y), 3) for y in ds.coords["lat"].values]
        lons = [round(float(x), 3) for x in ds.coords["lon"].values]

        # Values matrix: rounded to 2 decimals, NaN serialized as None
        values_list: list[list[float | None]] = [
            [None if np.isnan(v) else round(float(v), 2) for v in row] for row in vals_2d
        ]
        coverage_list: list[list[int]] = cov_2d.tolist()

        assert self.loader.manifest is not None
        return MapResponse(
            run_id=self.loader.manifest["active_run_id"],
            synthetic=self.loader.manifest["synthetic"],
            date=date,
            depth_m=target_depth,
            field=field,
            split=split,
            units="degC",
            lat=lats,
            lon=lons,
            values=values_list,
            coverage=coverage_list,
            stats=stats,
        )

    def get_profile(self, lat: float, lon: float, date: str) -> ProfileResponse:
        """Return 15-depth vertical temperature profile for lat, lon, date (PRD_BACKEND §4.4)."""
        split = self.get_date_split(date)
        ds = self.get_predictions_dataset()

        assert self.loader.manifest is not None
        reg = self.loader.manifest["region"]
        lats = ds.coords["lat"].values
        lons = ds.coords["lon"].values

        lat_idx, snapped_lat = snap_coordinate(lat, lats, reg["lat_min"], reg["lat_max"])
        lon_idx, snapped_lon = snap_coordinate(lon, lons, reg["lon_min"], reg["lon_max"])

        try:
            ds_cell = ds.sel(time=date).isel(lat=lat_idx, lon=lon_idx)
        except KeyError:
            raise DateUnavailableError(f"Date {date} is unavailable in dataset")

        coverage_code = int(ds_cell["coverage"].values)
        depths = [float(d) for d in self.loader.manifest["depths_m"]]

        raw_p = ds_cell["temp_pred"].values
        raw_g = ds_cell["temp_glorys"].values

        pred_list: list[float | None] = [None if np.isnan(v) else round(float(v), 2) for v in raw_p]

        if split == "live":
            glorys_list: list[float | None] = [None] * len(depths)
            error_list: list[float | None] = [None] * len(depths)
        else:
            glorys_list = [None if np.isnan(v) else round(float(v), 2) for v in raw_g]
            raw_err = raw_p - raw_g
            error_list = [None if np.isnan(v) else round(float(v), 2) for v in raw_err]

        return ProfileResponse(
            run_id=self.loader.manifest["active_run_id"],
            synthetic=self.loader.manifest["synthetic"],
            date=date,
            lat=snapped_lat,
            lon=snapped_lon,
            split=split,
            coverage=coverage_code,
            depths_m=depths,
            pred=pred_list,
            glorys=glorys_list,
            error=error_list,
        )

    def get_timeseries(
        self,
        lat: float,
        lon: float,
        depth_m: float | None = None,
        depth_index: int | None = None,
        start: str = "",
        end: str = "",
    ) -> TimeseriesResponse:
        """Return daily temperature time series for a location and depth (PRD_BACKEND §4.5)."""
        d_idx, target_depth = self.resolve_depth(depth_m, depth_index)
        start_dt = validate_date(start)
        end_dt = validate_date(end)

        if start_dt > end_dt:
            raise APIError(422, "invalid_time_range", "start date cannot be after end date")

        span_days = (end_dt - start_dt).days + 1
        if span_days > 366:
            raise RangeTooLargeError()

        ds = self.get_predictions_dataset()
        assert self.loader.manifest is not None
        reg = self.loader.manifest["region"]
        lats = ds.coords["lat"].values
        lons = ds.coords["lon"].values

        lat_idx, snapped_lat = snap_coordinate(lat, lats, reg["lat_min"], reg["lat_max"])
        lon_idx, snapped_lon = snap_coordinate(lon, lons, reg["lon_min"], reg["lon_max"])

        available = self.loader.manifest["available_dates"]
        avail_start = available["start"]
        avail_end = available["end"]
        avail_start_dt = validate_date(avail_start)
        avail_end_dt = validate_date(avail_end)

        if end_dt < avail_start_dt or start_dt > avail_end_dt:
            raise DateUnavailableError(
                f"Requested range [{start}, {end}] does not overlap "
                f"available dates [{avail_start}, {avail_end}]"
            )

        # Slice to dates that overlap available dates
        eff_start = max(start, avail_start)
        eff_end = min(end, avail_end)

        cell_series = ds.sel(time=slice(eff_start, eff_end)).isel(lat=lat_idx, lon=lon_idx)
        time_vals = pd.to_datetime(cell_series.coords["time"].values)
        dates_list = [t.strftime("%Y-%m-%d") for t in time_vals]

        p_vals = cell_series["temp_pred"].isel(depth=d_idx).values
        g_vals = cell_series["temp_glorys"].isel(depth=d_idx).values

        pred_list: list[float | None] = [
            None if np.isnan(v) else round(float(v), 2) for v in p_vals
        ]
        glorys_list: list[float | None] = [
            None if np.isnan(v) else round(float(v), 2) for v in g_vals
        ]

        splits_list = [self.get_date_split(d) for d in dates_list]

        return TimeseriesResponse(
            run_id=self.loader.manifest["active_run_id"],
            synthetic=self.loader.manifest["synthetic"],
            lat=snapped_lat,
            lon=snapped_lon,
            depth_m=target_depth,
            dates=dates_list,
            pred=pred_list,
            glorys=glorys_list,
            split=splits_list,
        )

    def get_region_summary(
        self,
        date: str,
        depth_m: float | None = None,
        depth_index: int | None = None,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
        region: Literal["arabian_sea", "bay_of_bengal", "equatorial"] | None = None,
        field: Literal["pred", "glorys", "error"] = "pred",
    ) -> RegionSummaryResponse:
        """Compute area-weighted aggregate summary over a sub-region (PRD_BACKEND §4.6)."""
        d_idx, target_depth = self.resolve_depth(depth_m, depth_index)
        split = self.get_date_split(date)

        if split == "live" and field in ("glorys", "error"):
            raise ReferenceUnavailableError()

        ds = self.get_predictions_dataset()
        lats = ds.coords["lat"].values
        lons = ds.coords["lon"].values

        mask_2d = get_region_mask(
            lats=lats,
            lons=lons,
            region=region,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )

        try:
            ds_date = ds.sel(time=date)
        except KeyError:
            raise DateUnavailableError(f"Date {date} is unavailable in dataset")

        cov_2d = ds_date["coverage"].values.astype(int)

        if field == "pred":
            vals_2d = ds_date["temp_pred"].isel(depth=d_idx).values.astype(np.float32)
        elif field == "glorys":
            vals_2d = ds_date["temp_glorys"].isel(depth=d_idx).values.astype(np.float32)
        elif field == "error":
            p = ds_date["temp_pred"].isel(depth=d_idx).values.astype(np.float32)
            g = ds_date["temp_glorys"].isel(depth=d_idx).values.astype(np.float32)
            vals_2d = p - g
        else:
            raise APIError(422, "invalid_field", f"Field '{field}' must be pred, glorys, or error")

        # Cells in region with coverage 1
        ocean_cells = (cov_2d == 1) & mask_2d
        n_total = int(np.count_nonzero(ocean_cells))

        valid_cells = ocean_cells & ~np.isnan(vals_2d)
        n_valid = int(np.count_nonzero(valid_cells))

        if n_valid > 0:
            weights_2d = compute_area_weights(lats, len(lons))
            w = weights_2d[valid_cells]
            x = vals_2d[valid_cells]
            sum_w = float(np.sum(w))

            mean_val = float(np.sum(w * x) / sum_w)
            var_val = float(np.sum(w * (x - mean_val) ** 2) / sum_w)
            std_val = float(np.sqrt(max(0.0, var_val)))

            min_val = float(np.min(x))
            max_val = float(np.max(x))

            out_mean = round(mean_val, 2)
            out_std = round(std_val, 2)
            out_min = round(min_val, 2)
            out_max = round(max_val, 2)
        else:
            out_mean = None
            out_std = None
            out_min = None
            out_max = None

        assert self.loader.manifest is not None
        return RegionSummaryResponse(
            run_id=self.loader.manifest["active_run_id"],
            synthetic=self.loader.manifest["synthetic"],
            date=date,
            depth_m=target_depth,
            field=field,
            mean=out_mean,
            min=out_min,
            max=out_max,
            std=out_std,
            n_valid=n_valid,
            n_total=n_total,
        )

    def get_metrics(
        self,
        split: Literal["val", "test"] = "val",
        group_by: Literal["depth", "season", "region"] = "depth",
    ) -> MetricsResponse:
        """Query evaluation metrics grouped by depth, season, or region."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        manifest = self.loader.manifest
        data_dir = self.loader.data_dir

        if split == "test":
            if not manifest.get("test_evaluated", False):
                raise TestNotEvaluatedError("Test split has not been evaluated")
            test_file = manifest.get("files", {}).get("metrics_test")
            if not test_file:
                raise TestNotEvaluatedError("Test metrics file is not available")
            files_to_read: list[tuple[str, str | None]] = [(test_file, None)]
        else:
            files_to_read = []
            files_dict = manifest.get("files", {})
            if group_by == "depth":
                active_f = files_dict.get("metrics_val")
                if active_f:
                    files_to_read.append((active_f, None))
                for bf in files_dict.get("baselines_val", []):
                    files_to_read.append((bf, None))
            elif group_by == "season":
                active_f = files_dict.get("metrics_val_by_season")
                if active_f:
                    files_to_read.append((active_f, "season"))
            elif group_by == "region":
                active_f = files_dict.get("metrics_val_by_region")
                if active_f:
                    files_to_read.append((active_f, "region"))

        rows: list[MetricRow] = []
        for rel_path, group_col in files_to_read:
            csv_path = data_dir / rel_path
            if not csv_path.exists():
                logger.warning("Metrics file %s does not exist", csv_path)
                continue

            cache_key = str(csv_path)
            if cache_key in self._cached_metrics:
                df = self._cached_metrics[cache_key]
            else:
                df = pd.read_csv(csv_path)
                self._cached_metrics[cache_key] = df

            for _, r in df.iterrows():
                depth_raw = r["depth_m"]
                try:
                    depth_m_val: float | str = float(depth_raw)
                except (ValueError, TypeError):
                    depth_m_val = str(depth_raw)

                group_val: str | None = None
                if group_col and group_col in r and pd.notna(r[group_col]):
                    group_val = str(r[group_col])

                rows.append(
                    MetricRow(
                        model=str(r["model"]),
                        split=str(r["split"]),
                        depth_m=depth_m_val,
                        group=group_val,
                        n=int(r["n"]),
                        mae_c=round(float(r["mae_c"]), 4),
                        rmse_c=round(float(r["rmse_c"]), 4),
                        bias_c=round(float(r["bias_c"]), 4),
                        skill_vs_clim_pct=round(float(r["skill_vs_clim_pct"]), 2),
                    )
                )

        return MetricsResponse(
            run_id=manifest["active_run_id"],
            synthetic=manifest["synthetic"],
            split=split,
            group_by=group_by,
            rows=rows,
        )

    def get_argo(self) -> ArgoResponse:
        """Query ARGO validation metrics table."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        manifest = self.loader.manifest
        data_dir = self.loader.data_dir

        argo_rel = manifest.get("files", {}).get("argo")
        if not argo_rel:
            raise APIError(404, "argo_not_available", "ARGO validation metrics not available")

        csv_path = data_dir / argo_rel
        if not csv_path.exists():
            raise APIError(404, "argo_not_available", f"ARGO file {argo_rel} not found")

        if self._cached_argo is not None:
            df = self._cached_argo
        else:
            df = pd.read_csv(csv_path)
            self._cached_argo = df

        def _opt_float(val: Any) -> float | None:
            if pd.isna(val) or val is None or val == "":
                return None
            try:
                f = float(val)
                return None if np.isnan(f) else round(f, 4)
            except (ValueError, TypeError):
                return None

        rows: list[ArgoMetricRow] = []
        for _, r in df.iterrows():
            rows.append(
                ArgoMetricRow(
                    depth_m=round(float(r["depth_m"]), 2),
                    n_profiles=int(r["n_profiles"]),
                    n_obs=int(r["n_obs"]),
                    mae_c=_opt_float(r.get("mae_c")),
                    rmse_c=_opt_float(r.get("rmse_c")),
                    bias_c=_opt_float(r.get("bias_c")),
                    glorys_mae_c=_opt_float(r.get("glorys_mae_c")),
                    glorys_rmse_c=_opt_float(r.get("glorys_rmse_c")),
                    grep_mae_c=_opt_float(r.get("grep_mae_c")),
                    grep_rmse_c=_opt_float(r.get("grep_rmse_c")),
                    armor3d_mae_c=_opt_float(r.get("armor3d_mae_c")),
                    armor3d_rmse_c=_opt_float(r.get("armor3d_rmse_c")),
                )
            )

        n_profiles_total = int(df["n_profiles"].max()) if not df.empty else 0
        n_obs_total = int(df["n_obs"].sum()) if not df.empty else 0

        return ArgoResponse(
            run_id=manifest["active_run_id"],
            synthetic=manifest["synthetic"],
            n_profiles_total=n_profiles_total,
            n_obs_total=n_obs_total,
            caveat=ARGO_CAVEAT,
            rows=rows,
        )

    def get_argo_profiles(
        self,
        date: str | None = None,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
    ) -> ArgoProfilesResponse:
        """Query matched ARGO profiles for map markers."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        manifest = self.loader.manifest
        data_dir = self.loader.data_dir

        if lat_min is not None and lat_max is not None and lat_min > lat_max:
            raise OutOfRegionError("Invalid bounding box: lat_min exceeds lat_max")
        if lon_min is not None and lon_max is not None and lon_min > lon_max:
            raise OutOfRegionError("Invalid bounding box: lon_min exceeds lon_max")

        matches_rel = manifest.get("files", {}).get("argo_matches")
        if not matches_rel:
            return ArgoProfilesResponse(
                run_id=manifest["active_run_id"],
                synthetic=manifest["synthetic"],
                profiles=[],
            )

        parquet_path = data_dir / matches_rel
        if not parquet_path.exists():
            return ArgoProfilesResponse(
                run_id=manifest["active_run_id"],
                synthetic=manifest["synthetic"],
                profiles=[],
            )

        if self._cached_argo_matches is not None:
            df = self._cached_argo_matches
        else:
            df = pd.read_parquet(parquet_path)
            self._cached_argo_matches = df

        filtered = df
        if date is not None:
            validate_date(date)
            filtered = filtered[filtered["time"].astype(str).str.startswith(date)]
        if lat_min is not None:
            filtered = filtered[filtered["lat"] >= lat_min]
        if lat_max is not None:
            filtered = filtered[filtered["lat"] <= lat_max]
        if lon_min is not None:
            filtered = filtered[filtered["lon"] >= lon_min]
        if lon_max is not None:
            filtered = filtered[filtered["lon"] <= lon_max]

        if filtered.empty:
            return ArgoProfilesResponse(
                run_id=manifest["active_run_id"],
                synthetic=manifest["synthetic"],
                profiles=[],
            )

        grouped = (
            filtered.groupby(["platform_number", "cycle", "time", "lat", "lon"], as_index=False)
            .size()
            .rename(columns={"size": "n_obs"})
        )
        grouped = grouped.iloc[:500]

        profiles = [
            ArgoProfileMatch(
                platform_number=str(r["platform_number"]),
                cycle=int(r["cycle"]),
                time=str(r["time"]),
                lat=round(float(r["lat"]), 4),
                lon=round(float(r["lon"]), 4),
                n_obs=int(r["n_obs"]),
            )
            for _, r in grouped.iterrows()
        ]

        return ArgoProfilesResponse(
            run_id=manifest["active_run_id"],
            synthetic=manifest["synthetic"],
            profiles=profiles,
        )

    def get_headline(self) -> HeadlineResponse:
        """Query headline success criterion table."""
        self._ensure_loaded()
        assert self.loader.manifest is not None
        manifest = self.loader.manifest
        data_dir = self.loader.data_dir

        headline_rel = manifest.get("files", {}).get("headline")
        if not headline_rel:
            raise HeadlineNotAvailableError("Headline evaluation metrics are not available")

        csv_path = data_dir / headline_rel
        if not csv_path.exists():
            raise HeadlineNotAvailableError(f"Headline file {headline_rel} not found")

        if self._cached_headline is not None:
            df = self._cached_headline
        else:
            df = pd.read_csv(csv_path)
            self._cached_headline = df

        def _safe_float(val: Any) -> float | None:
            if val is None or pd.isna(val) or val == "":
                return None
            try:
                f = float(val)
                return None if np.isnan(f) else round(f, 4)
            except (ValueError, TypeError):
                return None

        def _safe_int(val: Any) -> int | None:
            if val is None or pd.isna(val) or val == "":
                return None
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        rows: list[HeadlineRow] = []
        for _, r in df.iterrows():
            rows.append(
                HeadlineRow(
                    period=str(r["period"]),
                    product=str(r["product"]),
                    n_floats=_safe_int(r.get("n_floats")),
                    n_obs=_safe_int(r.get("n_obs")),
                    rmse_c=_safe_float(r.get("rmse_c")),
                    ci_low_c=_safe_float(r.get("ci_low_c")),
                    ci_high_c=_safe_float(r.get("ci_high_c")),
                    diff_vs_model_c=_safe_float(r.get("diff_vs_model_c")),
                    diff_ci_low_c=_safe_float(r.get("diff_ci_low_c")),
                    diff_ci_high_c=_safe_float(r.get("diff_ci_high_c")),
                )
            )

        reported_period = "test" if manifest.get("test_evaluated", False) else "val"
        competitors_with_data = [
            r
            for r in rows
            if r.period == reported_period
            and r.product.lower() != "model"
            and r.diff_ci_high_c is not None
        ]
        if competitors_with_data:
            met: bool | None = all(
                c.diff_ci_high_c is not None and c.diff_ci_high_c < 0.0
                for c in competitors_with_data
            )
        else:
            met = None

        return HeadlineResponse(
            run_id=manifest["active_run_id"],
            synthetic=manifest["synthetic"],
            criterion=HEADLINE_CRITERION,
            met=met,
            rows=rows,
        )


store = OutputStore()
