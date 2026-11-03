"""ARGO profiles as sparse input channels (PRD_AI_ML §5.1, §7, P4-6).

The 17 ARGO channels of a sample on day t are built from the profiles of non-held-out floats on
days t - window_days + 1 .. t, placed in their nearest ocean cell:

- ``argo_anom@<d>m`` (15): (T_argo - clim) / sigma_d at that depth, 0 where there is no value;
- ``argo_mask``: 1 in cells with a profile, else 0;
- ``argo_age``: (t - profile day + 1) / window_days in those cells, else 0.

When several profiles fall in one cell, the most recent one is used.

Which profiles are inputs, and which are hidden and become same-day ARGO targets, depends on the
mode (PRD §5.9: a profile used as a target is never in that sample's input channels):

- ``train``: ``train_drop_frac`` of the window's profiles are hidden, drawn again each epoch with
  a fixed seed; the hidden ones from day t are the targets.
- ``val``: the profiles of a fixed set of "score" floats (20% of the non-held-out floats, by
  float, fixed seed) are never inputs; their day-t profiles are the targets. The validation ARGO
  score that picks the best epoch therefore never sees its own floats as inputs.
- ``predict``: every profile in the window is an input; there are no targets.

Floats in the held-out list (``argo_eval_floats.json``) are refused at construction: they never
become inputs or targets, in any mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.argo import profile_levels, read_levels
from oceanscope_ml.evaluation.argo_validation import _nearest_cell

logger = logging.getLogger("oceanscope_ml.data.argo_inputs")

MODES = ("train", "val", "predict")
SCORE_FRACTION = 0.2
SCORE_SEED = 7
N_EXTRA = 2  # argo_mask, argo_age


@dataclass
class ArgoSample:
    planes: np.ndarray  # float32 [D + 2, H, W]
    target: np.ndarray  # float32 [D, H, W], 0 where no target
    target_mask: np.ndarray  # bool [D, H, W]


def _day_number(times: pd.DatetimeIndex | np.ndarray | pd.Series) -> np.ndarray:
    days = pd.DatetimeIndex(times).normalize().values.astype("datetime64[D]")
    return days.astype(np.int64)


class ArgoInputs:
    """Profiles of non-held-out floats on the store grid, as normalized anomalies."""

    def __init__(
        self,
        profiles: pd.DataFrame,
        ds: xr.Dataset,
        sigma_d: np.ndarray,
        holdout: set[str],
        window_days: int = 10,
        train_drop_frac: float = 0.3,
        seed: int = 42,
    ) -> None:
        if profiles["platform_number"].astype(str).isin(holdout).any():
            raise AssertionError("A held-out float reached the ARGO input channels (PRD §7)")
        self.window_days = int(window_days)
        self.train_drop_frac = float(train_drop_frac)
        self.seed = int(seed)
        self.epoch = 0
        n_depth = int(sigma_d.size)
        self.n_depth = n_depth
        self.n_lat, self.n_lon = int(ds.sizes["lat"]), int(ds.sizes["lon"])
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        empty = np.zeros(0, dtype=np.int64)
        self.day, self.i_lat, self.i_lon, self.float_id = empty, empty, empty, empty
        self.values = np.zeros((0, n_depth), dtype=np.float32)
        self.floats = np.array([], dtype=str)
        if profiles.empty:
            self.score_floats: set[str] = set()
            self._is_score = np.zeros(0, dtype=bool)
            return
        i_lat, lat_ok = _nearest_cell(lat, profiles["lat"].to_numpy(float))
        i_lon, lon_ok = _nearest_cell(lon, profiles["lon"].to_numpy(float))
        ocean = np.asarray(ds["ocean_mask"].values).astype(bool)
        keep = lat_ok & lon_ok
        keep[keep] &= ocean[i_lat[keep], i_lon[keep]]
        prof = profiles.loc[keep].reset_index(drop=True)
        i_lat, i_lon = i_lat[keep], i_lon[keep]
        doy = pd.DatetimeIndex(prof["day"]).day_of_year.to_numpy() - 1
        clim = (
            ds["clim"]
            .isel(
                dayofyear=xr.DataArray(doy, dims="p"),
                lat=xr.DataArray(i_lat, dims="p"),
                lon=xr.DataArray(i_lon, dims="p"),
            )
            .transpose("p", "depth")
            .values
        )
        temps = prof[[f"t_{d}" for d in range(n_depth)]].to_numpy(float)
        anomaly = (temps - np.asarray(clim, dtype=float)) / np.asarray(sigma_d, float)[None, :]
        good = np.isfinite(anomaly).any(axis=1)
        order = np.argsort(_day_number(prof["day"].values)[good], kind="stable")
        self.day = _day_number(prof["day"].values)[good][order]
        self.i_lat = i_lat[good][order]
        self.i_lon = i_lon[good][order]
        self.values = anomaly[good][order].astype(np.float32)
        names = prof["platform_number"].astype(str).to_numpy()[good][order]
        self.floats, self.float_id = np.unique(names, return_inverse=True)
        rng = np.random.default_rng(SCORE_SEED)
        n_score = max(1, int(round(SCORE_FRACTION * self.floats.size)))
        score = rng.choice(self.floats.size, size=n_score, replace=False)
        self._is_score = np.isin(self.float_id, score)
        self.score_floats = {str(self.floats[i]) for i in score}
        logger.info(
            "ARGO inputs: %d profiles from %d non-held-out floats (%d score floats for val)",
            self.day.size,
            self.floats.size,
            n_score,
        )

    def __len__(self) -> int:
        return int(self.day.size)

    def has_profiles(self, day: pd.Timestamp) -> bool:
        """True when the window ending on ``day`` holds at least one profile."""
        d = int(_day_number([day])[0])
        lo, hi = np.searchsorted(self.day, [d - self.window_days + 1, d + 1])
        return bool(hi > lo)

    def sample(self, day: pd.Timestamp, mode: str, index: int = 0) -> ArgoSample:
        """Input planes and same-day targets for one sample (see module docstring)."""
        if mode not in MODES:
            raise ValueError(f"ARGO input mode must be one of {MODES}, not {mode!r}")
        d = int(_day_number([day])[0])
        shape = (self.n_lat, self.n_lon)
        planes = np.zeros((self.n_depth + N_EXTRA, *shape), dtype=np.float32)
        target = np.zeros((self.n_depth, *shape), dtype=np.float32)
        target_mask = np.zeros(target.shape, dtype=bool)
        lo, hi = np.searchsorted(self.day, [d - self.window_days + 1, d + 1])
        rows = np.arange(lo, hi)
        if rows.size == 0:
            return ArgoSample(planes, target, target_mask)
        if mode == "train":
            rng = np.random.default_rng([self.seed, self.epoch, int(index)])
            hidden = rng.random(rows.size) < self.train_drop_frac
        elif mode == "val":
            hidden = self._is_score[rows]
        else:
            hidden = np.zeros(rows.size, dtype=bool)

        inputs = rows[~hidden]  # ascending day, so later writes are the most recent profile
        if inputs.size:
            il, jl = self.i_lat[inputs], self.i_lon[inputs]
            vals = self.values[inputs]
            planes[: self.n_depth, il, jl] = np.where(np.isfinite(vals), vals, 0.0).T
            planes[self.n_depth, il, jl] = 1.0
            age = (d - self.day[inputs] + 1).astype(np.float32) / self.window_days
            planes[self.n_depth + 1, il, jl] = age

        targets = rows[hidden & (self.day[rows] == d)]
        if targets.size:
            il, jl = self.i_lat[targets], self.i_lon[targets]
            vals = self.values[targets]
            finite = np.isfinite(vals)
            # Several hidden profiles in one cell: average them.
            sums = np.zeros(target.shape, dtype=np.float64)
            counts = np.zeros(target.shape, dtype=np.float64)
            for k in range(self.n_depth):
                ok = finite[:, k]
                np.add.at(sums[k], (il[ok], jl[ok]), vals[ok, k])
                np.add.at(counts[k], (il[ok], jl[ok]), 1.0)
            target_mask = counts > 0
            target = np.where(target_mask, sums / np.maximum(counts, 1.0), 0.0).astype(np.float32)
        return ArgoSample(planes, target, target_mask)


def enable_argo_input(config: ExperimentConfig) -> None:
    """Turn the ARGO input channels on in a loaded config (``inputs.argo_input.enabled``)."""
    inputs = config.inputs
    argo_cfg = dict(inputs.get("argo_input", {}) or {})
    argo_cfg["enabled"] = True
    inputs["argo_input"] = argo_cfg


def expand_input_weights(
    state: dict, base_names: list[str], new_names: list[str], key: str = "encoders.0.0.weight"
) -> dict:
    """Base model weights for a wider input: old channels copied by name, new ones zero.

    With the new channels at zero weight the expanded model gives exactly the base model's
    output, so fine-tuning starts from the base model.
    """
    missing = [n for n in base_names if n not in new_names]
    if missing:
        raise ValueError(f"The new channel list lacks base channels {missing[:5]}")
    weight = state[key]
    if weight.shape[1] != len(base_names):
        raise ValueError(f"{key} has {weight.shape[1]} inputs, base list has {len(base_names)}")
    new = weight.new_zeros((weight.shape[0], len(new_names), *weight.shape[2:]))
    position = {n: i for i, n in enumerate(new_names)}
    for i, name in enumerate(base_names):
        new[:, position[name]] = weight[:, i]
    out = dict(state)
    out[key] = new
    return out


def argo_inputs_for(
    config: ExperimentConfig,
    argo_paths: list[Path],
    holdout: set[str],
    ds: xr.Dataset,
    sigma_d: np.ndarray,
    depths_m: list[float],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> ArgoInputs:
    """ARGO input profiles of non-held-out floats, from ``start`` (default: the window before
    the training start) to ``end`` (default: the end of validation). Never the test period."""
    argo_cfg = config.inputs.get("argo_input", {}) or {}
    window = int(argo_cfg.get("window_days", 10))
    first = pd.Timestamp(config.splits.train[0]) - pd.Timedelta(days=window)
    start = max(start or first, pd.Timestamp(f"{int(argo_cfg.get('min_year', 2004))}-01-01"))
    end = end or pd.Timestamp(config.splits.val[1])
    if end >= pd.Timestamp(config.splits.test[0]):
        raise ValueError("ARGO inputs never read the sealed test period (PRD_AI_ML §0)")
    # One year at a time keeps memory small (and a float's cycle numbers stay unique per read).
    pieces = []
    for year in range(start.year, end.year + 1):
        lo = max(start, pd.Timestamp(f"{year}-01-01"))
        hi = min(end, pd.Timestamp(f"{year}-12-31"))
        year_paths = [p for p in argo_paths if _file_year(p) in (year, None)]
        for a, b in _split_at(lo, hi, config):
            levels = read_levels(year_paths, a, b)
            pieces.append(profile_levels(levels, depths_m, holdout))
    profiles = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return ArgoInputs(
        profiles,
        ds,
        sigma_d,
        holdout,
        window_days=window,
        train_drop_frac=float(argo_cfg.get("train_drop_frac", 0.3)),
        seed=int(config.train.get("seed", 42)),
    )


def _split_at(
    lo: pd.Timestamp, hi: pd.Timestamp, config: ExperimentConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[lo, hi] cut at the train and val starts, so each split's profiles are read on their own."""
    cuts = sorted(
        pd.Timestamp(getattr(config.splits, name)[0])
        for name in ("train", "val")
        if lo < pd.Timestamp(getattr(config.splits, name)[0]) <= hi
    )
    edges = [lo, *cuts]
    ends = [c - pd.Timedelta(days=1) for c in cuts] + [hi]
    return list(zip(edges, ends, strict=True))


def _file_year(path: Path) -> int | None:
    """Year of an ``argo_<YYYY-MM>.nc`` month file (``fetch_month``), else None (read always)."""
    stem = Path(path).stem
    if stem.startswith("argo_") and len(stem) == 12 and stem[5:9].isdigit():
        return int(stem[5:9])
    return None
