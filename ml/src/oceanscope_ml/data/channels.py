"""Input channel order for the U-Net (PRD_AI_ML §5.1).

``channel_list`` is the only place the channel order is defined. Every checkpoint stores the
names it returns, so inference can rebuild the same tensor. Nothing else may hardcode the order.

Order (§5.1): dynamic channels (core + forcing) for each history day, oldest to newest; then
long-history frames; then index channels; then ARGO channels; then static channels; then extra
channels. With the default config (ARGO and indices off) this gives 122 channels. The §5.1
figure of 143 assumes ARGO and indices are on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from oceanscope_ml.config import ExperimentConfig

logger = logging.getLogger("oceanscope_ml.data.channels")

EXTRA_KINDS = ("doy_sin", "doy_cos", "lat", "lon", "ocean_mask")
ARGO_EXTRA_NAMES = ("argo_mask", "argo_age")


@dataclass(frozen=True)
class ChannelSpec:
    """One input channel.

    ``source`` is the store variable (or the config name for extra channels). ``lag`` is the
    number of days before the target day. For long-history frames it is the first (oldest)
    lag of the window, and the frame is the mean over ``window_days`` days. ``depth_index``
    picks one level of a 3D static field or one ARGO level.
    """

    name: str
    kind: str
    source: str
    lag: int = 0
    window_days: int = 1
    depth_index: int | None = None


def _dynamic_channels(inputs: dict[str, Any]) -> list[ChannelSpec]:
    dynamic_vars = list(inputs.get("core", [])) + list(inputs.get("forcing", []))
    history_days = int(inputs.get("history_days", 1))
    channels: list[ChannelSpec] = []
    for lag in range(history_days - 1, -1, -1):  # oldest first
        for var in dynamic_vars:
            channels.append(ChannelSpec(name=f"{var}@t-{lag}", kind="dynamic", source=var, lag=lag))
    return channels


def _long_history_channels(inputs: dict[str, Any]) -> list[ChannelSpec]:
    long_cfg = inputs.get("long_history", {}) or {}
    total_days = int(long_cfg.get("days", 0))
    stride = int(long_cfg.get("stride", 5))
    if total_days <= 0:
        return []
    if str(long_cfg.get("reduce", "mean")) != "mean":
        raise ValueError("Only reduce: mean is implemented for long-history frames")
    # Frame k covers lags k*stride-stride+1 .. k*stride, so frames tile lags 1..total_days.
    n_frames = total_days // stride
    channels: list[ChannelSpec] = []
    for frame in range(n_frames, 0, -1):  # oldest first
        first_lag = frame * stride - stride + 1
        for var in long_cfg.get("vars", []):
            channels.append(
                ChannelSpec(
                    name=f"{var}@mean{first_lag}-{first_lag + stride - 1}",
                    kind="long_history",
                    source=str(var),
                    lag=first_lag,
                    window_days=stride,
                )
            )
    return channels


def _index_channels(inputs: dict[str, Any]) -> list[ChannelSpec]:
    if not bool(inputs.get("indices_enabled", False)):
        return []
    return [
        ChannelSpec(name=str(name), kind="index", source=str(name))
        for name in inputs.get("indices", [])
    ]


def _argo_channels(inputs: dict[str, Any], depths_m: list[float]) -> list[ChannelSpec]:
    argo_cfg = inputs.get("argo_input", {}) or {}
    if not bool(argo_cfg.get("enabled", False)):
        return []
    channels = [
        ChannelSpec(
            name=f"argo_anom@{depth:g}m",
            kind="argo",
            source="argo_anomaly",
            depth_index=i,
        )
        for i, depth in enumerate(depths_m)
    ]
    channels += [ChannelSpec(name=name, kind="argo", source=name) for name in ARGO_EXTRA_NAMES]
    return channels


def _static_channels(inputs: dict[str, Any], depths_m: list[float]) -> list[ChannelSpec]:
    channels: list[ChannelSpec] = []
    for var in inputs.get("static", []):
        if var == "bathymetry":
            channels.append(ChannelSpec(name="bathymetry", kind="static", source="bathymetry"))
        elif var == "woa_clim_profile":
            channels += [
                ChannelSpec(
                    name=f"woa@{depth:g}m",
                    kind="static",
                    source="woa_clim_profile",
                    depth_index=i,
                )
                for i, depth in enumerate(depths_m)
            ]
        else:
            raise ValueError(f"Unknown static input: {var}")
    return channels


def _extra_channels(inputs: dict[str, Any]) -> list[ChannelSpec]:
    names = list(inputs.get("extra", []))
    unknown = [n for n in names if n not in EXTRA_KINDS]
    if unknown:
        raise ValueError(f"Unknown extra inputs: {unknown}")
    return [ChannelSpec(name=n, kind="extra", source=n) for n in names]


def channel_list(config: ExperimentConfig) -> list[ChannelSpec]:
    """Return the input channels in the order the model expects, and log the count."""
    inputs = config.inputs
    depths = list(config.depths_m)
    channels = (
        _dynamic_channels(inputs)
        + _long_history_channels(inputs)
        + _index_channels(inputs)
        + _argo_channels(inputs, depths)
        + _static_channels(inputs, depths)
        + _extra_channels(inputs)
    )
    logger.info("Input channels: %d", len(channels))
    return channels


def channel_names(config: ExperimentConfig) -> list[str]:
    """Names of the input channels, in order. Stored in every checkpoint."""
    return [c.name for c in channel_list(config)]
