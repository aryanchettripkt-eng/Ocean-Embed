"""Configuration loader and schema for OceanScope ML pipeline (PRD_AI_ML §2)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RegionConfig:
    lon_min: float = 40.0
    lon_max: float = 100.0
    lat_min: float = 0.0
    lat_max: float = 30.0
    lon_convention: str = "-180_180"
    lat_order: str = "ascending"


@dataclass
class GridConfig:
    resolution_deg: float = 0.25


@dataclass
class TimeConfig:
    start: str = "1993-01-01"
    end: str = "2021-12-31"
    day_boundary: str = "UTC"


@dataclass
class SplitsConfig:
    train: list[str] = field(default_factory=lambda: ["1993-03-02", "2017-12-01"])
    val: list[str] = field(default_factory=lambda: ["2018-01-01", "2020-12-01"])
    test: list[str] = field(default_factory=lambda: ["2021-01-01", "2021-12-31"])
    gap_days: int = 30


@dataclass
class PathsConfig:
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    outputs: Path = Path("outputs")
    checkpoints: Path = Path("checkpoints")


@dataclass
class BaselinesConfig:
    max_train_samples: int = 500_000
    ridge_alpha: float = 1.0


@dataclass
class ExperimentConfig:
    project: str = "oceanscope-ai"
    dataset_version: str = "v1"
    region: RegionConfig = field(default_factory=RegionConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    depths_m: list[float] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    finetune_argo: dict[str, Any] = field(default_factory=dict)
    climatology: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)
    train: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    publish: dict[str, Any] = field(default_factory=dict)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    raw_dict: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    """Load configuration YAML and apply environment variable overrides."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Parse nested configs
    region_data = data.get("region", {})
    region = RegionConfig(
        lon_min=float(region_data.get("lon_min", 40.0)),
        lon_max=float(region_data.get("lon_max", 100.0)),
        lat_min=float(region_data.get("lat_min", 0.0)),
        lat_max=float(region_data.get("lat_max", 30.0)),
        lon_convention=str(region_data.get("lon_convention", "-180_180")),
        lat_order=str(region_data.get("lat_order", "ascending")),
    )

    grid = GridConfig(resolution_deg=float(data.get("grid", {}).get("resolution_deg", 0.25)))

    time_data = data.get("time", {})
    time_cfg = TimeConfig(
        start=str(time_data.get("start", "1993-01-01")),
        end=str(time_data.get("end", "2021-12-31")),
        day_boundary=str(time_data.get("day_boundary", "UTC")),
    )

    splits_data = data.get("splits", {})
    splits = SplitsConfig(
        train=list(splits_data.get("train", ["1993-03-02", "2017-12-01"])),
        val=list(splits_data.get("val", ["2018-01-01", "2020-12-01"])),
        test=list(splits_data.get("test", ["2021-01-01", "2021-12-31"])),
        gap_days=int(splits_data.get("gap_days", 30)),
    )

    # Path overrides from environment variables
    paths_data = data.get("paths", {})
    data_dir_env = os.environ.get("OCEANSCOPE_DATA_DIR")
    out_dir_env = os.environ.get("OCEANSCOPE_OUT_DIR")
    ckpt_dir_env = os.environ.get("OCEANSCOPE_CKPT_DIR")

    raw_path = (
        Path(data_dir_env) / "raw" if data_dir_env else Path(paths_data.get("raw", "data/raw"))
    )
    processed_path = (
        Path(data_dir_env) / "processed"
        if data_dir_env
        else Path(paths_data.get("processed", "data/processed"))
    )
    outputs_path = Path(out_dir_env) if out_dir_env else Path(paths_data.get("outputs", "outputs"))
    checkpoints_path = (
        Path(ckpt_dir_env) if ckpt_dir_env else Path(paths_data.get("checkpoints", "checkpoints"))
    )

    paths = PathsConfig(
        raw=raw_path,
        processed=processed_path,
        outputs=outputs_path,
        checkpoints=checkpoints_path,
    )

    baselines_data = data.get("baselines", {})
    baselines = BaselinesConfig(
        max_train_samples=int(baselines_data.get("max_train_samples", 500_000)),
        ridge_alpha=float(baselines_data.get("ridge_alpha", 1.0)),
    )

    return ExperimentConfig(
        project=str(data.get("project", "oceanscope-ai")),
        dataset_version=str(data.get("dataset_version", "v1")),
        region=region,
        grid=grid,
        time=time_cfg,
        splits=splits,
        depths_m=[float(d) for d in data.get("depths_m", [])],
        inputs=data.get("inputs", {}),
        target=data.get("target", {}),
        finetune_argo=data.get("finetune_argo", {}),
        climatology=data.get("climatology", {}),
        paths=paths,
        train=data.get("train", {}),
        inference=data.get("inference", {}),
        publish=data.get("publish", {}),
        baselines=baselines,
        raw_dict=data,
    )
