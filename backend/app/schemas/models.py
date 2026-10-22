"""Pydantic response and request models for OceanScope AI backend API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Detailed error object returned on all 4xx/5xx responses."""

    code: str = Field(..., description="Snake-case machine-readable error code")
    message: str = Field(..., description="Human-readable explanation of the error")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional additional context about the error",
    )


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Meta Schemas
# ---------------------------------------------------------------------------


class Region(BaseModel):
    """Bounding box of the modelled region."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class GridShape(BaseModel):
    """Shape of the 2D spatial grid (lat, lon)."""

    lat: int
    lon: int


class AvailableDates(BaseModel):
    """Date range covered by the dataset."""

    start: str = Field(..., description="Start date YYYY-MM-DD")
    end: str = Field(..., description="End date YYYY-MM-DD")


class Splits(BaseModel):
    """Start and end dates for train, validation, and test splits."""

    train: list[str] = Field(..., min_length=2, max_length=2)
    val: list[str] = Field(..., min_length=2, max_length=2)
    test: list[str] = Field(..., min_length=2, max_length=2)


class ColorRange(BaseModel):
    """2nd and 98th percentile temperature bounds for visualization at depth."""

    depth_m: float
    vmin: float
    vmax: float


class ErrorRange(BaseModel):
    """Fixed color scale bounds for reconstruction error fields."""

    vmin: float
    vmax: float


class Caveats(BaseModel):
    """Fixed scientific and data caveats returned to the UI."""

    glorys: str
    argo: str
    synthetic: str
    live: str


class ChatInfo(BaseModel):
    """Status of the LLM chat assistant."""

    enabled: bool
    provider_label: str | None = None


class MetaResponse(BaseModel):
    """Response model for GET /meta."""

    run_id: str
    synthetic: bool
    dataset_version: str
    model_name: str
    region: Region
    grid_resolution_deg: float
    grid_shape: GridShape
    depths_m: list[float]
    available_dates: AvailableDates
    latest_live_date: str | None = None
    splits: Splits
    test_evaluated: bool
    color_ranges: list[ColorRange]
    error_range: ErrorRange
    caveats: Caveats
    chat: ChatInfo


# ---------------------------------------------------------------------------
# Map Schemas
# ---------------------------------------------------------------------------


class MapStats(BaseModel):
    """Summary statistics for the returned 2D field."""

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    n_valid: int


class MapResponse(BaseModel):
    """Response model for GET /map."""

    run_id: str
    synthetic: bool
    date: str
    depth_m: float
    field: Literal["pred", "glorys", "error"]
    split: str
    units: str = "degC"
    lat: list[float]
    lon: list[float]
    values: list[list[float | None]]
    coverage: list[list[int]]
    stats: MapStats


# ---------------------------------------------------------------------------
# Profile Schemas
# ---------------------------------------------------------------------------


class ProfileResponse(BaseModel):
    """Response model for GET /profile."""

    run_id: str
    synthetic: bool
    date: str
    lat: float
    lon: float
    split: str
    coverage: int
    depths_m: list[float]
    pred: list[float | None]
    glorys: list[float | None]
    error: list[float | None]


# ---------------------------------------------------------------------------
# Time Series Schemas
# ---------------------------------------------------------------------------


class TimeseriesResponse(BaseModel):
    """Response model for GET /timeseries."""

    run_id: str
    synthetic: bool
    lat: float
    lon: float
    depth_m: float
    dates: list[str]
    pred: list[float | None]
    glorys: list[float | None]
    split: list[str]


# ---------------------------------------------------------------------------
# Region Summary Schemas
# ---------------------------------------------------------------------------


class RegionSummaryResponse(BaseModel):
    """Response model for GET /region-summary."""

    run_id: str
    synthetic: bool
    date: str
    depth_m: float
    field: Literal["pred", "glorys", "error"]
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    n_valid: int
    n_total: int


# ---------------------------------------------------------------------------
# Metrics Schemas
# ---------------------------------------------------------------------------


class MetricRow(BaseModel):
    """Single evaluation metric row."""

    model_config = ConfigDict(extra="ignore")

    model: str
    split: str
    depth_m: float | str
    group: str | None = None
    n: int
    mae_c: float
    rmse_c: float
    bias_c: float
    skill_vs_clim_pct: float


class MetricsResponse(BaseModel):
    """Response model for GET /metrics."""

    run_id: str
    synthetic: bool
    split: Literal["val", "test"]
    group_by: Literal["depth", "season", "region"]
    rows: list[MetricRow]


# ---------------------------------------------------------------------------
# ARGO Validation Schemas
# ---------------------------------------------------------------------------


class ArgoMetricRow(BaseModel):
    """Per-depth ARGO validation row comparing reconstructed temperature to matchups."""

    depth_m: float
    n_profiles: int
    n_obs: int
    mae_c: float
    rmse_c: float
    bias_c: float
    glorys_mae_c: float
    glorys_rmse_c: float
    grep_mae_c: float | None = None
    grep_rmse_c: float | None = None
    armor3d_mae_c: float | None = None
    armor3d_rmse_c: float | None = None


class ArgoResponse(BaseModel):
    """Response model for GET /argo."""

    run_id: str
    synthetic: bool
    n_profiles_total: int
    n_obs_total: int
    caveat: str
    rows: list[ArgoMetricRow]


class ArgoProfileMatch(BaseModel):
    """ARGO profile marker coordinate for map display."""

    platform_number: str
    cycle: int
    time: str
    lat: float
    lon: float
    n_obs: int


class ArgoProfilesResponse(BaseModel):
    """Response model for GET /argo/profiles."""

    run_id: str
    synthetic: bool
    profiles: list[ArgoProfileMatch]


# ---------------------------------------------------------------------------
# Headline Validation Schemas
# ---------------------------------------------------------------------------


class HeadlineRow(BaseModel):
    """Single competitor row in the pre-registered Bay of Bengal 50-300m evaluation."""

    period: str
    product: str
    n_floats: int
    n_obs: int
    rmse_c: float
    ci_low_c: float
    ci_high_c: float
    diff_vs_model_c: float
    diff_ci_low_c: float
    diff_ci_high_c: float


class HeadlineResponse(BaseModel):
    """Response model for GET /headline."""

    run_id: str
    synthetic: bool
    criterion: str
    met: bool | None
    rows: list[HeadlineRow]


# ---------------------------------------------------------------------------
# Chat Assistant Schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatContext(BaseModel):
    """Current explorer UI state passed as context to the assistant."""

    date: str | None = None
    depth_m: float | None = None
    lat: float | None = None
    lon: float | None = None


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    messages: list[ChatMessage] = Field(..., max_length=10)
    context: ChatContext | None = None


class ToolCallRecord(BaseModel):
    """Recorded tool call executed during assistant reasoning."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    """Response model for POST /chat."""

    answer: str
    tool_calls: list[ToolCallRecord]
    provider_label: str
    run_id: str
    synthetic: bool
