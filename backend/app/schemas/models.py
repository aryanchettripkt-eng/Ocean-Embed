"""Pydantic response and request models for OceanScope AI backend API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
# Health Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = "ok"
    run_id: str | None = None
    data_loaded: bool = False


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
    # Null when the served data is real (PRD_BACKEND §7; DECISIONS 2026-11-03).
    synthetic: str | None
    live: str
    ai: str
    postcard: str
    segment: str


class ChatInfo(BaseModel):
    """Status of the LLM chat assistant."""

    enabled: bool
    provider_label: str | None = None


class Features(BaseModel):
    """Per-feature availability of the Meta features (PRD_BACKEND §4.12)."""

    chat: bool
    voice: bool
    postcards: bool
    segment: bool
    segment_method: Literal["sam", "threshold"]
    beach: bool
    provider_label: str | None = None


class DateRange(BaseModel):
    """A run of consecutive days present in the served prediction store."""

    start: str = Field(..., description="First day YYYY-MM-DD")
    end: str = Field(..., description="Last day YYYY-MM-DD")


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
    date_ranges: list[DateRange] | None = Field(
        None,
        description=(
            "Runs of consecutive days the store holds, oldest first. Live mode appends recent "
            "days after the historical ones, leaving a gap inside available_dates; dates in a "
            "gap return 404 date_unavailable."
        ),
    )
    splits: Splits
    test_evaluated: bool
    color_ranges: list[ColorRange]
    error_range: ErrorRange
    caveats: Caveats
    chat: ChatInfo
    features: Features


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
    split: str | None = None
    units: str = "degC"
    lat: list[float]
    lon: list[float]
    values: list[list[float | None]]
    coverage: list[list[int]]
    stats: MapStats


class SectionResponse(BaseModel):
    """Response model for GET /section (PRD_BACKEND §4.11)."""

    run_id: str
    synthetic: bool
    date: str
    field: Literal["pred", "glorys", "error"]
    split: str | None = None
    units: str = "degC"
    axis: Literal["lat", "lon"] = Field(
        ...,
        description="The varying coordinate: 'lon' when lat was given, 'lat' when lon was given",
    )
    fixed: dict[str, float] = Field(
        ...,
        description="The snapped fixed coordinate (e.g. {'lat': 15.125} or {'lon': 65.125})",
    )
    positions: list[float] = Field(
        ...,
        description="Coordinates along the varying axis (lon or lat)",
    )
    depths_m: list[float] = Field(
        ...,
        description="All 15 model depths in meters",
    )
    values: list[list[float | None]] = Field(
        ...,
        description="Vertical section slice [depth][position], null where unavailable",
    )
    coverage: list[int] = Field(
        ...,
        description="Coverage codes [position] along the section line",
    )


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
    split: str | None = None
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
    split: list[str | None]


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
    mae_c: float | None = None
    rmse_c: float | None = None
    bias_c: float | None = None
    glorys_mae_c: float | None = None
    glorys_rmse_c: float | None = None
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
    n_floats: int | None = None
    n_obs: int | None = None
    rmse_c: float | None = None
    ci_low_c: float | None = None
    ci_high_c: float | None = None
    diff_vs_model_c: float | None = None
    diff_ci_low_c: float | None = None
    diff_ci_high_c: float | None = None


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

    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    depth_m: float | None = None
    lat: float | None = None
    lon: float | None = None


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    messages: list[ChatMessage] = Field(..., min_length=1, max_length=10)
    context: ChatContext | None = None

    @model_validator(mode="after")
    def _ends_with_user(self) -> ChatRequest:
        # The guide answers the last user turn; a trailing assistant turn would have the model
        # continue text the client wrote in the guide's voice.
        if self.messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return self


class ToolCallRecord(BaseModel):
    """Recorded tool call executed during assistant reasoning."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class UIAction(BaseModel):
    """A backend-validated UI tool call for the browser to apply (PRD_BACKEND §6.2)."""

    type: Literal["set_view", "select_point", "slice_at", "open_event"]
    args: dict[str, Any]


class ChatResponse(BaseModel):
    """Response model for POST /chat."""

    answer: str
    tool_calls: list[ToolCallRecord]
    ui_actions: list[UIAction] = Field(default_factory=list, max_length=3)
    grounded: bool = True
    provider_label: str
    run_id: str
    synthetic: bool


class TranscribeResponse(BaseModel):
    """Response model for POST /transcribe (PRD_BACKEND §4.13). The audio is never returned."""

    text: str
    duration_s: float | None = None
    provider_label: str
    run_id: str
    synthetic: bool


class EventBBox(BaseModel):
    """Bounding box of an event, inside the modelled region (PRD_BACKEND §4.14)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class EventItem(BaseModel):
    """One curated ocean event for the time machine (PRD_BACKEND §4.14)."""

    id: str
    title: str
    kind: str
    start: str
    end: str
    before: str
    after: str
    bbox: EventBBox
    suggested_depths_m: list[float]
    summary: str
    source_url: str
    source_name: str


class EventsResponse(BaseModel):
    """Response model for GET /events."""

    events: list[EventItem]
    run_id: str
    synthetic: bool


# ---------------------------------------------------------------------------
# Postcards, lasso and beach mode (PRD_BACKEND §4.15-4.17)
# ---------------------------------------------------------------------------


class PostcardRequest(BaseModel):
    """POST /postcard. No free text: every field is checked against the data (§4.15)."""

    model_config = ConfigDict(extra="forbid")

    date: str
    depth_m: float
    field: Literal["pred"] = "pred"
    event_id: str | None = Field(default=None, max_length=80)
    style: Literal["woodblock"] = "woodblock"


class PostcardCaption(BaseModel):
    """Numbers for the HTML caption the frontend overlays; never drawn into the image."""

    date: str
    depth_m: float
    mean_c: float | None = None
    source: str


class PostcardResponse(BaseModel):
    """Response model for POST /postcard."""

    image_url: str
    caption: PostcardCaption
    label: str
    prompt_version: str
    provider_label: str
    run_id: str
    synthetic: bool


class SegmentRequest(BaseModel):
    """POST /segment: the tapped point on our own rendered grid (§4.16)."""

    model_config = ConfigDict(extra="forbid")

    date: str
    depth_m: float
    field: Literal["pred", "glorys"] = "pred"
    lat: float
    lon: float


class SegmentMask(BaseModel):
    """Selected cells as run-length rows: [lat_index, first_lon, last_lon], both inclusive."""

    lat_index_ranges: list[list[int]]


class SegmentStats(BaseModel):
    """Area-weighted stats over the selected cells, same code as /region-summary."""

    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    n_valid: int
    area_km2: float


class SegmentResponse(BaseModel):
    """Response model for POST /segment."""

    method: Literal["sam", "threshold"]
    date: str
    depth_m: float
    field: Literal["pred", "glorys"]
    lat: float
    lon: float
    mask: SegmentMask
    stats: SegmentStats
    outline: list[list[float]]
    capped: bool = False
    caveat: str
    provider_label: str
    run_id: str
    synthetic: bool


class NearestOceanResponse(BaseModel):
    """Response model for GET /nearest-ocean (beach mode, §4.17). Snapped cell centre."""

    lat: float
    lon: float
    distance_km: float
    bearing_deg: float
    date: str
    run_id: str
    synthetic: bool


class GlassesAskResponse(BaseModel):
    """Response model for POST /glasses/ask (glasses ocean guide, §4.18)."""

    answer_text: str = Field(..., description="Short spoken answer for text-to-speech")
    question: str = Field(..., description="The question as typed or transcribed")
    temperature_c: float = Field(..., description="Model prediction, rounded to 0.1 °C")
    depth_asked_m: float | None = Field(None, description="Depth heard in the question, if any")
    depth_used_m: float = Field(..., description="Depth the temperature is for")
    levels_m: list[float] = Field(..., description="Model levels used (two when blended)")
    interpolated: bool = Field(..., description="True when blended between two levels")
    too_shallow: bool = Field(
        ..., description="True when no water that deep was found, so the deepest level was used"
    )
    lat: float
    lon: float
    distance_km: float
    bearing_deg: float
    direction: str = Field(..., description="16-point compass word from the user to the cell")
    date: str
    split: str | None = None
    caveats: list[str]
    run_id: str
    synthetic: bool
