"use client";

import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import {
  MapPin,
  SlidersHorizontal,
  Eye,
  EyeOff,
  AlertTriangle,
  CalendarRange,
  MessageCircle,
  Image as ImageIcon,
  Lasso,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

import { useArgoProfiles, useEvents, useMap, useMeta, useSection } from "../../../lib/api/hooks";
import { apiClient, type ChatResponse, type EventItem, type SegmentResponse } from "../../../lib/api/client";
import {
  applyEventPhase,
  applyUiActions,
  describeChanges,
  type CameraFocus,
  type CameraTarget,
  type EventPhase,
} from "../../../lib/guide/ui-actions";
import type { GuidePrompt, NavigateResult } from "../../../components/guide/GuideDrawer";
import {
  parseExplorerParams,
  type ExplorerState,
  type GridStyle,
  buildExplorerSearchParams,
} from "../../../lib/url-state";
import { DatePicker } from "../../../components/controls/DatePicker";
import { DepthSelector } from "../../../components/controls/DepthSelector";
import { FieldToggle, type FieldType } from "../../../components/controls/FieldToggle";
import { Colorbar } from "../../../components/map/Colorbar";
import { CoverageLegend } from "../../../components/map/CoverageLegend";
import { Banner } from "../../../components/common/Banner";
import { Loading } from "../../../components/common/Loading";
import { ErrorState } from "../../../components/common/ErrorState";
import { ErrorBoundary } from "../../../components/common/ErrorBoundary";
import { formatCoord } from "../../../lib/format";
import { getWallColorRange } from "../../../components/map3d/depthScale";
import { ViewSwitch } from "../../../components/map3d/ViewSwitch";
import { DepthRail } from "../../../components/map3d/DepthRail";

// MapView is statically included so MapLibre is counted directly in the bundle budget (PRD_FRONTEND §4)
import { MapView } from "../../../components/map/MapView";

// DetailsPanel is displayed only when a point is clicked; charts (Recharts) load on-demand
const DetailsPanel = dynamic(
  () => import("../../../components/panels/DetailsPanel").then((mod) => mod.DetailsPanel),
  {
    ssr: false,
    loading: () => (
      <div className="bg-white/90 dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 rounded-xl p-6 text-sm text-slate-700 dark:text-slate-300">
        Loading details...
      </div>
    ),
  }
);

// Meta features (PRD_FRONTEND §3.4) load only on first open, outside the initial bundle
// Lasso request abort. The backend gives SAM 6 s, then falls back to the threshold method.
const LASSO_TIMEOUT_MS = 20_000;

const GuideDrawer = dynamic(
  () => import("../../../components/guide/GuideDrawer").then((mod) => mod.GuideDrawer),
  { ssr: false }
);
const PostcardPanel = dynamic(
  () => import("../../../components/postcard/PostcardPanel").then((mod) => mod.PostcardPanel),
  { ssr: false }
);
const LassoPanel = dynamic(
  () => import("../../../components/lasso/LassoPanel").then((mod) => mod.LassoPanel),
  { ssr: false }
);
const TimeMachine = dynamic(
  () => import("../../../components/events/TimeMachine").then((mod) => mod.TimeMachine),
  { ssr: false }
);

function ExplorerContent() {
  const searchParams = useSearchParams();
  const { data: meta, isLoading: isMetaLoading, error: metaError, refetch: refetchMeta } = useMeta();

  // Initialize Explorer state from URL or fallback
  const [state, setState] = useState<ExplorerState>(() =>
    parseExplorerParams(searchParams, null)
  );

  const [showControls, setShowControls] = useState(true);
  const [showArgo, setShowArgo] = useState(false);

  // 3D Cutaway states
  const [isTouring, setIsTouring] = useState(false);
  const [isExploded, setIsExploded] = useState(false);
  const [isSmoothed, setIsSmoothed] = useState(false);
  const [showContours, setShowContours] = useState(true);

  // Meta features: guide drawer, time machine, camera moves, undo (PRD_FRONTEND §3.4a, c)
  const [guideOpen, setGuideOpen] = useState(false);
  const [guidePrompt, setGuidePrompt] = useState<GuidePrompt | null>(null);
  const [eventsOpen, setEventsOpen] = useState(false);
  const [activeEvent, setActiveEvent] = useState<{ event: EventItem; phase: EventPhase } | null>(null);
  const [focus, setFocus] = useState<CameraFocus | null>(null);
  const focusIdRef = useRef(0);
  const undoSnapshots = useRef(new Map<number, ExplorerState>());
  const undoIdRef = useRef(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  // Postcard request is a snapshot taken on click, so moving the map never re-prints
  const [postcard, setPostcard] = useState<{
    date: string;
    depth: number;
    eventId: string | null;
    eventTitle: string | null;
    nonce: number;
  } | null>(null);

  // Lasso (PRD_FRONTEND §3.4e)
  const [lassoActive, setLassoActive] = useState(false);
  const [lassoPending, setLassoPending] = useState<{ lat: number; lon: number } | null>(null);
  const [lassoResult, setLassoResult] = useState<SegmentResponse | null>(null);
  const [lassoError, setLassoError] = useState<string | null>(null);
  const lassoAbortRef = useRef<AbortController | null>(null);

  // Sync state once metadata loads
  useEffect(() => {
    if (meta) {
      setState((prev) => parseExplorerParams(searchParams, meta));
    }
  }, [meta, searchParams]);

  // Focus explorer heading on mount (PLAN_LANDING §6 item 5)
  useEffect(() => {
    const heading = document.getElementById("explorer-heading");
    if (heading) {
      heading.focus();
    }
  }, []);

  // Sync URL query string when state changes
  useEffect(() => {
    const params = buildExplorerSearchParams(state);
    const newUrl = `/explore?${params.toString()}`;
    window.history.replaceState(null, "", newUrl);
  }, [state]);

  // Query 2D map slice for active date, depth, field
  const {
    data: mapData,
    isLoading: isMapLoading,
    error: mapError,
    refetch: refetchMap,
  } = useMap(
    {
      date: state.date,
      depth_m: state.depth,
      field: state.field,
    },
    { enabled: !!state.date }
  );

  // Query ARGO profile markers when toggled
  const { data: argoData } = useArgoProfiles(
    { date: state.date },
    { enabled: showArgo }
  );

  // Query 4 boundary sections when in 3D Cutaway mode
  const is3D = state.view === "3d";
  const rgn = meta?.region;

  const { data: southSection, isError: isSouthError } = useSection(
    {
      date: state.date,
      field: state.field,
      lat: rgn?.lat_min,
    },
    { enabled: is3D && !!rgn && !!state.date }
  );

  const { data: northSection, isError: isNorthError } = useSection(
    {
      date: state.date,
      field: state.field,
      lat: rgn?.lat_max,
    },
    { enabled: is3D && !!rgn && !!state.date }
  );

  const { data: westSection, isError: isWestError } = useSection(
    {
      date: state.date,
      field: state.field,
      lon: rgn?.lon_min,
    },
    { enabled: is3D && !!rgn && !!state.date }
  );

  const { data: eastSection, isError: isEastError } = useSection(
    {
      date: state.date,
      field: state.field,
      lon: rgn?.lon_max,
    },
    { enabled: is3D && !!rgn && !!state.date }
  );

  const sections = useMemo(() => {
    return {
      south: southSection,
      north: northSection,
      west: westSection,
      east: eastSection,
    };
  }, [southSection, northSection, westSection, eastSection]);

  const wallsUnavailable =
    is3D && (isSouthError || isNorthError || isWestError || isEastError);

  // Determine top face color scale vmin and vmax
  const colorRange = useMemo(() => {
    if (state.field === "error") {
      return meta?.error_range ?? { vmin: -2.0, vmax: 2.0 };
    }
    const cr = meta?.color_ranges?.find((c) => Math.abs(c.depth_m - state.depth) < 0.1);
    return cr ?? { vmin: 15.0, vmax: 30.0 };
  }, [meta, state.depth, state.field]);

  // Column-wide wall color range
  const wallRange = useMemo(() => {
    if (state.view !== "3d" || !meta?.color_ranges) return undefined;
    return getWallColorRange(meta.color_ranges);
  }, [state.view, meta?.color_ranges]);

  // Split and live status derived directly from API response
  const isLive = mapData?.split === "live";

  // Automatically fall back to 'pred' field on live dates where GLORYS is unavailable
  useEffect(() => {
    if (mapData?.split === "live" && state.field !== "pred") {
      setState((prev) => ({ ...prev, field: "pred" }));
    }
  }, [mapData?.split, state.field]);

  const queryClient = useQueryClient();

  const handlePrefetchAhead = useCallback(
    (currentDate: string) => {
      if (!meta) return;
      const currentDays = Math.floor(new Date(currentDate + "T00:00:00Z").getTime() / 86400000);
      const endDays = Math.floor(new Date(meta.available_dates.end + "T00:00:00Z").getTime() / 86400000);
      for (let offset = 1; offset <= 3; offset++) {
        const nextDays = currentDays + offset;
        if (nextDays <= endDays) {
          const nextDate = new Date(nextDays * 86400000).toISOString().slice(0, 10);
          queryClient.prefetchQuery({
            queryKey: ["map", nextDate, state.depth, undefined, state.field],
            queryFn: () =>
              apiClient.getMap({
                date: nextDate,
                depth_m: state.depth,
                field: state.field,
              }),
            staleTime: Infinity,
          });
        }
      }
    },
    [meta, state.depth, state.field, queryClient]
  );

  const handleDateChange = useCallback((newDate: string) => {
    setState((prev) => ({ ...prev, date: newDate }));
  }, []);

  const handleDepthChange = useCallback((newDepth: number) => {
    setState((prev) => ({ ...prev, depth: newDepth }));
  }, []);

  const handleFieldChange = useCallback((newField: FieldType) => {
    setState((prev) => ({ ...prev, field: newField }));
  }, []);

  const lassoActiveRef = useRef(lassoActive);
  lassoActiveRef.current = lassoActive;

  const runLasso = useCallback(async (lat: number, lon: number) => {
    lassoAbortRef.current?.abort();
    const controller = new AbortController();
    lassoAbortRef.current = controller;
    const timer = setTimeout(() => controller.abort(), LASSO_TIMEOUT_MS);
    const s = stateRef.current;
    setLassoPending({ lat, lon });
    setLassoError(null);
    setLassoResult(null);
    try {
      const res = await apiClient.postSegment(
        { date: s.date, depth_m: s.depth, lat, lon, field: s.field === "glorys" ? "glorys" : "pred" },
        controller.signal
      );
      setLassoResult(res);
    } catch (err) {
      const { segmentErrorMessage } = await import("../../../components/lasso/LassoPanel");
      setLassoError(segmentErrorMessage(err, controller.signal.aborted));
    } finally {
      clearTimeout(timer);
      if (lassoAbortRef.current === controller) {
        lassoAbortRef.current = null;
        setLassoPending(null);
      }
    }
  }, []);

  const closeLasso = useCallback(() => {
    lassoAbortRef.current?.abort();
    setLassoActive(false);
    setLassoPending(null);
    setLassoResult(null);
    setLassoError(null);
  }, []);

  const handleSelectPoint = useCallback(
    (lat: number, lon: number) => {
      if (lassoActiveRef.current) {
        void runLasso(lat, lon);
        return;
      }
      setState((prev) => ({ ...prev, lat, lon }));
    },
    [runLasso]
  );

  const handleViewChange = useCallback((newView: "map" | "3d") => {
    setState((prev) => ({ ...prev, view: newView }));
    if (newView === "map") {
      setIsTouring(false);
    }
  }, []);

  const handleGridStyleChange = useCallback((newStyle: GridStyle) => {
    setState((prev) => ({ ...prev, gridStyle: newStyle }));
  }, []);

  const chatEnabled = !!meta?.features?.chat;
  const {
    data: eventsData,
    isLoading: isEventsLoading,
    isError: isEventsError,
  } = useEvents({ enabled: !!meta });
  const events = useMemo(() => eventsData?.events ?? [], [eventsData]);

  const moveCamera = useCallback((camera: CameraTarget | null) => {
    if (!camera) return;
    focusIdRef.current += 1;
    setFocus({ ...camera, id: focusIdRef.current });
  }, []);

  const openEventPhase = useCallback(
    (event: EventItem, phase: EventPhase, opts?: { pickDepth?: boolean }) => {
      if (!meta) return;
      const applied = applyEventPhase(stateRef.current, event, phase, meta, opts);
      setState(applied.state);
      moveCamera(applied.camera);
      setActiveEvent({ event, phase });
      setEventsOpen(true);
    },
    [meta, moveCamera]
  );

  const handleGuideNavigate = useCallback(
    (response: ChatResponse): NavigateResult => {
      if (!meta) return { summary: null, undoToken: null };
      const before = stateRef.current;
      const applied = applyUiActions(before, response.ui_actions, meta);
      let summary = describeChanges(applied.changes);
      let next = applied.state;
      let camera = applied.camera;

      if (applied.openEvent) {
        const ev = events.find((e) => e.id === applied.openEvent!.event_id);
        if (ev) {
          const phase = applied.openEvent.phase ?? "before";
          const evApplied = applyEventPhase(next, ev, phase, meta);
          next = evApplied.state;
          camera = evApplied.camera;
          setActiveEvent({ event: ev, phase });
          setEventsOpen(true);
          summary = `Opened ${ev.title} · ${phase}`;
        }
      }

      const moved = JSON.stringify(next) !== JSON.stringify(before) || camera !== null;
      if (!moved) return { summary: null, undoToken: null };
      setState(next);
      moveCamera(camera);
      undoIdRef.current += 1;
      undoSnapshots.current.set(undoIdRef.current, before);
      return { summary: summary ?? "Moved the map", undoToken: undoIdRef.current };
    },
    [meta, events, moveCamera]
  );

  const handleGuideUndo = useCallback(
    (token: number) => {
      const prev = undoSnapshots.current.get(token);
      if (!prev) return;
      undoSnapshots.current.delete(token);
      setState(prev);
      if (prev.lat !== null && prev.lon !== null) moveCamera({ kind: "center", lat: prev.lat, lon: prev.lon });
    },
    [moveCamera]
  );

  const handleExplainEvent = useCallback((event: EventItem) => {
    const s = stateRef.current;
    const b = event.bbox;
    setGuidePrompt({
      nonce: Date.now(),
      text:
        `Compare the water before and after ${event.title}: ${event.before} versus ${event.after}, ` +
        `at ${s.depth} m, in the box ${b.lat_min}–${b.lat_max}°N, ${b.lon_min}–${b.lon_max}°E. ` +
        `What changed, and can you step me through the phases?`,
    });
    setGuideOpen(true);
  }, []);

  const handleAskAboutArea = useCallback(async (seg: SegmentResponse) => {
    const { askAboutAreaPrompt } = await import("../../../components/lasso/LassoPanel");
    setGuidePrompt({ nonce: Date.now(), text: askAboutAreaPrompt(seg) });
    setGuideOpen(true);
  }, []);

  const openPostcard = useCallback((event?: EventItem | null) => {
    const s = stateRef.current;
    setPostcard({
      date: s.date,
      depth: s.depth,
      eventId: event?.id ?? null,
      eventTitle: event?.title ?? null,
      nonce: Date.now(),
    });
  }, []);

  if (isMetaLoading) {
    return (
      <>
        <h1 id="explorer-heading" tabIndex={-1} className="sr-only">
          OceanScope AI Subsurface Temperature Explorer
        </h1>
        <Loading fullScreen label="Loading OceanScope AI metadata..." />
      </>
    );
  }

  if (metaError || !meta) {
    return (
      <div className="flex-1 flex items-center justify-center p-6 bg-slate-50 dark:bg-slate-950">
        <ErrorState
          title="Could not connect to ocean model server"
          message={metaError instanceof Error ? metaError.message : "Metadata unavailable"}
          onRetry={() => refetchMeta()}
        />
      </div>
    );
  }

  const activeGridStyle: GridStyle = state.gridStyle ?? "dots";
  const hasSelection = state.lat !== null && state.lon !== null;

  return (
    <div className="relative flex-1 w-full h-[calc(100vh-3.5rem)] overflow-hidden bg-slate-100 dark:bg-[#070d12] transition-colors">
      {/* Main Accessible Heading for Focus Management (PLAN_LANDING §6 item 5) */}
      <h1 id="explorer-heading" tabIndex={-1} className="sr-only">
        OceanScope AI Subsurface Temperature Explorer
      </h1>

      {/* Full-bleed Map Viewport */}
      <div className="relative w-full h-full min-h-[400px]">
        <ErrorBoundary name="Interactive Map">
          <MapView
            mapData={mapData}
            region={meta.region}
            gridResolutionDeg={meta.grid_resolution_deg}
            vmin={colorRange.vmin}
            vmax={colorRange.vmax}
            selectedLat={state.lat}
            selectedLon={state.lon}
            onSelectPoint={handleSelectPoint}
            showArgo={showArgo}
            argoProfiles={argoData?.profiles ?? []}
            gridStyle={activeGridStyle}
            view={state.view}
            depth_m={state.depth}
            availableDepths={meta.depths_m}
            colorRanges={meta.color_ranges}
            sections={sections}
            isTouring={isTouring}
            onStopTour={() => setIsTouring(false)}
            isExploded={isExploded}
            isSmoothed={isSmoothed}
            showContours={showContours}
            focus={focus}
            outline={lassoActive ? lassoResult?.outline ?? null : null}
            pendingPoint={lassoPending}
          />
        </ErrorBoundary>

        {/* Top-center status pills (loading, error, side-wall fallback) */}
        <div className="pointer-events-none absolute top-4 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2">
          {isMapLoading && (
            <div className="bg-white/90 dark:bg-[#0b1116]/88 backdrop-blur-xl border border-slate-300 dark:border-white/12 rounded-full px-4 py-1.5 shadow-2xl flex items-center gap-2 text-xs text-cyan-700 dark:text-cyan-300 font-medium animate-pulse motion-reduce:animate-none">
              <span className="h-2 w-2 rounded-full bg-cyan-500 dark:bg-cyan-400" />
              <span>Fetching temperature slice...</span>
            </div>
          )}

          {mapError && (
            <div className="pointer-events-auto bg-white/95 dark:bg-[#0b1116]/92 backdrop-blur-xl border border-red-500/50 rounded-full px-4 py-1.5 shadow-2xl flex items-center gap-3 text-xs text-red-600 dark:text-red-200">
              <span>Failed to load slice.</span>
              <button
                type="button"
                onClick={() => refetchMap()}
                className="text-cyan-600 dark:text-cyan-400 underline font-semibold"
              >
                Retry
              </button>
            </div>
          )}

          {wallsUnavailable && (
            <div className="bg-amber-500/95 text-slate-950 font-medium text-xs px-3.5 py-1.5 rounded-full shadow-lg flex items-center gap-1.5 backdrop-blur-md">
              <AlertTriangle aria-hidden="true" className="h-3.5 w-3.5" />
              <span>Side walls unavailable for this date</span>
            </div>
          )}
        </div>

        {/* Floating Left Glass HUD Panel (Controls + Colorbar + Coverage Legend) */}
        <aside
          aria-label="Explorer controls"
          className={`absolute inset-x-3 bottom-16 md:inset-x-auto md:bottom-auto md:top-4 md:left-4 md:w-[348px] z-20 rounded-2xl border border-slate-200/90 dark:border-white/12 bg-white/90 dark:bg-[#0b1116]/88 backdrop-blur-xl shadow-2xl flex flex-col overflow-hidden transition-all duration-200 ${
            showControls
              ? "max-h-[54vh] md:max-h-[calc(100vh-5.5rem)]"
              : "max-h-11"
          }`}
        >
          {/* Compact HUD card header */}
          <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-slate-200/70 dark:border-white/10 shrink-0">
            <div className="flex items-center gap-2 text-xs font-semibold tracking-wide text-slate-800 dark:text-slate-100">
              <SlidersHorizontal aria-hidden="true" className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
              <span>Ocean Controls</span>
            </div>
            <button
              type="button"
              aria-expanded={showControls}
              onClick={() => setShowControls(!showControls)}
              className="inline-flex items-center gap-1 text-[11px] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white font-medium px-2 py-0.5 rounded-full hover:bg-slate-200/60 dark:hover:bg-white/10 transition-colors"
            >
              <span>{showControls ? "Hide" : "Show"}</span>
              {showControls ? (
                <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" />
              ) : (
                <ChevronUp aria-hidden="true" className="h-3.5 w-3.5" />
              )}
            </button>
          </div>

          {/* Scrollable controls & scale body */}
          <div className={`p-3.5 space-y-4 overflow-y-auto ${!showControls ? "hidden" : ""}`}>
            {/* Selected Point Status */}
            {state.lat !== null && state.lon !== null ? (
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-2.5 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2 text-cyan-950 dark:text-cyan-100">
                  <MapPin aria-hidden="true" className="h-4 w-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
                  <div>
                    <p className="font-semibold font-mono">{formatCoord(state.lat, state.lon)}</p>
                    <p className="text-[11px] text-cyan-700 dark:text-cyan-400/80">Selected cell centre</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setState((prev) => ({ ...prev, lat: null, lon: null }))}
                  className="text-[11px] text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white bg-white/80 dark:bg-white/[0.06] border border-slate-200 dark:border-white/12 px-2.5 py-1 rounded-full transition-colors"
                >
                  Clear
                </button>
              </div>
            ) : (
              <div className="bg-slate-100/80 dark:bg-white/[0.03] border border-slate-200/80 dark:border-white/10 rounded-xl p-2.5 text-xs text-slate-600 dark:text-slate-400 flex items-center gap-2">
                <MapPin aria-hidden="true" className="h-4 w-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
                <span>Click any ocean dot to inspect vertical profile &amp; time series.</span>
              </div>
            )}

            {/* Date Picker */}
            <DatePicker
              date={state.date}
              startDate={meta.available_dates.start}
              endDate={meta.available_dates.end}
              currentSplit={mapData?.split}
              latestLiveDate={meta.latest_live_date}
              splits={meta.splits}
              onChange={handleDateChange}
              prefetchAhead={handlePrefetchAhead}
            />

            {/* Depth Selector */}
            <DepthSelector
              depths={meta.depths_m}
              selectedDepth={state.depth}
              onChange={handleDepthChange}
            />

            {/* Field Toggle */}
            <FieldToggle
              value={state.field}
              onChange={handleFieldChange}
              isLive={isLive}
            />

            {/* Live caveat when on live dates */}
            {isLive && <Banner type="live" />}

            {/* Scale & Coverage Legend */}
            <div className="pt-2 border-t border-slate-200/80 dark:border-white/10 space-y-2">
              <Colorbar
                field={state.field}
                vmin={colorRange.vmin}
                vmax={colorRange.vmax}
                wallRange={wallRange}
              />
              <CoverageLegend gridStyle={activeGridStyle} />
            </div>
          </div>
        </aside>

        {/* Contextual Meta Feature Cards (Lasso, Postcard, Events Time Machine) */}
        {(eventsOpen || (meta.features.segment && lassoActive) || (meta.features.postcards && postcard)) && (
          <div className="absolute top-4 left-3 right-3 sm:right-auto md:left-[376px] z-30 flex flex-col items-start gap-2 pointer-events-auto max-w-[calc(100%-1.5rem)]">
            {meta.features.segment && lassoActive && (
              <LassoPanel
                pending={!!lassoPending}
                error={lassoError}
                result={lassoResult}
                onAsk={chatEnabled ? handleAskAboutArea : undefined}
                onClose={closeLasso}
              />
            )}
            {meta.features.postcards && postcard && (
              <PostcardPanel
                key={postcard.nonce}
                date={postcard.date}
                depth={postcard.depth}
                eventId={postcard.eventId}
                eventTitle={postcard.eventTitle}
                caveat={meta.caveats.postcard}
                onClose={() => setPostcard(null)}
              />
            )}
            {eventsOpen && (
              <TimeMachine
                meta={meta}
                events={events}
                isLoading={isEventsLoading}
                isError={isEventsError}
                active={activeEvent}
                currentDate={state.date}
                onOpen={openEventPhase}
                onSetDate={handleDateChange}
                onBack={() => setActiveEvent(null)}
                onClose={() => {
                  setEventsOpen(false);
                  setActiveEvent(null);
                }}
                onExplain={chatEnabled ? handleExplainEvent : undefined}
              />
            )}
          </div>
        )}

        {/* Bottom-Center Floating Glass Pill Toolbar (functional tools only) */}
        <div className="pointer-events-none absolute bottom-3 inset-x-3 md:inset-x-auto md:bottom-5 md:left-[calc(50%+96px)] lg:left-1/2 md:-translate-x-1/2 z-20 flex flex-wrap items-end justify-center gap-2">
          <ViewSwitch
            view={state.view}
            onChangeView={handleViewChange}
            gridStyle={activeGridStyle}
            onChangeGridStyle={handleGridStyleChange}
            isTouring={isTouring}
            onToggleTour={() => setIsTouring(!isTouring)}
            isExploded={isExploded}
            onToggleExploded={() => setIsExploded(!isExploded)}
            isSmoothed={isSmoothed}
            onToggleSmoothed={() => setIsSmoothed(!isSmoothed)}
            showContours={showContours}
            onToggleContours={() => setShowContours(!showContours)}
          />

          <div className="pointer-events-auto flex flex-wrap items-center gap-1 p-1 rounded-full bg-white/90 dark:bg-[#0b1116]/90 backdrop-blur-xl border border-slate-300/80 dark:border-white/12 shadow-2xl">
            <button
              type="button"
              aria-pressed={showArgo}
              onClick={() => setShowArgo(!showArgo)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                showArgo
                  ? "bg-amber-500 text-slate-950 font-semibold shadow-sm"
                  : "text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
            >
              {showArgo ? (
                <Eye aria-hidden="true" className="h-3.5 w-3.5" />
              ) : (
                <EyeOff aria-hidden="true" className="h-3.5 w-3.5 text-slate-400" />
              )}
              <span>ARGO Floats {argoData?.profiles ? `(${argoData.profiles.length})` : ""}</span>
            </button>

            {events.length > 0 && !eventsOpen && (
              <button
                type="button"
                onClick={() => setEventsOpen(true)}
                className="px-3 py-1.5 rounded-full text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10 inline-flex items-center gap-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400"
              >
                <CalendarRange aria-hidden="true" className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
                <span>Events</span>
              </button>
            )}

            {meta.features.postcards && !postcard && (
              <button
                type="button"
                onClick={() => openPostcard(activeEvent?.event)}
                className="px-3 py-1.5 rounded-full text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10 inline-flex items-center gap-1.5 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400"
              >
                <ImageIcon aria-hidden="true" className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
                <span>Make a postcard</span>
              </button>
            )}

            {meta.features.segment && (
              <button
                type="button"
                aria-pressed={lassoActive}
                onClick={() => (lassoActive ? closeLasso() : setLassoActive(true))}
                className={`px-3 py-1.5 rounded-full text-xs font-medium inline-flex items-center gap-1.5 transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                  lassoActive
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950 font-semibold shadow-sm"
                    : "text-slate-700 dark:text-slate-200 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10"
                }`}
              >
                <Lasso aria-hidden="true" className="h-3.5 w-3.5" />
                <span>Lasso</span>
              </button>
            )}

            {chatEnabled && !guideOpen && (
              <button
                type="button"
                onClick={() => setGuideOpen(true)}
                className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-semibold px-3.5 py-1.5 shadow-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              >
                <MessageCircle aria-hidden="true" className="h-3.5 w-3.5" />
                <span>Ask the ocean</span>
              </button>
            )}
          </div>
        </div>

        {/* 3D Depth Rail overlay on right edge */}
        {state.view === "3d" && meta.depths_m && (
          <div
            className={`absolute top-4 z-20 pointer-events-auto animate-in fade-in duration-300 ${
              hasSelection ? "right-4 lg:right-[452px]" : "right-4"
            }`}
          >
            <DepthRail
              availableDepths={meta.depths_m}
              selectedDepth={state.depth}
              onChangeDepth={handleDepthChange}
            />
          </div>
        )}

        {chatEnabled && guideOpen && (
          <GuideDrawer
            meta={meta}
            state={state}
            eventTitles={events.map((e) => e.title)}
            onNavigate={handleGuideNavigate}
            onUndo={handleGuideUndo}
            onClose={() => setGuideOpen(false)}
            prompt={guidePrompt}
          />
        )}

        {/* Details Panel overlay when lat/lon selected */}
        {state.lat !== null && state.lon !== null && (
          <div className="absolute top-4 right-4 z-30 w-[calc(100%-2rem)] sm:w-96 md:w-[420px] pointer-events-auto">
            <ErrorBoundary
              name="Point Details Panel"
              onReset={() => setState((prev) => ({ ...prev, lat: null, lon: null }))}
            >
              <DetailsPanel
                lat={state.lat}
                lon={state.lon}
                date={state.date}
                depth={state.depth}
                minDate={meta.available_dates.start}
                maxDate={meta.available_dates.end}
                onClose={() => setState((prev) => ({ ...prev, lat: null, lon: null }))}
                onSelectDepth={handleDepthChange}
              />
            </ErrorBoundary>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ExplorerPage() {
  return (
    <Suspense fallback={<Loading fullScreen label="Loading OceanScope Explorer..." />}>
      <ExplorerContent />
    </Suspense>
  );
}
