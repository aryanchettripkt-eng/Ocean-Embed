"use client";

import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import {
  MapPin,
  AlertTriangle,
  CalendarRange,
  MessageCircle,
  Image as ImageIcon,
  Lasso,
  ChevronDown,
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
import { formatCoord, formatDate, formatDepth } from "../../../lib/format";
import { getWallColorRange } from "../../../components/map3d/depthScale";
import { ToolbarDivider, ViewSwitch } from "../../../components/map3d/ViewSwitch";
import { DepthRail } from "../../../components/map3d/DepthRail";
import { Collapse, FlipIcon, Presence, Pulse } from "../../../components/common/Motion";
import { arrivalMapReady, finishArrival, isArrivalPending } from "../../../lib/arrival";
import { EASE_OUT, gsap } from "../../../lib/motion";

// MapView is statically included so MapLibre is counted directly in the bundle budget (PRD_FRONTEND §4)
import { MapView } from "../../../components/map/MapView";

// DetailsPanel is displayed only when a point is clicked; charts (Recharts) load on-demand
const DetailsPanel = dynamic(
  () => import("../../../components/panels/DetailsPanel").then((mod) => mod.DetailsPanel),
  {
    ssr: false,
    loading: () => (
      <div className="hud-glass rounded-2xl p-5 text-sm text-neutral-600 dark:text-neutral-300">
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

  const [hasMounted, setHasMounted] = useState(false);
  // Arriving from the landing CTA (src/lib/arrival.ts): the map plays its zoom-out/zoom-in intro
  // in 2D, then switches to the requested view; the HUD slides in as the camera turns inward.
  const [introActive, setIntroActive] = useState(() => isArrivalPending());
  useEffect(() => setHasMounted(true), []);

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

  // Fetch only once the depth is snapped to a model level (the pre-metadata default may not be one)
  const depthIsModelLevel = !!meta?.depths_m?.some((d) => Math.abs(d - state.depth) < 0.01);

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
    { enabled: !!state.date && depthIsModelLevel }
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

  // Header from the top, controls from the left, toolbar from the bottom, map controls from the
  // right. The from-states hide them inline before the page-level CSS hide is lifted.
  const playHudEntrance = useCallback(() => {
    const pick = (selector: string) => document.querySelector<HTMLElement>(selector);
    const items: [HTMLElement | null, gsap.TweenVars, number][] = [
      [pick('[data-hud="header"]'), { yPercent: -100 }, 0],
      [pick('[data-hud="controls"]'), { x: -64 }, 0.1],
      [pick('[data-hud="toolbar"]'), { y: 64 }, 0.18],
      [pick(".maplibregl-ctrl-bottom-right"), { x: 48 }, 0.26],
    ];
    const tl = gsap.timeline({ defaults: { duration: 0.75, ease: EASE_OUT } });
    for (const [el, from, at] of items) {
      if (!el) continue;
      tl.fromTo(
        el,
        { ...from, autoAlpha: 0 },
        { x: 0, y: 0, yPercent: 0, autoAlpha: 1, clearProps: "transform,opacity,visibility" },
        at
      );
    }
    finishArrival();
  }, []);

  const handleIntroDone = useCallback(() => setIntroActive(false), []);

  // If the slice can't load, don't keep the HUD hidden waiting for the intro
  useEffect(() => {
    if (!introActive || !mapError) return;
    arrivalMapReady();
    playHudEntrance();
    setIntroActive(false);
  }, [introActive, mapError, playHudEntrance]);

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

  // The header's useMeta can resolve before this boundary hydrates; render the server's loading
  // markup until mounted so the first client render matches it.
  if (!hasMounted || isMetaLoading) {
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
  const fieldLabel =
    state.field === "error" ? "Model − GLORYS" : state.field === "glorys" ? "GLORYS reanalysis" : "Model prediction";
  const clearSelection = () => setState((prev) => ({ ...prev, lat: null, lon: null }));
  const guideShown = chatEnabled && guideOpen;
  // Right-hand stack on wide screens: guide card (380 px) at the edge, details panel (400 px) beside
  // it from xl up; below xl the guide overlays the details panel.
  const detailsRight = guideShown ? "md:right-4 xl:right-[412px]" : "md:right-4";
  const railRight = hasSelection
    ? guideShown
      ? "right-4 lg:right-[432px] xl:right-[828px]"
      : "right-4 lg:right-[432px]"
    : guideShown
    ? "right-4 lg:right-[412px]"
    : "right-4";

  return (
    <div className="relative min-h-[480px] w-full flex-1 overflow-hidden bg-neutral-100 dark:bg-[#0a0c0d]">
      {/* Main Accessible Heading for Focus Management (PLAN_LANDING §6 item 5) */}
      <h1 id="explorer-heading" tabIndex={-1} className="sr-only">
        OceanScope AI Subsurface Temperature Explorer
      </h1>

      {/* Full-bleed map */}
      <div className="absolute inset-0">
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
            view={introActive ? "map" : state.view}
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
            intro={introActive}
            onIntroReady={arrivalMapReady}
            onIntroFlyIn={playHudEntrance}
            onIntroDone={handleIntroDone}
          />
        </ErrorBoundary>
      </div>

      {/* Status pills (loading, error, side-wall fallback) */}
      <div className="pointer-events-none absolute left-1/2 top-4 z-20 flex -translate-x-1/2 flex-col items-center gap-2 lg:left-[calc(50%+178px)]">
        <Presence show={isMapLoading} from="top">
          <div className="hud-glass flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-medium">
            <Pulse className="h-1.5 w-1.5 rounded-full bg-cyan-500 dark:bg-cyan-300" />
            <span>Fetching temperature slice...</span>
          </div>
        </Presence>

        <Presence show={!!mapError} from="top">
          <div className="hud-glass pointer-events-auto flex items-center gap-3 rounded-full px-3.5 py-1.5 text-xs text-red-700 dark:text-red-300">
            <span>Failed to load slice.</span>
            <button
              type="button"
              onClick={() => refetchMap()}
              className="font-semibold text-neutral-900 underline underline-offset-2 dark:text-white"
            >
              Retry
            </button>
          </div>
        </Presence>

        <Presence show={wallsUnavailable} from="top">
          <div className="flex items-center gap-1.5 rounded-full bg-amber-400 px-3.5 py-1.5 text-xs font-medium text-neutral-950 shadow-lg">
            <AlertTriangle aria-hidden="true" className="h-3.5 w-3.5" />
            <span>Side walls unavailable for this date</span>
          </div>
        </Presence>
      </div>

      {/* Controls card */}
      <aside
        aria-label="Explorer controls"
        data-hud="controls"
        className={`hud-glass absolute inset-x-3 bottom-[60px] z-20 flex flex-col overflow-hidden rounded-[20px] md:inset-x-auto md:bottom-auto md:left-4 md:top-4 md:w-[340px] ${
          showControls ? "max-h-[55%] md:max-h-[calc(100%-6.5rem)]" : ""
        }`}
      >
        {/* Title: what the map is showing right now */}
        <div className="flex shrink-0 items-start justify-between gap-3 px-4 pb-3 pt-3.5">
          <div className="min-w-0">
            <p className="hud-label">Ocean temperature</p>
            <p className="mt-1 text-[26px] font-light leading-none tracking-tight tabular-nums text-neutral-950 dark:text-white">
              {formatDepth(state.depth)}
              <span className="ml-1.5 text-neutral-400 dark:text-neutral-500">deep</span>
            </p>
            <p className="mt-1.5 truncate text-xs text-neutral-500 dark:text-neutral-400">
              {formatDate(state.date)} · {fieldLabel}
            </p>
          </div>
          <button
            type="button"
            aria-expanded={showControls}
            aria-controls="explorer-controls-body"
            aria-label={showControls ? "Hide controls" : "Show controls"}
            title={showControls ? "Hide controls" : "Show controls"}
            onClick={() => setShowControls(!showControls)}
            className="hud-icon-btn mt-0.5"
          >
            <FlipIcon open={showControls}>
              <ChevronDown aria-hidden="true" className="h-4 w-4" />
            </FlipIcon>
          </button>
        </div>

        <Collapse open={showControls} id="explorer-controls-body" className="min-h-0 overflow-y-auto overscroll-contain">
          {/* Selected cell, or how to pick one */}
          <div className="px-4 pb-3">
            {hasSelection ? (
              <div className="flex items-center justify-between gap-2 rounded-xl bg-cyan-500/10 px-3 py-2 ring-1 ring-inset ring-cyan-500/25">
                <div className="flex min-w-0 items-center gap-2">
                  <MapPin aria-hidden="true" className="h-4 w-4 shrink-0 text-cyan-700 dark:text-cyan-300" />
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold tabular-nums text-neutral-950 dark:text-white">
                      {formatCoord(state.lat!, state.lon!)}
                    </p>
                    <p className="text-[11px] text-neutral-600 dark:text-neutral-400">Selected cell centre</p>
                  </div>
                </div>
                <button type="button" onClick={clearSelection} className="hud-btn h-7 px-2.5">
                  Clear
                </button>
              </div>
            ) : (
              <p className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
                <MapPin aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
                <span>Click any ocean dot for its profile and time series.</span>
              </p>
            )}
          </div>

          <div className="hud-divider space-y-4 border-t px-4 py-3.5">
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

            <DepthSelector depths={meta.depths_m} selectedDepth={state.depth} onChange={handleDepthChange} />

            <FieldToggle value={state.field} onChange={handleFieldChange} isLive={isLive} />

            {isLive && <Banner type="live" />}
          </div>

          <div className="hud-divider space-y-3.5 border-t px-4 py-3.5">
            <Colorbar field={state.field} vmin={colorRange.vmin} vmax={colorRange.vmax} wallRange={wallRange} />
            <CoverageLegend gridStyle={activeGridStyle} />
          </div>
        </Collapse>
      </aside>

      {/* Meta feature cards (Lasso, Postcard, Events time machine) */}
      {/* Always mounted so each card can animate out; the column itself takes no clicks */}
      <div className="pointer-events-none absolute left-3 right-3 top-4 z-30 flex max-w-[calc(100%-1.5rem)] flex-col items-start gap-2 sm:right-auto md:left-[372px]">
          <Presence show={!!meta.features.segment && lassoActive} from="left" className="pointer-events-auto w-full sm:w-auto">
            <LassoPanel
              pending={!!lassoPending}
              error={lassoError}
              result={lassoResult}
              onAsk={chatEnabled ? handleAskAboutArea : undefined}
              onClose={closeLasso}
            />
          </Presence>
          <Presence show={!!meta.features.postcards && !!postcard} from="left" className="pointer-events-auto w-full sm:w-auto">
            {postcard && (
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
          </Presence>
          <Presence show={eventsOpen} from="left" className="pointer-events-auto w-full sm:w-auto">
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
          </Presence>
      </div>

      {/* Bottom toolbar: view, grid style, layers and tools in one pill */}
      <div data-hud="toolbar" className="pointer-events-none absolute inset-x-3 bottom-3 z-20 flex justify-center md:bottom-5">
        <ViewSwitch
          view={introActive ? "map" : state.view}
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
        >
          <ToolbarDivider />
          <button
            type="button"
            aria-pressed={showArgo}
            aria-label="ARGO Floats"
            title="Show ARGO float profiles for this date"
            onClick={() => setShowArgo(!showArgo)}
            className={`hud-btn ${showArgo ? "hud-btn-active" : ""}`}
          >
            <span
              aria-hidden="true"
              className={`h-2 w-2 rounded-full ${showArgo ? "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.9)]" : "bg-amber-500/60"}`}
            />
            <span className="hidden sm:inline">ARGO</span>
            {showArgo && argoData?.profiles && (
              <span className="tabular-nums opacity-60">{argoData.profiles.length}</span>
            )}
          </button>

          {events.length > 0 && !eventsOpen && (
            <button
              type="button"
              aria-label="Events"
              title="Ocean events time machine"
              onClick={() => setEventsOpen(true)}
              className="hud-btn"
            >
              <CalendarRange aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Events</span>
            </button>
          )}

          {meta.features.postcards && !postcard && (
            <button
              type="button"
              aria-label="Make a postcard"
              title="Make a postcard of this view"
              onClick={() => openPostcard(activeEvent?.event)}
              className="hud-btn"
            >
              <ImageIcon aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="hidden xl:inline">Postcard</span>
            </button>
          )}

          {meta.features.segment && (
            <button
              type="button"
              aria-pressed={lassoActive}
              aria-label="Lasso"
              title="Lasso a water mass: tap the map after turning this on"
              onClick={() => (lassoActive ? closeLasso() : setLassoActive(true))}
              className={`hud-btn ${lassoActive ? "hud-btn-active" : ""}`}
            >
              <Lasso aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="hidden xl:inline">Lasso</span>
            </button>
          )}

          {chatEnabled && !guideOpen && (
            <>
              <ToolbarDivider />
              <button
                type="button"
                onClick={() => setGuideOpen(true)}
                aria-label="Ask the ocean"
                className="hud-btn bg-cyan-400 font-semibold text-neutral-950 hover:bg-cyan-300 hover:text-neutral-950 dark:text-neutral-950 dark:hover:bg-cyan-300 dark:hover:text-neutral-950"
              >
                <MessageCircle aria-hidden="true" className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Ask the ocean</span>
              </button>
            </>
          )}
        </ViewSwitch>
      </div>

      {/* 3D depth rail on the right edge */}
      <Presence
        show={!introActive && state.view === "3d" && !!meta.depths_m}
        from="right"
        className={`pointer-events-auto absolute top-4 z-20 ${railRight}`}
      >
        <DepthRail availableDepths={meta.depths_m} selectedDepth={state.depth} onChangeDepth={handleDepthChange} />
      </Presence>

      <Presence
        show={guideShown}
        from="right"
        className="absolute inset-x-3 bottom-3 z-40 flex h-[75%] flex-col md:inset-x-auto md:bottom-[104px] md:right-4 md:top-4 md:h-auto md:w-[380px]"
      >
        <GuideDrawer
          meta={meta}
          state={state}
          eventTitles={events.map((e) => e.title)}
          onNavigate={handleGuideNavigate}
          onUndo={handleGuideUndo}
          onClose={() => setGuideOpen(false)}
          prompt={guidePrompt}
          placementClassName="min-h-0 w-full flex-1"
        />
      </Presence>

      {/* Details panel for the selected cell */}
      <Presence
        show={hasSelection}
        from="right"
        className={`pointer-events-none absolute bottom-[60px] right-3 top-3 z-30 flex w-[calc(100%-1.5rem)] flex-col sm:w-96 md:bottom-[104px] md:top-4 md:w-[400px] [&>*]:pointer-events-auto ${detailsRight}`}
      >
          <ErrorBoundary name="Point Details Panel" onReset={clearSelection}>
            <DetailsPanel
              className="pointer-events-auto max-h-full"
              lat={state.lat!}
              lon={state.lon!}
              date={state.date}
              depth={state.depth}
              minDate={meta.available_dates.start}
              maxDate={meta.available_dates.end}
              onClose={clearSelection}
              onSelectDepth={handleDepthChange}
            />
          </ErrorBoundary>
      </Presence>
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
