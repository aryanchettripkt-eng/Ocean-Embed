"use client";

import React, { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { MapPin, SlidersHorizontal, Eye, EyeOff, AlertTriangle } from "lucide-react";

import { useArgoProfiles, useMap, useMeta, useSection } from "../lib/api/hooks";
import { apiClient } from "../lib/api/client";
import { parseExplorerParams, type ExplorerState, buildExplorerSearchParams } from "../lib/url-state";
import { DatePicker } from "../components/controls/DatePicker";
import { DepthSelector } from "../components/controls/DepthSelector";
import { FieldToggle, type FieldType } from "../components/controls/FieldToggle";
import { Colorbar } from "../components/map/Colorbar";
import { CoverageLegend } from "../components/map/CoverageLegend";
import { Banner } from "../components/common/Banner";
import { Loading } from "../components/common/Loading";
import { ErrorState } from "../components/common/ErrorState";
import { ErrorBoundary } from "../components/common/ErrorBoundary";
import { formatCoord } from "../lib/format";
import { getWallColorRange } from "../components/map3d/depthScale";
import { ViewSwitch } from "../components/map3d/ViewSwitch";
import { DepthRail } from "../components/map3d/DepthRail";

// MapView is statically included so MapLibre is counted directly in the initial / budget (PRD_FRONTEND §4)
import { MapView } from "../components/map/MapView";

// DetailsPanel is displayed only when a point is clicked; charts (Recharts) load on-demand
const DetailsPanel = dynamic(
  () => import("../components/panels/DetailsPanel").then((mod) => mod.DetailsPanel),
  {
    ssr: false,
    loading: () => (
      <div className="bg-white/90 dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 rounded-xl p-6 text-sm text-slate-700 dark:text-slate-300">
        Loading details...
      </div>
    ),
  }
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

  // Sync state once metadata loads
  useEffect(() => {
    if (meta) {
      setState((prev) => parseExplorerParams(searchParams, meta));
    }
  }, [meta, searchParams]);

  // Sync URL query string when state changes
  useEffect(() => {
    const params = buildExplorerSearchParams(state);
    const newUrl = `/?${params.toString()}`;
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

  const handleSelectPoint = useCallback((lat: number, lon: number) => {
    setState((prev) => ({ ...prev, lat, lon }));
  }, []);

  const handleViewChange = useCallback((newView: "map" | "3d") => {
    setState((prev) => ({ ...prev, view: newView }));
    if (newView === "map") {
      setIsTouring(false);
    }
  }, []);

  if (isMetaLoading) {
    return <Loading fullScreen label="Loading OceanScope AI metadata..." />;
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

  return (
    <div className="relative flex-1 w-full h-[calc(100vh-3.5rem)] overflow-hidden flex flex-col md:flex-row bg-slate-100 dark:bg-slate-950 transition-colors">
      {/* Map Viewport */}
      <div className="relative flex-1 w-full h-[calc(100vh-3.5rem)] min-h-[400px]">
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
          />
        </ErrorBoundary>

        {/* View Switch: 2D Map | 3D Cutaway */}
        <div className="absolute top-4 left-4 z-20">
          <ViewSwitch
            view={state.view}
            onChangeView={handleViewChange}
            isTouring={isTouring}
            onToggleTour={() => setIsTouring(!isTouring)}
            isExploded={isExploded}
            onToggleExploded={() => setIsExploded(!isExploded)}
            isSmoothed={isSmoothed}
            onToggleSmoothed={() => setIsSmoothed(!isSmoothed)}
            showContours={showContours}
            onToggleContours={() => setShowContours(!showContours)}
          />
        </div>

        {/* 3D Depth Rail overlay on right edge */}
        {state.view === "3d" && meta.depths_m && (
          <div className="absolute top-20 right-4 z-20 pointer-events-auto animate-in fade-in duration-300">
            <DepthRail
              availableDepths={meta.depths_m}
              selectedDepth={state.depth}
              onChangeDepth={handleDepthChange}
            />
          </div>
        )}

        {/* Fallback badge when side walls are unavailable (PRD_FRONTEND §3.1b) */}
        {wallsUnavailable && (
          <div className="absolute top-16 left-4 z-20 bg-amber-500/90 text-slate-950 font-medium text-xs px-3 py-1.5 rounded-lg shadow-lg flex items-center gap-1.5 backdrop-blur-md">
            <AlertTriangle className="h-3.5 w-3.5" />
            <span>Side walls unavailable for this date</span>
          </div>
        )}

        {/* Map Loading overlay */}
        {isMapLoading && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 pointer-events-none bg-white/80 dark:bg-slate-950/80 backdrop-blur-md border border-slate-300 dark:border-slate-700/80 rounded-full px-4 py-1.5 shadow-2xl flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-300 font-medium animate-pulse">
            <span className="h-2 w-2 rounded-full bg-cyan-500 dark:bg-cyan-400" />
            <span>Fetching temperature slice...</span>
          </div>
        )}

        {/* Error overlay on map */}
        {mapError && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 bg-white/95 dark:bg-slate-950/90 border border-red-500/50 rounded-lg px-4 py-2 shadow-2xl flex items-center gap-3 text-xs text-red-600 dark:text-red-200">
            <span>Failed to load slice.</span>
            <button
              onClick={() => refetchMap()}
              className="text-cyan-600 dark:text-cyan-400 underline font-semibold"
            >
              Retry
            </button>
          </div>
        )}

        {/* Bottom overlays: Colorbar, Legend, and ARGO toggle */}
        <div className="absolute bottom-6 left-4 right-4 z-20 flex flex-col sm:flex-row items-end sm:items-end justify-between gap-3 pointer-events-none">
          <div className="pointer-events-auto flex flex-col gap-2 max-w-xs sm:max-w-sm w-full">
            <Colorbar
              field={state.field}
              vmin={colorRange.vmin}
              vmax={colorRange.vmax}
              wallRange={wallRange}
            />
          </div>

          <div className="pointer-events-auto flex flex-col sm:flex-row items-end gap-2">
            <button
              type="button"
              onClick={() => setShowArgo(!showArgo)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border shadow-lg backdrop-blur-md transition-all flex items-center gap-2 ${
                showArgo
                  ? "bg-amber-500 text-slate-950 border-amber-400 font-semibold"
                  : "bg-white/85 dark:bg-slate-950/85 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-800 hover:text-slate-950 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-900"
              }`}
            >
              {showArgo ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5 text-slate-400" />}
              <span>ARGO Floats {argoData?.profiles ? `(${argoData.profiles.length})` : ""}</span>
            </button>
            <CoverageLegend />
          </div>
        </div>

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

      {/* Control Panel (Sidebar on desktop, bottom sheet / drawer on mobile) */}
      <div
        className={`z-30 transition-all duration-300 border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-800/80 bg-white/95 dark:bg-slate-950/90 backdrop-blur-xl flex flex-col shadow-2xl md:w-96 w-full ${
          showControls ? "max-h-[60vh] md:max-h-full" : "max-h-12 md:max-h-full"
        }`}
      >
        {/* Toggle header on mobile */}
        <div className="flex md:hidden items-center justify-between px-4 py-2.5 border-b border-slate-200 dark:border-slate-800/60">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
            <SlidersHorizontal className="h-4 w-4 text-cyan-500 dark:text-cyan-400" />
            <span>Controls</span>
          </div>
          <button
            type="button"
            onClick={() => setShowControls(!showControls)}
            className="text-xs text-cyan-600 dark:text-cyan-400 font-medium"
          >
            {showControls ? "Hide" : "Show Controls"}
          </button>
        </div>

        {/* Scrollable controls body */}
        <div className={`p-4 space-y-5 overflow-y-auto ${!showControls ? "hidden md:block" : ""}`}>
          {/* Selected Point Status */}
          {state.lat !== null && state.lon !== null ? (
            <div className="bg-cyan-50 dark:bg-cyan-950/40 border border-cyan-200 dark:border-cyan-800/50 rounded-lg p-3 text-xs flex items-center justify-between">
              <div className="flex items-center gap-2 text-cyan-900 dark:text-cyan-200">
                <MapPin className="h-4 w-4 text-cyan-500 dark:text-cyan-400 shrink-0" />
                <div>
                  <p className="font-semibold">{formatCoord(state.lat, state.lon)}</p>
                  <p className="text-[11px] text-cyan-600 dark:text-cyan-400/80">Selected cell centre</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setState((prev) => ({ ...prev, lat: null, lon: null }))}
                className="text-[11px] text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-700/60 px-2 py-1 rounded transition-colors"
              >
                Clear
              </button>
            </div>
          ) : (
            <div className="bg-slate-100/70 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800/60 rounded-lg p-2.5 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
              <MapPin className="h-4 w-4 text-slate-400 dark:text-slate-500 shrink-0" />
              <span>Click any point on the map to inspect vertical profiles & time series.</span>
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
        </div>
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
