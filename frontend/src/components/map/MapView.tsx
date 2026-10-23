"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { type CanvasSource, type Map as MapLibreMap } from "maplibre-gl";
import { formatCoord, formatDepth, formatTemp } from "../../lib/format";
import { renderGridToCanvas } from "./GridCanvas";
import type { ArgoProfileMatch, MapResponse, SectionResponse } from "../../lib/api/client";
import type { WallId } from "../map3d/depthScale";
import type { CutawayLayer } from "../map3d/CutawayLayer";

export interface RegionBounds {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

interface MapViewProps {
  mapData?: MapResponse | null;
  region?: RegionBounds;
  gridResolutionDeg?: number;
  vmin: number;
  vmax: number;
  selectedLat: number | null;
  selectedLon: number | null;
  onSelectPoint: (lat: number, lon: number) => void;
  showArgo?: boolean;
  argoProfiles?: ArgoProfileMatch[];
  className?: string;

  // 3D Explorer props (PRD_FRONTEND §3.1b)
  view?: "map" | "3d";
  depth_m?: number;
  availableDepths?: number[];
  colorRanges?: { depth_m: number; vmin: number; vmax: number }[];
  sections?: Partial<Record<WallId, SectionResponse>>;
  isTouring?: boolean;
  onStopTour?: () => void;
  isExploded?: boolean;
  isSmoothed?: boolean;
  showContours?: boolean;
}

interface HoverInfo {
  x: number;
  y: number;
  lat: number;
  lon: number;
  value: number | null;
  coverage: number;
  depth_m?: number;
}

const DEFAULT_REGION: RegionBounds = {
  lat_min: 0.0,
  lat_max: 30.0,
  lon_min: 40.0,
  lon_max: 100.0,
};

const OPENFREEMAP_STYLES = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};

function getBasemapStyle(isLight: boolean): string {
  return isLight ? OPENFREEMAP_STYLES.light : OPENFREEMAP_STYLES.dark;
}

/**
 * Find the nearest index in an array of sorted coordinate values.
 */
function findNearestIndex(arr: number[], target: number): number {
  if (arr.length === 0) return 0;
  let bestIdx = 0;
  let bestDist = Math.abs(arr[0] - target);
  for (let i = 1; i < arr.length; i++) {
    const dist = Math.abs(arr[i] - target);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

export function MapView({
  mapData,
  region,
  vmin,
  vmax,
  selectedLat,
  selectedLon,
  onSelectPoint,
  showArgo = false,
  argoProfiles = [],
  className = "",
  view = "map",
  depth_m = 100.0,
  availableDepths,
  colorRanges,
  sections,
  isTouring = false,
  onStopTour,
  isExploded = false,
  isSmoothed = false,
  showContours = true,
}: MapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<MapLibreMap | null>(null);
  const selectionMarkerRef = useRef<maplibregl.Marker | null>(null);
  const argoMarkersRef = useRef<maplibregl.Marker[]>([]);

  const [mapLoaded, setMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [isLightMode, setIsLightMode] = useState<boolean>(false);
  const currentStyleRef = useRef<string | null>(null);

  // Reference to 3D Custom Layer instance
  const cutawayLayerRef = useRef<CutawayLayer | null>(null);

  const activeRegion = region ?? DEFAULT_REGION;
  const regionRef = useRef<RegionBounds>(activeRegion);
  useEffect(() => {
    regionRef.current = activeRegion;
  }, [activeRegion]);

  const mapDataRef = useRef(mapData);
  useEffect(() => {
    mapDataRef.current = mapData;
  }, [mapData]);

  const onSelectPointRef = useRef(onSelectPoint);
  useEffect(() => {
    onSelectPointRef.current = onSelectPoint;
  }, [onSelectPoint]);

  // Canvas source reference to avoid PNG dataURL encoding on main thread
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceId = "ocean-grid-source";
  const layerId = "ocean-grid-layer";

  // Monitor documentElement class changes to switch between light and dark basemap
  useEffect(() => {
    if (typeof document === "undefined") return;
    const checkLight = () =>
      document.documentElement.classList.contains("light") ||
      !document.documentElement.classList.contains("dark");
    setIsLightMode(checkLight());

    const observer = new MutationObserver(() => {
      setIsLightMode(checkLight());
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  const getCoordinates = useCallback(
    (rgn: RegionBounds): [[number, number], [number, number], [number, number], [number, number]] => [
      [rgn.lon_min, rgn.lat_max], // Top-left [lon, lat]
      [rgn.lon_max, rgn.lat_max], // Top-right
      [rgn.lon_max, rgn.lat_min], // Bottom-right
      [rgn.lon_min, rgn.lat_min], // Bottom-left
    ],
    []
  );

  const syncGridLayer = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map || !map.isStyleLoaded()) return;

    // In 3D mode, the 2D raster grid layer is hidden to avoid z-fighting
    if (view === "3d") {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", "none");
      }
      return;
    }

    const data = mapDataRef.current;
    const canvas = canvasRef.current;

    if (
      !data ||
      !canvas ||
      !data.values ||
      data.values.length === 0 ||
      !data.values[0] ||
      data.values[0].length === 0
    ) {
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
      return;
    }

    const coordinates = getCoordinates(regionRef.current);
    const existingSource = map.getSource(sourceId) as CanvasSource | undefined;
    const existingLayer = map.getLayer(layerId);

    if (existingSource && existingLayer) {
      existingSource.setCoordinates(coordinates);
      map.setLayoutProperty(layerId, "visibility", "visible");
      existingSource.play();
      requestAnimationFrame(() => {
        existingSource.pause();
      });
    } else {
      if (existingLayer) map.removeLayer(layerId);
      if (existingSource) map.removeSource(sourceId);

      map.addSource(sourceId, {
        type: "canvas",
        canvas,
        coordinates,
        animate: false,
      });

      map.addLayer({
        id: layerId,
        type: "raster",
        source: sourceId,
        paint: {
          "raster-opacity": 0.9,
          "raster-fade-duration": 0,
          "raster-resampling": "nearest",
        },
      });

      const newSource = map.getSource(sourceId) as CanvasSource | undefined;
      if (newSource) {
        newSource.play();
        requestAnimationFrame(() => {
          newSource.pause();
        });
      }
    }
  }, [getCoordinates, view]);

  // Lazy-load and attach 3D cutaway layer when view is "3d"
  const sync3DLayer = useCallback(async () => {
    const map = mapInstanceRef.current;
    if (!map || !map.isStyleLoaded()) return;

    if (view === "3d") {
      if (!cutawayLayerRef.current) {
        // Dynamically import CutawayLayer so it is NOT bundled into the initial / page load (PRD_FRONTEND §3.1b)
        const { CutawayLayer } = await import("../map3d/CutawayLayer");
        const layer = new CutawayLayer({
          region: regionRef.current,
          availableDepths: availableDepths ?? [0.5, 10, 20, 50, 100, 200, 500, 1000],
          colorRanges,
          isDark: !isLightMode,
        });

        if (!map.getLayer(layer.id)) {
          map.addLayer(layer);
        }
        cutawayLayerRef.current = layer;
      } else if (!map.getLayer(cutawayLayerRef.current.id)) {
        map.addLayer(cutawayLayerRef.current);
      }

      const layer = cutawayLayerRef.current;
      if (layer) {
        layer.setIsDark(!isLightMode);
        layer.setSmoothed(isSmoothed);
        layer.setExploded(isExploded);
        layer.setShowContours(showContours);

        const reducedMotion =
          typeof window !== "undefined" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        layer.setDepth(depth_m, reducedMotion);

        if (mapData) {
          layer.setMapData(mapData, vmin, vmax);
        }

        if (sections) {
          for (const wallId of ["south", "north", "west", "east"] as WallId[]) {
            const sec = sections[wallId];
            if (sec) {
              layer.setSectionData(wallId, sec);
            }
          }
        }
      }
    } else if (cutawayLayerRef.current) {
      if (map.getLayer(cutawayLayerRef.current.id)) {
        map.removeLayer(cutawayLayerRef.current.id);
      }
      cutawayLayerRef.current = null;
    }
  }, [
    view,
    availableDepths,
    colorRanges,
    isLightMode,
    isSmoothed,
    isExploded,
    showContours,
    depth_m,
    vmin,
    vmax,
    mapData,
    sections,
  ]);

  // Initialize MapLibre Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLight =
      typeof document !== "undefined" &&
      (document.documentElement.classList.contains("light") ||
        !document.documentElement.classList.contains("dark"));

    const initialStyle = getBasemapStyle(initialLight);
    currentStyleRef.current = initialStyle;

    const rgn = regionRef.current;
    const centerLon = (rgn.lon_min + rgn.lon_max) / 2;
    const centerLat = (rgn.lat_min + rgn.lat_max) / 2;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: initialStyle,
      center: [centerLon, centerLat],
      zoom: 4,
      minZoom: 2,
      maxZoom: 10,
      pitch: view === "3d" ? 50 : 0,
      bearing: view === "3d" ? -20 : 0,
      maxPitch: view === "3d" ? 85 : 0,
      dragRotate: view === "3d",
      pitchWithRotate: view === "3d",
      touchPitch: view === "3d",
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");

    const handleStyleReady = () => {
      if (map.isStyleLoaded()) {
        syncGridLayer();
        sync3DLayer();
      }
    };

    map.on("load", () => {
      setMapLoaded(true);
      handleStyleReady();
    });
    map.on("style.load", () => {
      setMapLoaded(true);
      handleStyleReady();
    });
    map.on("styledata", handleStyleReady);
    map.on("error", () => {
      setMapLoaded(true);
    });

    if (map.isStyleLoaded()) {
      setMapLoaded(true);
      handleStyleReady();
    }

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      cutawayLayerRef.current = null;
    };
  }, [syncGridLayer, sync3DLayer, view]);

  // Dynamically switch pitch, bearing, and controls between 2D Map and 3D Cutaway
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;

    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (view === "3d") {
      map.setMaxPitch(85);
      map.dragRotate.enable();
      map.touchPitch.enable();

      map.easeTo({
        pitch: 50,
        bearing: -20,
        duration: reducedMotion ? 0 : 800,
      });

      syncGridLayer();
      sync3DLayer();
    } else {
      map.easeTo({
        pitch: 0,
        bearing: 0,
        duration: reducedMotion ? 0 : 600,
      });

      map.setMaxPitch(0);
      map.dragRotate.disable();
      map.touchPitch.disable();

      syncGridLayer();
      sync3DLayer();
    }
  }, [view, mapLoaded, syncGridLayer, sync3DLayer]);

  // Update 3D layer on parameter changes
  useEffect(() => {
    if (view === "3d" && mapLoaded) {
      sync3DLayer();
    }
  }, [view, mapLoaded, sync3DLayer]);

  // Tour mode: auto-rotate bearing smoothly in 3D
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded || view !== "3d" || !isTouring) return;

    let tourRaf: number | null = null;

    const rotateStep = () => {
      const currentBearing = map.getBearing();
      map.setBearing((currentBearing + 0.15) % 360);
      tourRaf = requestAnimationFrame(rotateStep);
    };

    tourRaf = requestAnimationFrame(rotateStep);

    // Cancel tour on user interaction with map canvas
    const handleUserInteract = (e: any) => {
      const target = e?.originalEvent?.target as HTMLElement | null;
      if (
        target &&
        (target.closest("button") ||
          target.closest("input") ||
          target.closest("select") ||
          target.closest(".pointer-events-auto") ||
          target.closest(".maplibregl-ctrl"))
      ) {
        return;
      }
      if (onStopTour) onStopTour();
    };

    map.on("mousedown", handleUserInteract);
    map.on("touchstart", handleUserInteract);
    map.on("wheel", handleUserInteract);

    return () => {
      if (tourRaf !== null) cancelAnimationFrame(tourRaf);
      map.off("mousedown", handleUserInteract);
      map.off("touchstart", handleUserInteract);
      map.off("wheel", handleUserInteract);
    };
  }, [view, isTouring, mapLoaded, onStopTour]);

  // Dynamically update basemap style when theme toggles
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;

    const nextStyle = getBasemapStyle(isLightMode);
    if (currentStyleRef.current !== nextStyle) {
      currentStyleRef.current = nextStyle;
      map.setStyle(nextStyle);
    }
  }, [isLightMode, mapLoaded]);

  // Update canvas grid layer when mapData, vmin, or vmax change
  useEffect(() => {
    if (
      !mapData ||
      !mapData.values ||
      mapData.values.length === 0 ||
      !mapData.values[0] ||
      mapData.values[0].length === 0
    ) {
      syncGridLayer();
      return;
    }

    if (!canvasRef.current && typeof document !== "undefined") {
      canvasRef.current = document.createElement("canvas");
    }

    const colormap = mapData.field === "error" ? "balance" : "thermal";
    renderGridToCanvas(mapData.values, mapData.coverage, vmin, vmax, colormap, 4, canvasRef.current);
    syncGridLayer();

    if (view === "3d" && cutawayLayerRef.current) {
      cutawayLayerRef.current.setMapData(mapData, vmin, vmax);
      mapInstanceRef.current?.triggerRepaint();
    }
  }, [mapData, vmin, vmax, syncGridLayer, view]);

  // Throttled hover handler and single click path
  const hoverRafRef = useRef<number | null>(null);
  const lastHoverCellRef = useRef<{ r: number; c: number } | null>(null);
  const pendingMouseEventRef = useRef<maplibregl.MapMouseEvent | null>(null);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const handleMouseMove = (e: maplibregl.MapMouseEvent) => {
      pendingMouseEventRef.current = e;

      if (hoverRafRef.current !== null) {
        return;
      }

      hoverRafRef.current = requestAnimationFrame(() => {
        hoverRafRef.current = null;
        const evt = pendingMouseEventRef.current;
        const data = mapDataRef.current;
        const rgn = regionRef.current;

        if (
          !evt ||
          !data ||
          !data.lat ||
          !data.lon ||
          data.lat.length === 0 ||
          data.lon.length === 0
        ) {
          if (lastHoverCellRef.current !== null) {
            lastHoverCellRef.current = null;
            setHoverInfo(null);
          }
          return;
        }

        let cellLat: number;
        let cellLon: number;
        let r: number;
        let c: number;

        if (view === "3d" && cutawayLayerRef.current) {
          const canvas = map.getCanvas();
          const snapped = cutawayLayerRef.current.getSnappedCellAtScreenPoint(
            evt.point.x,
            evt.point.y,
            data.lat,
            data.lon,
            canvas.clientWidth || canvas.width,
            canvas.clientHeight || canvas.height
          );
          if (!snapped) {
            if (lastHoverCellRef.current !== null) {
              lastHoverCellRef.current = null;
              setHoverInfo(null);
            }
            return;
          }
          if (
            snapped.lon < rgn.lon_min ||
            snapped.lon > rgn.lon_max ||
            snapped.lat < rgn.lat_min ||
            snapped.lat > rgn.lat_max
          ) {
            if (lastHoverCellRef.current !== null) {
              lastHoverCellRef.current = null;
              setHoverInfo(null);
            }
            return;
          }
          cellLat = snapped.lat;
          cellLon = snapped.lon;
          r = findNearestIndex(data.lat, cellLat);
          c = findNearestIndex(data.lon, cellLon);
        } else {
          const { lng, lat } = evt.lngLat;
          if (
            lng < rgn.lon_min ||
            lng > rgn.lon_max ||
            lat < rgn.lat_min ||
            lat > rgn.lat_max
          ) {
            if (lastHoverCellRef.current !== null) {
              lastHoverCellRef.current = null;
              setHoverInfo(null);
            }
            return;
          }
          r = findNearestIndex(data.lat, lat);
          c = findNearestIndex(data.lon, lng);
          cellLat = data.lat[r];
          cellLon = data.lon[c];
        }

        // Skip state update if mouse is still in the same grid cell
        if (
          lastHoverCellRef.current &&
          lastHoverCellRef.current.r === r &&
          lastHoverCellRef.current.c === c
        ) {
          return;
        }

        lastHoverCellRef.current = { r, c };

        const val = data.values[r]?.[c] ?? null;
        const cov = data.coverage[r]?.[c] ?? 0;

        setHoverInfo({
          x: evt.point.x,
          y: evt.point.y,
          lat: cellLat,
          lon: cellLon,
          value: val,
          coverage: cov,
          depth_m,
        });
      });
    };

    const handleMouseLeave = () => {
      if (hoverRafRef.current !== null) {
        cancelAnimationFrame(hoverRafRef.current);
        hoverRafRef.current = null;
      }
      pendingMouseEventRef.current = null;
      lastHoverCellRef.current = null;
      setHoverInfo(null);
    };

    // Single click path: MapLibre click handler only
    const handleClick = (e: maplibregl.MapMouseEvent) => {
      const data = mapDataRef.current;
      const rgn = regionRef.current;
      let targetLat = e.lngLat.lat;
      let targetLon = e.lngLat.lng;

      if (view === "3d" && cutawayLayerRef.current && data?.lat && data?.lon) {
        const canvas = map.getCanvas();
        const snapped = cutawayLayerRef.current.getSnappedCellAtScreenPoint(
          e.point.x,
          e.point.y,
          data.lat,
          data.lon,
          canvas.clientWidth || canvas.width,
          canvas.clientHeight || canvas.height
        );
        if (snapped) {
          targetLat = snapped.lat;
          targetLon = snapped.lon;
        }
      }

      if (
        targetLon >= rgn.lon_min &&
        targetLon <= rgn.lon_max &&
        targetLat >= rgn.lat_min &&
        targetLat <= rgn.lat_max
      ) {
        if (data?.lat && data?.lon && data.lat.length > 0 && data.lon.length > 0) {
          const r = findNearestIndex(data.lat, targetLat);
          const c = findNearestIndex(data.lon, targetLon);
          onSelectPointRef.current(data.lat[r], data.lon[c]);
        } else {
          onSelectPointRef.current(targetLat, targetLon);
        }
      }
    };

    map.on("mousemove", handleMouseMove);
    map.on("mouseout", handleMouseLeave);
    map.on("click", handleClick);

    return () => {
      if (hoverRafRef.current !== null) {
        cancelAnimationFrame(hoverRafRef.current);
        hoverRafRef.current = null;
      }
      map.off("mousemove", handleMouseMove);
      map.off("mouseout", handleMouseLeave);
      map.off("click", handleClick);
    };
  }, [depth_m, view]);

  // Update selection marker
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;

    if (selectedLat !== null && selectedLon !== null) {
      if (!selectionMarkerRef.current) {
        const el = document.createElement("div");
        el.className = "relative flex items-center justify-center cursor-pointer";
        el.innerHTML = `
          <span class="animate-ping absolute inline-flex h-6 w-6 rounded-full bg-cyan-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-3.5 w-3.5 bg-cyan-400 border-2 border-white shadow-md"></span>
        `;
        selectionMarkerRef.current = new maplibregl.Marker({ element: el })
          .setLngLat([selectedLon, selectedLat])
          .addTo(map);
      } else {
        selectionMarkerRef.current.setLngLat([selectedLon, selectedLat]);
      }
    } else if (selectionMarkerRef.current) {
      selectionMarkerRef.current.remove();
      selectionMarkerRef.current = null;
    }
  }, [mapLoaded, selectedLat, selectedLon]);

  // Update ARGO markers
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;

    argoMarkersRef.current.forEach((m) => m.remove());
    argoMarkersRef.current = [];

    if (showArgo && argoProfiles.length > 0) {
      argoProfiles.forEach((p) => {
        const el = document.createElement("div");
        el.className = "group relative cursor-pointer";
        el.innerHTML = `
          <div class="h-3 w-3 rounded-full bg-amber-400 border border-slate-900 shadow-sm transition-transform group-hover:scale-125"></div>
        `;
        const marker = new maplibregl.Marker({ element: el })
          .setLngLat([p.lon, p.lat])
          .setPopup(
            new maplibregl.Popup({ offset: 10, closeButton: false }).setHTML(`
              <div class="text-xs p-1 text-slate-900">
                <p class="font-bold">ARGO Float ${p.platform_number}</p>
                <p>Cycle: ${p.cycle} (${p.n_obs} obs)</p>
                <p class="text-[10px] text-slate-500">${p.time.slice(0, 10)}</p>
              </div>
            `)
          )
          .addTo(map);

        argoMarkersRef.current.push(marker);
      });
    }
  }, [mapLoaded, showArgo, argoProfiles]);

  const sourceLabel =
    mapData?.field === "error"
      ? "Difference (Model − GLORYS)"
      : mapData?.field === "glorys"
      ? "GLORYS reanalysis"
      : "Model prediction";

  return (
    <div
      className={`relative w-full h-full min-h-[400px] overflow-hidden select-none bg-slate-100 dark:bg-slate-950 transition-colors ${className}`}
    >
      <div ref={mapContainerRef} className="w-full h-full min-h-[400px]" />

      {/* Hover tooltip */}
      {hoverInfo && (
        <div
          className="pointer-events-none absolute z-30 transform -translate-x-1/2 -translate-y-full mb-2 bg-white/95 dark:bg-slate-950/90 backdrop-blur-md border border-slate-300 dark:border-slate-700/80 rounded-lg px-2.5 py-1.5 shadow-2xl text-xs text-slate-800 dark:text-slate-200"
          style={{ left: hoverInfo.x, top: hoverInfo.y - 12 }}
        >
          <div className="font-mono text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
            <span>{formatCoord(hoverInfo.lat, hoverInfo.lon)}</span>
            {view === "3d" && hoverInfo.depth_m !== undefined && (
              <span className="text-cyan-500 font-semibold">· {formatDepth(hoverInfo.depth_m)}</span>
            )}
          </div>
          {hoverInfo.coverage === 0 ? (
            <div className="text-slate-500 font-medium text-[11px]">Land / Unmodelled</div>
          ) : hoverInfo.coverage === 2 ? (
            <div className="text-amber-500 dark:text-amber-400/90 font-medium text-[11px]">
              No prediction (insufficient input data)
            </div>
          ) : (
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="font-semibold text-cyan-600 dark:text-cyan-300 font-mono text-sm">
                {formatTemp(hoverInfo.value, 1)}
              </span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400">({sourceLabel})</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
