"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { type CanvasSource, type Map as MapLibreMap } from "maplibre-gl";
import { formatCoord, formatTemp } from "../../lib/format";
import { renderGridToCanvas } from "./GridCanvas";
import type { ArgoProfileMatch, MapResponse } from "../../lib/api/client";

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
}

interface HoverInfo {
  x: number;
  y: number;
  lat: number;
  lon: number;
  value: number | null;
  coverage: number;
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
}: MapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<MapLibreMap | null>(null);
  const selectionMarkerRef = useRef<maplibregl.Marker | null>(null);
  const argoMarkersRef = useRef<maplibregl.Marker[]>([]);

  const [mapLoaded, setMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [isLightMode, setIsLightMode] = useState<boolean>(false);
  const currentStyleRef = useRef<string | null>(null);

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
  }, [getCoordinates]);

  // Initialize MapLibre Map with locked 2D pitch and disabled drag-rotate
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
      pitch: 0,
      maxPitch: 0, // Flat 2D map view
      dragRotate: false, // Disable drag rotate in 2D map view
      pitchWithRotate: false,
      touchPitch: false,
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    const handleStyleReady = () => {
      if (map.isStyleLoaded()) {
        syncGridLayer();
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
    };
  }, [syncGridLayer]);

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
  }, [mapData, vmin, vmax, syncGridLayer]);

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
        return; // Already scheduled for this animation frame
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

        const r = findNearestIndex(data.lat, lat);
        const c = findNearestIndex(data.lon, lng);

        // Skip state update if mouse is still in the same grid cell
        if (
          lastHoverCellRef.current &&
          lastHoverCellRef.current.r === r &&
          lastHoverCellRef.current.c === c
        ) {
          return;
        }

        lastHoverCellRef.current = { r, c };

        const cellLat = data.lat[r];
        const cellLon = data.lon[c];
        const val = data.values[r]?.[c] ?? null;
        const cov = data.coverage[r]?.[c] ?? 0;

        setHoverInfo({
          x: evt.point.x,
          y: evt.point.y,
          lat: cellLat,
          lon: cellLon,
          value: val,
          coverage: cov,
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
      const { lng, lat } = e.lngLat;

      if (
        lng >= rgn.lon_min &&
        lng <= rgn.lon_max &&
        lat >= rgn.lat_min &&
        lat <= rgn.lat_max
      ) {
        if (data?.lat && data?.lon && data.lat.length > 0 && data.lon.length > 0) {
          const r = findNearestIndex(data.lat, lat);
          const c = findNearestIndex(data.lon, lng);
          onSelectPointRef.current(data.lat[r], data.lon[c]);
        } else {
          onSelectPointRef.current(lat, lng);
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
  }, []);

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
          <div className="font-mono text-[11px] text-slate-500 dark:text-slate-400">
            {formatCoord(hoverInfo.lat, hoverInfo.lon)}
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
