"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { type Map as MapLibreMap } from "maplibre-gl";
import { formatCoord, formatTemp } from "../../lib/format";
import { renderGridToCanvas } from "./GridCanvas";
import { isNorthIndianOcean } from "../../lib/oceanMask";
import type { ArgoProfileMatch, MapResponse } from "../../lib/api/client";

interface MapViewProps {
  mapData?: MapResponse | null;
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

const OPENFREEMAP_STYLES = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};

function getBasemapStyle(isLight: boolean): string {
  return isLight ? OPENFREEMAP_STYLES.light : OPENFREEMAP_STYLES.dark;
}

function getInitialThemeIsLight(): boolean {
  if (typeof window !== "undefined") {
    const saved = localStorage.getItem("theme");
    if (saved === "light") return true;
    if (saved === "dark") return false;
  }
  if (typeof document !== "undefined") {
    if (document.documentElement.classList.contains("light")) return true;
    if (document.documentElement.classList.contains("dark")) return false;
  }
  return false;
}

export function MapView({
  mapData,
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
  const [isLightMode, setIsLightMode] = useState<boolean>(getInitialThemeIsLight);
  const currentStyleRef = useRef<string | null>(null);

  const lastDataUrlRef = useRef<string | null>(null);
  const lastAppliedUrlRef = useRef<string | null>(null);

  // Monitor documentElement class changes to switch between light and dark basemap
  useEffect(() => {
    if (typeof document === "undefined") return;
    const checkLight = () => {
      const saved = typeof window !== "undefined" ? localStorage.getItem("theme") : null;
      if (saved === "light") return true;
      if (saved === "dark") return false;
      return document.documentElement.classList.contains("light");
    };
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

  const syncGridLayer = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const dataUrl = lastDataUrlRef.current;
    const sourceId = "ocean-grid-source";
    const layerId = "ocean-grid-layer";

    if (!dataUrl) {
      if (map.getLayer(layerId)) {
        map.removeLayer(layerId);
      }
      if (map.getSource(sourceId)) {
        map.removeSource(sourceId);
      }
      lastAppliedUrlRef.current = null;
      return;
    }

    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [40.0, 30.0], // Top-left [lon, lat]
      [100.0, 30.0], // Top-right
      [100.0, 0.0], // Bottom-right
      [40.0, 0.0], // Bottom-left
    ];

    try {
      const existingSource = map.getSource(sourceId) as maplibregl.ImageSource | undefined;
      const existingLayer = map.getLayer(layerId);

      if (existingSource && existingLayer) {
        if (lastAppliedUrlRef.current !== dataUrl) {
          existingSource.updateImage({ url: dataUrl });
          lastAppliedUrlRef.current = dataUrl;
        }
      } else {
        if (existingLayer) {
          map.removeLayer(layerId);
        }
        if (existingSource) {
          map.removeSource(sourceId);
        }

        map.addSource(sourceId, {
          type: "image",
          url: dataUrl,
          coordinates,
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

        lastAppliedUrlRef.current = dataUrl;
      }
    } catch (err) {
      console.warn("Could not sync ocean grid layer:", err);
    }
  }, []);

  // Initialize MapLibre Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLight = getInitialThemeIsLight();
    const initialStyle = getBasemapStyle(initialLight);
    currentStyleRef.current = initialStyle;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: initialStyle,
      center: [70.0, 15.0],
      zoom: 4,
      minZoom: 2,
      maxZoom: 10,
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");

    const handleStyleReady = () => {
      if (map.isStyleLoaded()) {
        syncGridLayer();
      }
    };

    map.on("load", () => {
      setMapLoaded(true);
      handleStyleReady();
    });

    map.on("styledata", handleStyleReady);
    map.on("idle", handleStyleReady);

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
      lastAppliedUrlRef.current = null;
      map.setStyle(nextStyle);
    }
  }, [isLightMode, mapLoaded]);

  // Update canvas grid layer when mapData, vmin, or vmax change
  useEffect(() => {
    if (!mapData) {
      lastDataUrlRef.current = null;
    } else {
      const colormap = mapData.field === "error" ? "balance" : "thermal";
      const nLat = mapData.lat?.length ?? mapData.values.length;
      const nLon = mapData.lon?.length ?? (mapData.values[0]?.length ?? 0);
      const maskedCoverage = mapData.coverage.map((row, r) =>
        row.map((cov, c) => {
          const lat = mapData.lat?.[r] ?? (30.0 / nLat) * (r + 0.5);
          const lon = mapData.lon?.[c] ?? 40.0 + (60.0 / nLon) * (c + 0.5);
          return isNorthIndianOcean(lat, lon) ? cov : 0;
        })
      );
      const canvas = renderGridToCanvas(mapData.values, maskedCoverage, vmin, vmax, colormap);
      lastDataUrlRef.current = canvas.toDataURL();
    }
    syncGridLayer();
  }, [mapData, vmin, vmax, syncGridLayer]);

  const mapDataRef = useRef(mapData);
  useEffect(() => {
    mapDataRef.current = mapData;
  }, [mapData]);

  const onSelectPointRef = useRef(onSelectPoint);
  useEffect(() => {
    onSelectPointRef.current = onSelectPoint;
  }, [onSelectPoint]);

  // Handle hover and click inspection over grid cells
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const selectAtLngLat = (lng: number, lat: number) => {
      if (!isNorthIndianOcean(lat, lng)) {
        return;
      }
      const data = mapDataRef.current;
      if (lng >= 40.0 && lng <= 100.0 && lat >= 0.0 && lat <= 30.0) {
        if (data && data.lat && data.lat.length > 0 && data.lon && data.lon.length > 0) {
          const nLat = data.lat.length;
          const nLon = data.lon.length;
          const r = Math.min(nLat - 1, Math.max(0, Math.floor(lat / (30.0 / nLat))));
          const c = Math.min(nLon - 1, Math.max(0, Math.floor((lng - 40.0) / (60.0 / nLon))));
          const snappedLat = data.lat[r];
          const snappedLon = data.lon[c];
          onSelectPointRef.current(snappedLat, snappedLon);
        } else {
          const snappedLat = Math.round(lat * 4) / 4;
          const snappedLon = Math.round(lng * 4) / 4;
          onSelectPointRef.current(snappedLat, snappedLon);
        }
      }
    };

    const handleMouseMove = (e: maplibregl.MapMouseEvent) => {
      const data = mapDataRef.current;
      if (!data) return;
      const { lng, lat } = e.lngLat;
      const lonMin = 40.0;
      const lonMax = 100.0;
      const latMin = 0.0;
      const latMax = 30.0;

      if (lng >= lonMin && lng <= lonMax && lat >= latMin && lat <= latMax) {
        const nLat = data.lat.length;
        const nLon = data.lon.length;
        const latRes = (latMax - latMin) / nLat;
        const lonRes = (lonMax - lonMin) / nLon;

        const r = Math.min(nLat - 1, Math.max(0, Math.floor((lat - latMin) / latRes)));
        const c = Math.min(nLon - 1, Math.max(0, Math.floor((lng - lonMin) / lonRes)));

        const cellLat = data.lat[r];
        const cellLon = data.lon[c];
        const isOcean = isNorthIndianOcean(cellLat, cellLon);
        const val = isOcean ? (data.values[r]?.[c] ?? null) : null;
        const cov = isOcean ? (data.coverage[r]?.[c] ?? 0) : 0;

        setHoverInfo({
          x: e.point.x,
          y: e.point.y,
          lat: cellLat,
          lon: cellLon,
          value: val,
          coverage: cov,
        });
      } else {
        setHoverInfo(null);
      }
    };

    const handleMouseLeave = () => {
      setHoverInfo(null);
    };

    const handleClick = (e: maplibregl.MapMouseEvent) => {
      selectAtLngLat(e.lngLat.lng, e.lngLat.lat);
    };

    const handleDomClick = (e: MouseEvent) => {
      const container = mapContainerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      try {
        const coords = map.unproject([x, y]);
        selectAtLngLat(coords.lng, coords.lat);
      } catch {
        // Unproject failed
      }
    };

    map.on("mousemove", handleMouseMove);
    map.on("mouseout", handleMouseLeave);
    map.on("click", handleClick);

    const containerEl = mapContainerRef.current;
    containerEl?.addEventListener("click", handleDomClick, true);

    return () => {
      map.off("mousemove", handleMouseMove);
      map.off("mouseout", handleMouseLeave);
      map.off("click", handleClick);
      containerEl?.removeEventListener("click", handleDomClick, true);
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

    // Clear previous argo markers
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
