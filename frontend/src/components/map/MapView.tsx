"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { type CanvasSource, type Map as MapLibreMap } from "maplibre-gl";
import { formatCoord, formatDepth, formatTemp } from "../../lib/format";
import { renderGridToCanvas } from "./GridCanvas";
import type { ArgoProfileMatch, MapResponse, SectionResponse } from "../../lib/api/client";
import type { GridStyle } from "../../lib/url-state";
import type { WallId } from "../map3d/depthScale";
import type { CutawayLayer } from "../map3d/CutawayLayer";
import type { CameraFocus } from "../../lib/guide/ui-actions";
import { gsap, loopPing, prefersReducedMotion } from "../../lib/motion";

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

  // 3D Explorer & grid rendering props (PRD_FRONTEND §3.1, §3.1b)
  gridStyle?: GridStyle;
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

  /** Camera move requested by the guide or the time machine (PRD_FRONTEND §3.4a, c). */
  focus?: CameraFocus | null;

  /** Lasso selection (PRD_FRONTEND §3.4e): closed [[lon, lat], ...] ring from /segment. */
  outline?: number[][] | null;
  /** Tapped point while /segment runs, shown as a pulsing ring. */
  pendingPoint?: { lat: number; lon: number } | null;

  /**
   * Landing → Explorer arrival (src/lib/arrival.ts). When true at mount, the map starts zoomed in
   * on open water with the grid hidden; once the slice has drawn it calls onIntroReady, zooms out
   * to the basin and flies into the region (onIntroFlyIn as it turns inward), then onIntroDone.
   */
  intro?: boolean;
  onIntroReady?: () => void;
  onIntroFlyIn?: () => void;
  onIntroDone?: () => void;
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

const LASSO_SOURCE = "lasso-outline";

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
 * Basemap labels kept in Dots mode: seas and countries for orientation. City, road and
 * state names are hidden there because they print over the dot matrix; Classic shows all.
 * Layer ids are from the OpenFreeMap dark and positron styles.
 */
const DOTS_KEEP_LABEL = /^(water_name|place_country|label_country)/;

function applyLabelDensity(map: MapLibreMap, style: GridStyle) {
  const layers = map.getStyle?.()?.layers ?? [];
  for (const layer of layers) {
    if (layer.type !== "symbol" || layer.id.startsWith(LASSO_SOURCE)) continue;
    const visible = style !== "dots" || DOTS_KEEP_LABEL.test(layer.id);
    const next = visible ? "visible" : "none";
    if (map.getLayoutProperty(layer.id, "visibility") !== next) {
      map.setLayoutProperty(layer.id, "visibility", next);
    }
  }
}

/**
 * True once the style JSON is parsed, i.e. sources and layers may be added. This is the flag
 * Style#addLayer checks; isStyleLoaded() also waits for tiles, so it reads false after every
 * camera move and would skip redraws. Falls back to isStyleLoaded() if the field ever goes away.
 */
function canEditStyle(map: MapLibreMap): boolean {
  const loaded = (map as unknown as { style?: { _loaded?: boolean } }).style?._loaded;
  return loaded ?? Boolean(map.isStyleLoaded());
}

/** Frame the model region, leaving room for the floating controls card on wide screens. */
function regionPadding(map: MapLibreMap) {
  return map.getContainer().clientWidth >= 1024
    ? { top: 32, bottom: 96, left: 388, right: 48 }
    : { top: 24, bottom: 120, left: 16, right: 16 };
}

function regionBounds(rgn: RegionBounds): [[number, number], [number, number]] {
  return [
    [rgn.lon_min, rgn.lat_min],
    [rgn.lon_max, rgn.lat_max],
  ];
}

function fitRegion(map: MapLibreMap, rgn: RegionBounds, animate: boolean) {
  map.fitBounds(regionBounds(rgn), { padding: regionPadding(map), duration: animate ? 600 : 0 });
}

/** Arrival: start this far in, on open water at the region's centre (a flat water-coloured screen). */
const INTRO_ZOOM = 9;
/** Arrival: how far below the region's framing zoom the camera pulls back (shows the basin). */
const INTRO_ZOOM_OUT = 1.7;

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
  gridStyle = "dots",
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
  focus = null,
  outline = null,
  pendingPoint = null,
  intro = false,
  onIntroReady,
  onIntroFlyIn,
  onIntroDone,
}: MapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<MapLibreMap | null>(null);
  const selectionMarkerRef = useRef<maplibregl.Marker | null>(null);
  const selectionPingRef = useRef<gsap.core.Tween | null>(null);
  const argoMarkersRef = useRef<maplibregl.Marker[]>([]);

  const [mapLoaded, setMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [isLightMode, setIsLightMode] = useState<boolean>(false);
  const currentStyleRef = useRef<string | null>(null);

  // Reference to 3D Custom Layer instance
  const cutawayLayerRef = useRef<CutawayLayer | null>(null);

  const activeRegion = region ?? DEFAULT_REGION;
  const regionRef = useRef<RegionBounds>(activeRegion);
  // Arrival state: whether this mount plays the intro, and the grid opacity it drives (null = normal)
  const introAtMountRef = useRef(intro);
  const gridOpacityRef = useRef<number | null>(intro ? 0 : null);
  const introStartedRef = useRef(false);
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
  const lastCanvasSizeRef = useRef<{ w: number; h: number } | null>(null);
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
    if (!map || !canEditStyle(map)) return;

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
    const resampling = gridStyle === "dots" ? "linear" : "nearest";
    const opacity = gridOpacityRef.current ?? (gridStyle === "dots" ? 0.96 : 0.9);
    const existingSource = map.getSource(sourceId) as CanvasSource | undefined;
    const existingLayer = map.getLayer(layerId);
    const sizeChanged =
      lastCanvasSizeRef.current !== null &&
      (lastCanvasSizeRef.current.w !== canvas.width || lastCanvasSizeRef.current.h !== canvas.height);

    if (existingSource && existingLayer && !sizeChanged) {
      existingSource.setCoordinates(coordinates);
      map.setLayoutProperty(layerId, "visibility", "visible");
      map.setPaintProperty(layerId, "raster-resampling", resampling);
      map.setPaintProperty(layerId, "raster-opacity", opacity);
      existingSource.play();
      requestAnimationFrame(() => {
        existingSource.pause();
      });
    } else {
      if (existingLayer) map.removeLayer(layerId);
      if (existingSource) map.removeSource(sourceId);

      lastCanvasSizeRef.current = { w: canvas.width, h: canvas.height };

      map.addSource(sourceId, {
        type: "canvas",
        canvas,
        coordinates,
        animate: false,
      });

      // Place grid layer beneath vector symbol labels so sea/country names stay legible
      const firstSymbolId = map
        .getStyle?.()
        ?.layers?.find((l) => l.type === "symbol" && l.id !== `${LASSO_SOURCE}-line`)?.id;

      map.addLayer(
        {
          id: layerId,
          type: "raster",
          source: sourceId,
          paint: {
            "raster-opacity": opacity,
            "raster-fade-duration": 0,
            "raster-resampling": resampling,
          },
        },
        firstSymbolId
      );

      const newSource = map.getSource(sourceId) as CanvasSource | undefined;
      if (newSource) {
        newSource.play();
        requestAnimationFrame(() => {
          newSource.pause();
        });
      }
    }
  }, [getCoordinates, gridStyle, view]);

  // Lazy-load and attach 3D cutaway layer when view is "3d"
  const sync3DLayer = useCallback(async () => {
    const map = mapInstanceRef.current;
    if (!map || !canEditStyle(map)) return;

    if (view === "3d") {
      if (!cutawayLayerRef.current) {
        // Dynamically import CutawayLayer so it is NOT bundled into the initial / page load (PRD_FRONTEND §3.1b)
        const { CutawayLayer } = await import("../map3d/CutawayLayer");
        const layer = new CutawayLayer({
          region: regionRef.current,
          availableDepths: availableDepths ?? [0.5, 10, 20, 50, 100, 200, 500, 1000],
          colorRanges,
          isDark: !isLightMode,
          dotMode: gridStyle === "dots",
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
        layer.setDotMode(gridStyle === "dots");
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
    gridStyle,
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

  // The map is created once; its style handlers call the latest sync functions through refs.
  // (Depending on the callbacks directly tore the whole map down on every new slice.)
  const syncGridLayerRef = useRef(syncGridLayer);
  syncGridLayerRef.current = syncGridLayer;
  const sync3DLayerRef = useRef(sync3DLayer);
  sync3DLayerRef.current = sync3DLayer;
  const initialViewRef = useRef(view);
  const gridStyleRef = useRef(gridStyle);
  gridStyleRef.current = gridStyle;

  // Initialize MapLibre Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;
    const view = initialViewRef.current;

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
      zoom: introAtMountRef.current ? INTRO_ZOOM : 4,
      minZoom: 2,
      maxZoom: 10,
      pitch: view === "3d" ? 50 : 0,
      bearing: view === "3d" ? -20 : 0,
      maxPitch: view === "3d" ? 85 : 0,
      dragRotate: view === "3d",
      pitchWithRotate: true,
      touchPitch: view === "3d",
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");

    const handleStyleReady = () => {
      if (canEditStyle(map)) {
        applyLabelDensity(map, gridStyleRef.current);
        syncGridLayerRef.current();
        void sync3DLayerRef.current();
        syncOutlineRef.current();
      }
    };

    map.on("load", () => {
      setMapLoaded(true);
      handleStyleReady();
      if (!introAtMountRef.current) fitRegion(map, rgn, false);
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
  }, []);

  // Basemap label density follows the grid style
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded || !canEditStyle(map)) return;
    applyLabelDensity(map, gridStyle);
  }, [gridStyle, mapLoaded]);

  // Arrival intro: once the slice has drawn under the overlay, zoom out to the basin and fly into
  // the region, fading the grid in on the way down. Any drag, wheel or tap skips to the end.
  const introCallbacksRef = useRef({ onIntroReady, onIntroFlyIn, onIntroDone });
  introCallbacksRef.current = { onIntroReady, onIntroFlyIn, onIntroDone };
  const introCleanupRef = useRef<(() => void) | null>(null);
  const hasSlice = !!mapData?.values?.length;

  // Stage 1 (as soon as the map exists, in parallel with the slice fetch): frame the pulled-back
  // view so its tiles load under the overlay. Zooming *in* later can always fall back to these
  // lower-zoom tiles, so only the pulled-back view needs preloading.
  const introPlanRef = useRef<{
    home: { center: maplibregl.LngLat; zoom: number };
    outZoom: number;
    target: maplibregl.LngLat;
    targetZoom: number;
    outReady: Promise<void>;
  } | null>(null);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!introAtMountRef.current || introStartedRef.current || !map || !mapLoaded) return;

    const normalOpacity = () => (gridStyleRef.current === "dots" ? 0.96 : 0.9);
    const setGridOpacity = (v: number | null) => {
      gridOpacityRef.current = v;
      if (map.getLayer(layerId)) map.setPaintProperty(layerId, "raster-opacity", v ?? normalOpacity());
    };
    // Label fade-in (300 ms by default) delays every `idle`; off during the intro only.
    // `_fadeDuration` is MapLibre internal (typed in its .d.ts); restored when the intro ends.
    const fadeDuration = map._fadeDuration;
    const restoreFade = () => {
      map._fadeDuration = fadeDuration;
    };

    // The page gave up on the intro (e.g. the slice failed): just frame the region
    if (!intro) {
      introStartedRef.current = true;
      restoreFade();
      setGridOpacity(null);
      fitRegion(map, regionRef.current, true);
      return;
    }

    let cancelled = false;
    const waitIdle = (capMs = 2000) =>
      new Promise<void>((resolve) => {
        const t = setTimeout(resolve, capMs);
        map.once("idle", () => {
          clearTimeout(t);
          resolve();
        });
        map.triggerRepaint();
      });

    if (!introPlanRef.current) {
      map._fadeDuration = 0;
      const framed = map.cameraForBounds(regionBounds(regionRef.current), { padding: regionPadding(map) });
      const targetZoom = framed?.zoom ?? 4;
      const target = framed?.center ? maplibregl.LngLat.convert(framed.center) : map.getCenter();
      const outZoom = Math.max(map.getMinZoom(), targetZoom - INTRO_ZOOM_OUT);
      const home = { center: map.getCenter(), zoom: map.getZoom() };
      let outReady: Promise<void> = Promise.resolve();
      if (!prefersReducedMotion()) {
        map.jumpTo({ center: home.center, zoom: outZoom });
        outReady = waitIdle();
      }
      introPlanRef.current = { home, outZoom, target, targetZoom, outReady };
    }

    // Stage 2 (slice ready): back to open water, wait for that one frame, reveal, fly
    if (!hasSlice) return;
    introStartedRef.current = true;
    const plan = introPlanRef.current;

    let timeline: gsap.core.Timeline | null = null;
    const skip = () => timeline?.progress(1);
    const finish = () => {
      restoreFade();
      setGridOpacity(null);
      introCallbacksRef.current.onIntroDone?.();
    };

    void (async () => {
      await plan.outReady;
      if (cancelled) return;

      if (prefersReducedMotion()) {
        map.jumpTo({ center: plan.target, zoom: plan.targetZoom });
        introCallbacksRef.current.onIntroReady?.();
        introCallbacksRef.current.onIntroFlyIn?.();
        finish();
        return;
      }

      map.jumpTo(plan.home);
      await waitIdle();
      if (cancelled) return;
      introCallbacksRef.current.onIntroReady?.();

      const cam = { lng: plan.home.center.lng, lat: plan.home.center.lat, zoom: plan.home.zoom };
      const grid = { v: 0 };
      timeline = gsap.timeline({
        delay: 0.2, // the overlay starts fading first
        onUpdate: () => map.jumpTo({ center: [cam.lng, cam.lat], zoom: cam.zoom }),
        onComplete: finish,
      });
      timeline
        // Pull back from open water until the basin and its coasts are in view…
        .to(cam, { zoom: plan.outZoom, duration: 1.15, ease: "power2.inOut" })
        .call(() => introCallbacksRef.current.onIntroFlyIn?.())
        // …then dive back in onto the model region
        .to(cam, { lng: plan.target.lng, lat: plan.target.lat, zoom: plan.targetZoom, duration: 1.35, ease: "power3.inOut" }, ">-0.15")
        .to(
          grid,
          { v: normalOpacity(), duration: 0.7, ease: "power1.out", onUpdate: () => setGridOpacity(grid.v) },
          "<0.55"
        );

      map.on("mousedown", skip);
      map.on("wheel", skip);
      map.on("touchstart", skip);
    })();

    introCleanupRef.current = () => {
      cancelled = true;
      map.off("mousedown", skip);
      map.off("wheel", skip);
      map.off("touchstart", skip);
      timeline?.kill();
      restoreFade();
    };
  }, [intro, mapLoaded, hasSlice]);

  useEffect(() => () => introCleanupRef.current?.(), []);

  // Guide / time machine camera moves: one short animation (≤ 1.2 s), none under reduced motion
  const lastFocusIdRef = useRef<number | null>(null);
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded || !focus || lastFocusIdRef.current === focus.id) return;
    lastFocusIdRef.current = focus.id;
    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = reducedMotion ? 0 : 1200;
    if (focus.kind === "bbox") {
      map.fitBounds(
        [
          [focus.bbox.lon_min, focus.bbox.lat_min],
          [focus.bbox.lon_max, focus.bbox.lat_max],
        ],
        { padding: 48, duration, maxZoom: 7 }
      );
    } else {
      map.easeTo({
        center: [focus.lon, focus.lat],
        zoom: focus.zoom ?? Math.max(map.getZoom(), 4.5),
        ...(focus.bearing !== undefined ? { bearing: focus.bearing } : {}),
        duration,
      });
    }
  }, [focus, mapLoaded]);

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

    // One full turn every 40 s (the old loop's 0.15° per frame at 60 fps), as a GSAP tween on a
    // bearing proxy. Under reduced motion the tour doesn't spin.
    const proxy = { bearing: map.getBearing() };
    const tourTween = prefersReducedMotion()
      ? null
      : gsap.to(proxy, {
          bearing: "+=360",
          duration: 40,
          ease: "none",
          repeat: -1,
          onUpdate: () => map.setBearing(proxy.bearing % 360),
        });

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
      tourTween?.kill();
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
    const scale = gridStyle === "dots" ? 8 : 4;
    renderGridToCanvas(mapData.values, mapData.coverage, vmin, vmax, colormap, scale, canvasRef.current, {
      gridStyle,
      isDark: !isLightMode,
    });
    syncGridLayer();

    if (view === "3d" && cutawayLayerRef.current) {
      cutawayLayerRef.current.setDotMode(gridStyle === "dots");
      cutawayLayerRef.current.setMapData(mapData, vmin, vmax);
      mapInstanceRef.current?.triggerRepaint();
    }
  }, [mapData, vmin, vmax, gridStyle, isLightMode, syncGridLayer, view]);

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
  }, [depth_m, view, mapLoaded]);

  // Lasso outline: an ink-style line with a soft fill, re-added after a theme switch
  const outlineRef = useRef(outline);
  outlineRef.current = outline;
  const syncOutline = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map || !canEditStyle(map)) return;
    const ring = outlineRef.current;
    const data: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features:
        ring && ring.length >= 4
          ? [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } }]
          : [],
    };
    const src = map.getSource(LASSO_SOURCE) as maplibregl.GeoJSONSource | undefined;
    if (src) {
      src.setData(data);
    } else {
      map.addSource(LASSO_SOURCE, { type: "geojson", data });
      map.addLayer({
        id: `${LASSO_SOURCE}-fill`,
        type: "fill",
        source: LASSO_SOURCE,
        paint: { "fill-color": "#f8fafc", "fill-opacity": 0.18 },
      });
      map.addLayer({
        id: `${LASSO_SOURCE}-line`,
        type: "line",
        source: LASSO_SOURCE,
        paint: { "line-color": "#0f172a", "line-width": 2.5, "line-dasharray": [3, 1.5] },
      });
    }
  }, []);
  const syncOutlineRef = useRef(syncOutline);
  syncOutlineRef.current = syncOutline;
  useEffect(() => {
    if (mapLoaded) syncOutline();
  }, [outline, mapLoaded, syncOutline]);

  const pendingMarkerRef = useRef<maplibregl.Marker | null>(null);
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded || !pendingPoint) return;
    const el = document.createElement("div");
    el.className = "relative flex items-center justify-center pointer-events-none";
    el.innerHTML = `<span data-ping class="absolute inline-flex h-10 w-10 rounded-full border-2 border-neutral-900 dark:border-white"></span>`;
    const marker = new maplibregl.Marker({ element: el }).setLngLat([pendingPoint.lon, pendingPoint.lat]).addTo(map);
    pendingMarkerRef.current = marker;
    const ping = loopPing(el.querySelector("[data-ping]")!);
    return () => {
      ping?.kill();
      marker.remove();
      if (pendingMarkerRef.current === marker) pendingMarkerRef.current = null;
    };
  }, [mapLoaded, pendingPoint]);

  // Update selection marker (HUD status badge pin)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapLoaded) return;

    if (selectedLat !== null && selectedLon !== null) {
      if (!selectionMarkerRef.current) {
        const el = document.createElement("div");
        el.className = "relative flex items-center justify-center cursor-pointer";
        el.setAttribute("aria-hidden", "true");
        el.innerHTML = `
          <span data-ping class="absolute inline-flex h-8 w-8 rounded-full bg-cyan-400/40"></span>
          <span data-pin class="relative inline-flex items-center justify-center rounded-full h-7 w-7 bg-[#111314]/90 ring-1 ring-white/20 shadow-[0_4px_16px_rgba(0,0,0,0.5)]">
            <span class="h-2.5 w-2.5 rounded-full bg-cyan-300 shadow-[0_0_8px_rgba(103,232,249,0.9)]"></span>
          </span>
        `;
        selectionMarkerRef.current = new maplibregl.Marker({ element: el })
          .setLngLat([selectedLon, selectedLat])
          .addTo(map);
        selectionPingRef.current = loopPing(el.querySelector("[data-ping]")!);
      } else {
        selectionMarkerRef.current.setLngLat([selectedLon, selectedLat]);
      }
      const pin = selectionMarkerRef.current.getElement().querySelector("[data-pin]");
      if (pin && !prefersReducedMotion()) {
        gsap.fromTo(pin, { scale: 0.4, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.45, ease: "back.out(2.2)", overwrite: "auto" });
      }
    } else if (selectionMarkerRef.current) {
      selectionPingRef.current?.kill();
      selectionPingRef.current = null;
      selectionMarkerRef.current.remove();
      selectionMarkerRef.current = null;
    }
  }, [mapLoaded, selectedLat, selectedLon]);

  // Update ARGO markers (HUD circular sensor pins)
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
          <div data-argo class="h-6 w-6 rounded-full bg-[#111314]/90 ring-1 ring-white/15 shadow-[0_4px_12px_rgba(0,0,0,0.5)] flex items-center justify-center transition-transform group-hover:scale-110">
            <span class="h-2 w-2 rounded-full bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.9)]"></span>
          </div>
        `;
        const marker = new maplibregl.Marker({ element: el })
          .setLngLat([p.lon, p.lat])
          .setPopup(
            new maplibregl.Popup({ offset: 12, closeButton: false }).setHTML(`
              <div class="text-xs leading-relaxed text-neutral-900 dark:text-neutral-100">
                <p class="font-semibold">ARGO float ${p.platform_number}</p>
                <p class="text-neutral-600 dark:text-neutral-400">Cycle ${p.cycle} · ${p.n_obs} obs · ${p.time.slice(0, 10)}</p>
              </div>
            `)
          )
          .addTo(map);

        argoMarkersRef.current.push(marker);
      });
      if (!prefersReducedMotion()) {
        const pins = argoMarkersRef.current.map((m) => m.getElement().querySelector("[data-argo]"));
        gsap.fromTo(
          pins,
          { scale: 0, opacity: 0 },
          { scale: 1, opacity: 1, duration: 0.35, ease: "back.out(2)", stagger: { amount: 0.4 }, clearProps: "transform,opacity" }
        );
      }
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
      className={`absolute inset-0 overflow-hidden select-none bg-neutral-100 dark:bg-[#0a0c0d] ${className}`}
      style={{ viewTransitionName: view === "3d" ? "ocean-section" : undefined }}
    >
      <div ref={mapContainerRef} className="absolute inset-0" />

      {/* Subtle edge vignette in dark dot-matrix mode so the ocean matrix blends into the HUD frame */}
      {!isLightMode && gridStyle === "dots" && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-10"
          style={{
            background:
              "radial-gradient(ellipse at 55% 45%, transparent 50%, rgba(10, 12, 13, 0.7) 100%)",
          }}
        />
      )}

      {/* Hover tooltip */}
      {hoverInfo && (
        <div
          className="hud-glass pointer-events-none absolute z-30 -translate-x-1/2 -translate-y-full rounded-xl px-3 py-2 text-xs"
          style={{ left: hoverInfo.x, top: hoverInfo.y - 14 }}
        >
          <div className="flex items-center gap-1.5 text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
            <span>{formatCoord(hoverInfo.lat, hoverInfo.lon)}</span>
            {view === "3d" && hoverInfo.depth_m !== undefined && (
              <span className="font-medium text-cyan-700 dark:text-cyan-300">· {formatDepth(hoverInfo.depth_m)}</span>
            )}
          </div>
          {hoverInfo.coverage === 0 ? (
            <div className="mt-0.5 text-[11px] font-medium text-neutral-500">Land / Unmodelled</div>
          ) : hoverInfo.coverage === 2 ? (
            <div className="mt-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-400">
              No prediction (insufficient input data)
            </div>
          ) : hoverInfo.value === null ? (
            <div className="mt-0.5 text-[11px] font-medium text-neutral-500">No value at this depth</div>
          ) : (
            <div className="mt-0.5 flex items-baseline gap-1.5">
              <span className="text-sm font-semibold tabular-nums text-neutral-950 dark:text-white">
                {formatTemp(hoverInfo.value, 1)}
              </span>
              <span className="text-[10px] text-neutral-500 dark:text-neutral-400">{sourceLabel}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
