"use client";

import React, { useEffect, useState } from "react";
import { Box, Grip, Grid2x2, Layers, Play, RotateCw, Sparkles, Map as MapIcon } from "lucide-react";
import type { GridStyle } from "../../lib/url-state";
import { isWebGL2Supported } from "./webglUtils";

interface ViewSwitchProps {
  view: "map" | "3d";
  onChangeView: (view: "map" | "3d") => void;
  gridStyle?: GridStyle;
  onChangeGridStyle?: (style: GridStyle) => void;
  isTouring?: boolean;
  onToggleTour?: () => void;
  isExploded?: boolean;
  onToggleExploded?: () => void;
  isSmoothed?: boolean;
  onToggleSmoothed?: () => void;
  showContours?: boolean;
  onToggleContours?: () => void;
}

export function ViewSwitch({
  view,
  onChangeView,
  gridStyle = "dots",
  onChangeGridStyle,
  isTouring = false,
  onToggleTour,
  isExploded = false,
  onToggleExploded,
  isSmoothed = false,
  onToggleSmoothed,
  showContours = true,
  onToggleContours,
}: ViewSwitchProps) {
  const [canUse3D, setCanUse3D] = useState(true);
  const [fallbackReason, setFallbackReason] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const checkSupport = () => {
      if (window.innerWidth < 768) {
        setCanUse3D(false);
        setFallbackReason("3D cutaway requires a tablet or desktop display (≥ 768px).");
      } else if (!isWebGL2Supported()) {
        setCanUse3D(false);
        setFallbackReason("3D cutaway requires WebGL2 support.");
      } else {
        setCanUse3D(true);
        setFallbackReason(null);
      }
    };

    checkSupport();
    window.addEventListener("resize", checkSupport);
    return () => window.removeEventListener("resize", checkSupport);
  }, []);

  return (
    <div
      className="flex flex-col items-center gap-1.5 pointer-events-auto"
      onMouseDown={(e) => e.stopPropagation()}
      onTouchStart={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {/* 3D-specific secondary controls pill above main switch when active */}
      {view === "3d" && (
        <div className="flex flex-wrap items-center justify-center gap-1 p-1 bg-white/90 dark:bg-[#0b1116]/90 backdrop-blur-xl border border-slate-300/80 dark:border-white/12 rounded-full shadow-2xl text-xs animate-in fade-in duration-200">
          {onToggleTour && (
            <button
              type="button"
              aria-pressed={isTouring}
              onClick={onToggleTour}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                isTouring
                  ? "bg-amber-500 text-slate-950 font-semibold"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
              title="Automatically rotate around the ocean block"
            >
              {isTouring ? (
                <RotateCw aria-hidden="true" className="h-3 w-3 animate-spin motion-reduce:animate-none" />
              ) : (
                <Play aria-hidden="true" className="h-3 w-3" />
              )}
              <span>Tour</span>
            </button>
          )}

          {onToggleExploded && (
            <button
              type="button"
              aria-pressed={isExploded}
              onClick={onToggleExploded}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                isExploded
                  ? "bg-cyan-500 text-slate-950 font-semibold"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
              title="Explode into stacked depth sheets"
            >
              <Layers aria-hidden="true" className="h-3 w-3" />
              <span>Exploded</span>
            </button>
          )}

          {onToggleSmoothed && (
            <button
              type="button"
              aria-pressed={isSmoothed}
              onClick={onToggleSmoothed}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                isSmoothed
                  ? "bg-cyan-500 text-slate-950 font-semibold"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
              title="Toggle between linear smoothed display and exact raw grid cells"
            >
              <Sparkles aria-hidden="true" className="h-3 w-3" />
              <span>{isSmoothed ? "Smoothed" : "Exact cells"}</span>
            </button>
          )}

          {onToggleContours && (
            <button
              type="button"
              aria-pressed={showContours}
              onClick={onToggleContours}
              className={`px-2.5 py-1 rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                showContours
                  ? "bg-cyan-500 text-slate-950 font-semibold"
                  : "text-slate-700 dark:text-slate-300 hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
              title="Toggle 2°C contours and 20°C isotherm lines"
            >
              <span>20°C Isotherm</span>
            </button>
          )}
        </div>
      )}

      {/* Primary View & Grid Style Segmented Capsule */}
      <div className="flex items-center gap-1 p-1 bg-white/90 dark:bg-[#0b1116]/90 backdrop-blur-xl border border-slate-300/80 dark:border-white/12 rounded-full shadow-2xl">
        <button
          type="button"
          aria-pressed={view === "map"}
          onClick={() => onChangeView("map")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
            view === "map"
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950 font-semibold shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
          }`}
        >
          <MapIcon aria-hidden="true" className="h-3.5 w-3.5" />
          <span>2D Map</span>
        </button>

        <button
          type="button"
          disabled={!canUse3D}
          aria-pressed={view === "3d"}
          onClick={() => canUse3D && onChangeView("3d")}
          title={fallbackReason ?? "Switch to 3D Ocean Cutaway"}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
            view === "3d"
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-950 font-semibold shadow-sm"
              : canUse3D
              ? "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              : "text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-60"
          }`}
        >
          <Box aria-hidden="true" className="h-3.5 w-3.5" />
          <span>3D Cutaway</span>
        </button>

        {onChangeGridStyle && (
          <>
            <span aria-hidden="true" className="h-4 w-px bg-slate-300 dark:bg-white/15 mx-0.5" />
            <button
              type="button"
              aria-pressed={gridStyle === "dots"}
              onClick={() => onChangeGridStyle("dots")}
              title="Dot-matrix grid display"
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-full text-xs font-medium transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                gridStyle === "dots"
                  ? "bg-cyan-500 text-slate-950 font-semibold shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              }`}
            >
              <Grip aria-hidden="true" className="h-3.5 w-3.5" />
              <span>Dots</span>
            </button>
            <button
              type="button"
              aria-pressed={gridStyle === "solid"}
              onClick={() => onChangeGridStyle("solid")}
              title="Classic continuous grid display"
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-full text-xs font-medium transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                gridStyle === "solid"
                  ? "bg-cyan-500 text-slate-950 font-semibold shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              }`}
            >
              <Grid2x2 aria-hidden="true" className="h-3.5 w-3.5" />
              <span>Classic</span>
            </button>
          </>
        )}
      </div>
    </div>
  );
}
