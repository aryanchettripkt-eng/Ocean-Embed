"use client";

import React, { useEffect, useState } from "react";
import { Box, Layers, Play, RotateCw, Sparkles, Map as MapIcon } from "lucide-react";
import { isWebGL2Supported } from "./webglUtils";

interface ViewSwitchProps {
  view: "map" | "3d";
  onChangeView: (view: "map" | "3d") => void;
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
      className="flex flex-col gap-2 pointer-events-auto"
      onMouseDown={(e) => e.stopPropagation()}
      onTouchStart={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {/* Primary View Segmented Toggle */}
      <div className="flex items-center p-1 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg">
        <button
          type="button"
          onClick={() => onChangeView("map")}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            view === "map"
              ? "bg-cyan-500 text-white font-semibold shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
          }`}
        >
          <MapIcon className="h-3.5 w-3.5" />
          <span>2D Map</span>
        </button>

        <button
          type="button"
          disabled={!canUse3D}
          onClick={() => canUse3D && onChangeView("3d")}
          title={fallbackReason ?? "Switch to 3D Ocean Cutaway"}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            view === "3d"
              ? "bg-cyan-500 text-white font-semibold shadow-sm"
              : canUse3D
              ? "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              : "text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-60"
          }`}
        >
          <Box className="h-3.5 w-3.5" />
          <span>3D Cutaway</span>
        </button>
      </div>

      {/* 3D-specific toolbar controls */}
      {view === "3d" && (
        <div className="flex flex-wrap items-center gap-1.5 p-1.5 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg text-xs animate-in fade-in duration-200">
          {onToggleTour && (
            <button
              type="button"
              onClick={onToggleTour}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
                isTouring
                  ? "bg-amber-500 text-slate-950 font-semibold"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
              title="Automatically rotate around the ocean block"
            >
              {isTouring ? <RotateCw className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
              <span>Tour</span>
            </button>
          )}

          {onToggleExploded && (
            <button
              type="button"
              onClick={onToggleExploded}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
                isExploded
                  ? "bg-cyan-500 text-white font-semibold"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
              title="Explode into stacked depth sheets"
            >
              <Layers className="h-3 w-3" />
              <span>Exploded</span>
            </button>
          )}

          {onToggleSmoothed && (
            <button
              type="button"
              onClick={onToggleSmoothed}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
                isSmoothed
                  ? "bg-cyan-500 text-white font-semibold"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
              title="Toggle between linear smoothed display and exact raw grid cells"
            >
              <Sparkles className="h-3 w-3" />
              <span>{isSmoothed ? "Smoothed" : "Exact cells"}</span>
            </button>
          )}

          {onToggleContours && (
            <button
              type="button"
              onClick={onToggleContours}
              className={`px-2 py-1 rounded-md transition-colors ${
                showContours
                  ? "bg-cyan-500 text-white font-semibold"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700"
              }`}
              title="Toggle 2°C contours and 20°C isotherm lines"
            >
              <span>20°C Isotherm</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
