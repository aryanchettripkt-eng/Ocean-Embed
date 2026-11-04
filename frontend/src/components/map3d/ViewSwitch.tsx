"use client";

import React, { useEffect, useState } from "react";
import { Box, Grip, Grid2x2, Layers, Play, RotateCw, Sparkles, Map as MapIcon, Waves } from "lucide-react";
import type { GridStyle } from "../../lib/url-state";
import { isWebGL2Supported } from "./webglUtils";
import { Presence, Spin } from "../common/Motion";

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
  /** Extra toolbar buttons (layers, Meta tools) rendered inside the same pill after a divider. */
  children?: React.ReactNode;
}

/** Thin vertical rule between toolbar groups. */
export function ToolbarDivider() {
  return <span aria-hidden="true" className="mx-0.5 h-4 w-px shrink-0 bg-black/10 dark:bg-white/10" />;
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
  children,
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

  const btn = (active: boolean) => `hud-btn ${active ? "hud-btn-active" : ""}`;

  return (
    <div
      className="pointer-events-auto flex max-w-full flex-col items-center gap-2"
      onMouseDown={(e) => e.stopPropagation()}
      onTouchStart={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {/* 3D-only options float just above the main toolbar */}
      <Presence show={view === "3d"} from="bottom" className="max-w-full">
        <div
          role="group"
          aria-label="3D options"
          className="hud-glass flex max-w-full items-center gap-0.5 overflow-x-auto rounded-full p-1"
        >
          {onToggleTour && (
            <button
              type="button"
              aria-pressed={isTouring}
              onClick={onToggleTour}
              className={btn(isTouring)}
              title="Automatically rotate around the ocean block"
            >
              {isTouring ? (
                <Spin>
                  <RotateCw aria-hidden="true" className="h-3.5 w-3.5" />
                </Spin>
              ) : (
                <Play aria-hidden="true" className="h-3.5 w-3.5" />
              )}
              <span>Tour</span>
            </button>
          )}
          {onToggleExploded && (
            <button
              type="button"
              aria-pressed={isExploded}
              onClick={onToggleExploded}
              className={btn(isExploded)}
              title="Explode into stacked depth sheets"
            >
              <Layers aria-hidden="true" className="h-3.5 w-3.5" />
              <span>Exploded</span>
            </button>
          )}
          {onToggleSmoothed && (
            <button
              type="button"
              aria-pressed={isSmoothed}
              onClick={onToggleSmoothed}
              className={btn(isSmoothed)}
              title="Toggle between linear smoothed display and exact raw grid cells"
            >
              <Sparkles aria-hidden="true" className="h-3.5 w-3.5" />
              <span>{isSmoothed ? "Smoothed" : "Exact cells"}</span>
            </button>
          )}
          {onToggleContours && (
            <button
              type="button"
              aria-pressed={showContours}
              onClick={onToggleContours}
              className={btn(showContours)}
              title="Toggle 2°C contours and 20°C isotherm lines"
            >
              <Waves aria-hidden="true" className="h-3.5 w-3.5" />
              <span>20°C Isotherm</span>
            </button>
          )}
        </div>
      </Presence>

      {/* Main toolbar: view, grid style, then the page's tools */}
      <div
        role="toolbar"
        aria-label="Map view and tools"
        className="hud-glass flex max-w-full items-center gap-0.5 overflow-x-auto rounded-full p-1 [scrollbar-width:none]"
      >
        <button
          type="button"
          aria-pressed={view === "map"}
          aria-label="2D Map"
          onClick={() => onChangeView("map")}
          className={btn(view === "map")}
          title="Flat map"
        >
          <MapIcon aria-hidden="true" className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">2D Map</span>
        </button>
        <button
          type="button"
          disabled={!canUse3D}
          aria-pressed={view === "3d"}
          aria-label="3D Cutaway"
          onClick={() => canUse3D && onChangeView("3d")}
          title={fallbackReason ?? "Switch to 3D Ocean Cutaway"}
          className={btn(view === "3d")}
        >
          <Box aria-hidden="true" className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">3D Cutaway</span>
        </button>

        {onChangeGridStyle && (
          <>
            <ToolbarDivider />
            <button
              type="button"
              aria-pressed={gridStyle === "dots"}
              aria-label="Dots"
              onClick={() => onChangeGridStyle("dots")}
              title="Dot-matrix grid: one dot per 0.25° cell"
              className={btn(gridStyle === "dots")}
            >
              <Grip aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Dots</span>
            </button>
            <button
              type="button"
              aria-pressed={gridStyle === "solid"}
              aria-label="Classic"
              onClick={() => onChangeGridStyle("solid")}
              title="Classic continuous grid"
              className={btn(gridStyle === "solid")}
            >
              <Grid2x2 aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Classic</span>
            </button>
          </>
        )}

        {children}
      </div>
    </div>
  );
}
