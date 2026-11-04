"use client";

import React from "react";
import type { GridStyle } from "../../lib/url-state";
import { getColormapGradientCss } from "../../lib/colormaps";

interface CoverageLegendProps {
  gridStyle?: GridStyle;
  className?: string;
}

export function CoverageLegend({ gridStyle = "dots", className = "" }: CoverageLegendProps) {
  const isDots = gridStyle === "dots";

  const item = "flex items-center gap-2";
  const oceanSwatch = { background: getColormapGradientCss("thermal") };
  const swatch = "flex h-3.5 w-3.5 shrink-0 items-center justify-center";

  return (
    <div className={`text-[11px] text-neutral-600 dark:text-neutral-400 ${className}`}>
      <div className="hud-label mb-1.5">Coverage</div>
      <ul className="flex flex-col gap-1.5">
        <li className={item}>
          {isDots ? (
            <span aria-hidden="true" data-testid="legend-ocean-dot" className={swatch}>
              <span className="h-3 w-3 rounded-full" style={oceanSwatch} />
            </span>
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-ocean-solid"
              className="h-3.5 w-3.5 shrink-0 rounded-[3px]"
              style={oceanSwatch}
            />
          )}
          <span>Valid ocean reconstruction</span>
        </li>
        <li className={item}>
          {isDots ? (
            <span aria-hidden="true" data-testid="legend-insufficient-ring" className={swatch}>
              <span className="h-3 w-3 rounded-full border-2 border-slate-500 dark:border-slate-400" />
            </span>
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-insufficient-hatch"
              className="h-3.5 w-3.5 shrink-0 rounded-[3px]"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, #6e7887, #6e7887 2px, #282d37 2px, #282d37 4px)",
              }}
            />
          )}
          <span>Insufficient inputs (&lt;7 of 9 valid)</span>
        </li>
        <li className={item}>
          {isDots ? (
            <span aria-hidden="true" data-testid="legend-land-dot" className={swatch}>
              <span className="h-1.5 w-1.5 rounded-full bg-[#948a7c] dark:bg-[#7e674c]" />
            </span>
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-land-solid"
              className="h-3.5 w-3.5 shrink-0 rounded-[3px] border border-dashed border-neutral-400 dark:border-neutral-600"
            />
          )}
          <span>Land, or no value at this depth</span>
        </li>
      </ul>
    </div>
  );
}
