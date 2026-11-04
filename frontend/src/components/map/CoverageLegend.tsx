"use client";

import React from "react";
import type { GridStyle } from "../../lib/url-state";

interface CoverageLegendProps {
  gridStyle?: GridStyle;
  className?: string;
}

export function CoverageLegend({ gridStyle = "dots", className = "" }: CoverageLegendProps) {
  const isDots = gridStyle === "dots";

  return (
    <div
      className={`bg-white/88 dark:bg-[#0b1116]/88 backdrop-blur-xl border border-slate-200/90 dark:border-white/10 rounded-xl p-2.5 shadow-xl text-slate-700 dark:text-slate-200 text-xs ${className}`}
    >
      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1.5">
        Coverage Legend
      </div>
      <div className="flex flex-col gap-1.5 text-[11px] text-slate-700 dark:text-slate-300">
        <div className="flex items-center gap-2">
          {isDots ? (
            <span
              aria-hidden="true"
              data-testid="legend-ocean-dot"
              className="w-3.5 h-3.5 rounded-full border border-cyan-400/70 bg-gradient-to-tr from-blue-600 via-teal-400 to-amber-400 shadow-sm shadow-cyan-500/30 shrink-0"
            />
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-ocean-solid"
              className="w-3.5 h-3.5 rounded border border-cyan-500/60 bg-gradient-to-r from-blue-600 to-amber-500 shrink-0"
            />
          )}
          <span>Valid ocean reconstruction</span>
        </div>
        <div className="flex items-center gap-2">
          {isDots ? (
            <span
              aria-hidden="true"
              data-testid="legend-insufficient-ring"
              className="w-3.5 h-3.5 flex items-center justify-center shrink-0"
            >
              <span className="w-3 h-3 rounded-full border-2 border-slate-500 dark:border-slate-400 bg-transparent" />
            </span>
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-insufficient-hatch"
              className="w-3.5 h-3.5 rounded border border-slate-500 dark:border-slate-600 shrink-0"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, #475569, #475569 2px, #1e293b 2px, #1e293b 4px)",
              }}
            />
          )}
          <span>Insufficient inputs (&lt;7 of 9 valid)</span>
        </div>
        <div className="flex items-center gap-2">
          {isDots ? (
            <span
              aria-hidden="true"
              data-testid="legend-land-dot"
              className="w-3.5 h-3.5 flex items-center justify-center shrink-0"
            >
              <span className="w-2 h-2 rounded-full bg-[#7e674c] dark:bg-[#8a7052]" />
            </span>
          ) : (
            <span
              aria-hidden="true"
              data-testid="legend-land-solid"
              className="w-3.5 h-3.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-950/80 shrink-0"
            />
          )}
          <span>Land / Unmodelled region</span>
        </div>
      </div>
    </div>
  );
}
