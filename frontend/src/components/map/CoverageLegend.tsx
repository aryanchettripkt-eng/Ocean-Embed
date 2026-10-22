"use client";

import React from "react";

interface CoverageLegendProps {
  className?: string;
}

export function CoverageLegend({ className = "" }: CoverageLegendProps) {
  return (
    <div
      className={`bg-slate-950/85 backdrop-blur-md border border-slate-800/80 rounded-lg p-2.5 shadow-xl text-slate-200 text-xs ${className}`}
    >
      <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
        Coverage Legend
      </div>
      <div className="flex flex-col gap-1.5 text-[11px] text-slate-300">
        <div className="flex items-center gap-2">
          <span className="w-3.5 h-3.5 rounded border border-cyan-500/60 bg-gradient-to-r from-blue-600 to-amber-500 shrink-0" />
          <span>Valid ocean reconstruction</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="w-3.5 h-3.5 rounded border border-slate-600 shrink-0"
            style={{
              backgroundImage:
                "repeating-linear-gradient(45deg, #475569, #475569 2px, #1e293b 2px, #1e293b 4px)",
            }}
          />
          <span>Insufficient inputs (&lt;7 of 9 valid)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3.5 h-3.5 rounded border border-slate-700 bg-slate-950/80 shrink-0" />
          <span>Land / Unmodelled region</span>
        </div>
      </div>
    </div>
  );
}
