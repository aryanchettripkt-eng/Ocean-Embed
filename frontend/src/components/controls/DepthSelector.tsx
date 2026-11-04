"use client";

import React from "react";
import { ArrowDownCircle } from "lucide-react";
import { formatDepth } from "../../lib/format";

interface DepthSelectorProps {
  depths: number[];
  selectedDepth: number;
  onChange: (depth: number) => void;
  className?: string;
}

export function DepthSelector({
  depths,
  selectedDepth,
  onChange,
  className = "",
}: DepthSelectorProps) {
  const currentIndex = depths.findIndex((d) => Math.abs(d - selectedDepth) < 0.05);
  const activeIdx = currentIndex >= 0 ? currentIndex : 0;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between">
        <label
          htmlFor="explorer-depth-slider"
          className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5"
        >
          <ArrowDownCircle aria-hidden="true" className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
          <span>Depth</span>
        </label>
        <span className="text-[11px] font-mono font-medium text-cyan-700 dark:text-cyan-300 bg-cyan-500/12 border border-cyan-500/30 px-2 py-0.5 rounded-full">
          {formatDepth(selectedDepth)} ({activeIdx + 1}/{depths.length})
        </span>
      </div>

      {/* Slider selector for smooth scrubbing */}
      <div className="px-0.5">
        <input
          id="explorer-depth-slider"
          type="range"
          min={0}
          max={depths.length - 1}
          step={1}
          value={activeIdx}
          onChange={(e) => {
            const idx = parseInt(e.target.value, 10);
            if (depths[idx] !== undefined) {
              onChange(depths[idx]);
            }
          }}
          className="w-full h-1.5 bg-slate-200 dark:bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-500 dark:accent-cyan-400 transition-all"
        />
      </div>

      {/* Quick selection chips */}
      <div className="flex gap-1 overflow-x-auto pb-1 pt-0.5 scrollbar-thin scrollbar-thumb-slate-700">
        {depths.map((d, idx) => {
          const isSelected = idx === activeIdx;
          return (
            <button
              key={d}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onChange(d)}
              className={`shrink-0 px-2 py-1 rounded-full text-[11px] font-mono transition-all ${
                isSelected
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/25"
                  : "bg-slate-100 dark:bg-white/[0.04] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-200/70 dark:hover:bg-white/10 border border-slate-200/80 dark:border-white/10"
              }`}
            >
              {formatDepth(d)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
