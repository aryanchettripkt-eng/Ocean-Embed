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
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <ArrowDownCircle className="h-3.5 w-3.5 text-cyan-400" />
          <span>Depth</span>
        </label>
        <span className="text-xs font-mono font-medium text-cyan-300 bg-cyan-950/60 border border-cyan-800/60 px-2 py-0.5 rounded">
          {formatDepth(selectedDepth)} (Level {activeIdx + 1}/15)
        </span>
      </div>

      {/* Slider selector for smooth scrubbing */}
      <div className="px-1">
        <input
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
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 hover:accent-cyan-300 transition-all"
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
              onClick={() => onChange(d)}
              className={`shrink-0 px-2 py-1 rounded text-[11px] font-mono transition-all ${
                isSelected
                  ? "bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/20"
                  : "bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800/80"
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
