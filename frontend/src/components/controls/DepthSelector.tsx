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

  const fillPct = `${depths.length > 1 ? (activeIdx / (depths.length - 1)) * 100 : 0}%`;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between">
        <label htmlFor="explorer-depth-slider" className="hud-label flex items-center gap-1.5">
          <ArrowDownCircle aria-hidden="true" className="h-3.5 w-3.5" />
          <span>Depth</span>
        </label>
        <span className="text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
          <span className="font-semibold text-neutral-900 dark:text-white">{formatDepth(selectedDepth)}</span>
          <span className="ml-1">
            level {activeIdx + 1}/{depths.length}
          </span>
        </span>
      </div>

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
        className="hud-range"
        style={{ "--fill": fillPct } as React.CSSProperties}
      />

      {/* Every model level at a glance */}
      <div className="grid grid-cols-5 gap-1">
        {depths.map((d, idx) => {
          const isSelected = idx === activeIdx;
          return (
            <button
              key={d}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onChange(d)}
              className={`h-7 rounded-full px-1 text-[11px] tabular-nums transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                isSelected
                  ? "bg-neutral-900 font-semibold text-white dark:bg-white dark:text-neutral-950"
                  : "text-neutral-600 hover:bg-black/[0.05] hover:text-neutral-950 dark:text-neutral-400 dark:hover:bg-white/[0.08] dark:hover:text-white"
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
