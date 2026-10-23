"use client";

import React, { useCallback, useMemo } from "react";
import { getDepthTicks, type DepthTick } from "./depthScale";
import { formatDepth } from "../../lib/format";

interface DepthRailProps {
  availableDepths: number[];
  selectedDepth: number;
  onChangeDepth: (depth_m: number) => void;
}

export function DepthRail({
  availableDepths,
  selectedDepth,
  onChangeDepth,
}: DepthRailProps) {
  const ticks = useMemo(() => getDepthTicks(availableDepths), [availableDepths]);
  const maxDepth = useMemo(
    () => Math.max(...availableDepths, 1000),
    [availableDepths]
  );

  // Position ratio for the selected depth on sqrt scale: 0 (top/surface) to 1 (bottom/floor)
  const currentRatio = useMemo(() => {
    return Math.sqrt(Math.max(0, selectedDepth)) / Math.sqrt(maxDepth);
  }, [selectedDepth, maxDepth]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      const currentIndex = availableDepths.findIndex(
        (d) => Math.abs(d - selectedDepth) < 0.1
      );
      if (currentIndex === -1) return;

      if (e.key === "ArrowUp") {
        e.preventDefault();
        if (currentIndex > 0) onChangeDepth(availableDepths[currentIndex - 1]);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        if (currentIndex < availableDepths.length - 1)
          onChangeDepth(availableDepths[currentIndex + 1]);
      } else if (e.key === "PageUp") {
        e.preventDefault();
        const nextIdx = Math.max(0, currentIndex - 3);
        onChangeDepth(availableDepths[nextIdx]);
      } else if (e.key === "PageDown") {
        e.preventDefault();
        const nextIdx = Math.min(availableDepths.length - 1, currentIndex + 3);
        onChangeDepth(availableDepths[nextIdx]);
      } else if (e.key === "Home") {
        e.preventDefault();
        onChangeDepth(availableDepths[0]);
      } else if (e.key === "End") {
        e.preventDefault();
        onChangeDepth(availableDepths[availableDepths.length - 1]);
      }
    },
    [availableDepths, selectedDepth, onChangeDepth]
  );

  // Handle mouse wheel over rail
  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      e.preventDefault();
      const currentIndex = availableDepths.findIndex(
        (d) => Math.abs(d - selectedDepth) < 0.1
      );
      if (currentIndex === -1) return;

      if (e.deltaY > 0 && currentIndex < availableDepths.length - 1) {
        onChangeDepth(availableDepths[currentIndex + 1]);
      } else if (e.deltaY < 0 && currentIndex > 0) {
        onChangeDepth(availableDepths[currentIndex - 1]);
      }
    },
    [availableDepths, selectedDepth, onChangeDepth]
  );

  return (
    <div
      tabIndex={0}
      role="slider"
      aria-label="Ocean Depth Rail"
      aria-valuemin={availableDepths[0] || 0}
      aria-valuemax={maxDepth}
      aria-valuenow={selectedDepth}
      aria-valuetext={formatDepth(selectedDepth)}
      onKeyDown={handleKeyDown}
      onWheel={handleWheel}
      className="relative flex items-center justify-between py-2 px-3 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border border-slate-200 dark:border-slate-800 rounded-xl shadow-xl select-none outline-none focus:ring-2 focus:ring-cyan-500 transition-all cursor-pointer h-64 w-32"
    >
      {/* Ticks and labels on the left */}
      <div className="relative flex-1 h-full flex flex-col justify-between text-[11px] font-mono text-slate-500 dark:text-slate-400">
        {ticks.map((tick: DepthTick) => {
          const isSelected = Math.abs(tick.depth_m - selectedDepth) < 0.1;
          const topPercent = tick.ratio * 100;
          return (
            <div
              key={tick.depth_m}
              onClick={() => onChangeDepth(tick.depth_m)}
              style={{ top: `${topPercent}%` }}
              className={`absolute left-0 -translate-y-1/2 flex items-center gap-1.5 transition-colors hover:text-cyan-600 dark:hover:text-cyan-400 ${
                isSelected
                  ? "text-cyan-600 dark:text-cyan-300 font-bold"
                  : "text-slate-500 dark:text-slate-400"
              }`}
            >
              <div
                className={`w-2 h-0.5 rounded-full ${
                  isSelected ? "bg-cyan-500 dark:bg-cyan-400 w-3" : "bg-slate-300 dark:bg-slate-700"
                }`}
              />
              <span>{tick.label}</span>
            </div>
          );
        })}
      </div>

      {/* Vertical Rail track on the right */}
      <div className="relative w-1.5 h-full bg-slate-200 dark:bg-slate-800 rounded-full ml-2">
        {/* Fill down to current ratio */}
        <div
          className="absolute top-0 left-0 w-full bg-cyan-500/40 dark:bg-cyan-400/40 rounded-full"
          style={{ height: `${currentRatio * 100}%` }}
        />

        {/* Draggable/Current Depth Indicator Thumb */}
        <div
          style={{ top: `${currentRatio * 100}%` }}
          className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-cyan-500 dark:bg-cyan-400 border-2 border-white dark:border-slate-900 rounded-full shadow-md transition-all pointer-events-none"
        />
      </div>
    </div>
  );
}
