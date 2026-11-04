"use client";

import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { getDepthTicks, type DepthTick } from "./depthScale";
import { formatDepth } from "../../lib/format";
import { gsap, prefersReducedMotion, EASE_OUT } from "../../lib/motion";

interface DepthRailProps {
  availableDepths: number[];
  selectedDepth: number;
  onChangeDepth: (depth_m: number) => void;
}

/** Wheel distance (px) that counts as one level, so a trackpad swipe doesn't skip the whole column. */
const WHEEL_STEP_PX = 60;
/** Tick labels closer than this (as a fraction of the rail) to the current-value pill are hidden. */
const LABEL_CLEARANCE = 0.07;

/** Rail position (0 = surface, 1 = deepest) on the same sqrt(depth) scale as the tick labels. */
export function depthToRailRatio(depth_m: number, maxDepth_m: number): number {
  if (maxDepth_m <= 0) return 0;
  return Math.min(1, Math.sqrt(Math.max(0, depth_m)) / Math.sqrt(maxDepth_m));
}

/** Index of the model level nearest to a rail position. */
export function nearestLevelIndex(ratio: number, depths: number[], maxDepth_m: number): number {
  let best = 0;
  let bestDiff = Infinity;
  depths.forEach((d, i) => {
    const diff = Math.abs(depthToRailRatio(d, maxDepth_m) - ratio);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = i;
    }
  });
  return best;
}

/** Index of the model level nearest to a depth (the selected depth may not be exactly a level). */
export function nearestDepthIndex(depth_m: number, depths: number[]): number {
  let best = 0;
  depths.forEach((d, i) => {
    if (Math.abs(d - depth_m) < Math.abs(depths[best] - depth_m)) best = i;
  });
  return best;
}

export function DepthRail({ availableDepths, selectedDepth, onChangeDepth }: DepthRailProps) {
  const depths = useMemo(() => [...availableDepths].sort((a, b) => a - b), [availableDepths]);
  const ticks = useMemo(() => getDepthTicks(depths), [depths]);
  const maxDepth = depths.length ? depths[depths.length - 1] : 1;
  const currentIndex = nearestDepthIndex(selectedDepth, depths);
  const currentRatio = depthToRailRatio(selectedDepth, maxDepth);

  const rootRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const valueRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const lastSentRef = useRef<number | null>(null);

  // Latest values for the native wheel listener
  const stateRef = useRef({ currentIndex, depths, onChangeDepth });
  stateRef.current = { currentIndex, depths, onChangeDepth };

  const stepTo = useCallback((index: number) => {
    const { depths: ds, onChangeDepth: change, currentIndex: cur } = stateRef.current;
    const clamped = Math.max(0, Math.min(ds.length - 1, index));
    if (clamped !== cur) change(ds[clamped]);
  }, []);

  // Thumb, fill and value pill glide to the new depth
  useEffect(() => {
    const pct = `${currentRatio * 100}%`;
    const targets = [thumbRef.current, valueRef.current].filter(Boolean);
    const duration = prefersReducedMotion() ? 0 : 0.35;
    gsap.to(targets, { top: pct, duration, ease: EASE_OUT, overwrite: "auto" });
    if (fillRef.current) gsap.to(fillRef.current, { height: pct, duration, ease: EASE_OUT, overwrite: "auto" });
  }, [currentRatio]);

  // Native, non-passive wheel listener: React's onWheel is passive, so preventDefault was ignored.
  // Deltas accumulate so a trackpad swipe moves one level per WHEEL_STEP_PX instead of per event.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    let acc = 0;
    let resetTimer: ReturnType<typeof setTimeout> | null = null;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const px = e.deltaMode === 1 ? e.deltaY * WHEEL_STEP_PX : e.deltaY;
      acc += px;
      if (resetTimer) clearTimeout(resetTimer);
      resetTimer = setTimeout(() => (acc = 0), 200);
      if (Math.abs(acc) >= WHEEL_STEP_PX) {
        const dir = acc > 0 ? 1 : -1;
        acc = 0;
        stepTo(stateRef.current.currentIndex + dir);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      if (resetTimer) clearTimeout(resetTimer);
    };
  }, [stepTo]);

  const pickAt = useCallback(
    (clientY: number) => {
      const area = areaRef.current;
      if (!area) return;
      const rect = area.getBoundingClientRect();
      if (rect.height <= 0) return;
      const ratio = Math.min(1, Math.max(0, (clientY - rect.top) / rect.height));
      const idx = nearestLevelIndex(ratio, depths, maxDepth);
      if (lastSentRef.current !== idx) {
        lastSentRef.current = idx;
        if (idx !== currentIndex) onChangeDepth(depths[idx]);
      }
    },
    [depths, maxDepth, currentIndex, onChangeDepth]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const keys: Record<string, number> = {
      ArrowUp: currentIndex - 1,
      ArrowLeft: currentIndex - 1,
      ArrowDown: currentIndex + 1,
      ArrowRight: currentIndex + 1,
      PageUp: currentIndex - 3,
      PageDown: currentIndex + 3,
      Home: 0,
      End: depths.length - 1,
    };
    if (!(e.key in keys)) return;
    e.preventDefault();
    stepTo(keys[e.key]);
  };

  return (
    <div
      ref={rootRef}
      tabIndex={0}
      role="slider"
      aria-label="Ocean Depth Rail"
      aria-orientation="vertical"
      aria-valuemin={depths[0] ?? 0}
      aria-valuemax={maxDepth}
      aria-valuenow={selectedDepth}
      aria-valuetext={`${formatDepth(selectedDepth)}, level ${currentIndex + 1} of ${depths.length}`}
      onKeyDown={handleKeyDown}
      className="hud-glass w-[136px] select-none rounded-[20px] px-3 pb-4 pt-3 outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
    >
      <div className="mb-3 flex items-baseline justify-between">
        <span className="hud-label">Depth</span>
        <span className="text-[10px] tabular-nums text-neutral-500">
          {currentIndex + 1}/{depths.length}
        </span>
      </div>

      {/* Click, tap or drag anywhere on the rail; snaps to the nearest model level */}
      <div
        ref={areaRef}
        className="relative h-56 cursor-ns-resize touch-none"
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          draggingRef.current = true;
          lastSentRef.current = currentIndex;
          e.currentTarget.setPointerCapture?.(e.pointerId);
          pickAt(e.clientY);
        }}
        onPointerMove={(e) => {
          if (draggingRef.current) pickAt(e.clientY);
        }}
        onPointerUp={(e) => {
          draggingRef.current = false;
          if (e.currentTarget.hasPointerCapture?.(e.pointerId)) e.currentTarget.releasePointerCapture?.(e.pointerId);
        }}
        onPointerCancel={() => (draggingRef.current = false)}
      >
        {/* Tick labels (a subset of levels) */}
        {ticks.map((tick: DepthTick) => {
          const hidden = Math.abs(tick.ratio - currentRatio) < LABEL_CLEARANCE;
          return (
            <span
              key={tick.depth_m}
              aria-hidden="true"
              style={{ top: `${tick.ratio * 100}%` }}
              className={`pointer-events-none absolute right-6 -translate-y-1/2 whitespace-nowrap text-[11px] tabular-nums text-neutral-500 transition-opacity duration-150 dark:text-neutral-400 ${
                hidden ? "opacity-0" : "opacity-100"
              }`}
            >
              {tick.label}
            </span>
          );
        })}

        {/* Track, one notch per model level, fill and thumb */}
        <div className="absolute bottom-0 right-[7px] top-0 w-[3px] rounded-full bg-black/10 dark:bg-white/10" />
        {depths.map((d) => (
          <span
            key={d}
            aria-hidden="true"
            style={{ top: `${depthToRailRatio(d, maxDepth) * 100}%` }}
            className="pointer-events-none absolute right-[4px] h-px w-[9px] bg-black/25 dark:bg-white/25"
          />
        ))}
        <div
          ref={fillRef}
          className="absolute right-[7px] top-0 w-[3px] rounded-full bg-cyan-600 dark:bg-cyan-300"
          style={{ height: `${currentRatio * 100}%` }}
        />
        <div
          ref={thumbRef}
          style={{ top: `${currentRatio * 100}%` }}
          className="pointer-events-none absolute right-[8.5px] h-3.5 w-3.5 -translate-y-1/2 translate-x-1/2 rounded-full bg-neutral-900 shadow-[0_0_0_4px_rgba(34,211,238,0.22),0_1px_3px_rgba(0,0,0,0.4)] dark:bg-white"
        />

        {/* Current value pill, follows the thumb */}
        <div
          ref={valueRef}
          aria-hidden="true"
          style={{ top: `${currentRatio * 100}%` }}
          className="pointer-events-none absolute right-6 -translate-y-1/2 whitespace-nowrap rounded-full bg-neutral-900 px-2 py-0.5 text-[11px] font-semibold tabular-nums text-white shadow-sm dark:bg-white dark:text-neutral-950"
        >
          {formatDepth(selectedDepth)}
        </div>
      </div>
    </div>
  );
}
