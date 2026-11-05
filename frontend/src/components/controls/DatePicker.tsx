"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Calendar,
  ChevronLeft,
  ChevronRight,
  Play,
  Square,
  Sparkles,
} from "lucide-react";
import { SplitBadge } from "../common/SplitBadge";
import { formatDate } from "../../lib/format";
import { ServedDays, type DayRange } from "../../lib/date-ranges";

interface DatePickerProps {
  date: string;
  startDate: string;
  endDate: string;
  currentSplit?: string | null;
  latestLiveDate?: string | null;
  /** Runs of served days (/meta date_ranges); the controls skip the gaps between them. */
  dateRanges?: DayRange[] | null;
  splits?: {
    train?: string[];
    val?: string[];
    test?: string[];
  } | null;
  onChange: (date: string) => void;
  prefetchAhead?: (date: string) => void;
  className?: string;
}

function getSplitForDate(
  date: string,
  splits?: { train?: string[]; val?: string[]; test?: string[] } | null
): string | null {
  if (splits?.test && splits.test.length >= 2 && date >= splits.test[0] && date <= splits.test[1]) {
    return "test";
  }
  if (splits?.val && splits.val.length >= 2 && date >= splits.val[0] && date <= splits.val[1]) {
    return "val";
  }
  if (splits?.train && splits.train.length >= 2 && date >= splits.train[0] && date <= splits.train[1]) {
    return "train";
  }
  return null;
}

export function DatePicker({
  date,
  startDate,
  endDate,
  currentSplit,
  latestLiveDate,
  dateRanges,
  splits,
  onChange,
  prefetchAhead,
  className = "",
}: DatePickerProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const served = useMemo(
    () => new ServedDays(startDate, endDate, dateRanges),
    [startDate, endDate, dateRanges]
  );
  const totalDays = Math.max(1, served.count - 1);
  const currentSliderValue = served.indexOf(date);
  const nextDate = served.step(date, 1);
  const prevDate = served.step(date, -1);

  // Determine split of current date
  const split = currentSplit ?? getSplitForDate(date, splits);

  // Animation player: 2 frames/sec (500ms per frame)
  useEffect(() => {
    if (!isPlaying) return;

    // Prefetch immediately when starting play
    prefetchAhead?.(date);

    const interval = setInterval(() => {
      const targetStr = nextDate ?? served.first;
      onChange(targetStr);
      prefetchAhead?.(targetStr);
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying, date, nextDate, served, onChange, prefetchAhead]);

  const canStepPrev = prevDate !== null;
  const canStepNext = nextDate !== null;

  const handleStep = (step: 1 | -1) => {
    const target = step === 1 ? nextDate : prevDate;
    if (target) onChange(target);
  };

  const fillPct = `${(currentSliderValue / totalDays) * 100}%`;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="explorer-date-input" className="hud-label flex items-center gap-1.5">
          <Calendar aria-hidden="true" className="h-3.5 w-3.5" />
          <span>Date</span>
        </label>
        <div className="flex items-center gap-1.5">
          <SplitBadge split={split} />
          {latestLiveDate && (
            <button
              type="button"
              onClick={() => onChange(latestLiveDate)}
              className="inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-700 transition-colors hover:bg-emerald-500/20 dark:text-emerald-300"
              title="Jump to latest live date"
            >
              <Sparkles aria-hidden="true" className="h-3 w-3" />
              <span>Latest</span>
            </button>
          )}
        </div>
      </div>

      {/* Step buttons around the native date input */}
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          disabled={!canStepPrev}
          onClick={() => handleStep(-1)}
          className="hud-icon-btn"
          title="Previous day (-1d)"
          aria-label="Previous day"
        >
          <ChevronLeft aria-hidden="true" className="h-4 w-4" />
        </button>

        <div className="flex h-8 min-w-0 flex-1 items-center justify-center gap-2 rounded-full bg-black/[0.04] px-2 dark:bg-white/[0.05]">
          <input
            id="explorer-date-input"
            type="date"
            min={startDate}
            max={endDate}
            value={date}
            onChange={(e) => {
              if (e.target.value) onChange(served.snap(e.target.value));
            }}
            className="min-w-0 cursor-pointer rounded bg-transparent px-1 text-center text-sm font-medium tabular-nums text-neutral-900 focus:outline-none focus-visible:ring-1 focus-visible:ring-cyan-500 dark:text-neutral-100 dark:[color-scheme:dark]"
          />
          <span className="hidden whitespace-nowrap text-[11px] text-neutral-500 sm:inline dark:text-neutral-400">
            {formatDate(date)}
          </span>
        </div>

        <button
          type="button"
          disabled={!canStepNext}
          onClick={() => handleStep(1)}
          className="hud-icon-btn"
          title="Next day (+1d)"
          aria-label="Next day"
        >
          <ChevronRight aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>

      {/* Play + scrubber */}
      <div className="flex items-center gap-2.5">
        <button
          type="button"
          aria-pressed={isPlaying}
          onClick={() => setIsPlaying(!isPlaying)}
          className={`hud-icon-btn ${isPlaying ? "!border-transparent !bg-amber-400 !text-neutral-950" : ""}`}
          title={isPlaying ? "Pause animation" : "Play daily animation (2 fps)"}
          aria-label="Play daily animation"
        >
          {isPlaying ? (
            <Square aria-hidden="true" className="h-3 w-3 fill-current" />
          ) : (
            <Play aria-hidden="true" className="ml-0.5 h-3 w-3 fill-current" />
          )}
        </button>

        <input
          type="range"
          aria-label="Date scrubber"
          min={0}
          max={totalDays}
          step={1}
          value={currentSliderValue}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            onChange(served.at(val));
          }}
          className="hud-range"
          style={{ "--fill": fillPct } as React.CSSProperties}
        />
      </div>

      <div className="flex justify-between pl-[38px] text-[10px] tabular-nums text-neutral-500">
        <span>{formatDate(startDate)}</span>
        <span>{formatDate(endDate)}</span>
      </div>
    </div>
  );
}
