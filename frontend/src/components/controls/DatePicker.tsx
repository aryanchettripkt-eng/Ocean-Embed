"use client";

import React, { useEffect, useState } from "react";
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

interface DatePickerProps {
  date: string;
  startDate: string;
  endDate: string;
  currentSplit?: string | null;
  latestLiveDate?: string | null;
  splits?: {
    train?: string[];
    val?: string[];
    test?: string[];
  } | null;
  onChange: (date: string) => void;
  prefetchAhead?: (date: string) => void;
  className?: string;
}

function parseDateToDays(dateStr: string): number {
  return Math.floor(new Date(dateStr + "T00:00:00Z").getTime() / 86400000);
}

function daysToDateStr(days: number): string {
  const d = new Date(days * 86400000);
  return d.toISOString().slice(0, 10);
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
  splits,
  onChange,
  prefetchAhead,
  className = "",
}: DatePickerProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  const startDays = parseDateToDays(startDate);
  const endDays = parseDateToDays(endDate);
  const currentDays = parseDateToDays(date);

  const totalDays = Math.max(1, endDays - startDays);
  const currentSliderValue = Math.max(0, Math.min(totalDays, currentDays - startDays));

  // Determine split of current date
  const split = currentSplit ?? getSplitForDate(date, splits);

  // Animation player: 2 frames/sec (500ms per frame)
  useEffect(() => {
    if (!isPlaying) return;

    // Prefetch immediately when starting play
    prefetchAhead?.(daysToDateStr(currentDays));

    const interval = setInterval(() => {
      const nextDays = currentDays + 1;
      const targetStr = nextDays > endDays ? daysToDateStr(startDays) : daysToDateStr(nextDays);
      onChange(targetStr);
      prefetchAhead?.(targetStr);
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying, currentDays, endDays, startDays, onChange, prefetchAhead]);

  const canStepPrev = currentDays > startDays;
  const canStepNext = currentDays < endDays;

  const handleStep = (step: number) => {
    const target = currentDays + step;
    if (target >= startDays && target <= endDays) {
      onChange(daysToDateStr(target));
    }
  };

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {/* Top Header Row */}
      <div className="flex items-center justify-between">
        <label
          htmlFor="explorer-date-input"
          className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5"
        >
          <Calendar aria-hidden="true" className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
          <span>Date</span>
        </label>
        <div className="flex items-center gap-2">
          <SplitBadge split={split} />
          {latestLiveDate && (
            <button
              type="button"
              onClick={() => onChange(latestLiveDate)}
              className="text-[11px] font-medium text-emerald-700 dark:text-emerald-300 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 px-2 py-0.5 rounded-full transition-all flex items-center gap-1"
              title="Jump to latest live date"
            >
              <Sparkles aria-hidden="true" className="h-3 w-3" />
              <span>Latest</span>
            </button>
          )}
        </div>
      </div>

      {/* Date display & step buttons */}
      <div className="flex items-center justify-between bg-slate-100/90 dark:bg-white/[0.04] border border-slate-200/90 dark:border-white/10 rounded-xl p-1 shadow-sm">
        <button
          type="button"
          disabled={!canStepPrev}
          onClick={() => handleStep(-1)}
          className="p-1 rounded-lg text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          title="Previous day (-1d)"
          aria-label="Previous day"
        >
          <ChevronLeft aria-hidden="true" className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-1.5">
          <input
            id="explorer-date-input"
            type="date"
            min={startDate}
            max={endDate}
            value={date}
            onChange={(e) => {
              if (e.target.value) onChange(e.target.value);
            }}
            className="bg-transparent text-xs sm:text-sm font-semibold text-slate-900 dark:text-slate-100 text-center border-none focus:outline-none focus:ring-1 focus:ring-cyan-500 rounded px-1 cursor-pointer"
          />
          <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono hidden sm:inline">
            ({formatDate(date)})
          </span>
        </div>

        <button
          type="button"
          disabled={!canStepNext}
          onClick={() => handleStep(1)}
          className="p-1 rounded-lg text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          title="Next day (+1d)"
          aria-label="Next day"
        >
          <ChevronRight aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>

      {/* Date slider and Play button */}
      <div className="flex items-center gap-2 pt-0.5">
        <button
          type="button"
          aria-pressed={isPlaying}
          onClick={() => setIsPlaying(!isPlaying)}
          className={`p-1.5 rounded-full border text-xs font-medium transition-all flex items-center justify-center shrink-0 ${
            isPlaying
              ? "bg-amber-500 text-slate-950 border-amber-400 shadow-md shadow-amber-500/20"
              : "bg-slate-100 dark:bg-white/[0.06] text-slate-700 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white border-slate-300 dark:border-white/12 hover:bg-slate-200 dark:hover:bg-white/12"
          }`}
          title={isPlaying ? "Pause animation" : "Play daily animation (2 fps)"}
          aria-label="Play daily animation"
        >
          {isPlaying ? (
            <Square aria-hidden="true" className="h-3 w-3 fill-current" />
          ) : (
            <Play aria-hidden="true" className="h-3 w-3 fill-current" />
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
            onChange(daysToDateStr(startDays + val));
          }}
          className="w-full h-1.5 bg-slate-200 dark:bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-500 dark:accent-cyan-400 transition-all"
        />
      </div>

      <div className="flex justify-between text-[10px] text-slate-500 font-mono px-0.5 -mt-1">
        <span>{startDate}</span>
        <span>{endDate}</span>
      </div>
    </div>
  );
}
