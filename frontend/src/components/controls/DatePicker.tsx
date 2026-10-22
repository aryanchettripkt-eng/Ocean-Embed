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

    const interval = setInterval(() => {
      const nextDays = currentDays + 1;
      if (nextDays > endDays) {
        // Loop back to start
        onChange(daysToDateStr(startDays));
      } else {
        onChange(daysToDateStr(nextDays));
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying, currentDays, endDays, startDays, onChange]);

  const canStepPrev = currentDays > startDays;
  const canStepNext = currentDays < endDays;

  const handleStep = (step: number) => {
    const target = currentDays + step;
    if (target >= startDays && target <= endDays) {
      onChange(daysToDateStr(target));
    }
  };

  return (
    <div className={`flex flex-col gap-2.5 ${className}`}>
      {/* Top Header Row */}
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Calendar className="h-3.5 w-3.5 text-cyan-400" />
          <span>Date</span>
        </label>
        <div className="flex items-center gap-2">
          <SplitBadge split={split} />
          {latestLiveDate && (
            <button
              type="button"
              onClick={() => onChange(latestLiveDate)}
              className="text-[11px] font-medium text-emerald-400 hover:text-emerald-300 bg-emerald-950/50 hover:bg-emerald-900/50 border border-emerald-700/60 px-2 py-0.5 rounded transition-all flex items-center gap-1"
              title="Jump to latest live date"
            >
              <Sparkles className="h-3 w-3" />
              <span>Latest</span>
            </button>
          )}
        </div>
      </div>

      {/* Date display & step buttons */}
      <div className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-lg p-1.5 shadow-sm">
        <button
          type="button"
          disabled={!canStepPrev}
          onClick={() => handleStep(-1)}
          className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          title="Previous day (-1d)"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-2">
          <input
            type="date"
            min={startDate}
            max={endDate}
            value={date}
            onChange={(e) => {
              if (e.target.value) onChange(e.target.value);
            }}
            className="bg-transparent text-sm font-semibold text-slate-100 text-center border-none focus:outline-none focus:ring-1 focus:ring-cyan-500 rounded px-1.5 cursor-pointer"
          />
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            ({formatDate(date)})
          </span>
        </div>

        <button
          type="button"
          disabled={!canStepNext}
          onClick={() => handleStep(1)}
          className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          title="Next day (+1d)"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      {/* Date slider and Play button */}
      <div className="flex items-center gap-2 pt-1">
        <button
          type="button"
          onClick={() => setIsPlaying(!isPlaying)}
          className={`p-1.5 rounded-lg border text-xs font-medium transition-all flex items-center justify-center shrink-0 ${
            isPlaying
              ? "bg-amber-500 text-slate-950 border-amber-400 shadow-md shadow-amber-500/20"
              : "bg-slate-800 text-slate-300 hover:text-white border-slate-700 hover:bg-slate-700"
          }`}
          title={isPlaying ? "Pause animation" : "Play daily animation (2 fps)"}
        >
          {isPlaying ? <Square className="h-3.5 w-3.5 fill-current" /> : <Play className="h-3.5 w-3.5 fill-current" />}
        </button>

        <input
          type="range"
          min={0}
          max={totalDays}
          step={1}
          value={currentSliderValue}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            onChange(daysToDateStr(startDays + val));
          }}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 hover:accent-cyan-300 transition-all"
        />
      </div>

      <div className="flex justify-between text-[10px] text-slate-500 font-mono px-0.5 -mt-1">
        <span>{startDate}</span>
        <span>{endDate}</span>
      </div>
    </div>
  );
}
