"use client";

import React, { useMemo } from "react";
import Link from "next/link";
import { X, ExternalLink, Activity, Layers, BarChart2 } from "lucide-react";

import { useMetrics, useProfile, useTimeseries } from "../../lib/api/hooks";
import { formatCoord, formatDate, formatDepth, formatTemp } from "../../lib/format";
import { SplitBadge } from "../common/SplitBadge";
import { Loading } from "../common/Loading";
import { ErrorState } from "../common/ErrorState";
import { ProfileChart } from "./ProfileChart";
import { TimeSeriesChart } from "./TimeSeriesChart";

interface DetailsPanelProps {
  lat: number;
  lon: number;
  date: string;
  depth: number;
  minDate?: string;
  maxDate?: string;
  onClose: () => void;
  onSelectDepth?: (depth: number) => void;
  className?: string;
}

function calculateDateRange(
  dateStr: string,
  minDate?: string,
  maxDate?: string
): { start: string; end: string } {
  try {
    const d = new Date(dateStr + "T00:00:00Z");
    const startD = new Date(d.getTime() - 90 * 86400 * 1000);
    const endD = new Date(d.getTime() + 90 * 86400 * 1000);

    let start = startD.toISOString().slice(0, 10);
    let end = endD.toISOString().slice(0, 10);

    if (minDate && start < minDate) start = minDate;
    if (maxDate && end > maxDate) end = maxDate;

    return { start, end };
  } catch {
    return {
      start: minDate ?? "2019-01-01",
      end: maxDate ?? "2020-01-10",
    };
  }
}

export function DetailsPanel({
  lat,
  lon,
  date,
  depth,
  minDate,
  maxDate,
  onClose,
  onSelectDepth,
  className = "",
}: DetailsPanelProps) {
  // Fetch vertical temperature profile
  const {
    data: profile,
    isLoading: isProfileLoading,
    error: profileError,
    refetch: refetchProfile,
  } = useProfile({ lat, lon, date });

  // Calculate ±90 day window
  const dateRange = useMemo(
    () => calculateDateRange(date, minDate, maxDate),
    [date, minDate, maxDate]
  );

  // Fetch time series at selected depth
  const {
    data: timeseries,
    isLoading: isTimeseriesLoading,
    error: timeseriesError,
    refetch: refetchTimeseries,
  } = useTimeseries({
    lat,
    lon,
    start: dateRange.start,
    end: dateRange.end,
    depth_m: depth,
  });

  // Fetch depth-wise validation metrics
  const { data: metricsData } = useMetrics({ split: "val", group_by: "depth" });

  // Find accuracy metrics for selected depth
  const depthMetric = useMemo(() => {
    if (!metricsData?.rows) return null;
    return (
      metricsData.rows.find(
        (r) =>
          r.depth_m !== null &&
          r.depth_m !== "all" &&
          Math.abs(Number(r.depth_m) - depth) < 0.1 &&
          r.model !== "climatology" &&
          r.model !== "ridge"
      ) ?? null
    );
  }, [metricsData, depth]);

  const split = profile?.split ?? (date >= "2020-01-09" ? "live" : "val");

  return (
    <div
      className={`flex flex-col bg-slate-950/95 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl overflow-hidden z-30 transition-all ${className}`}
    >
      {/* Panel Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-slate-900/60">
        <div className="flex items-center gap-2.5">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-slate-100 font-mono">
                {formatCoord(lat, lon)}
              </span>
              <SplitBadge split={split} />
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Target Date: <span className="text-slate-300 font-medium">{formatDate(date)}</span>
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close details panel"
          className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Panel Content (Scrollable) */}
      <div className="p-4 space-y-6 overflow-y-auto max-h-[calc(100vh-12rem)] md:max-h-[calc(100vh-8rem)]">
        {/* Section 1: Vertical Profile */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5 uppercase tracking-wider">
              <Layers className="h-3.5 w-3.5 text-cyan-400" />
              <span>Vertical Temperature Profile</span>
            </h3>
            <span className="text-[11px] text-slate-400">0 – 1500 m</span>
          </div>

          {isProfileLoading ? (
            <div className="h-56 flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-850">
              <Loading label="Loading vertical profile..." />
            </div>
          ) : profileError || !profile ? (
            <div className="p-4 bg-slate-900/40 rounded-xl border border-slate-850">
              <ErrorState
                title="Profile unavailable"
                message="Could not load vertical temperature data for this location."
                onRetry={() => refetchProfile()}
              />
            </div>
          ) : (
            <div className="p-2 bg-slate-900/40 rounded-xl border border-slate-800/60">
              <ProfileChart
                profile={profile}
                selectedDepth={depth}
                onSelectDepth={onSelectDepth}
              />
            </div>
          )}
        </div>

        {/* Section 2: Time Series Chart */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5 uppercase tracking-wider">
              <Activity className="h-3.5 w-3.5 text-cyan-400" />
              <span>Time Series at {formatDepth(depth)}</span>
            </h3>
            <span className="text-[11px] text-slate-400">±90 Days</span>
          </div>

          {isTimeseriesLoading ? (
            <div className="h-44 flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-850">
              <Loading label="Loading time series..." />
            </div>
          ) : timeseriesError || !timeseries ? (
            <div className="p-4 bg-slate-900/40 rounded-xl border border-slate-850">
              <ErrorState
                title="Time series unavailable"
                message="Could not load temporal profile for this period."
                onRetry={() => refetchTimeseries()}
              />
            </div>
          ) : (
            <div className="p-2 bg-slate-900/40 rounded-xl border border-slate-800/60">
              <TimeSeriesChart timeseries={timeseries} selectedDate={date} />
            </div>
          )}
        </div>

        {/* Section 3: Depth Accuracy Benchmark */}
        <div className="p-3 bg-cyan-950/30 border border-cyan-800/40 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-cyan-300 flex items-center gap-1.5">
              <BarChart2 className="h-3.5 w-3.5 text-cyan-400" />
              <span>How accurate is {formatDepth(depth)}?</span>
            </span>
            <Link
              href="/validation"
              className="text-[11px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1 hover:underline"
            >
              <span>Full metrics</span>
              <ExternalLink className="h-3 w-3" />
            </Link>
          </div>

          {depthMetric ? (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
                <p className="text-[10px] text-slate-400">Validation RMSE</p>
                <p className="font-mono font-semibold text-cyan-300">
                  {formatTemp(depthMetric.rmse_c ?? 0)}
                </p>
              </div>
              <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
                <p className="text-[10px] text-slate-400">Mean Bias (vs GLORYS)</p>
                <p className="font-mono font-semibold text-slate-200">
                  {formatTemp(depthMetric.bias_c ?? 0)}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-[11px] text-slate-400">
              Validation error benchmarks available across 15 depth tiers.
            </p>
          )}

          <p className="text-[10px] text-slate-500 italic">
            Evaluated on held-out 2018 validation split against GLORYS12 reanalysis.
          </p>
        </div>
      </div>
    </div>
  );
}
