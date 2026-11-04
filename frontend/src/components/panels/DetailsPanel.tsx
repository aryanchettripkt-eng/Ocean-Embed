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
      start: minDate ?? dateStr,
      end: maxDate ?? dateStr,
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
          r.model?.toLowerCase() === "model"
      ) ?? null
    );
  }, [metricsData, depth]);

  const split = profile?.split ?? null;

  return (
    <div
      className={`dark hud-glass flex flex-col rounded-[20px] overflow-hidden z-30 ${className}`}
    >
      {/* Panel Header */}
      <div className="flex items-start justify-between gap-3 px-4 pt-3.5 pb-3 border-b border-white/[0.07]">
        <div className="flex items-center gap-2.5">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-white tabular-nums">
                {formatCoord(lat, lon)}
              </span>
              <SplitBadge split={split} />
            </div>
            <p className="text-xs text-neutral-400 mt-1">
              {formatDate(date)} · {formatDepth(depth)}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close details panel"
          className="hud-icon-btn"
        >
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>

      {/* Panel Content (Scrollable) */}
      <div className="min-h-0 flex-1 p-4 space-y-6 overflow-y-auto overscroll-contain">
        {/* Section 1: Vertical Profile */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium text-neutral-200 flex items-center gap-1.5">
              <Layers aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>Vertical Temperature Profile</span>
            </h3>
            <span className="text-[11px] text-slate-400">0 – 1500 m</span>
          </div>

          {isProfileLoading ? (
            <div className="h-56 flex items-center justify-center rounded-xl bg-white/[0.03]">
              <Loading label="Loading vertical profile..." />
            </div>
          ) : profileError || !profile ? (
            <div className="p-4 rounded-xl bg-white/[0.03]">
              <ErrorState
                title="Profile unavailable"
                message="Could not load vertical temperature data for this location."
                onRetry={() => refetchProfile()}
              />
            </div>
          ) : (
            <div className="p-2 rounded-xl bg-white/[0.03]">
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
            <h3 className="text-xs font-medium text-neutral-200 flex items-center gap-1.5">
              <Activity aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>Time Series at {formatDepth(depth)}</span>
            </h3>
            <span className="text-[11px] text-slate-400">±90 Days</span>
          </div>

          {isTimeseriesLoading ? (
            <div className="h-44 flex items-center justify-center rounded-xl bg-white/[0.03]">
              <Loading label="Loading time series..." />
            </div>
          ) : timeseriesError || !timeseries ? (
            <div className="p-4 rounded-xl bg-white/[0.03]">
              <ErrorState
                title="Time series unavailable"
                message="Could not load temporal profile for this period."
                onRetry={() => refetchTimeseries()}
              />
            </div>
          ) : (
            <div className="p-2 rounded-xl bg-white/[0.03]">
              <TimeSeriesChart timeseries={timeseries} selectedDate={date} />
            </div>
          )}
        </div>

        {/* Section 3: Depth Accuracy Benchmark */}
        <div className="p-3 rounded-xl bg-white/[0.03] ring-1 ring-inset ring-white/[0.06] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-neutral-200 flex items-center gap-1.5">
              <BarChart2 aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>How accurate is {formatDepth(depth)}?</span>
            </span>
            <Link
              href="/validation"
              className="text-[11px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1 hover:underline"
            >
              <span>Full metrics</span>
              <ExternalLink aria-hidden="true" className="h-3 w-3" />
            </Link>
          </div>

          {depthMetric ? (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-black/30 p-2 rounded-lg">
                <p className="text-[10px] text-slate-400">Validation RMSE</p>
                <p className="font-semibold tabular-nums text-white">
                  {formatTemp(depthMetric.rmse_c ?? 0)}
                </p>
              </div>
              <div className="bg-black/30 p-2 rounded-lg">
                <p className="text-[10px] text-slate-400">Mean Bias (vs GLORYS)</p>
                <p className="font-semibold tabular-nums text-white">
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
