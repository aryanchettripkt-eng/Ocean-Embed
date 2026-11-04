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

  const chartBox = "min-h-0 flex-1 rounded-xl bg-black/[0.025] p-2 dark:bg-white/[0.03]";
  const statusBox = "flex min-h-0 flex-1 items-center justify-center rounded-xl bg-black/[0.025] dark:bg-white/[0.03]";
  const sectionTitle = "flex items-center gap-1.5 text-xs font-medium text-neutral-800 dark:text-neutral-200";

  return (
    // Sized by its container: the two chart sections shrink from their preferred heights to a
    // readable minimum so everything fits without a scrollbar (scrolls only below that minimum).
    <div className={`hud-glass z-30 flex min-h-0 flex-col overflow-hidden rounded-[20px] ${className}`}>
      {/* Header */}
      <div className="hud-divider flex shrink-0 items-start justify-between gap-3 border-b px-4 pb-3 pt-3.5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold tabular-nums text-neutral-950 dark:text-white">
              {formatCoord(lat, lon)}
            </span>
            <SplitBadge split={split} />
          </div>
          <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            {formatDate(date)} · {formatDepth(depth)}
          </p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close details panel" className="hud-icon-btn">
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overscroll-contain p-3.5 [scrollbar-width:thin]">
        {/* Vertical profile */}
        <section className="flex min-h-[170px] flex-[1_1_280px] flex-col gap-1.5">
          <div className="flex shrink-0 items-center justify-between">
            <h3 className={sectionTitle}>
              <Layers aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>Vertical Temperature Profile</span>
            </h3>
            <span className="text-[11px] text-neutral-500">0 – 1500 m</span>
          </div>

          {isProfileLoading ? (
            <div className={statusBox}>
              <Loading label="Loading vertical profile..." />
            </div>
          ) : profileError || !profile ? (
            <div className={statusBox}>
              <ErrorState
                title="Profile unavailable"
                message="Could not load vertical temperature data for this location."
                onRetry={() => refetchProfile()}
              />
            </div>
          ) : (
            <div className={chartBox}>
              <ProfileChart profile={profile} selectedDepth={depth} onSelectDepth={onSelectDepth} />
            </div>
          )}
        </section>

        {/* Time series */}
        <section className="flex min-h-[150px] flex-[1_1_220px] flex-col gap-1.5">
          <div className="flex shrink-0 items-center justify-between">
            <h3 className={sectionTitle}>
              <Activity aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>Time Series at {formatDepth(depth)}</span>
            </h3>
            <span className="text-[11px] text-neutral-500">±90 Days</span>
          </div>

          {isTimeseriesLoading ? (
            <div className={statusBox}>
              <Loading label="Loading time series..." />
            </div>
          ) : timeseriesError || !timeseries ? (
            <div className={statusBox}>
              <ErrorState
                title="Time series unavailable"
                message="Could not load temporal profile for this period."
                onRetry={() => refetchTimeseries()}
              />
            </div>
          ) : (
            <div className={chartBox}>
              <TimeSeriesChart timeseries={timeseries} selectedDate={date} />
            </div>
          )}
        </section>

        {/* Accuracy at this depth */}
        <section className="shrink-0 space-y-1.5 rounded-xl bg-black/[0.025] px-3 py-2.5 dark:bg-white/[0.03]">
          <div className="flex items-center justify-between gap-2">
            <span className={sectionTitle}>
              <BarChart2 aria-hidden="true" className="h-3.5 w-3.5 text-neutral-500" />
              <span>How accurate is {formatDepth(depth)}?</span>
            </span>
            <Link
              href="/validation"
              className="flex items-center gap-1 text-[11px] text-cyan-700 hover:underline dark:text-cyan-300"
            >
              <span>Full metrics</span>
              <ExternalLink aria-hidden="true" className="h-3 w-3" />
            </Link>
          </div>

          {depthMetric ? (
            <dl className="flex items-baseline gap-5 text-xs">
              <div className="flex items-baseline gap-1.5">
                <dt className="text-[11px] text-neutral-500 dark:text-neutral-400">Validation RMSE</dt>
                <dd className="font-semibold tabular-nums text-neutral-950 dark:text-white">
                  {formatTemp(depthMetric.rmse_c ?? 0)}
                </dd>
              </div>
              <div className="flex items-baseline gap-1.5">
                <dt className="text-[11px] text-neutral-500 dark:text-neutral-400">Mean bias vs GLORYS</dt>
                <dd className="font-semibold tabular-nums text-neutral-950 dark:text-white">
                  {formatTemp(depthMetric.bias_c ?? 0)}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
              Validation error benchmarks available across 15 depth tiers.
            </p>
          )}

          <p className="text-[10px] italic text-neutral-500">
            Evaluated on held-out 2018 validation split against GLORYS12 reanalysis.
          </p>
        </section>
      </div>
    </div>
  );
}
