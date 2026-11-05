"use client";

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { formatDate, formatSplit, formatTemp } from "../../lib/format";
import type { TimeseriesResponse } from "../../lib/api/client";
import { useIsDark } from "../../lib/use-dark-mode";
import { CHART_TOOLTIP_CLASS, chartTheme } from "./chartTheme";

interface TimeSeriesChartProps {
  timeseries: TimeseriesResponse;
  selectedDate: string;
}

interface TooltipPayloadItem {
  value: number | null;
  dataKey: string;
  name: string;
  payload: {
    date: string;
    split: string;
    pred: number | null;
    glorys: number | null;
  };
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
}

function TimeSeriesTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  const { date, split, pred, glorys } = data;

  return (
    <div className={CHART_TOOLTIP_CLASS}>
      <div className="flex items-center justify-between gap-3 border-b border-black/[0.06] pb-1 font-medium text-neutral-900 dark:border-white/[0.08] dark:text-white">
        <span>{formatDate(date)}</span>
        <span className="rounded-full bg-black/[0.05] px-1.5 py-0.5 text-[10px] font-normal text-neutral-600 dark:bg-white/[0.08] dark:text-neutral-300">
          {formatSplit(split).label}
        </span>
      </div>
      <div className="flex items-center justify-between gap-3 text-neutral-600 dark:text-neutral-300">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-cyan-600 dark:bg-cyan-400" />
          <span>Model prediction</span>
        </span>
        <span className="font-medium tabular-nums text-neutral-950 dark:text-white">
          {pred !== null ? formatTemp(pred) : "n/a"}
        </span>
      </div>
      {glorys !== null && (
        <div className="flex items-center justify-between gap-3 text-neutral-600 dark:text-neutral-300">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-violet-600 dark:bg-violet-400" />
            <span>GLORYS reanalysis</span>
          </span>
          <span className="font-medium tabular-nums text-neutral-950 dark:text-white">{formatTemp(glorys)}</span>
        </div>
      )}
      {pred !== null && glorys !== null && (
        <div className="flex items-center justify-between gap-3 border-t border-black/[0.06] pt-1 text-[11px] text-neutral-500 dark:border-white/[0.08] dark:text-neutral-400">
          <span>Model − GLORYS</span>
          <span className="tabular-nums text-amber-700 dark:text-amber-300">{formatTemp(pred - glorys)}</span>
        </div>
      )}
    </div>
  );
}

export function TimeSeriesChart({ timeseries, selectedDate }: TimeSeriesChartProps) {
  const theme = chartTheme(useIsDark());
  const chartData = useMemo(() => {
    return timeseries.dates.map((date, idx) => ({
      date,
      displayDate: date.slice(5), // "MM-DD"
      split: timeseries.split[idx] ?? "val",
      pred: timeseries.pred[idx] ?? null,
      glorys: timeseries.glorys[idx] ?? null,
    }));
  }, [timeseries]);

  const hasGlorys = useMemo(() => {
    return timeseries.glorys.some((v) => v !== null);
  }, [timeseries.glorys]);

  const { minTemp, maxTemp } = useMemo(() => {
    const validVals = [
      ...timeseries.pred.filter((v): v is number => v !== null),
      ...timeseries.glorys.filter((v): v is number => v !== null),
    ];
    if (validVals.length === 0) return { minTemp: 15, maxTemp: 30 };
    const min = Math.floor(Math.min(...validVals) * 10) / 10 - 0.2;
    const max = Math.ceil(Math.max(...validVals) * 10) / 10 + 0.2;
    return { minTemp: Number(min.toFixed(1)), maxTemp: Number(max.toFixed(1)) };
  }, [timeseries]);

  return (
    <div className="flex h-full min-h-0 w-full flex-col gap-1">
      <div className="flex items-center justify-between gap-2 px-1 text-[11px] text-neutral-600 dark:text-neutral-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="h-0.5 w-2.5 rounded-full bg-cyan-600 dark:bg-cyan-400" />
            <span>Model prediction</span>
          </span>
          {hasGlorys ? (
            <span className="flex items-center gap-1">
              <span className="h-0.5 w-2.5 rounded-full bg-violet-600 dark:bg-violet-400" />
              <span>GLORYS</span>
            </span>
          ) : (
            <span className="italic text-neutral-500">(GLORYS n/a for live)</span>
          )}
        </div>
        <span className="text-[10px] text-amber-700 dark:text-amber-400">Target: {formatDate(selectedDate)}</span>
      </div>

      <div className="min-h-0 w-full flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={theme.grid} />
            <XAxis
              dataKey="displayDate"
              tick={{ fill: theme.tick, fontSize: 10 }}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
            />
            <YAxis
              type="number"
              domain={[minTemp, maxTemp]}
              tick={{ fill: theme.tick, fontSize: 10 }}
              tickFormatter={(v) => `${Number(v).toFixed(1)}°`}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
              width={42}
            />
            <Tooltip content={<TimeSeriesTooltip />} />
            <ReferenceLine
              x={selectedDate.slice(5)}
              stroke={theme.marker}
              strokeDasharray="3 3"
              strokeWidth={1.5}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Model prediction"
              stroke={theme.pred}
              strokeWidth={2}
              dot={{ r: 2, fill: theme.pred }}
              activeDot={{ r: 4.5, fill: theme.predActive }}
              isAnimationActive={false}
            />
            {hasGlorys && (
              <Line
                type="monotone"
                dataKey="glorys"
                name="GLORYS reanalysis"
                stroke={theme.glorys}
                strokeWidth={1.8}
                strokeDasharray="3 3"
                dot={{ r: 2, fill: theme.glorys }}
                activeDot={{ r: 4, fill: theme.glorysActive }}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
