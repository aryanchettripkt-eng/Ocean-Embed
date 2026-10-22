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
    <div className="bg-slate-900/95 border border-slate-700/80 rounded-lg p-3 shadow-xl backdrop-blur-md text-xs space-y-1.5 min-w-[190px]">
      <div className="font-semibold text-slate-200 border-b border-slate-800 pb-1 flex items-center justify-between">
        <span>{formatDate(date)}</span>
        <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
          {formatSplit(split).label}
        </span>
      </div>
      <div className="flex items-center justify-between text-slate-300">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span>Model prediction:</span>
        </span>
        <span className="font-mono font-medium text-cyan-300">
          {pred !== null ? formatTemp(pred) : "n/a"}
        </span>
      </div>
      {glorys !== null && (
        <div className="flex items-center justify-between text-slate-300">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            <span>GLORYS reanalysis:</span>
          </span>
          <span className="font-mono font-medium text-indigo-300">
            {formatTemp(glorys)}
          </span>
        </div>
      )}
      {pred !== null && glorys !== null && (
        <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
          <span>Diff (Model − GLORYS):</span>
          <span className="font-mono text-amber-300">
            {formatTemp(pred - glorys)}
          </span>
        </div>
      )}
    </div>
  );
}

export function TimeSeriesChart({ timeseries, selectedDate }: TimeSeriesChartProps) {
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
    <div className="w-full flex flex-col gap-2">
      <div className="flex items-center justify-between text-[11px] text-slate-400 px-1">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-0.5 bg-cyan-400 rounded-full" />
            <span className="text-cyan-300">Model prediction</span>
          </span>
          {hasGlorys ? (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-0.5 bg-indigo-400 rounded-full border-dashed" />
              <span className="text-indigo-300">GLORYS</span>
            </span>
          ) : (
            <span className="text-slate-500 italic">(GLORYS n/a for live)</span>
          )}
        </div>
        <span className="text-amber-400/90 text-[10px]">
          Target: {selectedDate}
        </span>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 15, bottom: 5, left: -10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
            <XAxis
              dataKey="displayDate"
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              axisLine={{ stroke: "#475569" }}
            />
            <YAxis
              type="number"
              domain={[minTemp, maxTemp]}
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              tickFormatter={(v) => `${v}°`}
              axisLine={{ stroke: "#475569" }}
              width={38}
            />
            <Tooltip content={<TimeSeriesTooltip />} />
            <ReferenceLine
              x={selectedDate.slice(5)}
              stroke="#f59e0b"
              strokeDasharray="3 3"
              strokeWidth={1.5}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Model prediction"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={{ r: 2, fill: "#22d3ee" }}
              activeDot={{ r: 4.5, fill: "#06b6d4" }}
              isAnimationActive={false}
            />
            {hasGlorys && (
              <Line
                type="monotone"
                dataKey="glorys"
                name="GLORYS reanalysis"
                stroke="#a78bfa"
                strokeWidth={1.8}
                strokeDasharray="3 3"
                dot={{ r: 2, fill: "#a78bfa" }}
                activeDot={{ r: 4, fill: "#8b5cf6" }}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
