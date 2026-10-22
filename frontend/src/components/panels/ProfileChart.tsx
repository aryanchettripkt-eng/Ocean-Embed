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
import { formatDepth, formatTemp } from "../../lib/format";
import type { ProfileResponse } from "../../lib/api/client";

interface ProfileChartProps {
  profile: ProfileResponse;
  selectedDepth: number;
  onSelectDepth?: (depth: number) => void;
}

interface TooltipPayloadItem {
  value: number | null;
  dataKey: string;
  name: string;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}

function ProfileTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || label === undefined) return null;

  const predItem = payload?.find((p) => p.dataKey === "pred");
  const glorysItem = payload?.find((p) => p.dataKey === "glorys");

  const predVal = predItem?.value;
  const glorysVal = glorysItem?.value;

  return (
    <div className="bg-slate-900/95 border border-slate-700/80 rounded-lg p-3 shadow-xl backdrop-blur-md text-xs space-y-1.5 min-w-[180px]">
      <div className="font-semibold text-slate-200 border-b border-slate-800 pb-1 flex justify-between">
        <span>Depth</span>
        <span className="text-cyan-400">{formatDepth(label)}</span>
      </div>
      <div className="flex items-center justify-between text-slate-300">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span>Model prediction:</span>
        </span>
        <span className="font-mono font-medium text-cyan-300">
          {predVal !== null && predVal !== undefined ? formatTemp(predVal) : "n/a"}
        </span>
      </div>
      {glorysVal !== null && glorysVal !== undefined && (
        <div className="flex items-center justify-between text-slate-300">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            <span>GLORYS reanalysis:</span>
          </span>
          <span className="font-mono font-medium text-indigo-300">
            {formatTemp(glorysVal)}
          </span>
        </div>
      )}
      {predVal !== null &&
        predVal !== undefined &&
        glorysVal !== null &&
        glorysVal !== undefined && (
          <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
            <span>Diff (Model − GLORYS):</span>
            <span className="font-mono text-amber-300">
              {formatTemp(predVal - glorysVal)}
            </span>
          </div>
        )}
    </div>
  );
}

export function ProfileChart({ profile, selectedDepth }: ProfileChartProps) {
  const chartData = useMemo(() => {
    return profile.depths_m.map((depth, idx) => ({
      depth_m: depth,
      formattedDepth: formatDepth(depth),
      pred: profile.pred[idx] ?? null,
      glorys: profile.glorys[idx] ?? null,
    }));
  }, [profile]);

  const hasGlorys = useMemo(() => {
    return profile.glorys.some((v) => v !== null);
  }, [profile.glorys]);

  // Compute temp min/max for horizontal axis
  const { minTemp, maxTemp } = useMemo(() => {
    const validVals = [
      ...profile.pred.filter((v): v is number => v !== null),
      ...profile.glorys.filter((v): v is number => v !== null),
    ];
    if (validVals.length === 0) return { minTemp: 0, maxTemp: 35 };
    const min = Math.floor(Math.min(...validVals)) - 1;
    const max = Math.ceil(Math.max(...validVals)) + 1;
    return { minTemp: min, maxTemp: max };
  }, [profile]);

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
              <span className="text-indigo-300">GLORYS reanalysis</span>
            </span>
          ) : (
            <span className="text-slate-500 italic">(GLORYS n/a for live date)</span>
          )}
        </div>
        <span className="text-amber-400/90 text-[10px]">
          Selected: {formatDepth(selectedDepth)}
        </span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 15, bottom: 10, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
            <XAxis
              type="number"
              domain={[minTemp, maxTemp]}
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              tickFormatter={(v) => `${v}°`}
              axisLine={{ stroke: "#475569" }}
            />
            <YAxis
              type="number"
              dataKey="depth_m"
              reversed={true}
              domain={[0, "dataMax"]}
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              tickFormatter={(d) => `${d}m`}
              axisLine={{ stroke: "#475569" }}
              width={42}
            />
            <Tooltip content={<ProfileTooltip />} />
            <ReferenceLine
              y={selectedDepth}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Model prediction"
              stroke="#22d3ee"
              strokeWidth={2}
              dot={{ r: 2.5, fill: "#22d3ee" }}
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
