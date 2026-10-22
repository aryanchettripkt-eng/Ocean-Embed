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
  Legend,
} from "recharts";
import { formatDepth } from "../../lib/format";
import type { MetricsResponse } from "../../lib/api/client";

interface DepthErrorChartProps {
  metrics?: MetricsResponse | null;
  isLoading?: boolean;
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

function DepthTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || label === undefined || !payload) return null;

  return (
    <div className="bg-slate-900/95 border border-slate-700/80 rounded-lg p-3 shadow-xl backdrop-blur-md text-xs space-y-1.5 min-w-[200px]">
      <div className="font-semibold text-slate-200 border-b border-slate-800 pb-1 flex justify-between">
        <span>Depth Tier</span>
        <span className="text-cyan-400 font-mono">{formatDepth(label)}</span>
      </div>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center justify-between text-slate-300">
          <span className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span>{entry.name}:</span>
          </span>
          <span className="font-mono font-medium" style={{ color: entry.color }}>
            {entry.value !== null ? `${Number(entry.value).toFixed(2)} °C` : "n/a"}
          </span>
        </div>
      ))}
    </div>
  );
}

export function DepthErrorChart({ metrics, isLoading }: DepthErrorChartProps) {
  const chartData = useMemo(() => {
    if (!metrics?.rows) return [];

    // Group rows by depth_m (excluding "all")
    const depthMap = new Map<
      number,
      {
        depth_m: number;
        modelRmse: number | null;
        climRmse: number | null;
        ridgeRmse: number | null;
      }
    >();

    metrics.rows.forEach((r) => {
      if (r.depth_m === "all" || r.depth_m === null) return;
      const d = Number(r.depth_m);
      if (!depthMap.has(d)) {
        depthMap.set(d, {
          depth_m: d,
          modelRmse: null,
          climRmse: null,
          ridgeRmse: null,
        });
      }
      const entry = depthMap.get(d)!;
      const modelName = r.model.toLowerCase();
      if (modelName === "climatology") {
        entry.climRmse = r.rmse_c ?? null;
      } else if (modelName === "ridge") {
        entry.ridgeRmse = r.rmse_c ?? null;
      } else {
        entry.modelRmse = r.rmse_c ?? null;
      }
    });

    return Array.from(depthMap.values()).sort((a, b) => a.depth_m - b.depth_m);
  }, [metrics]);

  if (isLoading) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 h-80 animate-pulse" />
    );
  }

  if (chartData.length === 0) {
    return null;
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-xl space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs">
        <div>
          <h3 className="font-semibold text-slate-100 text-sm">
            Validation RMSE vs Depth (0–1500 m)
          </h3>
          <p className="text-slate-400 text-[11px]">
            Lower error is better. Note model improvement through the thermocline (50–300 m).
          </p>
        </div>
        <div className="text-[11px] font-mono text-cyan-400">
          Evaluated vs GLORYS12
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
            <XAxis
              type="number"
              domain={[0, "auto"]}
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              tickFormatter={(v) => `${v}°C`}
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
              width={45}
            />
            <Tooltip content={<DepthTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
              iconType="circle"
            />
            <Line
              type="monotone"
              dataKey="modelRmse"
              name="OceanScope Model"
              stroke="#22d3ee"
              strokeWidth={2.5}
              dot={{ r: 3, fill: "#22d3ee" }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="ridgeRmse"
              name="Ridge Baseline"
              stroke="#f59e0b"
              strokeWidth={1.8}
              strokeDasharray="4 4"
              dot={{ r: 2.5, fill: "#f59e0b" }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="climRmse"
              name="Climatology Baseline"
              stroke="#94a3b8"
              strokeWidth={1.8}
              strokeDasharray="2 2"
              dot={{ r: 2, fill: "#94a3b8" }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
