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
import { useIsDark } from "../../lib/use-dark-mode";
import { CHART_TOOLTIP_CLASS, chartTheme } from "../panels/chartTheme";
import { Pulse } from "../common/Motion";

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
    <div className={`${CHART_TOOLTIP_CLASS} min-w-[200px]`}>
      <div className="flex justify-between border-b border-black/[0.06] pb-1 font-medium text-neutral-900 dark:border-white/[0.08] dark:text-white">
        <span>Depth Tier</span>
        <span className="tabular-nums">{formatDepth(label)}</span>
      </div>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center justify-between gap-3 text-neutral-600 dark:text-neutral-300">
          <span className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span>{entry.name}:</span>
          </span>
          <span className="font-medium tabular-nums text-neutral-950 dark:text-white">
            {entry.value !== null ? `${Number(entry.value).toFixed(2)} °C` : "n/a"}
          </span>
        </div>
      ))}
    </div>
  );
}

export function DepthErrorChart({ metrics, isLoading }: DepthErrorChartProps) {
  const isDark = useIsDark();
  const theme = chartTheme(isDark);
  const climColor = isDark ? "#a3a3a3" : "#737373";
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
      <Pulse className="block h-80 rounded-[20px] bg-black/[0.05] dark:bg-white/[0.05]" />
    );
  }

  if (chartData.length === 0) {
    return null;
  }

  return (
    <section className="hud-glass space-y-3 rounded-[20px] p-5 sm:p-6">
      <div className="flex flex-col justify-between gap-1 text-xs sm:flex-row sm:items-end">
        <div>
          <h3 className="text-lg font-medium tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif]">
            Validation RMSE vs Depth (0–1500 m)
          </h3>
          <p className="text-[11px] text-neutral-500 dark:text-neutral-400">Lower error is better.</p>
        </div>
        <span className="hud-label">Evaluated vs GLORYS12</span>
      </div>

      <div className="h-72 w-full rounded-xl bg-black/[0.025] p-2 dark:bg-white/[0.03]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={theme.grid} />
            <XAxis
              type="number"
              domain={[0, "auto"]}
              tick={{ fill: theme.tick, fontSize: 10 }}
              tickFormatter={(v) => `${v}°C`}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
            />
            <YAxis
              type="number"
              dataKey="depth_m"
              reversed={true}
              domain={[0, "dataMax"]}
              tick={{ fill: theme.tick, fontSize: 10 }}
              tickFormatter={(d) => formatDepth(d).replace(" ", "\u00a0")}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
              width={52}
            />
            <Tooltip content={<DepthTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px", color: theme.tick }}
              iconType="circle"
            />
            <Line
              type="monotone"
              dataKey="modelRmse"
              name="OceanScope Model"
              stroke={theme.pred}
              strokeWidth={2.5}
              dot={{ r: 3, fill: theme.pred }}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="ridgeRmse"
              name="Ridge Baseline"
              stroke={theme.marker}
              strokeWidth={1.8}
              strokeDasharray="4 4"
              dot={{ r: 2.5, fill: theme.marker }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="climRmse"
              name="Climatology Baseline"
              stroke={climColor}
              strokeWidth={1.8}
              strokeDasharray="2 2"
              dot={{ r: 2, fill: climColor }}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
