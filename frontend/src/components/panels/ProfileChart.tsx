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
import { useIsDark } from "../../lib/use-dark-mode";
import { CHART_TOOLTIP_CLASS, chartTheme } from "./chartTheme";

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
    <div className={CHART_TOOLTIP_CLASS}>
      <div className="flex justify-between border-b border-black/[0.06] pb-1 font-medium text-neutral-900 dark:border-white/[0.08] dark:text-white">
        <span>Depth</span>
        <span className="tabular-nums">{formatDepth(label)}</span>
      </div>
      <div className="flex items-center justify-between gap-3 text-neutral-600 dark:text-neutral-300">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-cyan-600 dark:bg-cyan-400" />
          <span>Model prediction</span>
        </span>
        <span className="font-medium tabular-nums text-neutral-950 dark:text-white">
          {predVal !== null && predVal !== undefined ? formatTemp(predVal) : "n/a"}
        </span>
      </div>
      {glorysVal !== null && glorysVal !== undefined && (
        <div className="flex items-center justify-between gap-3 text-neutral-600 dark:text-neutral-300">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-violet-600 dark:bg-violet-400" />
            <span>GLORYS reanalysis</span>
          </span>
          <span className="font-medium tabular-nums text-neutral-950 dark:text-white">{formatTemp(glorysVal)}</span>
        </div>
      )}
      {predVal !== null &&
        predVal !== undefined &&
        glorysVal !== null &&
        glorysVal !== undefined && (
          <div className="flex items-center justify-between gap-3 border-t border-black/[0.06] pt-1 text-[11px] text-neutral-500 dark:border-white/[0.08] dark:text-neutral-400">
            <span>Model − GLORYS</span>
            <span className="tabular-nums text-amber-700 dark:text-amber-300">{formatTemp(predVal - glorysVal)}</span>
          </div>
        )}
    </div>
  );
}

export function ProfileChart({ profile, selectedDepth }: ProfileChartProps) {
  const theme = chartTheme(useIsDark());
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
              <span>GLORYS reanalysis</span>
            </span>
          ) : (
            <span className="italic text-neutral-500">(GLORYS n/a for live date)</span>
          )}
        </div>
        <span className="text-[10px] text-amber-700 dark:text-amber-400">Selected: {formatDepth(selectedDepth)}</span>
      </div>

      <div className="min-h-0 w-full flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            layout="vertical"
            data={chartData}
            margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={theme.grid} />
            <XAxis
              type="number"
              domain={[minTemp, maxTemp]}
              tick={{ fill: theme.tick, fontSize: 10 }}
              tickFormatter={(v) => `${v}°`}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
            />
            <YAxis
              type="number"
              dataKey="depth_m"
              reversed={true}
              domain={[0, "dataMax"]}
              tick={{ fill: theme.tick, fontSize: 10 }}
              // Non-breaking space: Recharts wraps tick text at spaces when the axis is narrow
              tickFormatter={(d) => formatDepth(d).replace(" ", "\u00a0")}
              axisLine={{ stroke: theme.axis }}
              tickLine={{ stroke: theme.axis }}
              width={52}
            />
            <Tooltip content={<ProfileTooltip />} />
            <ReferenceLine
              y={selectedDepth}
              stroke={theme.marker}
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Model prediction"
              stroke={theme.pred}
              strokeWidth={2}
              dot={{ r: 2.5, fill: theme.pred }}
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
