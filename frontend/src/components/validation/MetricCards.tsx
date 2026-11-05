"use client";

import React, { useMemo } from "react";
import { Activity, BarChart, CheckCircle, TrendingUp } from "lucide-react";
import type { MetricsResponse } from "../../lib/api/client";
import { Pulse } from "../common/Motion";

interface MetricCardsProps {
  metrics?: MetricsResponse | null;
  isLoading?: boolean;
}

export function MetricCards({ metrics, isLoading }: MetricCardsProps) {
  const overallRow = useMemo(() => {
    if (!metrics?.rows) return null;
    return (
      metrics.rows.find(
        (r) =>
          r.depth_m === "all" &&
          r.model !== "climatology" &&
          r.model !== "ridge"
      ) ?? null
    );
  }, [metrics]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Pulse key={i} className="block h-28 rounded-2xl bg-black/[0.05] dark:bg-white/[0.05]" />
        ))}
      </div>
    );
  }

  if (!overallRow) {
    return null;
  }

  const cards = [
    {
      title: "Overall Val RMSE",
      value: `${overallRow.rmse_c?.toFixed(2) ?? "n/a"} °C`,
      desc: "Root-mean-square error (all depths)",
      icon: <Activity aria-hidden="true" className="h-4 w-4 text-cyan-700 dark:text-cyan-300" />,
      highlight: true,
    },
    {
      title: "Mean Absolute Error",
      value: `${overallRow.mae_c?.toFixed(2) ?? "n/a"} °C`,
      desc: "Average absolute discrepancy",
      icon: <BarChart aria-hidden="true" className="h-4 w-4 text-violet-700 dark:text-violet-300" />,
    },
    {
      title: "Mean Bias",
      value: `${overallRow.bias_c !== null && overallRow.bias_c !== undefined ? (overallRow.bias_c > 0 ? "+" : "") + overallRow.bias_c.toFixed(2) : "n/a"} °C`,
      desc: "Systematic temperature offset",
      icon: <CheckCircle aria-hidden="true" className="h-4 w-4 text-emerald-700 dark:text-emerald-300" />,
    },
    {
      title: "Skill vs Climatology",
      value: `${overallRow.skill_vs_clim_pct !== null && overallRow.skill_vs_clim_pct !== undefined ? overallRow.skill_vs_clim_pct.toFixed(1) + "%" : "n/a"}`,
      desc: "MSE reduction over WOA23 baseline",
      icon: <TrendingUp aria-hidden="true" className="h-4 w-4 text-amber-700 dark:text-amber-300" />,
      highlight: true,
    },
  ];

  return (
    <section className="space-y-3">
      <div className="flex flex-col gap-1 px-1 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="hud-label">Domain-Wide Validation Summary</h2>
        <span className="text-[11px] tabular-nums text-neutral-500 dark:text-neutral-400">
          Evaluated across {overallRow.n.toLocaleString()} grid cell observations
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {cards.map((card) => (
          <div
            key={card.title}
            className={`hud-glass rounded-2xl p-4 ${card.highlight ? "ring-1 ring-inset ring-cyan-500/30" : ""}`}
          >
            <div className="mb-3 flex items-center justify-between gap-2">
              <span className="text-xs text-neutral-600 dark:text-neutral-400">{card.title}</span>
              {card.icon}
            </div>
            <div className="text-2xl font-medium tabular-nums tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif] sm:text-[28px]">
              {card.value}
            </div>
            <p className="mt-1 text-[11px] text-neutral-500">{card.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
