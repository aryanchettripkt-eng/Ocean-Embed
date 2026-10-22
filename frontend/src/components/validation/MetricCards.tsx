"use client";

import React, { useMemo } from "react";
import { Activity, BarChart, CheckCircle, TrendingUp } from "lucide-react";
import type { MetricsResponse } from "../../lib/api/client";

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
          <div
            key={i}
            className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 h-28 animate-pulse"
          />
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
      icon: <Activity className="h-4 w-4 text-cyan-400" />,
      highlight: true,
    },
    {
      title: "Mean Absolute Error",
      value: `${overallRow.mae_c?.toFixed(2) ?? "n/a"} °C`,
      desc: "Average absolute discrepancy",
      icon: <BarChart className="h-4 w-4 text-indigo-400" />,
    },
    {
      title: "Mean Bias",
      value: `${overallRow.bias_c !== null && overallRow.bias_c !== undefined ? (overallRow.bias_c > 0 ? "+" : "") + overallRow.bias_c.toFixed(2) : "n/a"} °C`,
      desc: "Systematic temperature offset",
      icon: <CheckCircle className="h-4 w-4 text-emerald-400" />,
    },
    {
      title: "Skill vs Climatology",
      value: `${overallRow.skill_vs_clim_pct !== null && overallRow.skill_vs_clim_pct !== undefined ? overallRow.skill_vs_clim_pct.toFixed(1) + "%" : "n/a"}`,
      desc: "MSE reduction over WOA23 baseline",
      icon: <TrendingUp className="h-4 w-4 text-amber-400" />,
      highlight: true,
    },
  ];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-400 px-1">
        <span className="font-semibold uppercase tracking-wider text-slate-300">
          Domain-Wide Validation Summary
        </span>
        <span className="font-mono text-[11px]">
          Evaluated across {overallRow.n.toLocaleString()} grid cell observations
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card) => (
          <div
            key={card.title}
            className={`p-4 rounded-xl border backdrop-blur-md shadow-lg transition-all ${
              card.highlight
                ? "bg-slate-900/90 border-cyan-500/30"
                : "bg-slate-900/60 border-slate-800/80"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400 font-medium">{card.title}</span>
              {card.icon}
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-100">
              {card.value}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">{card.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
