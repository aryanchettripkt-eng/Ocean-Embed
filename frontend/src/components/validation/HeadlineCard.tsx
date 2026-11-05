"use client";

import React from "react";
import { CheckCircle2, XCircle, Clock, Award } from "lucide-react";
import type { HeadlineResponse } from "../../lib/api/client";
import { Pulse } from "../common/Motion";

export interface HeadlineCardProps {
  headline?: HeadlineResponse | null;
  isLoading?: boolean;
  reportedPeriod?: string;
}

export function HeadlineCard({
  headline,
  isLoading,
  reportedPeriod,
}: HeadlineCardProps) {
  if (isLoading) {
    return (
      <div className="hud-glass rounded-[20px] p-6" data-testid="headline-loading">
        <Pulse className="mb-4 block h-6 w-48 rounded bg-black/[0.06] dark:bg-white/[0.06]" />
        <Pulse className="mb-2 block h-4 w-full rounded bg-black/[0.06] dark:bg-white/[0.06]" />
        <Pulse className="block h-4 w-3/4 rounded bg-black/[0.06] dark:bg-white/[0.06]" />
      </div>
    );
  }

  if (!headline) {
    return null;
  }

  const period =
    reportedPeriod ??
    (headline.rows.some((r) => r.period === "val")
      ? "val"
      : headline.rows[0]?.period ?? "val");

  const rows = headline.rows.filter((r) => r.period === period);

  // Determine verdict display
  const verdictConfig = (() => {
    if (headline.met === true) {
      return {
        label: "Met",
        bg: "bg-emerald-500/10 ring-emerald-500/35 text-emerald-800 dark:text-emerald-200",
        icon: <CheckCircle2 aria-hidden="true" className="h-5 w-5 text-emerald-600 dark:text-emerald-300" />,
      };
    }
    if (headline.met === false) {
      return {
        label: "Not met",
        bg: "bg-rose-500/10 ring-rose-500/35 text-rose-800 dark:text-rose-200",
        icon: <XCircle aria-hidden="true" className="h-5 w-5 text-rose-600 dark:text-rose-300" />,
      };
    }
    return {
      label: "Not yet evaluated",
      bg: "bg-amber-500/10 ring-amber-500/35 text-amber-800 dark:text-amber-200",
      icon: <Clock aria-hidden="true" className="h-5 w-5 text-amber-600 dark:text-amber-300" />,
    };
  })();

  // Find min and max for dot-and-whisker scale from valid numbers
  const validValues = rows.flatMap((r) =>
    [r.ci_low_c, r.ci_high_c, r.rmse_c].filter(
      (v): v is number => v !== null && v !== undefined && !isNaN(v)
    )
  );
  const minVal =
    validValues.length > 0
      ? Math.floor(Math.min(...validValues) * 10) / 10 - 0.05
      : 0;
  const maxVal =
    validValues.length > 0
      ? Math.ceil(Math.max(...validValues) * 10) / 10 + 0.05
      : 1;
  const valSpan = Math.max(maxVal - minVal, 0.1);

  return (
    <div
      data-testid="headline-card"
      className="hud-glass relative overflow-hidden rounded-[20px] p-5 sm:p-6"
    >
      {/* Decorative gradient glow */}
      <div className="absolute top-0 right-0 -mt-8 -mr-8 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="hud-divider flex flex-col justify-between gap-4 border-b pb-5 md:flex-row md:items-start">
        <div className="space-y-1.5 max-w-3xl">
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-cyan-700 dark:text-cyan-300">
            <Award aria-hidden="true" className="h-4 w-4" />
            <span>Pre-Registered Primary Success Criterion</span>
            <span className="text-neutral-400">•</span>
            <span className="text-neutral-500 dark:text-neutral-400">{period} split</span>
          </div>
          <h2 className="text-lg font-medium leading-snug tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif] sm:text-xl">
            {headline.criterion}
          </h2>
          <p className="text-xs text-neutral-600 dark:text-neutral-400">
            Float-bootstrap 95% confidence intervals against held-out independent ARGO floats in the Bay of Bengal (50–300 m thermocline).
          </p>
        </div>

        <div className="shrink-0">
          <div
            data-testid="headline-verdict"
            className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ring-1 ring-inset ${verdictConfig.bg}`}
          >
            {verdictConfig.icon}
            <span>Verdict: {verdictConfig.label}</span>
          </div>
        </div>
      </div>

      {/* Competitor Comparison Dot-and-Whisker Visualization */}
      <div className="mt-6 space-y-4">
        <div className="flex items-center justify-between px-1 text-xs text-neutral-500 dark:text-neutral-400">
          <span className="font-medium">Product / Model</span>
          <span className="text-[11px]">
            RMSE (°C) [95% CI] &amp; Diff vs Model
          </span>
        </div>

        <div className="space-y-3">
          {rows.map((row) => {
            const isModel =
              row.product.toLowerCase().includes("oceanscope") ||
              row.product.toLowerCase().includes("model") ||
              row.product.includes("SYNTHETIC");

            const hasData =
              row.rmse_c !== null &&
              row.rmse_c !== undefined &&
              row.ci_low_c !== null &&
              row.ci_low_c !== undefined &&
              row.ci_high_c !== null &&
              row.ci_high_c !== undefined;

            const leftPct = hasData
              ? ((row.ci_low_c! - minVal) / valSpan) * 100
              : 0;
            const rightPct = hasData
              ? ((row.ci_high_c! - minVal) / valSpan) * 100
              : 0;
            const dotPct = hasData
              ? ((row.rmse_c! - minVal) / valSpan) * 100
              : 0;
            const widthPct = Math.max(rightPct - leftPct, 1.5);

            return (
              <div
                key={row.product}
                data-testid={`headline-row-${row.product}`}
                className={`rounded-xl p-3.5 ring-1 ring-inset transition-colors ${
                  isModel
                    ? "bg-cyan-500/[0.07] ring-cyan-500/30"
                    : "bg-black/[0.025] ring-black/[0.05] dark:bg-white/[0.03] dark:ring-white/[0.06]"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2 text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-semibold ${
                        isModel ? "text-cyan-800 dark:text-cyan-200" : "text-neutral-900 dark:text-neutral-100"
                      }`}
                    >
                      {row.product}
                    </span>
                    {isModel && (
                      <span className="rounded-full bg-cyan-500/15 px-2 py-0.5 text-[10px] font-medium text-cyan-800 dark:text-cyan-200">
                        Our Model
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-[11px] tabular-nums">
                    <span className="font-semibold text-neutral-950 dark:text-white">
                      {row.rmse_c !== null && row.rmse_c !== undefined
                        ? `${row.rmse_c.toFixed(2)} °C`
                        : "n/a"}
                    </span>
                    <span className="text-neutral-500">
                      {row.ci_low_c !== null &&
                      row.ci_low_c !== undefined &&
                      row.ci_high_c !== null &&
                      row.ci_high_c !== undefined
                        ? `[${row.ci_low_c.toFixed(2)}, ${row.ci_high_c.toFixed(2)}]`
                        : "[n/a, n/a]"}
                    </span>
                    {!isModel && (
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                          row.diff_vs_model_c !== null &&
                          row.diff_vs_model_c !== undefined &&
                          row.diff_vs_model_c > 0
                            ? "bg-emerald-500/10 text-emerald-800 ring-1 ring-inset ring-emerald-500/30 dark:text-emerald-200"
                            : "bg-black/[0.05] text-neutral-600 dark:bg-white/[0.06] dark:text-neutral-400"
                        }`}
                      >
                        {row.diff_vs_model_c !== null &&
                        row.diff_vs_model_c !== undefined
                          ? `${row.diff_vs_model_c > 0 ? "+" : ""}${row.diff_vs_model_c.toFixed(2)} °C`
                          : "n/a"}
                      </span>
                    )}
                  </div>
                </div>

                {/* Horizontal Dot and Whisker Bar */}
                <div className="relative flex h-4 w-full items-center rounded-full bg-black/[0.05] px-1 dark:bg-white/[0.06]">
                  {hasData ? (
                    <>
                      {/* CI Range Whisker */}
                      <div
                        className={`absolute h-1.5 rounded-full ${
                          isModel ? "bg-cyan-600/50 dark:bg-cyan-400/50" : "bg-neutral-400/60 dark:bg-neutral-500/60"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, leftPct))}%`,
                          width: `${Math.max(0, Math.min(100 - leftPct, widthPct))}%`,
                        }}
                      />
                      {/* CI Whisker Ends */}
                      <div
                        className={`absolute h-3 w-0.5 ${
                          isModel ? "bg-cyan-600 dark:bg-cyan-300" : "bg-neutral-500"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, leftPct))}%`,
                        }}
                      />
                      <div
                        className={`absolute h-3 w-0.5 ${
                          isModel ? "bg-cyan-600 dark:bg-cyan-300" : "bg-neutral-500"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, rightPct))}%`,
                        }}
                      />
                      {/* Point Estimate Dot */}
                      <div
                        className={`absolute h-3 w-3 rounded-full border-2 transform -translate-x-1.5 shadow-md ${
                          isModel
                            ? "bg-cyan-500 border-white dark:bg-cyan-300 dark:border-neutral-900"
                            : "bg-violet-500 border-white dark:bg-violet-300 dark:border-neutral-900"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, dotPct))}%`,
                        }}
                      />
                    </>
                  ) : (
                    <span className="pl-2 text-[10px] italic text-neutral-500">
                      n/a — not evaluated for this period
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
