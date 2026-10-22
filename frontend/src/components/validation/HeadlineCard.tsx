"use client";

import React from "react";
import { CheckCircle2, XCircle, Clock, Award } from "lucide-react";
import type { HeadlineResponse } from "../../lib/api/client";

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
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 animate-pulse" data-testid="headline-loading">
        <div className="h-6 w-48 bg-slate-800 rounded mb-4" />
        <div className="h-4 w-full bg-slate-800 rounded mb-2" />
        <div className="h-4 w-3/4 bg-slate-800 rounded" />
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
        bg: "bg-emerald-950/60 border-emerald-500/40 text-emerald-300",
        icon: <CheckCircle2 className="h-5 w-5 text-emerald-400" />,
      };
    }
    if (headline.met === false) {
      return {
        label: "Not met",
        bg: "bg-rose-950/60 border-rose-500/40 text-rose-300",
        icon: <XCircle className="h-5 w-5 text-rose-400" />,
      };
    }
    return {
      label: "Not yet evaluated",
      bg: "bg-amber-950/60 border-amber-500/40 text-amber-300",
      icon: <Clock className="h-5 w-5 text-amber-400" />,
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
      className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-6 shadow-2xl backdrop-blur-xl relative overflow-hidden"
    >
      {/* Decorative gradient glow */}
      <div className="absolute top-0 right-0 -mt-8 -mr-8 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div className="space-y-1.5 max-w-3xl">
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Award className="h-4 w-4" />
            <span>Pre-Registered Primary Success Criterion</span>
            <span className="text-slate-500">•</span>
            <span className="text-slate-400 uppercase">{period} split</span>
          </div>
          <h2 className="text-base sm:text-lg font-semibold text-slate-100 leading-snug">
            {headline.criterion}
          </h2>
          <p className="text-xs text-slate-400">
            Float-bootstrap 95% confidence intervals against held-out independent ARGO floats in the Bay of Bengal (50–300 m thermocline).
          </p>
        </div>

        <div className="shrink-0">
          <div
            data-testid="headline-verdict"
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border font-semibold text-sm shadow-lg ${verdictConfig.bg}`}
          >
            {verdictConfig.icon}
            <span>Verdict: {verdictConfig.label}</span>
          </div>
        </div>
      </div>

      {/* Competitor Comparison Dot-and-Whisker Visualization */}
      <div className="mt-6 space-y-4">
        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <span className="font-medium">Product / Model</span>
          <span className="font-mono text-[11px]">
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
                className={`p-3.5 rounded-xl border transition-all ${
                  isModel
                    ? "bg-cyan-950/40 border-cyan-500/40 shadow-lg"
                    : "bg-slate-950/50 border-slate-800/80"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2 text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-semibold ${
                        isModel ? "text-cyan-300 font-mono" : "text-slate-200"
                      }`}
                    >
                      {row.product}
                    </span>
                    {isModel && (
                      <span className="text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-1.5 py-0.2 rounded font-medium">
                        Our Model
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 font-mono text-[11px]">
                    <span className="text-slate-200 font-semibold">
                      {row.rmse_c !== null && row.rmse_c !== undefined
                        ? `${row.rmse_c.toFixed(2)} °C`
                        : "n/a"}
                    </span>
                    <span className="text-slate-500">
                      {row.ci_low_c !== null &&
                      row.ci_low_c !== undefined &&
                      row.ci_high_c !== null &&
                      row.ci_high_c !== undefined
                        ? `[${row.ci_low_c.toFixed(2)}, ${row.ci_high_c.toFixed(2)}]`
                        : "[n/a, n/a]"}
                    </span>
                    {!isModel && (
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                          row.diff_vs_model_c !== null &&
                          row.diff_vs_model_c !== undefined &&
                          row.diff_vs_model_c > 0
                            ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800/60"
                            : "bg-slate-800 text-slate-400"
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
                <div className="relative w-full h-4 bg-slate-900 rounded-full border border-slate-800/80 flex items-center px-1">
                  {hasData ? (
                    <>
                      {/* CI Range Whisker */}
                      <div
                        className={`absolute h-1.5 rounded-full ${
                          isModel ? "bg-cyan-500/50" : "bg-slate-600/60"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, leftPct))}%`,
                          width: `${Math.max(0, Math.min(100 - leftPct, widthPct))}%`,
                        }}
                      />
                      {/* CI Whisker Ends */}
                      <div
                        className={`absolute h-3 w-0.5 ${
                          isModel ? "bg-cyan-400" : "bg-slate-400"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, leftPct))}%`,
                        }}
                      />
                      <div
                        className={`absolute h-3 w-0.5 ${
                          isModel ? "bg-cyan-400" : "bg-slate-400"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, rightPct))}%`,
                        }}
                      />
                      {/* Point Estimate Dot */}
                      <div
                        className={`absolute h-3 w-3 rounded-full border-2 transform -translate-x-1.5 shadow-md ${
                          isModel
                            ? "bg-cyan-300 border-cyan-900"
                            : "bg-indigo-300 border-indigo-900"
                        }`}
                        style={{
                          left: `${Math.max(0, Math.min(100, dotPct))}%`,
                        }}
                      />
                    </>
                  ) : (
                    <span className="text-[10px] text-slate-500 italic pl-2">
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
