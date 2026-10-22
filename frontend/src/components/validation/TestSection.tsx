"use client";

import React from "react";
import { Lock, ShieldCheck } from "lucide-react";
import type { MetricsResponse } from "../../lib/api/client";

interface TestSectionProps {
  testEvaluated: boolean;
  testMetrics?: MetricsResponse | null;
}

export function TestSection({ testEvaluated, testMetrics }: TestSectionProps) {
  if (testEvaluated && testMetrics) {
    return (
      <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-xl space-y-4">
        <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
          <ShieldCheck className="h-4 w-4" />
          <span>Final Test Split Evaluation (Held-Out)</span>
        </div>
        <p className="text-xs text-slate-300">
          Evaluated on the sealed test period after final checkpoint selection.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="space-y-1">
        <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold uppercase tracking-wider">
          <Lock className="h-4 w-4" />
          <span>Sealed Test Protocol (PRD_AI_ML §0)</span>
        </div>
        <h3 className="text-base font-semibold text-slate-200">
          Final test-period evaluation not yet run.
        </h3>
        <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
          To prevent data leakage, hyperparameter snooping, and benchmark degradation, the 2019 test split is permanently sealed until model training and validation decisions are complete.
        </p>
      </div>
      <div className="shrink-0">
        <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-950/40 border border-amber-800/50 text-amber-300 text-xs font-mono font-medium">
          <Lock className="h-3 w-3" />
          Sealed
        </span>
      </div>
    </div>
  );
}
