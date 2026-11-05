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
      <section className="hud-glass space-y-3 rounded-[20px] p-5 sm:p-6">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">
          <ShieldCheck aria-hidden="true" className="h-4 w-4" />
          <span>Final Test Split Evaluation (Held-Out)</span>
        </div>
        <p className="text-xs text-neutral-700 dark:text-neutral-300">
          Evaluated on the sealed test period after final checkpoint selection.
        </p>
      </section>
    );
  }

  return (
    <section className="hud-glass flex flex-col items-start justify-between gap-4 rounded-[20px] p-5 sm:flex-row sm:items-center sm:p-6">
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300">
          <Lock aria-hidden="true" className="h-4 w-4" />
          <span>Sealed Test Protocol (PRD_AI_ML §0)</span>
        </div>
        <h3 className="text-base font-medium text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif]">
          Final test-period evaluation not yet run.
        </h3>
        <p className="max-w-2xl text-xs leading-relaxed text-neutral-600 dark:text-neutral-400">
          To prevent data leakage, hyperparameter snooping, and benchmark degradation, the 2019 test split is permanently sealed until model training and validation decisions are complete.
        </p>
      </div>
      <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-800 ring-1 ring-inset ring-amber-500/30 dark:text-amber-200">
        <Lock aria-hidden="true" className="h-3 w-3" />
        Sealed
      </span>
    </section>
  );
}
