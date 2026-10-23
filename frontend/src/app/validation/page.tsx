"use client";

import React from "react";
import {
  useArgo,
  useHeadline,
  useMeta,
  useMetrics,
} from "../../lib/api/hooks";
import { Banner } from "../../components/common/Banner";
import { Loading } from "../../components/common/Loading";
import { ErrorState } from "../../components/common/ErrorState";
import { ErrorBoundary } from "../../components/common/ErrorBoundary";
import { HeadlineCard } from "../../components/validation/HeadlineCard";
import { MetricCards } from "../../components/validation/MetricCards";
import { DepthErrorChart } from "../../components/validation/DepthErrorChart";
import { MetricsTable } from "../../components/validation/MetricsTable";
import { ArgoSection } from "../../components/validation/ArgoSection";
import { TestSection } from "../../components/validation/TestSection";

export default function ValidationPage() {
  const { data: meta, isLoading: isMetaLoading, error: metaError, refetch: refetchMeta } = useMeta();
  const { data: headline, isLoading: isHeadlineLoading } = useHeadline();
  const { data: depthMetrics, isLoading: isDepthMetricsLoading } = useMetrics({
    split: "val",
    group_by: "depth",
  });
  const { data: seasonMetrics } = useMetrics({ split: "val", group_by: "season" });
  const { data: regionMetrics } = useMetrics({ split: "val", group_by: "region" });
  const { data: argo, isLoading: isArgoLoading } = useArgo();

  if (isMetaLoading) {
    return <Loading fullScreen label="Loading validation benchmarks..." />;
  }

  if (metaError || !meta) {
    return (
      <div className="flex-1 flex items-center justify-center p-6 min-h-[60vh] bg-slate-50 dark:bg-slate-950">
        <ErrorState
          title="Could not load validation data"
          message={metaError instanceof Error ? metaError.message : "API unavailable"}
          onRetry={() => refetchMeta()}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 py-8 px-4 sm:px-6 lg:px-8 transition-colors">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Synthetic Data Warning Banner */}
        {meta.synthetic && <Banner type="synthetic" />}

        {/* Page Header */}
        <div className="space-y-2 border-b border-slate-200 dark:border-slate-800 pb-5">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
            <span>Model Performance &amp; Scientific Integrity</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Validation &amp; Baseline Benchmarks
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 max-w-3xl leading-relaxed">
            Rigorous held-out validation across 15 depth tiers in the North Indian Ocean (40°–100°E, 0°–30°N). Every figure and table is sourced directly from precomputed model evaluations and verified against physical reanalyses and in-situ ARGO profiles.
          </p>
        </div>

        {/* GLORYS Reanalysis Caveat */}
        {meta.caveats?.glorys && (
          <div className="bg-white/90 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-xl p-4 text-xs text-slate-700 dark:text-slate-300 leading-relaxed shadow-sm">
            <strong className="text-cyan-600 dark:text-cyan-300 font-semibold">GLORYS Reference Caveat: </strong>
            {meta.caveats.glorys}
          </div>
        )}

        {/* 1. Pre-Registered Primary Headline Criterion */}
        <ErrorBoundary name="Headline Criterion Card">
          <HeadlineCard
            headline={headline}
            isLoading={isHeadlineLoading}
            reportedPeriod={meta.test_evaluated ? "test" : "val"}
          />
        </ErrorBoundary>

        {/* 2. Overall Validation Summary Metric Cards */}
        <ErrorBoundary name="Validation Metric Summary">
          <MetricCards metrics={depthMetrics} isLoading={isDepthMetricsLoading} />
        </ErrorBoundary>

        {/* 3. Depth Error Curve (RMSE vs Depth) */}
        <ErrorBoundary name="Depth Error Curves">
          <DepthErrorChart metrics={depthMetrics} isLoading={isDepthMetricsLoading} />
        </ErrorBoundary>

        {/* 4. Detailed Error Breakdown Tables (Depth, Season, Region) */}
        <ErrorBoundary name="Metrics Breakdown Tables">
          <MetricsTable
            depthMetrics={depthMetrics}
            seasonMetrics={seasonMetrics}
            regionMetrics={regionMetrics}
            isLoading={isDepthMetricsLoading}
          />
        </ErrorBoundary>

        {/* 5. Independent ARGO Float Matchup Comparison */}
        <ErrorBoundary name="ARGO Matchup Evaluation">
          <ArgoSection argo={argo} isLoading={isArgoLoading} />
        </ErrorBoundary>

        {/* 6. Sealed Test Split Status */}
        <ErrorBoundary name="Test Split Status">
          <TestSection
            testEvaluated={meta.test_evaluated}
          />
        </ErrorBoundary>
      </div>
    </div>
  );
}
