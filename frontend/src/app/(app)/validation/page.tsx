"use client";

import React from "react";
import {
  useArgo,
  useHeadline,
  useMeta,
  useMetrics,
} from "../../../lib/api/hooks";
import { PageHero, heroActionClass } from "../../../components/common/PageHero";
import { manrope } from "../../../components/landing/fonts";
import Link from "next/link";
import { Info, Waves } from "lucide-react";
import { Loading } from "../../../components/common/Loading";
import { ErrorState } from "../../../components/common/ErrorState";
import { ErrorBoundary } from "../../../components/common/ErrorBoundary";
import { HeadlineCard } from "../../../components/validation/HeadlineCard";
import { MetricCards } from "../../../components/validation/MetricCards";
import { DepthErrorChart } from "../../../components/validation/DepthErrorChart";
import { MetricsTable } from "../../../components/validation/MetricsTable";
import { ArgoSection } from "../../../components/validation/ArgoSection";
import { TestSection } from "../../../components/validation/TestSection";

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
      <div className="flex min-h-[60vh] flex-1 items-center justify-center bg-neutral-100 p-6 dark:bg-[#0a0c0d]">
        <ErrorState
          title="Could not load validation data"
          message={metaError instanceof Error ? metaError.message : "API unavailable"}
          onRetry={() => refetchMeta()}
        />
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex-1 bg-neutral-100 text-neutral-900 dark:bg-[#0a0c0d] dark:text-neutral-100 ${manrope.variable}`}>
      <PageHero
        eyebrow="Model Performance & Scientific Integrity"
        title="Validation & Baseline Benchmarks"
        actions={
          <>
            <Link href="/explore" className={heroActionClass}>
              <Waves aria-hidden="true" className="h-3.5 w-3.5" /> Open the Explorer
            </Link>
            <Link href="/about#method" className={heroActionClass}>
              How the model works
            </Link>
          </>
        }
      >
        Rigorous held-out validation across {meta.depths_m.length} depth tiers in the North Indian Ocean (
        {meta.region.lon_min}°–{meta.region.lon_max}°E, {meta.region.lat_min}°–{meta.region.lat_max}°N). Every figure
        and table is sourced directly from precomputed model evaluations and verified against physical reanalyses and
        in-situ ARGO profiles.
      </PageHero>

      <div className="relative mx-auto -mt-20 max-w-6xl space-y-6 px-4 pb-16 sm:px-6 lg:px-8">
        {/* GLORYS Reanalysis Caveat */}
        {meta.caveats?.glorys && (
          <div className="hud-glass flex items-start gap-3 rounded-2xl px-4 py-3 text-xs leading-relaxed text-neutral-700 dark:text-neutral-300">
            <Info aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-cyan-700 dark:text-cyan-300" />
            <p>
              <strong className="font-medium text-neutral-950 dark:text-white">GLORYS Reference Caveat: </strong>
              {meta.caveats.glorys}
            </p>
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
