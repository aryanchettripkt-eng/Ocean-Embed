"use client";

import React from "react";
import { Compass, AlertCircle } from "lucide-react";
import { formatDepth } from "../../lib/format";
import type { ArgoResponse } from "../../lib/api/client";
import { Pulse } from "../common/Motion";

interface ArgoSectionProps {
  argo?: ArgoResponse | null;
  isLoading?: boolean;
}

export function ArgoSection({ argo, isLoading }: ArgoSectionProps) {
  if (isLoading) {
    return (
      <Pulse className="block h-64 rounded-[20px] bg-black/[0.05] dark:bg-white/[0.05]" />
    );
  }

  if (!argo) {
    return null;
  }

  const num = (v: number | null | undefined) => (v !== null && v !== undefined ? `${v.toFixed(2)} °C` : "n/a");

  return (
    <section className="hud-glass space-y-4 rounded-[20px] p-5 sm:p-6">
      <div className="hud-divider flex flex-col justify-between gap-2 border-b pb-4 sm:flex-row sm:items-end">
        <div>
          <div className="hud-label flex items-center gap-2">
            <Compass aria-hidden="true" className="h-3.5 w-3.5" />
            <span>Independent In-Situ Validation</span>
          </div>
          <h3 className="mt-1 text-lg font-medium tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif]">
            Held-Out ARGO Float Matchup Comparison
          </h3>
        </div>
        <div className="text-xs tabular-nums text-neutral-500 dark:text-neutral-400">
          {argo.n_profiles_total} profiles matched ({argo.n_obs_total} total depth observations)
        </div>
      </div>

      {/* ARGO Caveat Box */}
      <div className="flex items-start gap-3 rounded-xl bg-amber-500/10 px-3.5 py-3 text-xs text-amber-900 ring-1 ring-inset ring-amber-500/25 dark:text-amber-100">
        <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-300" />
        <p className="leading-relaxed">
          <strong className="font-semibold">Observation Independence Caveat: </strong>
          {argo.caveat}
        </p>
      </div>

      {/* Per-depth error comparison table */}
      <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-black/[0.06] dark:ring-white/[0.07]">
        <table className="w-full border-collapse text-left text-xs">
          <thead className="bg-black/[0.03] text-neutral-500 dark:bg-white/[0.04] dark:text-neutral-400">
            <tr>
              <th className="whitespace-nowrap p-3 font-medium">Depth</th>
              <th className="whitespace-nowrap p-3 text-right font-medium text-cyan-700 dark:text-cyan-300">OceanScope RMSE</th>
              <th className="whitespace-nowrap p-3 text-right font-medium text-violet-700 dark:text-violet-300">GLORYS12 RMSE</th>
              <th className="whitespace-nowrap p-3 text-right font-medium">GREP RMSE</th>
              <th className="whitespace-nowrap p-3 text-right font-medium">ARMOR3D RMSE</th>
              <th className="whitespace-nowrap p-3 text-right font-medium">N Obs</th>
            </tr>
          </thead>
          <tbody className="hud-divider divide-y tabular-nums">
            {argo.rows.map((row) => (
              <tr key={row.depth_m} className="transition-colors hover:bg-black/[0.03] dark:hover:bg-white/[0.04]">
                <td className="p-3 font-medium text-neutral-900 dark:text-neutral-100">{formatDepth(row.depth_m)}</td>
                <td className="p-3 text-right font-semibold text-cyan-700 dark:text-cyan-300">{num(row.rmse_c)}</td>
                <td className="p-3 text-right text-violet-700 dark:text-violet-300">{num(row.glorys_rmse_c)}</td>
                <td className="p-3 text-right text-neutral-600 dark:text-neutral-400">{num(row.grep_rmse_c)}</td>
                <td className="p-3 text-right text-neutral-600 dark:text-neutral-400">{num(row.armor3d_rmse_c)}</td>
                <td className="p-3 text-right text-neutral-500">{row.n_obs}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
