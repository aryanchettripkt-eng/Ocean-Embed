"use client";

import React from "react";
import { Compass, AlertCircle } from "lucide-react";
import { formatDepth } from "../../lib/format";
import type { ArgoResponse } from "../../lib/api/client";

interface ArgoSectionProps {
  argo?: ArgoResponse | null;
  isLoading?: boolean;
}

export function ArgoSection({ argo, isLoading }: ArgoSectionProps) {
  if (isLoading) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 h-64 animate-pulse" />
    );
  }

  if (!argo) {
    return null;
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-xl space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Compass className="h-4 w-4" />
            <span>Independent In-Situ Validation</span>
          </div>
          <h3 className="text-base font-semibold text-slate-100 mt-0.5">
            Held-Out ARGO Float Matchup Comparison
          </h3>
        </div>
        <div className="text-xs text-slate-400 font-mono">
          {argo.n_profiles_total} profiles matched ({argo.n_obs_total} total depth observations)
        </div>
      </div>

      {/* ARGO Caveat Box */}
      <div className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-3 flex items-start gap-3 text-xs text-amber-200/90">
        <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong className="font-semibold text-amber-300">Observation Independence Caveat: </strong>
          {argo.caveat}
        </p>
      </div>

      {/* Per-depth error comparison table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800/80">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="bg-slate-950/80 text-slate-400 font-medium border-b border-slate-800">
            <tr>
              <th className="p-3 whitespace-nowrap">Depth</th>
              <th className="p-3 text-right whitespace-nowrap text-cyan-300">
                OceanScope RMSE
              </th>
              <th className="p-3 text-right whitespace-nowrap text-indigo-300">
                GLORYS12 RMSE
              </th>
              <th className="p-3 text-right whitespace-nowrap text-slate-300">
                GREP RMSE
              </th>
              <th className="p-3 text-right whitespace-nowrap text-slate-300">
                ARMOR3D RMSE
              </th>
              <th className="p-3 text-right whitespace-nowrap">N Obs</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {argo.rows.map((row) => (
              <tr
                key={row.depth_m}
                className="hover:bg-slate-800/40 transition-colors"
              >
                <td className="p-3 font-sans font-medium text-slate-200">
                  {formatDepth(row.depth_m)}
                </td>
                <td className="p-3 text-right font-semibold text-cyan-300">
                  {row.rmse_c !== null && row.rmse_c !== undefined
                    ? `${row.rmse_c.toFixed(2)} °C`
                    : "n/a"}
                </td>
                <td className="p-3 text-right text-indigo-300">
                  {row.glorys_rmse_c !== null && row.glorys_rmse_c !== undefined
                    ? `${row.glorys_rmse_c.toFixed(2)} °C`
                    : "n/a"}
                </td>
                <td className="p-3 text-right text-slate-400">
                  {row.grep_rmse_c !== null && row.grep_rmse_c !== undefined
                    ? `${row.grep_rmse_c.toFixed(2)} °C`
                    : "n/a"}
                </td>
                <td className="p-3 text-right text-slate-400">
                  {row.armor3d_rmse_c !== null && row.armor3d_rmse_c !== undefined
                    ? `${row.armor3d_rmse_c.toFixed(2)} °C`
                    : "n/a"}
                </td>
                <td className="p-3 text-right text-slate-500">{row.n_obs}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
