"use client";

import React, { useMemo, useState } from "react";
import { ArrowUpDown, Table as TableIcon } from "lucide-react";
import { formatDepth } from "../../lib/format";
import type { MetricsResponse } from "../../lib/api/client";

interface MetricsTableProps {
  depthMetrics?: MetricsResponse | null;
  seasonMetrics?: MetricsResponse | null;
  regionMetrics?: MetricsResponse | null;
  isLoading?: boolean;
}

type TabType = "depth" | "season" | "region";
type SortField = "group" | "model" | "n" | "mae_c" | "rmse_c" | "bias_c" | "skill_vs_clim_pct";
type SortOrder = "asc" | "desc";

export function MetricsTable({
  depthMetrics,
  seasonMetrics,
  regionMetrics,
  isLoading,
}: MetricsTableProps) {
  const [activeTab, setActiveTab] = useState<TabType>("depth");
  const [sortField, setSortField] = useState<SortField>("rmse_c");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");

  const currentMetrics = useMemo(() => {
    switch (activeTab) {
      case "season":
        return seasonMetrics;
      case "region":
        return regionMetrics;
      default:
        return depthMetrics;
    }
  }, [activeTab, depthMetrics, seasonMetrics, regionMetrics]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  };

  const sortedRows = useMemo(() => {
    if (!currentMetrics?.rows) return [];

    return [...currentMetrics.rows].sort((a, b) => {
      let aVal: unknown = "";
      let bVal: unknown = "";

      switch (sortField) {
        case "group":
          aVal = activeTab === "depth" ? (a.depth_m === "all" ? 9999 : Number(a.depth_m ?? 0)) : (a.group ?? "");
          bVal = activeTab === "depth" ? (b.depth_m === "all" ? 9999 : Number(b.depth_m ?? 0)) : (b.group ?? "");
          break;
        case "model":
          aVal = a.model;
          bVal = b.model;
          break;
        case "n":
          aVal = a.n;
          bVal = b.n;
          break;
        case "mae_c":
          aVal = a.mae_c ?? 999;
          bVal = b.mae_c ?? 999;
          break;
        case "rmse_c":
          aVal = a.rmse_c ?? 999;
          bVal = b.rmse_c ?? 999;
          break;
        case "bias_c":
          aVal = a.bias_c ?? 999;
          bVal = b.bias_c ?? 999;
          break;
        case "skill_vs_clim_pct":
          aVal = a.skill_vs_clim_pct ?? -999;
          bVal = b.skill_vs_clim_pct ?? -999;
          break;
      }

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortOrder === "asc" ? aVal - bVal : bVal - aVal;
      }
      return sortOrder === "asc"
        ? String(aVal).localeCompare(String(bVal))
        : String(bVal).localeCompare(String(aVal));
    });
  }, [currentMetrics, sortField, sortOrder, activeTab]);

  return (
    <section className="hud-glass space-y-4 rounded-[20px] p-5 sm:p-6">
      {/* Header and Tab Selector */}
      <div className="hud-divider flex flex-col justify-between gap-3 border-b pb-4 sm:flex-row sm:items-center">
        <div className="flex items-center gap-2">
          <TableIcon aria-hidden="true" className="h-4 w-4 text-cyan-700 dark:text-cyan-300" />
          <h3 className="text-lg font-medium tracking-tight text-neutral-950 dark:text-white [font-family:var(--font-manrope),sans-serif]">
            Validation Error Breakdown
          </h3>
        </div>

        {/* Tab Buttons */}
        <div role="group" aria-label="Breakdown" className="inline-flex gap-0.5 rounded-full bg-black/[0.04] p-0.5 dark:bg-white/[0.05]">
          {(["depth", "season", "region"] as TabType[]).map((tab) => (
            <button
              key={tab}
              type="button"
              aria-pressed={activeTab === tab}
              onClick={() => setActiveTab(tab)}
              className={`h-7 rounded-full px-3 text-xs font-medium capitalize transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                activeTab === tab
                  ? "bg-neutral-900 text-white shadow-sm dark:bg-white dark:text-neutral-950"
                  : "text-neutral-600 hover:text-neutral-950 dark:text-neutral-400 dark:hover:text-white"
              }`}
            >
              By {tab}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex h-48 items-center justify-center text-xs text-neutral-500">
          Loading metrics table...
        </div>
      ) : sortedRows.length === 0 ? (
        <div className="p-6 text-center text-xs text-neutral-500">
          No metrics available for this breakdown.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-black/[0.06] dark:ring-white/[0.07]">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-black/[0.03] font-medium text-neutral-500 dark:bg-white/[0.04] dark:text-neutral-400">
              <tr>
                <th
                  onClick={() => handleSort("group")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    <span>{activeTab === "depth" ? "Depth" : "Group"}</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("model")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    <span>Model</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("n")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>N Samples</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("mae_c")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>MAE (°C)</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("rmse_c")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>RMSE (°C)</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("bias_c")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Bias (°C)</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("skill_vs_clim_pct")}
                  className="cursor-pointer select-none whitespace-nowrap p-3 font-medium hover:text-neutral-900 dark:hover:text-white text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Skill vs Clim</span>
                    <ArrowUpDown aria-hidden="true" className="h-3 w-3 opacity-60" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="hud-divider divide-y tabular-nums">
              {sortedRows.map((row, idx) => {
                const isModel =
                  row.model.toLowerCase().includes("model") ||
                  row.model.includes("SYNTHETIC");
                const groupLabel =
                  activeTab === "depth"
                    ? row.depth_m === "all"
                      ? "All Depths"
                      : formatDepth(Number(row.depth_m))
                    : row.group ?? "All";

                return (
                  <tr
                    key={`${row.model}-${row.depth_m}-${row.group}-${idx}`}
                    className={`transition-colors hover:bg-black/[0.03] dark:hover:bg-white/[0.04] ${
                      isModel ? "bg-cyan-500/[0.06]" : ""
                    }`}
                  >
                    <td className="p-3 font-medium text-neutral-900 dark:text-neutral-100">
                      {groupLabel}
                    </td>
                    <td className="p-3">
                      <span
                        className={
                          isModel
                            ? "font-semibold text-cyan-800 dark:text-cyan-200"
                            : "text-neutral-600 dark:text-neutral-400"
                        }
                      >
                        {row.model}
                      </span>
                    </td>
                    <td className="p-3 text-right text-neutral-500 dark:text-neutral-400">
                      {row.n.toLocaleString()}
                    </td>
                    <td className="p-3 text-right text-neutral-900 dark:text-neutral-100">
                      {row.mae_c !== null && row.mae_c !== undefined
                        ? row.mae_c.toFixed(2)
                        : "n/a"}
                    </td>
                    <td className="p-3 text-right font-semibold text-cyan-700 dark:text-cyan-300">
                      {row.rmse_c !== null && row.rmse_c !== undefined
                        ? row.rmse_c.toFixed(2)
                        : "n/a"}
                    </td>
                    <td className="p-3 text-right text-neutral-700 dark:text-neutral-300">
                      {row.bias_c !== null && row.bias_c !== undefined
                        ? (row.bias_c > 0 ? "+" : "") + row.bias_c.toFixed(2)
                        : "n/a"}
                    </td>
                    <td className="p-3 text-right">
                      {row.skill_vs_clim_pct !== null &&
                      row.skill_vs_clim_pct !== undefined ? (
                        <span
                          className={
                            row.skill_vs_clim_pct > 0
                              ? "font-semibold text-emerald-700 dark:text-emerald-300"
                              : "text-neutral-500"
                          }
                        >
                          {row.skill_vs_clim_pct.toFixed(1)}%
                        </span>
                      ) : (
                        "n/a"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
