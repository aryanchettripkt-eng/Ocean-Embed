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
    <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-xl space-y-4">
      {/* Header and Tab Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2">
          <TableIcon className="h-4 w-4 text-cyan-400" />
          <h3 className="font-semibold text-slate-100 text-sm">
            Validation Error Breakdown
          </h3>
        </div>

        {/* Tab Buttons */}
        <div className="inline-flex rounded-lg bg-slate-950 p-1 border border-slate-800">
          {(["depth", "season", "region"] as TabType[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all capitalize ${
                activeTab === tab
                  ? "bg-slate-800 text-cyan-300 font-semibold shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              By {tab}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="h-48 flex items-center justify-center text-slate-500 text-xs">
          Loading metrics table...
        </div>
      ) : sortedRows.length === 0 ? (
        <div className="p-6 text-center text-slate-500 text-xs">
          No metrics available for this breakdown.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950/80 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th
                  onClick={() => handleSort("group")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"
                >
                  <div className="flex items-center gap-1">
                    <span>{activeTab === "depth" ? "Depth" : "Group"}</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("model")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"
                >
                  <div className="flex items-center gap-1">
                    <span>Model</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("n")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>N Samples</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("mae_c")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>MAE (°C)</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("rmse_c")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>RMSE (°C)</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("bias_c")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Bias (°C)</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("skill_vs_clim_pct")}
                  className="p-3 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap text-right"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Skill vs Clim</span>
                    <ArrowUpDown className="h-3 w-3 opacity-60" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
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
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isModel ? "bg-cyan-950/20" : ""
                    }`}
                  >
                    <td className="p-3 font-sans font-medium text-slate-200">
                      {groupLabel}
                    </td>
                    <td className="p-3 font-sans">
                      <span
                        className={
                          isModel
                            ? "text-cyan-300 font-semibold"
                            : "text-slate-400"
                        }
                      >
                        {row.model}
                      </span>
                    </td>
                    <td className="p-3 text-right text-slate-400">
                      {row.n.toLocaleString()}
                    </td>
                    <td className="p-3 text-right text-slate-200">
                      {row.mae_c !== null && row.mae_c !== undefined
                        ? row.mae_c.toFixed(2)
                        : "n/a"}
                    </td>
                    <td className="p-3 text-right text-cyan-300 font-semibold">
                      {row.rmse_c !== null && row.rmse_c !== undefined
                        ? row.rmse_c.toFixed(2)
                        : "n/a"}
                    </td>
                    <td className="p-3 text-right text-slate-300">
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
                              ? "text-emerald-400 font-semibold"
                              : "text-slate-500"
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
    </div>
  );
}
