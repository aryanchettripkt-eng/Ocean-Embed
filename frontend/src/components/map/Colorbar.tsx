"use client";

import React from "react";
import { getColormapGradientCss, type ColormapName } from "../../lib/colormaps";
import { formatTemp } from "../../lib/format";

interface ColorbarProps {
  field: "pred" | "glorys" | "error";
  vmin: number;
  vmax: number;
  wallRange?: { vmin: number; vmax: number };
  className?: string;
}

export function Colorbar({ field, vmin, vmax, wallRange, className = "" }: ColorbarProps) {
  const colormap: ColormapName = field === "error" ? "balance" : "thermal";
  const gradientCss = getColormapGradientCss(colormap);
  const mid = (vmin + vmax) / 2;

  const title =
    field === "error"
      ? "Difference (Model − GLORYS)"
      : field === "glorys"
      ? "GLORYS Reanalysis Temperature"
      : "Model Predicted Temperature";

  const showWallRange =
    wallRange &&
    (Math.abs(wallRange.vmin - vmin) > 0.1 || Math.abs(wallRange.vmax - vmax) > 0.1);

  return (
    <div
      className={`bg-white/88 dark:bg-[#0b1116]/88 backdrop-blur-xl border border-slate-200/90 dark:border-white/10 rounded-xl p-2.5 shadow-xl text-slate-800 dark:text-slate-200 ${className}`}
    >
      <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
        <span className="truncate">{title}</span>
        <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono">(°C)</span>
      </div>

      {/* Gradient bar */}
      <div
        className="w-full h-2.5 rounded-full border border-slate-300/80 dark:border-white/15 shadow-inner"
        style={{ background: gradientCss }}
      />

      {/* Numerical markers */}
      <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 dark:text-slate-400 mt-1">
        <span>{formatTemp(vmin, 1)}</span>
        <span>{formatTemp(mid, 1)}</span>
        <span>{formatTemp(vmax, 1)}</span>
      </div>

      {showWallRange && (
        <div className="mt-1.5 pt-1.5 border-t border-slate-200 dark:border-white/10 flex items-center justify-between text-[10px] text-slate-500 dark:text-slate-400 font-mono">
          <span>Top face: {formatTemp(vmin, 1)}–{formatTemp(vmax, 1)}</span>
          <span>Walls: {formatTemp(wallRange.vmin, 1)}–{formatTemp(wallRange.vmax, 1)}</span>
        </div>
      )}
    </div>
  );
}
