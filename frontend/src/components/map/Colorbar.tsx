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
    <div className={`text-neutral-800 dark:text-neutral-200 ${className}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="hud-label truncate">{title}</span>
        <span className="text-[11px] text-neutral-500">°C</span>
      </div>

      <div
        className="h-2 w-full rounded-full ring-1 ring-inset ring-black/10 dark:ring-white/10"
        style={{ background: gradientCss }}
      />

      <div className="mt-1.5 flex items-center justify-between text-[11px] tabular-nums text-neutral-600 dark:text-neutral-400">
        <span>{formatTemp(vmin, 1)}</span>
        <span>{formatTemp(mid, 1)}</span>
        <span>{formatTemp(vmax, 1)}</span>
      </div>

      {showWallRange && (
        <div className="mt-1.5 flex items-center justify-between text-[10px] tabular-nums text-neutral-500">
          <span>Top face {formatTemp(vmin, 1)}–{formatTemp(vmax, 1)}</span>
          <span>Walls {formatTemp(wallRange.vmin, 1)}–{formatTemp(wallRange.vmax, 1)}</span>
        </div>
      )}
    </div>
  );
}
