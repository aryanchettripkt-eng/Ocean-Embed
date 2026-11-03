"use client";

/**
 * Lasso stats panel (PRD_FRONTEND §3.4e). Shows the /segment stats with the method,
 * provider label and SEGMENT_CAVEAT verbatim; "Ask about this area" hands the selection's
 * bbox and stats to the guide.
 */

import React from "react";
import { Loader2, MessageCircle, X } from "lucide-react";

import { ApiError, type SegmentResponse } from "../../lib/api/client";
import { formatDepth, formatTemp } from "../../lib/format";
import type { BBox } from "../../lib/guide/ui-actions";

export function methodLabel(method: SegmentResponse["method"]): string {
  return method === "sam" ? "Selected by SAM 3.1" : "Selected by similar temperature (±0.5 °C)";
}

export function outlineBBox(outline: number[][]): BBox | null {
  if (!outline.length) return null;
  const lons = outline.map((p) => p[0]);
  const lats = outline.map((p) => p[1]);
  return {
    lat_min: Math.min(...lats),
    lat_max: Math.max(...lats),
    lon_min: Math.min(...lons),
    lon_max: Math.max(...lons),
  };
}

export function segmentErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.code === "point_not_ocean") return "That spot has no ocean value. Tap on the water.";
    if (err.status === 429) return "The lasso is busy right now. Try again in a minute.";
    if (err.status === 503) return "The lasso is switched off at the moment.";
    if (err.status === 502) return "The selection service didn't answer. Please try again.";
  }
  return "Couldn't select that area. Please try again.";
}

export function askAboutAreaPrompt(seg: SegmentResponse): string {
  const b = outlineBBox(seg.outline);
  const box = b
    ? `the box ${b.lat_min.toFixed(2)}–${b.lat_max.toFixed(2)}°N, ${b.lon_min.toFixed(2)}–${b.lon_max.toFixed(2)}°E`
    : `the area around ${seg.lat}°N, ${seg.lon}°E`;
  return `Tell me about ${box} at ${formatDepth(seg.depth_m)} on ${seg.date}. What stands out there compared with the water around it?`;
}

interface LassoPanelProps {
  pending: boolean;
  error: string | null;
  result: SegmentResponse | null;
  onAsk?: (seg: SegmentResponse) => void;
  onClose: () => void;
}

export function LassoPanel({ pending, error, result, onAsk, onClose }: LassoPanelProps) {
  const s = result?.stats;
  return (
    <section
      aria-label="Lasso selection"
      className="w-full sm:w-[320px] rounded-xl border border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-950/95 backdrop-blur-xl shadow-2xl"
    >
      <header className="flex items-center justify-between px-3 py-2 border-b border-slate-200 dark:border-slate-800">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Lasso</h2>
        <button type="button" onClick={onClose} aria-label="Close lasso" className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-900">
          <X className="h-4 w-4 text-slate-500" />
        </button>
      </header>
      <div className="p-3 text-xs text-slate-700 dark:text-slate-300 space-y-2" aria-live="polite">
        {!pending && !error && !result && <p>Tap the water to select an area.</p>}
        {pending && (
          <p role="status" className="flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Selecting…
          </p>
        )}
        {error && !pending && <p className="text-amber-700 dark:text-amber-300">{error}</p>}
        {result && s && !pending && (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1">
              <dt className="text-slate-500">Mean</dt>
              <dd className="font-semibold">{formatTemp(s.mean)}</dd>
              <dt className="text-slate-500">Min / max</dt>
              <dd>
                {formatTemp(s.min)} / {formatTemp(s.max)}
              </dd>
              <dt className="text-slate-500">Std. dev.</dt>
              <dd>{s.std !== null && s.std !== undefined ? `${s.std.toFixed(2)} °C` : "n/a"}</dd>
              <dt className="text-slate-500">Area</dt>
              <dd>{Math.round(s.area_km2).toLocaleString("en-GB")} km²</dd>
              <dt className="text-slate-500">Depth</dt>
              <dd>{formatDepth(result.depth_m)}</dd>
            </dl>
            <p className="text-[11px] text-slate-500">
              {methodLabel(result.method)} · {result.provider_label}
              {result.capped ? " · area capped" : ""}
            </p>
            <p className="text-[10px] leading-snug text-slate-500">{result.caveat}</p>
            {onAsk && (
              <button
                type="button"
                onClick={() => onAsk(result)}
                className="w-full inline-flex items-center justify-center gap-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium px-3 py-1.5"
              >
                <MessageCircle className="h-3.5 w-3.5" /> Ask about this area
              </button>
            )}
          </>
        )}
      </div>
    </section>
  );
}

export default LassoPanel;
