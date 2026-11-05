"use client";

/**
 * Lasso stats panel (PRD_FRONTEND §3.4e). Shows the /segment stats with the method,
 * provider label and SEGMENT_CAVEAT verbatim; "Ask about this area" hands the selection's
 * bbox and stats to the guide.
 */

import React from "react";
import { MessageCircle, X } from "lucide-react";

import { ApiError, type SegmentResponse } from "../../lib/api/client";
import { formatDepth, formatTemp } from "../../lib/format";
import type { BBox } from "../../lib/guide/ui-actions";
import { TypingDots } from "../common/TypingDots";

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

export function segmentErrorMessage(err: unknown, timedOut = false): string {
  if (timedOut) return "The lasso took too long to answer. Please try again.";
  if (err instanceof ApiError) {
    if (err.code === "point_not_ocean") return "That spot has no ocean value. Tap on the water.";
    if (err.status === 429) return "The lasso is busy right now. Try again in a minute.";
    if (err.status === 503) return "The lasso is switched off at the moment.";
    if (err.status === 502) return "The selection service didn't answer. Please try again.";
    if (err.status === 404) return "There is no map for that date and depth. Pick another date.";
    // Name the code so a report from the site says what went wrong.
    return `Couldn't select that area (${err.code || err.status}). Please try again.`;
  }
  return "Couldn't reach the lasso service. Check your connection and try again.";
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
      className="hud-glass w-full overflow-hidden rounded-[20px] sm:w-[320px]"
    >
      <header className="hud-divider flex items-center justify-between gap-3 border-b px-4 py-3">
        <h2 className="text-sm font-semibold text-neutral-950 dark:text-white">Lasso</h2>
        <button type="button" onClick={onClose} aria-label="Close lasso" className="hud-icon-btn">
          <X aria-hidden="true" className="h-4 w-4" />
        </button>
      </header>
      <div className="space-y-2.5 p-4 text-xs text-neutral-700 dark:text-neutral-300" aria-live="polite">
        {!pending && !error && !result && <p>Tap the water to select an area.</p>}
        {pending && (
          <p role="status" className="flex items-center gap-2.5 text-neutral-500 dark:text-neutral-400">
            <TypingDots label="Selecting" /> <span aria-hidden="true">Selecting…</span>
          </p>
        )}
        {error && !pending && <p className="rounded-xl bg-amber-500/10 px-3 py-2 text-amber-800 ring-1 ring-inset ring-amber-500/25 dark:text-amber-200">{error}</p>}
        {result && s && !pending && (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 tabular-nums">
              <dt className="text-neutral-500 dark:text-neutral-400">Mean</dt>
              <dd className="font-semibold text-neutral-950 dark:text-white">{formatTemp(s.mean)}</dd>
              <dt className="text-neutral-500 dark:text-neutral-400">Min / max</dt>
              <dd>
                {formatTemp(s.min)} / {formatTemp(s.max)}
              </dd>
              <dt className="text-neutral-500 dark:text-neutral-400">Std. dev.</dt>
              <dd>{s.std !== null && s.std !== undefined ? `${s.std.toFixed(2)} °C` : "n/a"}</dd>
              <dt className="text-neutral-500 dark:text-neutral-400">Area</dt>
              <dd>{Math.round(s.area_km2).toLocaleString("en-GB")} km²</dd>
              <dt className="text-neutral-500 dark:text-neutral-400">Depth</dt>
              <dd>{formatDepth(result.depth_m)}</dd>
            </dl>
            <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
              {methodLabel(result.method)} · {result.provider_label}
              {result.capped ? " · area capped" : ""}
            </p>
            <p className="text-[10px] leading-snug text-neutral-500">{result.caveat}</p>
            {onAsk && (
              <button
                type="button"
                onClick={() => onAsk(result)}
                className="hud-btn w-full bg-cyan-400 font-semibold text-neutral-950 hover:bg-cyan-300 hover:text-neutral-950 dark:text-neutral-950 dark:hover:bg-cyan-300 dark:hover:text-neutral-950"
              >
                <MessageCircle aria-hidden="true" className="h-3.5 w-3.5" /> Ask about this area
              </button>
            )}
          </>
        )}
      </div>
    </section>
  );
}

export default LassoPanel;
