/**
 * Navigator: apply the guide's `ui_actions` to the explorer state (PRD_FRONTEND §3.4a,
 * PRD_BACKEND §6.2). The backend validates and snaps every action; the frontend still
 * re-checks ranges, applies only the four known types, and ignores anything else.
 *
 * Pure functions only, so the whole navigator is unit-tested without a map.
 */

import type { EventItem, MetaResponse, UIAction } from "../api/client";
import { findClosestDepth, type ExplorerState } from "../url-state";
import { formatCoord, formatDate, formatDepth } from "../format";

export type EventPhase = "before" | "during" | "after";

export interface BBox {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

/** Where the camera should go after the state change. `id` changes on every request. */
export type CameraTarget =
  | { kind: "center"; lat: number; lon: number; zoom?: number; bearing?: number }
  | { kind: "bbox"; bbox: BBox };
export type CameraFocus = CameraTarget & { id: number };

export interface AppliedActions {
  state: ExplorerState;
  camera: CameraTarget | null;
  openEvent: { event_id: string; phase?: EventPhase } | null;
  /** Human-readable parts for the "Moved to …" chip; empty when nothing moved. */
  changes: string[];
  /** Action types that were skipped (unknown type or out-of-range values). */
  skipped: string[];
}

const MAX_ACTIONS = 3;
const FIELD_LABELS: Record<ExplorerState["field"], string> = {
  pred: "Prediction",
  glorys: "GLORYS",
  error: "Error",
};

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function inRegion(lat: number, lon: number, meta: MetaResponse): boolean {
  const r = meta.region;
  return lat >= r.lat_min && lat <= r.lat_max && lon >= r.lon_min && lon <= r.lon_max;
}

function dateAvailable(date: unknown, meta: MetaResponse): date is string {
  return (
    typeof date === "string" &&
    /^\d{4}-\d{2}-\d{2}$/.test(date) &&
    date >= meta.available_dates.start &&
    date <= meta.available_dates.end
  );
}

/**
 * Apply up to three UI actions in order. Returns the next explorer state and side effects;
 * the caller keeps the previous state for Undo.
 */
export function applyUiActions(
  current: ExplorerState,
  actions: UIAction[] | null | undefined,
  meta: MetaResponse
): AppliedActions {
  let state: ExplorerState = { ...current };
  let camera: AppliedActions["camera"] = null;
  let openEvent: AppliedActions["openEvent"] = null;
  const changes: string[] = [];
  const skipped: string[] = [];

  for (const action of (actions ?? []).slice(0, MAX_ACTIONS)) {
    const args = (action?.args ?? {}) as Record<string, unknown>;
    switch (action?.type) {
      case "set_view": {
        let used = false;
        if ("date" in args) {
          if (dateAvailable(args.date, meta)) {
            state.date = args.date;
            changes.push(formatDate(args.date));
            used = true;
          }
        }
        if (isNum(args.depth_m)) {
          state.depth = findClosestDepth(args.depth_m, meta.depths_m);
          changes.push(formatDepth(state.depth));
          used = true;
        }
        if (args.field === "pred" || args.field === "glorys" || args.field === "error") {
          state.field = args.field;
          if (args.field !== current.field) changes.push(FIELD_LABELS[args.field]);
          used = true;
        }
        if (args.mode === "map" || args.mode === "3d" || args.mode === "layers") {
          // "layers" is the exploded 3D view; the explorer shows it as 3D.
          const view = args.mode === "map" ? "map" : "3d";
          if (view !== state.view) changes.push(view === "3d" ? "3D view" : "Map view");
          state.view = view;
          used = true;
        }
        const center = args.center as { lat?: unknown; lon?: unknown } | undefined;
        if (center && isNum(center.lat) && isNum(center.lon) && inRegion(center.lat, center.lon, meta)) {
          camera = {
            kind: "center",
            lat: center.lat,
            lon: center.lon,
            zoom: isNum(args.zoom) ? Math.min(8, Math.max(3, args.zoom)) : undefined,
            bearing: isNum(args.bearing) ? args.bearing : undefined,
          };
          changes.push(formatCoord(center.lat, center.lon));
          used = true;
        }
        if (!used) skipped.push("set_view");
        break;
      }
      case "select_point": {
        if (isNum(args.lat) && isNum(args.lon) && inRegion(args.lat, args.lon, meta)) {
          state.lat = Math.round(args.lat * 1000) / 1000;
          state.lon = Math.round(args.lon * 1000) / 1000;
          if (!camera) camera = { kind: "center", lat: args.lat, lon: args.lon };
          changes.push(formatCoord(args.lat, args.lon));
        } else {
          skipped.push("select_point");
        }
        break;
      }
      case "open_event": {
        if (typeof args.event_id === "string" && args.event_id) {
          const phase =
            args.phase === "before" || args.phase === "during" || args.phase === "after"
              ? args.phase
              : undefined;
          openEvent = { event_id: args.event_id, phase };
        } else {
          skipped.push("open_event");
        }
        break;
      }
      default:
        // slice_at needs the optional 3D section-slice tool, which is not built yet;
        // unknown types are ignored by contract (PRD_FRONTEND §3.4a).
        skipped.push(String(action?.type ?? "unknown"));
    }
  }

  return { state, camera, openEvent, changes, skipped };
}

/** Chip text, e.g. "Moved to 15 Jan 2019 · 50 m · 12.375°N, 85.125°E". */
export function describeChanges(changes: string[]): string | null {
  const unique = Array.from(new Set(changes));
  if (unique.length === 0) return null;
  return `Moved to ${unique.join(" · ")}`;
}

/** The date a time machine phase shows. "During" uses the event start. */
export function phaseDate(event: EventItem, phase: EventPhase): string {
  if (phase === "before") return event.before;
  if (phase === "after") return event.after;
  return event.start;
}

/**
 * Apply an event phase: date, first suggested depth, and a camera move to the event bbox.
 * Returns null for the date when the phase date is outside this deployment's data.
 */
export function applyEventPhase(
  current: ExplorerState,
  event: EventItem,
  phase: EventPhase,
  meta: MetaResponse,
  opts: { pickDepth?: boolean } = {}
): { state: ExplorerState; camera: AppliedActions["camera"]; dateAvailable: boolean } {
  const date = phaseDate(event, phase);
  const ok = dateAvailable(date, meta);
  const depth =
    opts.pickDepth !== false && event.suggested_depths_m.length > 0
      ? findClosestDepth(event.suggested_depths_m[0], meta.depths_m)
      : current.depth;
  return {
    state: { ...current, date: ok ? date : current.date, depth },
    camera: { kind: "bbox", bbox: event.bbox },
    dateAvailable: ok,
  };
}
