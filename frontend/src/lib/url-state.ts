/**
 * Explorer URL query state syncing and validation (PRD_FRONTEND §3.1).
 */

import type { components } from "./api/types";

export type MetaResponse = components["schemas"]["MetaResponse"];

export interface ExplorerState {
  date: string;
  depth: number;
  field: "pred" | "glorys" | "error";
  lat: number | null;
  lon: number | null;
  view: "map" | "3d";
}

export const DEFAULT_FALLBACK_STATE: ExplorerState = {
  date: "2020-01-08",
  depth: 100.0,
  field: "pred",
  lat: null,
  lon: null,
  view: "map",
};

/**
 * Find the depth level closest to the target in meters.
 */
export function findClosestDepth(target: number, available: number[]): number {
  if (!available || available.length === 0) return target;
  let closest = available[0];
  let minDiff = Math.abs(target - closest);
  for (let i = 1; i < available.length; i++) {
    const diff = Math.abs(target - available[i]);
    if (diff < minDiff) {
      minDiff = diff;
      closest = available[i];
    }
  }
  return closest;
}

/**
 * Derive initial explorer state from URL search params and loaded metadata.
 */
export function parseExplorerParams(
  searchParams: URLSearchParams | Record<string, string | string[] | undefined>,
  meta?: MetaResponse | null
): ExplorerState {
  const getParam = (key: string): string | undefined => {
    if (searchParams instanceof URLSearchParams) {
      return searchParams.get(key) ?? undefined;
    }
    const val = searchParams[key];
    return Array.isArray(val) ? val[0] : val;
  };

  const rawDate = getParam("date");
  const rawDepth = getParam("depth");
  const rawField = getParam("field");
  const rawLat = getParam("lat");
  const rawLon = getParam("lon");
  const rawView = getParam("view");

  // Default depth: closest to 100m from meta.depths_m
  const defaultDepth = meta?.depths_m ? findClosestDepth(100.0, meta.depths_m) : DEFAULT_FALLBACK_STATE.depth;

  // Default date: latest available date that lies in the val split (min of val end and available end)
  let defaultDate = DEFAULT_FALLBACK_STATE.date;
  if (meta?.available_dates) {
    const availStart = meta.available_dates.start;
    const availEnd = meta.available_dates.end;
    if (meta.splits?.val && meta.splits.val.length >= 2) {
      const valStart = meta.splits.val[0];
      const valEnd = meta.splits.val[1];
      const candidate = valEnd < availEnd ? valEnd : availEnd;
      const earliest = valStart > availStart ? valStart : availStart;
      defaultDate = candidate >= earliest ? candidate : availEnd;
    } else {
      defaultDate = availEnd;
    }
  }

  // Validate date
  let date = defaultDate;
  if (rawDate && /^\d{4}-\d{2}-\d{2}$/.test(rawDate)) {
    if (meta?.available_dates) {
      if (rawDate >= meta.available_dates.start && rawDate <= meta.available_dates.end) {
        date = rawDate;
      }
    } else {
      date = rawDate;
    }
  }

  // Validate depth
  let depth = defaultDepth;
  if (rawDepth) {
    const parsed = parseFloat(rawDepth);
    if (!isNaN(parsed)) {
      depth = meta?.depths_m ? findClosestDepth(parsed, meta.depths_m) : parsed;
    }
  }

  // Validate field
  let field: "pred" | "glorys" | "error" = "pred";
  if (rawField === "pred" || rawField === "glorys" || rawField === "error") {
    field = rawField;
  }

  // Validate lat/lon
  let lat: number | null = null;
  let lon: number | null = null;
  if (rawLat && rawLon) {
    const pLat = parseFloat(rawLat);
    const pLon = parseFloat(rawLon);
    if (!isNaN(pLat) && !isNaN(pLon)) {
      if (meta?.region) {
        if (
          pLat >= meta.region.lat_min &&
          pLat <= meta.region.lat_max &&
          pLon >= meta.region.lon_min &&
          pLon <= meta.region.lon_max
        ) {
          lat = Math.round(pLat * 1000) / 1000;
          lon = Math.round(pLon * 1000) / 1000;
        }
      } else {
        lat = Math.round(pLat * 1000) / 1000;
        lon = Math.round(pLon * 1000) / 1000;
      }
    }
  }

  // Validate view
  let view: "map" | "3d" = "map";
  if (rawView === "3d" || rawView === "map") {
    view = rawView;
  }

  return { date, depth, field, lat, lon, view };
}

/**
 * Build URLSearchParams from ExplorerState.
 */
export function buildExplorerSearchParams(state: ExplorerState): URLSearchParams {
  const params = new URLSearchParams();
  params.set("date", state.date);
  params.set("depth", state.depth.toString());
  params.set("field", state.field);
  if (state.lat !== null && state.lon !== null) {
    params.set("lat", state.lat.toFixed(3));
    params.set("lon", state.lon.toFixed(3));
  }
  if (state.view && state.view !== "map") {
    params.set("view", state.view);
  }
  return params;
}
