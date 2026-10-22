/**
 * React Query hooks for OceanScope AI API endpoints (PRD_FRONTEND §2, §4, §5).
 */

import { useQuery } from "@tanstack/react-query";
import {
  apiClient,
  type ArgoProfilesResponse,
  type ArgoResponse,
  type HeadlineResponse,
  type MapResponse,
  type MetaResponse,
  type MetricsResponse,
  type ProfileResponse,
  type RegionSummaryResponse,
  type TimeseriesResponse,
} from "./client";

/**
 * Metadata hook with cold-backend retry backoff.
 */
export function useMeta() {
  return useQuery<MetaResponse>({
    queryKey: ["meta"],
    queryFn: () => apiClient.getMeta(),
    staleTime: Infinity,
    retry: 5,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 15000),
  });
}

/**
 * 2D horizontal temperature field slice.
 */
export function useMap(
  params: {
    date: string;
    depth_m?: number;
    depth_index?: number;
    field?: "pred" | "glorys" | "error";
  },
  options?: { enabled?: boolean }
) {
  return useQuery<MapResponse>({
    queryKey: ["map", params.date, params.depth_m, params.depth_index, params.field ?? "pred"],
    queryFn: () => apiClient.getMap(params),
    staleTime: Infinity,
    enabled: options?.enabled !== false && !!params.date,
  });
}

/**
 * 15-depth vertical temperature profile at a point.
 */
export function useProfile(
  params: { lat: number; lon: number; date: string } | null,
  options?: { enabled?: boolean }
) {
  return useQuery<ProfileResponse>({
    queryKey: ["profile", params?.lat, params?.lon, params?.date],
    queryFn: () => {
      if (!params) throw new Error("Missing profile coordinates or date");
      return apiClient.getProfile(params);
    },
    staleTime: Infinity,
    enabled: options?.enabled !== false && !!params && params.lat !== null && params.lon !== null && !!params.date,
  });
}

/**
 * Time series at a point and depth (±90 days).
 */
export function useTimeseries(
  params: {
    lat: number;
    lon: number;
    start: string;
    end: string;
    depth_m?: number;
    depth_index?: number;
  } | null,
  options?: { enabled?: boolean }
) {
  return useQuery<TimeseriesResponse>({
    queryKey: [
      "timeseries",
      params?.lat,
      params?.lon,
      params?.start,
      params?.end,
      params?.depth_m,
      params?.depth_index,
    ],
    queryFn: () => {
      if (!params) throw new Error("Missing timeseries parameters");
      return apiClient.getTimeseries(params);
    },
    staleTime: Infinity,
    enabled:
      options?.enabled !== false &&
      !!params &&
      params.lat !== null &&
      params.lon !== null &&
      !!params.start &&
      !!params.end,
  });
}

/**
 * Area-weighted regional summary statistics.
 */
export function useRegionSummary(
  params: {
    date: string;
    depth_m?: number;
    depth_index?: number;
    field?: "pred" | "glorys" | "error";
    region?: "arabian_sea" | "bay_of_bengal" | "equatorial";
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  },
  options?: { enabled?: boolean }
) {
  return useQuery<RegionSummaryResponse>({
    queryKey: [
      "region-summary",
      params.date,
      params.depth_m,
      params.depth_index,
      params.field,
      params.region,
      params.lat_min,
      params.lat_max,
      params.lon_min,
      params.lon_max,
    ],
    queryFn: () => apiClient.getRegionSummary(params),
    staleTime: Infinity,
    enabled: options?.enabled !== false && !!params.date,
  });
}

/**
 * Model and baseline validation metrics.
 */
export function useMetrics(params?: {
  split?: "val" | "test";
  group_by?: "depth" | "season" | "region";
}) {
  const split = params?.split ?? "val";
  const groupBy = params?.group_by ?? "depth";
  return useQuery<MetricsResponse>({
    queryKey: ["metrics", split, groupBy],
    queryFn: () => apiClient.getMetrics({ split, group_by: groupBy }),
    staleTime: Infinity,
  });
}

/**
 * Held-out ARGO float validation table.
 */
export function useArgo() {
  return useQuery<ArgoResponse>({
    queryKey: ["argo"],
    queryFn: () => apiClient.getArgo(),
    staleTime: Infinity,
  });
}

/**
 * Matched ARGO profile markers for map display.
 */
export function useArgoProfiles(
  params?: {
    date?: string;
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  },
  options?: { enabled?: boolean }
) {
  return useQuery<ArgoProfilesResponse>({
    queryKey: [
      "argo-profiles",
      params?.date,
      params?.lat_min,
      params?.lat_max,
      params?.lon_min,
      params?.lon_max,
    ],
    queryFn: () => apiClient.getArgoProfiles(params),
    staleTime: Infinity,
    enabled: options?.enabled !== false,
  });
}

/**
 * Pre-registered primary headline criterion result.
 */
export function useHeadline() {
  return useQuery<HeadlineResponse>({
    queryKey: ["headline"],
    queryFn: () => apiClient.getHeadline(),
    staleTime: Infinity,
  });
}
