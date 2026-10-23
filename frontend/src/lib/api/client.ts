/**
 * Typed HTTP client for OceanScope AI backend API (PRD_FRONTEND §0, §2, §5).
 */

import type { components } from "./types";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type MetaResponse = components["schemas"]["MetaResponse"];
export type MapResponse = components["schemas"]["MapResponse"];
export type ProfileResponse = components["schemas"]["ProfileResponse"];
export type TimeseriesResponse = components["schemas"]["TimeseriesResponse"];
export type RegionSummaryResponse = components["schemas"]["RegionSummaryResponse"];
export type MetricsResponse = components["schemas"]["MetricsResponse"];
export type ArgoResponse = components["schemas"]["ArgoResponse"];
export type ArgoProfilesResponse = components["schemas"]["ArgoProfilesResponse"];
export type ArgoProfileMatch = components["schemas"]["ArgoProfileMatch"];
export type HeadlineResponse = components["schemas"]["HeadlineResponse"];
export type SectionResponse = components["schemas"]["SectionResponse"];

export class ApiError extends Error {
  code: string;
  status: number;
  details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function getApiBaseUrl(): string {
  let envUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (envUrl) {
    // Strip trailing slashes if any
    envUrl = envUrl.replace(/\/+$/, "");
    if (!envUrl.endsWith("/api/v1")) {
      envUrl = `${envUrl}/api/v1`;
    }
    return envUrl;
  }
  return "http://localhost:8000/api/v1";
}

export async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const base = getApiBaseUrl();
  const url = `${base}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;

  const res = await fetch(url, {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let errorCode = "unknown_error";
    let errorMessage = `HTTP error ${res.status}`;
    let errorDetails: unknown = null;

    try {
      const errBody = await res.json();
      if (errBody?.error) {
        errorCode = errBody.error.code ?? errorCode;
        errorMessage = errBody.error.message ?? errorMessage;
        errorDetails = errBody.error.details ?? null;
      } else if (errBody?.detail) {
        errorMessage = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      // Body not JSON
    }

    throw new ApiError(res.status, errorCode, errorMessage, errorDetails);
  }

  return res.json() as Promise<T>;
}

export const apiClient = {
  getHealth(): Promise<HealthResponse> {
    return fetchApi<HealthResponse>("/health");
  },

  getMeta(): Promise<MetaResponse> {
    return fetchApi<MetaResponse>("/meta");
  },

  getMap(params: {
    date: string;
    depth_m?: number;
    depth_index?: number;
    field?: "pred" | "glorys" | "error";
  }): Promise<MapResponse> {
    const q = new URLSearchParams({ date: params.date });
    if (params.depth_m !== undefined) q.set("depth_m", params.depth_m.toString());
    if (params.depth_index !== undefined) q.set("depth_index", params.depth_index.toString());
    if (params.field) q.set("field", params.field);
    return fetchApi<MapResponse>(`/map?${q.toString()}`);
  },

  getSection(params: {
    date: string;
    field?: "pred" | "glorys" | "error";
    lat?: number;
    lon?: number;
  }): Promise<SectionResponse> {
    const q = new URLSearchParams({ date: params.date });
    if (params.field) q.set("field", params.field);
    if (params.lat !== undefined) q.set("lat", params.lat.toString());
    if (params.lon !== undefined) q.set("lon", params.lon.toString());
    return fetchApi<SectionResponse>(`/section?${q.toString()}`);
  },

  getProfile(params: { lat: number; lon: number; date: string }): Promise<ProfileResponse> {
    const q = new URLSearchParams({
      lat: params.lat.toString(),
      lon: params.lon.toString(),
      date: params.date,
    });
    return fetchApi<ProfileResponse>(`/profile?${q.toString()}`);
  },

  getTimeseries(params: {
    lat: number;
    lon: number;
    start: string;
    end: string;
    depth_m?: number;
    depth_index?: number;
  }): Promise<TimeseriesResponse> {
    const q = new URLSearchParams({
      lat: params.lat.toString(),
      lon: params.lon.toString(),
      start: params.start,
      end: params.end,
    });
    if (params.depth_m !== undefined) q.set("depth_m", params.depth_m.toString());
    if (params.depth_index !== undefined) q.set("depth_index", params.depth_index.toString());
    return fetchApi<TimeseriesResponse>(`/timeseries?${q.toString()}`);
  },

  getRegionSummary(params: {
    date: string;
    depth_m?: number;
    depth_index?: number;
    field?: "pred" | "glorys" | "error";
    region?: "arabian_sea" | "bay_of_bengal" | "equatorial";
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  }): Promise<RegionSummaryResponse> {
    const q = new URLSearchParams({ date: params.date });
    if (params.depth_m !== undefined) q.set("depth_m", params.depth_m.toString());
    if (params.depth_index !== undefined) q.set("depth_index", params.depth_index.toString());
    if (params.field) q.set("field", params.field);
    if (params.region) q.set("region", params.region);
    if (params.lat_min !== undefined) q.set("lat_min", params.lat_min.toString());
    if (params.lat_max !== undefined) q.set("lat_max", params.lat_max.toString());
    if (params.lon_min !== undefined) q.set("lon_min", params.lon_min.toString());
    if (params.lon_max !== undefined) q.set("lon_max", params.lon_max.toString());
    return fetchApi<RegionSummaryResponse>(`/region-summary?${q.toString()}`);
  },

  getMetrics(params?: {
    split?: "val" | "test";
    group_by?: "depth" | "season" | "region";
  }): Promise<MetricsResponse> {
    const q = new URLSearchParams();
    if (params?.split) q.set("split", params.split);
    if (params?.group_by) q.set("group_by", params.group_by);
    const qs = q.toString();
    return fetchApi<MetricsResponse>(`/metrics${qs ? `?${qs}` : ""}`);
  },

  getArgo(): Promise<ArgoResponse> {
    return fetchApi<ArgoResponse>("/argo");
  },

  getArgoProfiles(params?: {
    date?: string;
    lat_min?: number;
    lat_max?: number;
    lon_min?: number;
    lon_max?: number;
  }): Promise<ArgoProfilesResponse> {
    const q = new URLSearchParams();
    if (params?.date) q.set("date", params.date);
    if (params?.lat_min !== undefined) q.set("lat_min", params.lat_min.toString());
    if (params?.lat_max !== undefined) q.set("lat_max", params.lat_max.toString());
    if (params?.lon_min !== undefined) q.set("lon_min", params.lon_min.toString());
    if (params?.lon_max !== undefined) q.set("lon_max", params.lon_max.toString());
    const qs = q.toString();
    return fetchApi<ArgoProfilesResponse>(`/argo/profiles${qs ? `?${qs}` : ""}`);
  },

  getHeadline(): Promise<HeadlineResponse> {
    return fetchApi<HeadlineResponse>("/headline");
  },
};
