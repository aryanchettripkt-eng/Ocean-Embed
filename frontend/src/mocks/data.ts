/**
 * Mock data fixtures derived directly from backend fixtures (PRD_FRONTEND §2, §5).
 * All payloads are dumped from backend TestClient runs to prevent mock drift.
 */

import type {
  ArgoProfilesResponse,
  ArgoResponse,
  HeadlineResponse,
  HealthResponse,
  MapResponse,
  MetaResponse,
  MetricsResponse,
  ProfileResponse,
  RegionSummaryResponse,
  SectionResponse,
  TimeseriesResponse,
} from "../lib/api/client";

import argoJson from "./argo.json";
import argoProfilesJson from "./argo_profiles.json";
import headlineJson from "./headline.json";
import healthJson from "./health.json";
import mapJson from "./map.json";
import metaJson from "./meta.json";
import metricsJson from "./metrics.json";
import profileJson from "./profile.json";
import regionSummaryJson from "./region_summary.json";
import sectionJson from "./section.json";
import timeseriesJson from "./timeseries.json";

export const mockHealth = healthJson as unknown as HealthResponse;
export const mockMeta = metaJson as unknown as MetaResponse;
export const mockMap = mapJson as unknown as MapResponse;
export const mockProfile = profileJson as unknown as ProfileResponse;
export const mockTimeseries = timeseriesJson as unknown as TimeseriesResponse;
export const mockRegionSummary = regionSummaryJson as unknown as RegionSummaryResponse;
export const mockMetrics = metricsJson as unknown as MetricsResponse;
export const mockArgo = argoJson as unknown as ArgoResponse;
export const mockArgoProfiles = argoProfilesJson as unknown as ArgoProfilesResponse;
export const mockHeadline = headlineJson as unknown as HeadlineResponse;
export const mockSection = sectionJson as unknown as SectionResponse;

export function createMockMap(
  date = "2020-01-05",
  depth_m = 100.0,
  field: "pred" | "glorys" | "error" = "pred"
): MapResponse {
  const isLive = date >= "2020-01-09";
  return {
    ...mockMap,
    date,
    depth_m,
    field,
    split: isLive ? "live" : "val",
  };
}

export function createMockProfile(lat = 12.5, lon = 65.5, date = "2020-01-05"): ProfileResponse {
  const isLive = date >= "2020-01-09";
  return {
    ...mockProfile,
    date,
    lat,
    lon,
    split: isLive ? "live" : "val",
    glorys: isLive ? mockProfile.glorys.map(() => null) : mockProfile.glorys,
    error: isLive ? mockProfile.error.map(() => null) : mockProfile.error,
  };
}

export function createMockTimeseries(
  lat = 12.5,
  lon = 65.5,
  start = "2020-01-01",
  end = "2020-01-10"
): TimeseriesResponse {
  return {
    ...mockTimeseries,
    lat,
    lon,
  };
}
