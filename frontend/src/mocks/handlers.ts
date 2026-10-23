/**
 * MSW request handlers for development and offline testing (PRD_FRONTEND §2, §5).
 */

import { http, HttpResponse } from "msw";
import {
  createMockMap,
  createMockProfile,
  createMockTimeseries,
  mockArgo,
  mockArgoProfiles,
  mockHeadline,
  mockHealth,
  mockMeta,
  mockMetrics,
  mockRegionSummary,
  mockSection,
} from "./data";

export const handlers = [
  // Health
  http.get("*/api/v1/health", () => {
    return HttpResponse.json(mockHealth);
  }),

  // Meta
  http.get("*/api/v1/meta", () => {
    return HttpResponse.json(mockMeta);
  }),

  // Map
  http.get("*/api/v1/map", ({ request }) => {
    const url = new URL(request.url);
    const date = url.searchParams.get("date") ?? "2020-01-05";
    const depth_m = parseFloat(url.searchParams.get("depth_m") ?? "100.0");
    const field = (url.searchParams.get("field") ?? "pred") as "pred" | "glorys" | "error";

    const isOutOfRange =
      date < mockMeta.available_dates.start || date > mockMeta.available_dates.end;
    if (isOutOfRange) {
      return HttpResponse.json(
        {
          error: {
            code: "date_unavailable",
            message: `Requested date is outside available dates (${mockMeta.available_dates.start} to ${mockMeta.available_dates.end})`,
            details: null,
          },
        },
        { status: 404 }
      );
    }

    const isLive = mockMeta.latest_live_date ? date >= "2020-01-09" : false;
    if (isLive && (field === "glorys" || field === "error")) {
      return HttpResponse.json(
        {
          error: {
            code: "reference_unavailable",
            message: "GLORYS reference is not available for live dates",
            details: null,
          },
        },
        { status: 404 }
      );
    }

    return HttpResponse.json(createMockMap(date, depth_m, field));
  }),

  // Section
  http.get("*/api/v1/section", ({ request }) => {
    const url = new URL(request.url);
    const date = url.searchParams.get("date") ?? "2020-01-05";
    const field = (url.searchParams.get("field") ?? "pred") as "pred" | "glorys" | "error";
    const hasLat = url.searchParams.has("lat");
    const hasLon = url.searchParams.has("lon");

    if ((!hasLat && !hasLon) || (hasLat && hasLon)) {
      return HttpResponse.json(
        {
          error: {
            code: "invalid_section",
            message: "Provide exactly one of 'lat' or 'lon', not both or neither",
            details: null,
          },
        },
        { status: 422 }
      );
    }

    const isLive = mockMeta.latest_live_date ? date >= "2020-01-09" : false;
    if (isLive && (field === "glorys" || field === "error")) {
      return HttpResponse.json(
        {
          error: {
            code: "reference_unavailable",
            message: "GLORYS reference is not available for live dates",
            details: null,
          },
        },
        { status: 404 }
      );
    }

    return HttpResponse.json(mockSection);
  }),

  // Profile
  http.get("*/api/v1/profile", ({ request }) => {
    const url = new URL(request.url);
    const lat = parseFloat(url.searchParams.get("lat") ?? "12.5");
    const lon = parseFloat(url.searchParams.get("lon") ?? "65.5");
    const date = url.searchParams.get("date") ?? "2020-01-05";

    return HttpResponse.json(createMockProfile(lat, lon, date));
  }),

  // Timeseries
  http.get("*/api/v1/timeseries", ({ request }) => {
    const url = new URL(request.url);
    const lat = parseFloat(url.searchParams.get("lat") ?? "12.5");
    const lon = parseFloat(url.searchParams.get("lon") ?? "65.5");
    const start = url.searchParams.get("start") ?? "2020-01-01";
    const end = url.searchParams.get("end") ?? "2020-01-10";

    return HttpResponse.json(createMockTimeseries(lat, lon, start, end));
  }),

  // Region Summary
  http.get("*/api/v1/region-summary", () => {
    return HttpResponse.json(mockRegionSummary);
  }),

  // Metrics
  http.get("*/api/v1/metrics", ({ request }) => {
    const url = new URL(request.url);
    const split = url.searchParams.get("split") ?? "val";
    const groupBy = (url.searchParams.get("group_by") ?? "depth") as "depth" | "season" | "region";

    if (split === "test") {
      return HttpResponse.json(
        {
          error: {
            code: "test_not_evaluated",
            message: "Test split has not been evaluated",
            details: null,
          },
        },
        { status: 404 }
      );
    }

    return HttpResponse.json({
      ...mockMetrics,
      group_by: groupBy,
    });
  }),

  // ARGO
  http.get("*/api/v1/argo", () => {
    return HttpResponse.json(mockArgo);
  }),

  // ARGO Profiles
  http.get("*/api/v1/argo/profiles", ({ request }) => {
    const url = new URL(request.url);
    const date = url.searchParams.get("date");
    if (date === "2020-01-03") {
      return HttpResponse.json({
        ...mockArgoProfiles,
        profiles: [mockArgoProfiles.profiles[0]],
      });
    }
    return HttpResponse.json(mockArgoProfiles);
  }),

  // Headline
  http.get("*/api/v1/headline", () => {
    return HttpResponse.json(mockHeadline);
  }),
];
