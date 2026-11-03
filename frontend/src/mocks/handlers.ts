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
import type { ChatRequest } from "../lib/api/client";
import type { PostcardRequest, SegmentRequest } from "../lib/api/client";
import {
  MOCK_POSTCARD_SVG,
  mockChatResponse,
  mockEvents,
  mockNearestOcean,
  mockPostcard,
  mockSegment,
  mockTranscribe,
  withMockFeatures,
} from "./guide";

const mockFeaturesOn = process.env.NEXT_PUBLIC_MOCK_META_FEATURES === "true";

export const handlers = [
  // Health
  http.get("*/api/v1/health", () => {
    return HttpResponse.json(mockHealth);
  }),

  // Meta
  http.get("*/api/v1/meta", () => {
    return HttpResponse.json(mockFeaturesOn ? withMockFeatures(mockMeta) : mockMeta);
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

  // Meta features (PRD_BACKEND §4.12–4.14), offline
  http.post("*/api/v1/chat", async ({ request }) => {
    const body = (await request.json()) as ChatRequest;
    await new Promise((r) => setTimeout(r, 600));
    return HttpResponse.json(mockChatResponse(body));
  }),

  http.get("*/api/v1/events", () => {
    return HttpResponse.json(mockEvents);
  }),

  http.post("*/api/v1/transcribe", async () => {
    await new Promise((r) => setTimeout(r, 400));
    return HttpResponse.json(mockTranscribe);
  }),

  http.post("*/api/v1/postcard", async ({ request }) => {
    const body = (await request.json()) as PostcardRequest;
    await new Promise((r) => setTimeout(r, 1500));
    return HttpResponse.json(mockPostcard(body));
  }),

  http.get("*/api/v1/postcard/:name", () => {
    return new HttpResponse(MOCK_POSTCARD_SVG, { headers: { "Content-Type": "image/svg+xml" } });
  }),

  http.post("*/api/v1/segment", async ({ request }) => {
    const body = (await request.json()) as SegmentRequest;
    await new Promise((r) => setTimeout(r, 500));
    return HttpResponse.json(mockSegment(body));
  }),

  http.get("*/api/v1/nearest-ocean", ({ request }) => {
    const url = new URL(request.url);
    const bearing = url.searchParams.get("bearing_deg");
    return HttpResponse.json(
      mockNearestOcean(
        parseFloat(url.searchParams.get("lat") ?? "13"),
        parseFloat(url.searchParams.get("lon") ?? "80.3"),
        bearing === null ? null : parseFloat(bearing)
      )
    );
  }),
];
