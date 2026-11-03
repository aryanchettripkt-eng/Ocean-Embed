/**
 * Offline mocks for the Meta features (PRD_FRONTEND §3.4). No Meta key or network:
 * the guide answers from a few fixed patterns so the drawer, navigator and time machine
 * can be built and demoed before the key is in Render secrets.
 */

import type {
  ChatRequest,
  ChatResponse,
  EventsResponse,
  MetaResponse,
  NearestOceanResponse,
  PostcardRequest,
  PostcardResponse,
  SegmentRequest,
  SegmentResponse,
  TranscribeResponse,
} from "../lib/api/client";
import eventsJson from "./events.json";

const PROVIDER = "Mock guide (offline) · Muse Spark not connected";

export const mockEvents: EventsResponse = {
  events: (eventsJson as { events: EventsResponse["events"] }).events,
  run_id: "synthetic_v1",
  synthetic: true,
};

/** Meta with the feature flags switched on, for `NEXT_PUBLIC_MOCK_META_FEATURES=true`. */
export function withMockFeatures(meta: MetaResponse): MetaResponse {
  return {
    ...meta,
    chat: { enabled: true, provider_label: PROVIDER },
    features: {
      ...meta.features,
      chat: true,
      voice: true,
      postcards: true,
      segment: true,
      segment_method: "threshold",
      beach: true,
      provider_label: PROVIDER,
    },
  };
}

export function mockChatResponse(req: ChatRequest): ChatResponse {
  const last = [...req.messages].reverse().find((m) => m.role === "user")?.content.toLowerCase() ?? "";
  const ctx = req.context ?? {};
  const base = { provider_label: PROVIDER, run_id: "synthetic_v1", synthetic: true };

  if (last.includes("ignore your instructions") || last.includes("atlantis")) {
    return {
      ...base,
      answer: "I can only answer from the model outputs for the North Indian Ocean, and I couldn't find data for that.",
      grounded: false,
      tool_calls: [],
      ui_actions: [],
    };
  }

  const event = mockEvents.events.find((e) => last.includes(e.title.toLowerCase()) || last.includes(e.id));
  if (event) {
    return {
      ...base,
      answer: `Opening ${event.title}. Use Before, During and After to compare the water. (Mock answer.)`,
      grounded: true,
      tool_calls: [{ name: "list_events", arguments: {}, result: { count: mockEvents.events.length } }],
      ui_actions: [{ type: "open_event", args: { event_id: event.id, phase: "before" } }],
    };
  }

  if (last.includes("warm")) {
    return {
      ...base,
      answer: "The warmest surface water on this date is in the south-east Bay of Bengal, about 29.0 °C. (Mock answer.)",
      grounded: true,
      tool_calls: [
        {
          name: "get_region_summary",
          arguments: { date: ctx.date, depth_m: 0.5, region: "bay_of_bengal" },
          result: { max: 29.0, mean: 28.1, unit: "degC" },
        },
      ],
      ui_actions: [
        { type: "set_view", args: { depth_m: 0.5, center: { lat: 8.5, lon: 88.5 }, zoom: 5 } },
        { type: "select_point", args: { lat: 8.5, lon: 88.5 } },
      ],
    };
  }

  const lat = ctx.lat ?? 15.5;
  const lon = ctx.lon ?? 87.5;
  return {
    ...base,
    answer: `At ${ctx.depth_m ?? 100} m near ${lat}°N, ${lon}°E the model shows about 21.4 °C on ${ctx.date ?? "this date"}. (Mock answer.)`,
    grounded: true,
    tool_calls: [
      {
        name: "get_profile",
        arguments: { lat, lon, date: ctx.date },
        result: { depth_m: ctx.depth_m ?? 100, temp_pred: 21.4 },
      },
    ],
    ui_actions: [{ type: "select_point", args: { lat, lon } }],
  };
}

export const mockTranscribe: TranscribeResponse = {
  text: "Show me the warmest water today",
  duration_s: 2.1,
  provider_label: "Mock transcription (offline)",
  run_id: "synthetic_v1",
  synthetic: true,
};

export function mockPostcard(req: PostcardRequest): PostcardResponse {
  return {
    image_url: `/api/v1/postcard/mock-${req.date}.svg`,
    caption: { date: req.date, depth_m: req.depth_m, mean_c: 24.6, source: "OceanScope model (synthetic demo data)" },
    label: "Mock postcard (offline) · Muse Image not connected",
    prompt_version: "mock",
    provider_label: "Mock image (offline)",
    run_id: "synthetic_v1",
    synthetic: true,
  };
}

/** A plain SVG standing in for the generated image. It carries no numbers, like the real one. */
export const MOCK_POSTCARD_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512" viewBox="0 0 768 512">
<rect width="768" height="512" fill="#e9dfc8"/>
<path d="M0 300 Q 96 260 192 300 T 384 300 T 576 300 T 768 300 V512 H0Z" fill="#2b6f8f"/>
<path d="M0 360 Q 96 320 192 360 T 384 360 T 576 360 T 768 360 V512 H0Z" fill="#1b4b66"/>
<circle cx="600" cy="140" r="60" fill="#d9843b"/>
<text x="24" y="40" font-family="sans-serif" font-size="20" fill="#3b3b3b">Mock postcard (synthetic)</text>
</svg>`;

export function mockSegment(req: SegmentRequest): SegmentResponse {
  const h = 1.5;
  const [a, b, c, d] = [req.lon - h, req.lon + h, req.lat - h, req.lat + h];
  return {
    method: "threshold",
    date: req.date,
    depth_m: req.depth_m,
    field: req.field ?? "pred",
    lat: req.lat,
    lon: req.lon,
    mask: { lat_index_ranges: [] },
    stats: { mean: 24.1, min: 23.6, max: 24.6, std: 0.28, n_valid: 9, area_km2: 110000 },
    outline: [
      [a, c],
      [b, c],
      [b, d],
      [a, d],
      [a, c],
    ],
    caveat: "The outline is a visual selection, not a scientific boundary. Statistics are computed from the model grid inside it.",
    provider_label: "Mock lasso (offline)",
    capped: false,
    run_id: "synthetic_v1",
    synthetic: true,
  };
}

export function mockNearestOcean(lat: number, lon: number, bearing: number | null): NearestOceanResponse {
  return {
    lat: Math.min(29.5, Math.max(0.5, lat - 0.5)),
    lon: Math.min(99.5, Math.max(40.5, lon + 0.5)),
    distance_km: 62,
    bearing_deg: bearing ?? 135,
    date: "2020-01-10",
    run_id: "synthetic_v1",
    synthetic: true,
  };
}
