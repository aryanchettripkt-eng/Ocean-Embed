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
    features: { ...meta.features, chat: true, voice: true, provider_label: PROVIDER },
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
