import "@testing-library/jest-dom/vitest";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ApiError, apiClient, type ChatResponse, type EventItem, type MetaResponse } from "../src/lib/api/client";
import {
  applyEventPhase,
  applyUiActions,
  describeChanges,
  phaseDate,
} from "../src/lib/guide/ui-actions";
import type { ExplorerState } from "../src/lib/url-state";
import { GuideDrawer, guideErrorMessage, starterQuestions } from "../src/components/guide/GuideDrawer";
import { TimeMachine, eventDays } from "../src/components/events/TimeMachine";
import { mockMeta } from "../src/mocks/data";
import { mockChatResponse, mockEvents, withMockFeatures } from "../src/mocks/guide";

const meta: MetaResponse = withMockFeatures(mockMeta);
const base: ExplorerState = { date: "2020-01-05", depth: 100, field: "pred", lat: null, lon: null, view: "map" };

const event: EventItem = {
  id: "test-event",
  title: "Test Cyclone",
  kind: "cyclone",
  start: "2020-01-04",
  end: "2020-01-06",
  before: "2020-01-01",
  after: "2021-02-01",
  bbox: { lat_min: 10, lat_max: 20, lon_min: 80, lon_max: 90 },
  suggested_depths_m: [48, 100],
  summary: "A test event from the backend.",
  source_url: "https://example.org/report",
  source_name: "Example Agency",
};

function chat(partial: Partial<ChatResponse>): ChatResponse {
  return {
    answer: "ok",
    grounded: true,
    tool_calls: [],
    ui_actions: [],
    provider_label: "Muse Spark",
    run_id: "r",
    synthetic: true,
    ...partial,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("applyUiActions (navigator)", () => {
  it("applies set_view and select_point in order and snaps depth", () => {
    const out = applyUiActions(
      base,
      [
        { type: "set_view", args: { date: "2020-01-08", depth_m: 48, field: "error", mode: "3d", center: { lat: 12, lon: 85 }, zoom: 20 } },
        { type: "select_point", args: { lat: 12.5, lon: 85.5 } },
      ],
      meta
    );
    expect(out.state).toMatchObject({ date: "2020-01-08", depth: 50, field: "error", view: "3d", lat: 12.5, lon: 85.5 });
    expect(out.camera).toEqual({ kind: "center", lat: 12, lon: 85, zoom: 8, bearing: undefined });
    expect(describeChanges(out.changes)).toMatch(/^Moved to .*50 m/);
  });

  it("ignores unknown types, out-of-range dates and points outside the region", () => {
    const out = applyUiActions(
      base,
      [
        { type: "set_view", args: { date: "2030-01-01" } },
        { type: "select_point", args: { lat: -40, lon: 10 } },
        // @ts-expect-error unknown type from a misbehaving model
        { type: "delete_everything", args: {} },
      ],
      meta
    );
    expect(out.state).toEqual(base);
    expect(out.changes).toEqual([]);
    expect(out.skipped).toEqual(["set_view", "select_point", "delete_everything"]);
  });

  it("applies at most three actions and returns open_event", () => {
    const out = applyUiActions(
      base,
      [
        { type: "set_view", args: { depth_m: 5 } },
        { type: "set_view", args: { depth_m: 10 } },
        { type: "open_event", args: { event_id: "x", phase: "after" } },
        { type: "set_view", args: { depth_m: 1000 } },
      ],
      meta
    );
    expect(out.state.depth).toBe(10);
    expect(out.openEvent).toEqual({ event_id: "x", phase: "after" });
  });
});

describe("applyEventPhase (time machine)", () => {
  it("sets the phase date, the first suggested depth and the bbox camera", () => {
    const out = applyEventPhase(base, event, "during", meta);
    expect(out.state.date).toBe("2020-01-04");
    expect(out.state.depth).toBe(50);
    expect(out.camera).toEqual({ kind: "bbox", bbox: event.bbox });
    expect(out.dateAvailable).toBe(true);
  });

  it("keeps the date when the phase is outside the data, and keeps depth when asked", () => {
    const out = applyEventPhase({ ...base, depth: 200 }, event, "after", meta, { pickDepth: false });
    expect(out.state.date).toBe(base.date);
    expect(out.state.depth).toBe(200);
    expect(out.dateAvailable).toBe(false);
    expect(phaseDate(event, "after")).toBe("2021-02-01");
  });

  it("lists event days inclusively", () => {
    expect(eventDays(event)).toEqual(["2020-01-04", "2020-01-05", "2020-01-06"]);
  });
});

describe("GuideDrawer", () => {
  it("sends context, applies grounded ui_actions and offers Undo", async () => {
    const post = vi.spyOn(apiClient, "postChat").mockResolvedValue(
      chat({
        answer: "Here it is.",
        tool_calls: [{ name: "get_profile", arguments: { lat: 1 }, result: { t: 2 } }],
        ui_actions: [{ type: "set_view", args: { depth_m: 50 } }],
      })
    );
    const onNavigate = vi.fn(() => ({ summary: "Moved to 50 m", undoToken: 7 }));
    const onUndo = vi.fn();
    render(
      <GuideDrawer meta={meta} state={{ ...base, lat: 12, lon: 85 }} onNavigate={onNavigate} onUndo={onUndo} onClose={() => {}} />
    );

    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "How warm is it?" } });
    fireEvent.click(screen.getByLabelText("Send"));

    expect(await screen.findByText("Here it is.")).toBeInTheDocument();
    expect(post).toHaveBeenCalledWith(
      {
        messages: [{ role: "user", content: "How warm is it?" }],
        context: { date: "2020-01-05", depth_m: 100, lat: 12, lon: 85 },
      },
      expect.any(AbortSignal)
    );
    expect(onNavigate).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Moved to 50 m")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Data used (1)"));
    expect(screen.getByText("get_profile")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Undo"));
    expect(onUndo).toHaveBeenCalledWith(7);
    expect(screen.getByText("Moved back")).toBeInTheDocument();
    // AI caveat verbatim
    expect(screen.getByText(meta.caveats.ai)).toBeInTheDocument();
  });

  it("shows ungrounded answers with a badge and does not move the map", async () => {
    vi.spyOn(apiClient, "postChat").mockResolvedValue(
      chat({ answer: "I couldn't verify that.", grounded: false, ui_actions: [{ type: "set_view", args: { depth_m: 5 } }] })
    );
    const onNavigate = vi.fn();
    render(<GuideDrawer meta={meta} state={base} onNavigate={onNavigate} onUndo={() => {}} onClose={() => {}} />);
    fireEvent.click(screen.getByText("Show me the warmest water today"));
    expect(await screen.findByText("I couldn't verify that.")).toBeInTheDocument();
    expect(screen.getByText("Couldn't verify")).toBeInTheDocument();
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("shows a friendly message on 429", async () => {
    vi.spyOn(apiClient, "postChat").mockRejectedValue(new ApiError(429, "rate_limited", "slow down"));
    render(<GuideDrawer meta={meta} state={base} onNavigate={vi.fn()} onUndo={() => {}} onClose={() => {}} />);
    fireEvent.click(screen.getByText("Show me the warmest water today"));
    expect(await screen.findByText(/a lot of questions/)).toBeInTheDocument();
  });

  it("sends a prompt passed in (time machine Explain)", async () => {
    const post = vi.spyOn(apiClient, "postChat").mockResolvedValue(chat({ answer: "Compared." }));
    render(
      <GuideDrawer
        meta={meta}
        state={base}
        prompt={{ text: "Explain the event", nonce: 1 }}
        onNavigate={vi.fn(() => ({ summary: null, undoToken: null }))}
        onUndo={() => {}}
        onClose={() => {}}
      />
    );
    expect(await screen.findByText("Compared.")).toBeInTheDocument();
    expect(post.mock.calls[0][0].messages).toEqual([{ role: "user", content: "Explain the event" }]);
  });

  it("times out after 25 s", async () => {
    vi.useFakeTimers();
    vi.spyOn(apiClient, "postChat").mockImplementation(
      (_b, signal) =>
        new Promise((_res, rej) => signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError"))))
    );
    render(<GuideDrawer meta={meta} state={base} onNavigate={vi.fn()} onUndo={() => {}} onClose={() => {}} />);
    fireEvent.click(screen.getByText("Show me the warmest water today"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(25_000);
    });
    vi.useRealTimers();
    await waitFor(() => expect(screen.getByText(/took too long/)).toBeInTheDocument());
  });

  it("builds starter questions from context and event titles", () => {
    expect(starterQuestions(base, ["Cyclone Fani"])).toEqual([
      "What's happening at 100 m in the Bay of Bengal?",
      "Show me the warmest water today",
      "Take me to Cyclone Fani",
    ]);
    expect(guideErrorMessage(new ApiError(503, "x", "y"))).toMatch(/switched off/);
  });
});

describe("TimeMachine", () => {
  it("lists events from the API with their source and opens one", () => {
    const onOpen = vi.fn();
    render(
      <TimeMachine
        meta={meta}
        events={[event]}
        active={null}
        currentDate={base.date}
        onOpen={onOpen}
        onSetDate={vi.fn()}
        onBack={vi.fn()}
        onClose={vi.fn()}
      />
    );
    expect(screen.getByText(/Example Agency/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Test Cyclone"));
    expect(onOpen).toHaveBeenCalledWith(event, "before", { pickDepth: true });
  });

  it("shows phases, the source link, a note for dates outside the data, and Explain", () => {
    const onOpen = vi.fn();
    const onExplain = vi.fn();
    render(
      <TimeMachine
        meta={meta}
        events={[event]}
        active={{ event, phase: "after" }}
        currentDate={base.date}
        onOpen={onOpen}
        onSetDate={vi.fn()}
        onBack={vi.fn()}
        onClose={vi.fn()}
        onExplain={onExplain}
      />
    );
    expect(screen.getByRole("tab", { selected: true })).toHaveTextContent("After");
    expect(screen.getByText(/is not in the data on this site yet/)).toBeInTheDocument();
    expect(screen.getByRole("link")).toHaveAttribute("href", "https://example.org/report");
    fireEvent.click(screen.getByRole("tab", { name: /During/ }));
    expect(onOpen).toHaveBeenCalledWith(event, "during", { pickDepth: false });
    fireEvent.click(screen.getByText(/Explain what changed/));
    expect(onExplain).toHaveBeenCalledWith(event);
  });

  it("hides Explain when the guide is off", () => {
    render(
      <TimeMachine
        meta={mockMeta}
        events={[event]}
        active={{ event, phase: "during" }}
        currentDate="2020-01-04"
        onOpen={vi.fn()}
        onSetDate={vi.fn()}
        onBack={vi.fn()}
        onClose={vi.fn()}
      />
    );
    expect(screen.queryByText(/Explain what changed/)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/Day 1 of 3/)).toBeInTheDocument();
  });
});

describe("offline mocks", () => {
  it("routes event questions to open_event and unknown places to ungrounded", () => {
    const fani = mockChatResponse({ messages: [{ role: "user", content: "Take me to Cyclone Fani" }] });
    expect(fani.ui_actions?.[0]).toEqual({ type: "open_event", args: { event_id: "cyclone-fani-2019", phase: "before" } });
    expect(mockChatResponse({ messages: [{ role: "user", content: "How warm is Atlantis?" }] }).grounded).toBe(false);
    expect(mockEvents.events.length).toBeGreaterThan(0);
  });
});

describe("describeChanges", () => {
  it("drops repeated parts", () => {
    expect(describeChanges(["0.5 m", "8.500°N, 88.500°E", "8.500°N, 88.500°E"])).toBe("Moved to 0.5 m · 8.500°N, 88.500°E");
    expect(describeChanges([])).toBeNull();
  });
});
