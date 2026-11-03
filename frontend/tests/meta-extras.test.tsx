import "@testing-library/jest-dom/vitest";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import { ApiError, apiClient, resolveApiUrl, type MetaResponse } from "../src/lib/api/client";
import { PostcardPanel, captionLines, postcardErrorMessage } from "../src/components/postcard/PostcardPanel";
import {
  LassoPanel,
  askAboutAreaPrompt,
  methodLabel,
  outlineBBox,
  segmentErrorMessage,
} from "../src/components/lasso/LassoPanel";
import { compassWord, describeDistance } from "../src/components/beach/BeachMode";
import { mockMeta } from "../src/mocks/data";
import { mockPostcard, mockSegment, withMockFeatures } from "../src/mocks/guide";

vi.mock("next/navigation", () => ({ usePathname: () => "/explore" }));

const meta: MetaResponse = withMockFeatures(mockMeta);

afterEach(() => {
  vi.restoreAllMocks();
});

describe("postcards", () => {
  it("sends only date, depth and event id, then shows the HTML caption and the caveat", async () => {
    const post = vi
      .spyOn(apiClient, "postPostcard")
      .mockResolvedValue(mockPostcard({ date: "2020-01-05", depth_m: 50 }));
    render(
      <PostcardPanel date="2020-01-05" depth={50} eventId="ev-1" eventTitle="Event" caveat={meta.caveats.postcard} onClose={() => {}} />
    );
    expect(screen.getByRole("status")).toHaveTextContent(/Printing/);
    expect(await screen.findByRole("img")).toHaveAttribute("src", resolveApiUrl("/api/v1/postcard/mock-2020-01-05.svg"));
    expect(post.mock.calls[0][0]).toEqual({ date: "2020-01-05", depth_m: 50, event_id: "ev-1" });
    expect(screen.getByText(meta.caveats.postcard)).toBeInTheDocument();
    expect(screen.getByText(/Mean temperature/)).toBeInTheDocument();
    expect(screen.getByText("Copy link")).toBeInTheDocument();
  });

  it("formats caption lines without a mean when it is null", () => {
    const r = mockPostcard({ date: "2020-01-05", depth_m: 0.5 });
    r.caption.mean_c = null;
    expect(captionLines(r)).toHaveLength(2);
  });

  it("maps errors to friendly messages", () => {
    expect(postcardErrorMessage(new ApiError(503, "postcard_disabled", "x"), false)).toMatch(/switched off/);
    expect(postcardErrorMessage(null, true)).toMatch(/too long/);
  });
});

describe("lasso", () => {
  const seg = mockSegment({ date: "2020-01-05", depth_m: 50, lat: 15, lon: 88 });

  it("shows stats, method, provider and the caveat, and asks the guide", () => {
    const onAsk = vi.fn();
    render(<LassoPanel pending={false} error={null} result={seg} onAsk={onAsk} onClose={() => {}} />);
    expect(screen.getByText("Selected by similar temperature (±0.5 °C) · Mock lasso (offline)")).toBeInTheDocument();
    expect(screen.getByText(seg.caveat)).toBeInTheDocument();
    fireEvent.click(screen.getByText("Ask about this area"));
    expect(onAsk).toHaveBeenCalledWith(seg);
  });

  it("shows only a message on error, with no stats", () => {
    render(<LassoPanel pending={false} error="Tap on the water." result={null} onClose={() => {}} />);
    expect(screen.getByText("Tap on the water.")).toBeInTheDocument();
    expect(screen.queryByText("Mean")).not.toBeInTheDocument();
  });

  it("builds the bbox and the guide prompt from the outline", () => {
    expect(outlineBBox(seg.outline)).toEqual({ lat_min: 13.5, lat_max: 16.5, lon_min: 86.5, lon_max: 89.5 });
    expect(askAboutAreaPrompt(seg)).toContain("13.50–16.50°N, 86.50–89.50°E");
    expect(methodLabel("sam")).toBe("Selected by SAM 3.1");
    expect(segmentErrorMessage(new ApiError(422, "point_not_ocean", "x"))).toMatch(/water/);
  });
});

describe("beach mode", () => {
  it("names the direction and distance", () => {
    expect(compassWord(0)).toBe("north");
    expect(compassWord(91)).toBe("east");
    expect(compassWord(359)).toBe("north");
    expect(describeDistance({ distance_km: 61.6, bearing_deg: 135 })).toBe("62 km south-east of you");
  });
});

describe("resolveApiUrl", () => {
  it("prefixes backend paths with the API origin and keeps absolute URLs", () => {
    expect(resolveApiUrl("/api/v1/postcard/a.webp")).toBe("http://localhost:8000/api/v1/postcard/a.webp");
    expect(resolveApiUrl("https://cdn.example/x.webp")).toBe("https://cdn.example/x.webp");
  });
});
