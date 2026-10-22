import { describe, expect, it } from "vitest";
import {
  buildExplorerSearchParams,
  findClosestDepth,
  parseExplorerParams,
} from "../src/lib/url-state";
import { mockMeta } from "../src/mocks/data";

describe("url-state.ts explorer query sync", () => {
  it("finds closest depth level", () => {
    const depths = [0.5, 5, 10, 20, 50, 75, 100, 150, 200];
    expect(findClosestDepth(98, depths)).toBe(100);
    expect(findClosestDepth(45, depths)).toBe(50);
    expect(findClosestDepth(0.4, depths)).toBe(0.5);
  });

  it("parses valid URL params with metadata", () => {
    const params = new URLSearchParams({
      date: "2020-01-05",
      depth: "100",
      field: "glorys",
      lat: "12.500",
      lon: "65.500",
    });

    const state = parseExplorerParams(params, mockMeta);
    expect(state.date).toBe("2020-01-05");
    expect(state.depth).toBe(100.0);
    expect(state.field).toBe("glorys");
    expect(state.lat).toBe(12.5);
    expect(state.lon).toBe(65.5);
  });

  it("falls back gracefully when URL params are invalid or out of bounds", () => {
    const invalidParams = new URLSearchParams({
      date: "1980-01-01", // outside available_dates
      depth: "99999", // not a valid depth
      field: "invalid_field",
      lat: "85.0", // outside region
      lon: "150.0", // outside region
    });

    const state = parseExplorerParams(invalidParams, mockMeta);
    // Date falls back to latest val date (2020-12-01 capped or available_dates)
    expect(state.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(state.field).toBe("pred");
    expect(state.depth).toBe(1000.0); // closest depth to 99999 is max depth 1000m
    expect(state.lat).toBeNull();
    expect(state.lon).toBeNull();
  });

  it("round-trips state through buildExplorerSearchParams", () => {
    const original = {
      date: "2020-01-03",
      depth: 50.0,
      field: "error" as const,
      lat: 14.25,
      lon: 68.75,
    };

    const sp = buildExplorerSearchParams(original);
    const parsed = parseExplorerParams(sp, mockMeta);

    expect(parsed.date).toBe(original.date);
    expect(parsed.depth).toBe(original.depth);
    expect(parsed.field).toBe(original.field);
    expect(parsed.lat).toBe(original.lat);
    expect(parsed.lon).toBe(original.lon);
  });
});
