import { describe, expect, it } from "vitest";
import { formatCoord, formatDate, formatDepth, formatSplit, formatTemp } from "../src/lib/format";

describe("format.ts oceanographic formatters", () => {
  it("formats temperature with custom decimals", () => {
    expect(formatTemp(28.456, 1)).toBe("28.5 °C");
    expect(formatTemp(28.456, 2)).toBe("28.46 °C");
    expect(formatTemp(null)).toBe("—");
    expect(formatTemp(undefined)).toBe("—");
    expect(formatTemp(NaN)).toBe("—");
  });

  it("formats ocean depth according to PRD rule (< 10m 1 dec, >= 10m integer)", () => {
    expect(formatDepth(0.49)).toBe("0.5 m");
    expect(formatDepth(5.0)).toBe("5.0 m");
    expect(formatDepth(9.9)).toBe("9.9 m");
    expect(formatDepth(10.0)).toBe("10 m");
    expect(formatDepth(99.8)).toBe("100 m");
    expect(formatDepth(1000.0)).toBe("1000 m");
    expect(formatDepth(null)).toBe("—");
    expect(formatDepth(undefined)).toBe("—");
  });

  it("formats coordinates with cardinal directions and 3 decimals", () => {
    expect(formatCoord(12.375, 65.125)).toBe("12.375°N, 65.125°E");
    expect(formatCoord(-5.5, -40.25)).toBe("5.500°S, 40.250°W");
    expect(formatCoord(0, 0)).toBe("0.000°N, 0.000°E");
    expect(formatCoord(null, 65.0)).toBe("—");
  });

  it("formats date strings into readable labels", () => {
    expect(formatDate("2020-01-05")).toBe("05 Jan 2020");
    expect(formatDate("2021-12-31")).toBe("31 Dec 2021");
    expect(formatDate(null)).toBe("—");
  });

  it("formats split badges", () => {
    expect(formatSplit("val")).toEqual({ label: "Validation", variant: "val" });
    expect(formatSplit("test")).toEqual({ label: "Test", variant: "test" });
    expect(formatSplit("train")).toEqual({ label: "Training (in-sample)", variant: "train" });
    expect(formatSplit("live")).toEqual({ label: "Live", variant: "live" });
  });
});
