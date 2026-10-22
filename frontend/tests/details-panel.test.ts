import { describe, expect, it } from "vitest";
import { formatCoord, formatDate, formatDepth, formatTemp } from "../src/lib/format";
import { createMockProfile, createMockTimeseries, mockMetrics } from "../src/mocks/data";

describe("DetailsPanel data logic and formatting", () => {
  it("formats coordinate and depth labels strictly according to PRD standards", () => {
    // PRD: coordinates like "12.375°N, 65.125°E"
    expect(formatCoord(12.375, 65.125)).toBe("12.375°N, 65.125°E");
    expect(formatCoord(-5.5, -45.25)).toBe("5.500°S, 45.250°W");

    // PRD: depth < 10m one decimal, >= 10m integer
    expect(formatDepth(0.5)).toBe("0.5 m");
    expect(formatDepth(5.0)).toBe("5.0 m");
    expect(formatDepth(10.0)).toBe("10 m");
    expect(formatDepth(100.0)).toBe("100 m");
    expect(formatDepth(1450.0)).toBe("1450 m");
  });

  it("extracts and formats profile points for both validation and live dates", () => {
    // Validation date
    const valProfile = createMockProfile(12.5, 65.5, "2020-01-05");
    expect(valProfile.split).toBe("val");
    expect(valProfile.depths_m.length).toBe(15);
    expect(valProfile.pred.every((p) => p !== null)).toBe(true);
    expect(valProfile.glorys.every((g) => g !== null)).toBe(true);

    // Live date
    const liveProfile = createMockProfile(12.5, 65.5, "2020-01-09");
    expect(liveProfile.split).toBe("live");
    expect(liveProfile.glorys.every((g) => g === null)).toBe(true);
    expect(liveProfile.pred.every((p) => p !== null)).toBe(true);
  });

  it("extracts timeseries points and matches target date", () => {
    const ts = createMockTimeseries(12.5, 65.5, "2020-01-01", "2020-01-10");
    expect(ts.dates.length).toBe(10);
    expect(ts.depth_m).toBe(100.0);
    expect(ts.pred.length).toBe(10);
    expect(ts.glorys[8]).toBeNull(); // live date in mock
    expect(ts.glorys[9]).toBeNull(); // live date in mock
  });

  it("finds depth metric row and extracts RMSE and bias for selected depth", () => {
    const targetDepth = 100.0;
    const row = mockMetrics.rows.find(
      (r) =>
        r.depth_m !== null &&
        r.depth_m !== "all" &&
        Math.abs(Number(r.depth_m) - targetDepth) < 0.1 &&
        r.model === "SYNTHETIC_DEMO"
    );

    expect(row).toBeDefined();
    expect(row?.rmse_c).toBe(0.29);
    expect(row?.bias_c).toBe(0.29);
    expect(formatTemp(row?.rmse_c ?? 0)).toBe("0.3 °C");
  });
});
