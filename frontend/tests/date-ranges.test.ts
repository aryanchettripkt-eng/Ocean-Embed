import { describe, expect, it } from "vitest";
import { ServedDays } from "../src/lib/date-ranges";

// Historical days, then live days after a gap (as live mode leaves them).
const ranges = [
  { start: "2020-10-30", end: "2020-11-01" },
  { start: "2026-10-28", end: "2026-10-29" },
];

describe("ServedDays", () => {
  const served = new ServedDays("2020-10-30", "2026-10-29", ranges);

  it("counts only served days and maps the slider across the gap", () => {
    expect(served.count).toBe(5);
    expect([0, 1, 2, 3, 4].map((i) => served.at(i))).toEqual([
      "2020-10-30",
      "2020-10-31",
      "2020-11-01",
      "2026-10-28",
      "2026-10-29",
    ]);
    expect(served.indexOf("2026-10-28")).toBe(3);
    expect(served.indexOf("2023-05-05")).toBe(3); // in the gap: next served day
  });

  it("steps over the gap in both directions and stops at the ends", () => {
    expect(served.step("2020-11-01", 1)).toBe("2026-10-28");
    expect(served.step("2026-10-28", -1)).toBe("2020-11-01");
    expect(served.step("2026-10-29", 1)).toBeNull();
    expect(served.step("2020-10-30", -1)).toBeNull();
  });

  it("snaps a typed gap date to the next served day", () => {
    expect(served.snap("2024-01-01")).toBe("2026-10-28");
    expect(served.snap("2020-10-31")).toBe("2020-10-31");
  });

  it("falls back to one continuous range without date_ranges", () => {
    const plain = new ServedDays("2020-01-01", "2020-01-10", null);
    expect(plain.count).toBe(10);
    expect(plain.step("2020-01-05", 1)).toBe("2020-01-06");
    expect(plain.first).toBe("2020-01-01");
    expect(plain.last).toBe("2020-01-10");
  });
});
