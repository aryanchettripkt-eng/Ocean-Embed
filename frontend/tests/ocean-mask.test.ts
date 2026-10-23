import { describe, expect, it } from "vitest";
import { isNorthIndianOcean } from "../src/lib/oceanMask";

describe("isNorthIndianOcean geographical mask", () => {
  it("correctly identifies ocean points in the Arabian Sea and Bay of Bengal", () => {
    // Arabian Sea open water
    expect(isNorthIndianOcean(15.0, 65.0)).toBe(true);
    expect(isNorthIndianOcean(18.0, 68.0)).toBe(true);
    expect(isNorthIndianOcean(10.0, 60.0)).toBe(true);

    // Bay of Bengal open water
    expect(isNorthIndianOcean(15.0, 88.0)).toBe(true);
    expect(isNorthIndianOcean(12.0, 85.0)).toBe(true);
    expect(isNorthIndianOcean(18.0, 90.0)).toBe(true);

    // Equatorial Indian Ocean
    expect(isNorthIndianOcean(2.0, 75.0)).toBe(true);
    expect(isNorthIndianOcean(1.0, 55.0)).toBe(true);
    expect(isNorthIndianOcean(3.0, 90.0)).toBe(true);
  });

  it("masks out northern landmasses (Pakistan, Afghanistan, Iran, Nepal, Tibet)", () => {
    // Pakistan & Afghanistan (previously bleeding)
    expect(isNorthIndianOcean(26.0, 67.0)).toBe(false); // Karachi / Pakistan
    expect(isNorthIndianOcean(30.0, 70.0)).toBe(false); // Pakistan / Punjab
    expect(isNorthIndianOcean(30.0, 65.0)).toBe(false); // Afghanistan border
    expect(isNorthIndianOcean(28.0, 62.0)).toBe(false); // Balochistan / Iran border

    // Iran
    expect(isNorthIndianOcean(30.0, 55.0)).toBe(false); // Iran
    expect(isNorthIndianOcean(27.0, 58.0)).toBe(false); // Southern Iran inland

    // Northern India & Himalayas & Tibet
    expect(isNorthIndianOcean(28.6, 77.2)).toBe(false); // New Delhi
    expect(isNorthIndianOcean(26.8, 80.9)).toBe(false); // Lucknow
    expect(isNorthIndianOcean(27.7, 85.3)).toBe(false); // Kathmandu / Nepal
    expect(isNorthIndianOcean(29.6, 91.1)).toBe(false); // Lhasa / Tibet
  });

  it("masks out terrestrial peninsular India and Sri Lanka", () => {
    expect(isNorthIndianOcean(19.0, 73.0)).toBe(false); // Mumbai inland
    expect(isNorthIndianOcean(13.0, 77.6)).toBe(false); // Bangalore
    expect(isNorthIndianOcean(17.4, 78.5)).toBe(false); // Hyderabad
    expect(isNorthIndianOcean(13.1, 80.3)).toBe(false); // Chennai inland
    expect(isNorthIndianOcean(7.5, 80.5)).toBe(false);  // Sri Lanka
  });

  it("masks out Arabia, Bangladesh, and Myanmar", () => {
    expect(isNorthIndianOcean(24.7, 46.7)).toBe(false); // Riyadh / Saudi Arabia
    expect(isNorthIndianOcean(23.8, 90.4)).toBe(false); // Dhaka / Bangladesh
    expect(isNorthIndianOcean(16.8, 96.1)).toBe(false); // Yangon / Myanmar
  });

  it("rejects coordinates outside the 40°–100°E, 0°–30°N domain", () => {
    expect(isNorthIndianOcean(-1.0, 75.0)).toBe(false);
    expect(isNorthIndianOcean(31.0, 75.0)).toBe(false);
    expect(isNorthIndianOcean(15.0, 39.0)).toBe(false);
    expect(isNorthIndianOcean(15.0, 101.0)).toBe(false);
  });
});
