import { describe, expect, it } from "vitest";
import {
  BALANCE_LUT,
  getColorHex,
  getColorRgba,
  getColormapGradientCss,
  THERMAL_LUT,
} from "../src/lib/colormaps";

describe("colormaps.ts cmocean thermal and balance", () => {
  it("contains 256 RGB entries in LUTs", () => {
    expect(THERMAL_LUT.length).toBe(256);
    expect(BALANCE_LUT.length).toBe(256);
  });

  it("handles null and NaN values with transparent RGBA", () => {
    expect(getColorRgba(null, 20, 30, "thermal")).toEqual([0, 0, 0, 0]);
    expect(getColorRgba(undefined, 20, 30, "thermal")).toEqual([0, 0, 0, 0]);
    expect(getColorRgba(NaN, 20, 30, "thermal")).toEqual([0, 0, 0, 0]);
    expect(getColorHex(null, 20, 30, "thermal")).toBe("transparent");
  });

  it("clamps values to boundary colors when out of range", () => {
    const minColor = getColorRgba(10, 20, 30, "thermal");
    expect(minColor).toEqual([...THERMAL_LUT[0], 255]);

    const maxColor = getColorRgba(40, 20, 30, "thermal");
    expect(maxColor).toEqual([...THERMAL_LUT[255], 255]);
  });

  it("returns hex strings for valid values", () => {
    const hex = getColorHex(25, 20, 30, "thermal");
    expect(hex).toMatch(/^#[0-9a-f]{6}$/);
  });

  it("generates CSS linear-gradient string", () => {
    const css = getColormapGradientCss("thermal");
    expect(css).toContain("linear-gradient(to right,");
    expect(css).toContain("rgb(");
  });
});
