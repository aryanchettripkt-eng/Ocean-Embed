import { describe, expect, it, vi } from "vitest";
import { renderGridToCanvas } from "../src/components/map/GridCanvas";
import { getColorRgba } from "../src/lib/colormaps";

describe("renderGridToCanvas", () => {
  it("creates a canvas with correct scaled dimensions and maps ocean/land/hatch pixels", () => {
    // 2 x 2 grid
    // Row 0 (lat_min): col 0 is land (cov 0), col 1 is valid (cov 1, val 25)
    // Row 1 (lat_max): col 0 is hatch (cov 2, val null), col 1 is valid (cov 1, val 20)
    const values: (number | null)[][] = [
      [null, 25.0],
      [null, 20.0],
    ];
    const coverage: number[][] = [
      [0, 1],
      [2, 1],
    ];

    const vmin = 10;
    const vmax = 30;
    const scale = 2;
    const width = 2 * scale; // 4
    const height = 2 * scale; // 4
    const pixelData = new Uint8ClampedArray(width * height * 4);

    const mockCtx = {
      createImageData: vi.fn().mockReturnValue({
        width,
        height,
        data: pixelData,
      }),
      putImageData: vi.fn(),
    };

    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
    };

    vi.stubGlobal("document", {
      createElement: vi.fn().mockImplementation((tag: string) => {
        if (tag === "canvas") return mockCanvas;
        return {};
      }),
    });

    const canvas = renderGridToCanvas(values, coverage, vmin, vmax, "thermal", scale);

    expect(canvas.width).toBe(4);
    expect(canvas.height).toBe(4);
    expect(mockCtx.putImageData).toHaveBeenCalledTimes(1);

    // Row 1 in values (lat_max) is top row in canvas (yBase = 0, y in [0, 1])
    // Col 0 at y=0 is cov 2 (hatch pattern): alpha should be 230
    const hatchIdx = (0 * width + 0) * 4;
    expect(pixelData[hatchIdx + 3]).toBe(230);

    // Col 1 at y=0 is cov 1, val 20
    const expectedColor20 = getColorRgba(20.0, vmin, vmax, "thermal");
    const val20Idx = (0 * width + 2) * 4;
    expect(pixelData[val20Idx]).toBe(expectedColor20[0]);
    expect(pixelData[val20Idx + 1]).toBe(expectedColor20[1]);
    expect(pixelData[val20Idx + 2]).toBe(expectedColor20[2]);
    expect(pixelData[val20Idx + 3]).toBe(expectedColor20[3]);

    // Row 0 in values (lat_min) is bottom row in canvas (yBase = (2 - 1 - 0)*2 = 2, y in [2, 3])
    // Col 0 at y=2 is cov 0 (land): all zeros
    const landIdx = (2 * width + 0) * 4;
    expect(pixelData[landIdx]).toBe(0);
    expect(pixelData[landIdx + 1]).toBe(0);
    expect(pixelData[landIdx + 2]).toBe(0);
    expect(pixelData[landIdx + 3]).toBe(0);

    // Col 1 at y=2 is cov 1, val 25
    const expectedColor25 = getColorRgba(25.0, vmin, vmax, "thermal");
    const val25Idx = (2 * width + 2) * 4;
    expect(pixelData[val25Idx]).toBe(expectedColor25[0]);
    expect(pixelData[val25Idx + 1]).toBe(expectedColor25[1]);
    expect(pixelData[val25Idx + 2]).toBe(expectedColor25[2]);
    expect(pixelData[val25Idx + 3]).toBe(expectedColor25[3]);

    vi.unstubAllGlobals();
  });

  it("renders dot-matrix mode with ocean dots, hollow insufficient-input rings, and muted land dots", () => {
    // 2 x 2 grid:
    // Row 0 (lat_min -> bottom row yBase = 8): col 0 is land (cov 0), col 1 is valid ocean (cov 1, val 25)
    // Row 1 (lat_max -> top row yBase = 0): col 0 is insufficient (cov 2), col 1 is valid ocean (cov 1, val 20)
    const values: (number | null)[][] = [
      [null, 25.0],
      [null, 20.0],
    ];
    const coverage: number[][] = [
      [0, 1],
      [2, 1],
    ];

    const vmin = 10;
    const vmax = 30;
    const scale = 8;
    const width = 2 * scale; // 16
    const height = 2 * scale; // 16
    const pixelData = new Uint8ClampedArray(width * height * 4);

    const mockCtx = {
      createImageData: vi.fn().mockReturnValue({
        width,
        height,
        data: pixelData,
      }),
      putImageData: vi.fn(),
    };

    const mockCanvas = {
      width: 0,
      height: 0,
      getContext: vi.fn().mockReturnValue(mockCtx),
    };

    vi.stubGlobal("document", {
      createElement: vi.fn().mockImplementation((tag: string) => {
        if (tag === "canvas") return mockCanvas;
        return {};
      }),
    });

    renderGridToCanvas(values, coverage, vmin, vmax, "thermal", scale, null, {
      gridStyle: "dots",
      isDark: true,
    });

    // Helper to read RGBA at pixel (px, py)
    const rgbaAt = (px: number, py: number) => {
      const i = (py * width + px) * 4;
      return [pixelData[i], pixelData[i + 1], pixelData[i + 2], pixelData[i + 3]];
    };

    // 1. Top-left cell (col 0, row 1 = cov 2 hollow ring):
    // Corner (0, 0) is dark obsidian matrix backing [7, 13, 18, 218]
    expect(rgbaAt(0, 0)).toEqual([7, 13, 18, 218]);
    // Center (4, 4) of hollow ring is also backing (hollow interior!)
    expect(rgbaAt(4, 4)).toEqual([7, 13, 18, 218]);
    // Ring stroke at (6, 4) (cx = +0.3125, cy = +0.0625 -> dist ~ 0.318 in [0.19, 0.37]) is bright slate
    const ringStroke = rgbaAt(6, 4);
    expect(ringStroke[0]).toBeGreaterThan(100);
    expect(ringStroke[1]).toBeGreaterThan(110);
    expect(ringStroke[2]).toBeGreaterThan(130);

    // 2. Top-right cell (col 1, row 1 = cov 1 ocean dot, val 20):
    // Corner (8, 0) is dark obsidian backing
    expect(rgbaAt(8, 0)).toEqual([7, 13, 18, 218]);
    // Center (12, 4) is colored by cmocean thermal (>= base color due to subtle core boost)
    const baseOcean20 = getColorRgba(20.0, vmin, vmax, "thermal");
    const dotCenter20 = rgbaAt(12, 4);
    expect(dotCenter20[0]).toBeGreaterThanOrEqual(baseOcean20[0]);
    expect(dotCenter20[1]).toBeGreaterThanOrEqual(baseOcean20[1]);
    expect(dotCenter20[3]).toBe(255);

    // 3. Bottom-left cell (col 0, row 0 = cov 0 land dot):
    // Corner (0, 8) is dark obsidian backing
    expect(rgbaAt(0, 8)).toEqual([7, 13, 18, 218]);
    // Center (4, 12) has warm bronze-earth stipple dot (R > G > B > backing)
    const landCenter = rgbaAt(4, 12);
    expect(landCenter[0]).toBeGreaterThan(landCenter[1]);
    expect(landCenter[1]).toBeGreaterThan(landCenter[2]);
    expect(landCenter[0]).toBeGreaterThan(80);

    vi.unstubAllGlobals();
  });

  it("draws a valid cell with no value at this depth as a land dot, not an insufficient-input ring", () => {
    // 1 x 2 grid: col 0 is cov 1 with a null value (e.g. shallower than the selected depth), col 1 is cov 2
    const values: (number | null)[][] = [[null, null]];
    const coverage: number[][] = [[1, 2]];
    const scale = 8;
    const width = 2 * scale;
    const height = scale;
    const pixelData = new Uint8ClampedArray(width * height * 4);
    const mockCtx = {
      createImageData: vi.fn().mockReturnValue({ width, height, data: pixelData }),
      putImageData: vi.fn(),
    };
    const mockCanvas = { width: 0, height: 0, getContext: vi.fn().mockReturnValue(mockCtx) };
    vi.stubGlobal("document", {
      createElement: vi.fn().mockImplementation((tag: string) => (tag === "canvas" ? mockCanvas : {})),
    });

    renderGridToCanvas(values, coverage, 10, 30, "thermal", scale, null, { gridStyle: "dots", isDark: true });

    const rgbaAt = (px: number, py: number) => {
      const i = (py * width + px) * 4;
      return [pixelData[i], pixelData[i + 1], pixelData[i + 2], pixelData[i + 3]];
    };

    // Null-valued cov 1 cell: filled warm land dot at the centre, no ring stroke
    const nullCenter = rgbaAt(4, 4);
    expect(nullCenter[0]).toBeGreaterThan(nullCenter[2]);
    expect(nullCenter[0]).toBeGreaterThan(80);
    expect(rgbaAt(6, 4)).toEqual([7, 13, 18, 218]);

    // cov 2 cell next to it keeps the hollow ring (backing at the centre, stroke at the edge)
    expect(rgbaAt(12, 4)).toEqual([7, 13, 18, 218]);
    expect(rgbaAt(14, 4)[2]).toBeGreaterThan(130);

    vi.unstubAllGlobals();
  });
});
