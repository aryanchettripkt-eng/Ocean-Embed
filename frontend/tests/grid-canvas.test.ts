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
});
