/**
 * Client-side canvas renderer for 2D temperature grid fields (PRD_FRONTEND §2, §3.1).
 */

import { getColorRgba, type ColormapName } from "../../lib/colormaps";

export function renderGridToCanvas(
  values: (number | null)[][],
  coverage: number[][],
  vmin: number,
  vmax: number,
  colormap: ColormapName = "thermal",
  scale = 4, // Supersample each cell into scale x scale pixels for sharp hatch patterns
  targetCanvas?: HTMLCanvasElement | null
): HTMLCanvasElement {
  const nLat = values.length;
  const nLon = values[0]?.length ?? 0;

  const canvas = targetCanvas || document.createElement("canvas");
  const targetWidth = nLon * scale;
  const targetHeight = nLat * scale;

  if (canvas.width !== targetWidth) canvas.width = targetWidth;
  if (canvas.height !== targetHeight) canvas.height = targetHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(canvas.width, canvas.height);
  const data = imgData.data;

  for (let r = 0; r < nLat; r++) {
    // In canvas, y=0 is top (lat_max), while in data index 0 is bottom (lat_min)
    const yBase = (nLat - 1 - r) * scale;

    for (let c = 0; c < nLon; c++) {
      const xBase = c * scale;
      const cov = coverage[r]?.[c] ?? 0;
      const val = values[r]?.[c];

      let rgba: [number, number, number, number] = [0, 0, 0, 0];
      if (cov === 1) {
        rgba = getColorRgba(val, vmin, vmax, colormap);
      }

      // Draw scale x scale block
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) {
          const px = xBase + dx;
          const py = yBase + dy;
          const idx = (py * canvas.width + px) * 4;

          if (cov === 0) {
            // Land / unmodelled: transparent
            data[idx] = 0;
            data[idx + 1] = 0;
            data[idx + 2] = 0;
            data[idx + 3] = 0;
          } else if (cov === 2) {
            // Insufficient inputs: diagonal hatch pattern
            const isStripe = (px + py) % 4 < 2;
            if (isStripe) {
              data[idx] = 110;
              data[idx + 1] = 120;
              data[idx + 2] = 135;
              data[idx + 3] = 230;
            } else {
              data[idx] = 40;
              data[idx + 1] = 45;
              data[idx + 2] = 55;
              data[idx + 3] = 230;
            }
          } else {
            // Valid ocean cell
            data[idx] = rgba[0];
            data[idx + 1] = rgba[1];
            data[idx + 2] = rgba[2];
            data[idx + 3] = rgba[3];
          }
        }
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}
