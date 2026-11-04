/**
 * Client-side canvas renderer for 2D temperature grid fields (PRD_FRONTEND §2, §3.1).
 * Supports both dot-matrix ("dots") and classic continuous block ("solid") modes.
 */

import { getColorRgba, type ColormapName } from "../../lib/colormaps";
import type { GridStyle } from "../../lib/url-state";

export interface GridRenderOptions {
  gridStyle?: GridStyle;
  isDark?: boolean;
}

interface SubCellMask {
  oceanAlpha: number;
  coreBoost: number;
  ringAlpha: number;
  landAlpha: number;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function buildSubCellLut(scale: number): SubCellMask[] {
  const lut: SubCellMask[] = new Array(scale * scale);
  const aa = Math.max(0.06, 0.85 / scale);

  for (let dy = 0; dy < scale; dy++) {
    const cy = (dy + 0.5) / scale - 0.5;
    for (let dx = 0; dx < scale; dx++) {
      const cx = (dx + 0.5) / scale - 0.5;
      const dist = Math.hypot(cx, cy);

      const oceanAlpha = clamp01((0.38 + 0.5 * aa - dist) / aa);
      const coreBoost = dist < 0.18 ? (1 - dist / 0.18) * 0.12 : 0;
      const ringOuter = clamp01((0.37 + 0.5 * aa - dist) / aa);
      const ringInner = clamp01((dist - (0.19 - 0.5 * aa)) / aa);
      const ringAlpha = ringOuter * ringInner;
      const landAlpha = clamp01((0.24 + 0.5 * aa - dist) / aa) * 0.82;

      lut[dy * scale + dx] = {
        oceanAlpha,
        coreBoost,
        ringAlpha,
        landAlpha,
      };
    }
  }
  return lut;
}

function blendOverBacking(
  fgR: number,
  fgG: number,
  fgB: number,
  fgWeight: number,
  bgR: number,
  bgG: number,
  bgB: number,
  bgA255: number
): [number, number, number, number] {
  if (fgWeight <= 0.001) {
    return [bgR, bgG, bgB, bgA255];
  }
  const aFg = clamp01(fgWeight);
  const aBg = bgA255 / 255;
  const outA = aFg + aBg * (1 - aFg);
  if (outA <= 0.0001) {
    return [0, 0, 0, 0];
  }
  const invOutA = 1 / outA;
  const r = Math.round((fgR * aFg + bgR * aBg * (1 - aFg)) * invOutA);
  const g = Math.round((fgG * aFg + bgG * aBg * (1 - aFg)) * invOutA);
  const b = Math.round((fgB * aFg + bgB * aBg * (1 - aFg)) * invOutA);
  return [r, g, b, Math.round(outA * 255)];
}

export function renderGridToCanvas(
  values: (number | null)[][],
  coverage: number[][],
  vmin: number,
  vmax: number,
  colormap: ColormapName = "thermal",
  scale = 4, // Supersample each cell into scale x scale pixels for sharp patterns
  targetCanvas?: HTMLCanvasElement | null,
  options?: GridRenderOptions
): HTMLCanvasElement {
  const nLat = values.length;
  const nLon = values[0]?.length ?? 0;
  const gridStyle: GridStyle = options?.gridStyle ?? "solid";
  const isDark = options?.isDark ?? true;

  const canvas = targetCanvas || document.createElement("canvas");
  const targetWidth = nLon * scale;
  const targetHeight = nLat * scale;

  if (canvas.width !== targetWidth) canvas.width = targetWidth;
  if (canvas.height !== targetHeight) canvas.height = targetHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  const imgData = ctx.createImageData(canvas.width, canvas.height);
  const data = imgData.data;

  if (gridStyle === "dots") {
    const subLut = buildSubCellLut(scale);
    const bgR = isDark ? 7 : 241;
    const bgG = isDark ? 13 : 245;
    const bgB = isDark ? 18 : 249;
    const bgA = isDark ? 218 : 212;

    const landR = isDark ? 126 : 148;
    const landG = isDark ? 103 : 138;
    const landB = isDark ? 76 : 124;

    const ringR = isDark ? 148 : 71;
    const ringG = isDark ? 163 : 85;
    const ringB = isDark ? 184 : 105;

    for (let r = 0; r < nLat; r++) {
      const yBase = (nLat - 1 - r) * scale;

      for (let c = 0; c < nLon; c++) {
        const xBase = c * scale;
        const cov = coverage[r]?.[c] ?? 0;
        const val = values[r]?.[c];

        let rgba: [number, number, number, number] = [0, 0, 0, 0];
        if (cov === 1) {
          rgba = getColorRgba(val, vmin, vmax, colormap);
        }

        for (let dy = 0; dy < scale; dy++) {
          const py = yBase + dy;
          for (let dx = 0; dx < scale; dx++) {
            const px = xBase + dx;
            const idx = (py * canvas.width + px) * 4;
            const mask = subLut[dy * scale + dx];

            let out: [number, number, number, number];
            if (cov === 0) {
              out = blendOverBacking(landR, landG, landB, mask.landAlpha, bgR, bgG, bgB, bgA);
            } else if (cov === 2 || rgba[3] === 0) {
              out = blendOverBacking(ringR, ringG, ringB, mask.ringAlpha * 0.92, bgR, bgG, bgB, bgA);
            } else {
              const boost = mask.coreBoost * 255;
              const dotR = Math.min(255, Math.round(rgba[0] + boost));
              const dotG = Math.min(255, Math.round(rgba[1] + boost));
              const dotB = Math.min(255, Math.round(rgba[2] + boost));
              out = blendOverBacking(dotR, dotG, dotB, mask.oceanAlpha, bgR, bgG, bgB, bgA);
            }

            data[idx] = out[0];
            data[idx + 1] = out[1];
            data[idx + 2] = out[2];
            data[idx + 3] = out[3];
          }
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

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
