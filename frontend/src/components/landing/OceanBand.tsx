"use client";

import React, { useEffect, useRef, useState, useMemo } from "react";
import { useMeta, useSection } from "../../lib/api/hooks";
import { findNearestCoord, getWallColorRange } from "../map3d/depthScale";
import { THERMAL_LUT } from "../../lib/colormaps";
import { ThermalBand } from "./ThermalBand";
import styles from "./landing.module.css";

// Standard 4x4 Bayer Matrix
const BAYER_4X4: number[][] = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

/**
 * Phase C: Real Data Subsurface Ocean Band (PLAN_LANDING §7).
 * Fetches GET /section at ~15°N for the latest available date.
 * Renders on a <canvas> with sqrt(depth) scaling, Bayer 4x4 ordered dithering
 * using the cmocean thermal LUT, 2°C dashed contours, and the 20°C isotherm line.
 * Vector ThermalBand stays underneath as graceful fallback with zero layout shift.
 */
export function OceanBand() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasReady, setCanvasReady] = useState(false);
  const { data: meta } = useMeta();

  // Find latitude row nearest 15.0°N
  const targetLat = useMemo(() => {
    if (!meta?.region) return 15.0;
    const lats: number[] = [];
    const minLat = meta.region.lat_min;
    const maxLat = meta.region.lat_max;
    const res = meta.grid_resolution_deg || 0.25;
    for (let lat = minLat; lat <= maxLat + 1e-5; lat += res) {
      lats.push(Math.round(lat * 1000) / 1000);
    }
    return findNearestCoord(15.0, lats);
  }, [meta]);

  // Use latest live date or latest available date
  const targetDate = useMemo(() => {
    if (!meta) return "";
    return meta.latest_live_date ?? meta.available_dates?.end ?? "";
  }, [meta]);

  const { data: section } = useSection(
    {
      date: targetDate,
      field: "pred",
      lat: targetLat,
    },
    { enabled: !!meta && !!targetDate }
  );

  useEffect(() => {
    if (!section || !section.values || !section.positions || !section.depths_m) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = 1600;
    const height = 500;
    canvas.width = width;
    canvas.height = height;

    const { vmin, vmax } = getWallColorRange(meta?.color_ranges);
    const range = vmax > vmin ? vmax - vmin : 1;

    const depths = section.depths_m;
    const maxDepth = Math.max(...depths);
    const positions = section.positions;
    const numPositions = positions.length;
    const numDepths = depths.length;

    // Create ImageData
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Helper: sample temperature at (posIdx, depth_m)
    function sampleTemp(pIdx: number, dMeters: number): number | null {
      let d0 = 0;
      let d1 = 0;
      if (dMeters <= depths[0]) {
        d0 = 0;
        d1 = 0;
      } else if (dMeters >= depths[numDepths - 1]) {
        d0 = numDepths - 1;
        d1 = numDepths - 1;
      } else {
        for (let i = 0; i < numDepths - 1; i++) {
          if (dMeters >= depths[i] && dMeters <= depths[i + 1]) {
            d0 = i;
            d1 = i + 1;
            break;
          }
        }
      }

      const p0 = Math.floor(pIdx);
      const p1 = Math.min(numPositions - 1, p0 + 1);
      const fracP = pIdx - p0;

      const fracD =
        d0 === d1 ? 0 : (dMeters - depths[d0]) / (depths[d1] - depths[d0]);

      const v00 = section!.values[d0]?.[p0] ?? null;
      const v01 = section!.values[d0]?.[p1] ?? null;
      const v10 = section!.values[d1]?.[p0] ?? null;
      const v11 = section!.values[d1]?.[p1] ?? null;

      if (v00 === null || v01 === null || v10 === null || v11 === null) {
        return v00 ?? v01 ?? v10 ?? v11;
      }

      const top = v00 * (1 - fracP) + v01 * fracP;
      const bot = v10 * (1 - fracP) + v11 * fracP;
      return top * (1 - fracD) + bot * fracD;
    }

    // Render pixel grid with ordered Bayer 4x4 dithering
    for (let y = 0; y < height; y++) {
      const ratioY = y / (height - 1);
      const dMeters = ratioY * ratioY * maxDepth;

      for (let x = 0; x < width; x++) {
        const pIdx = (x / (width - 1)) * (numPositions - 1);
        const temp = sampleTemp(pIdx, dMeters);

        const pixelIndex = (y * width + x) * 4;

        if (temp === null || isNaN(temp)) {
          data[pixelIndex] = 20;
          data[pixelIndex + 1] = 53;
          data[pixelIndex + 2] = 59;
          data[pixelIndex + 3] = 230;
          continue;
        }

        const frac = (temp - vmin) / range;
        const bayer = (BAYER_4X4[y % 4][x % 4] / 16 - 0.5) * (1 / 32);
        const ditheredFrac = Math.max(0, Math.min(1, frac + bayer));
        const lutIdx = Math.max(0, Math.min(255, Math.floor(ditheredFrac * 255)));
        const rgb = THERMAL_LUT[lutIdx] ?? [20, 53, 59];

        data[pixelIndex] = rgb[0];
        data[pixelIndex + 1] = rgb[1];
        data[pixelIndex + 2] = rgb[2];
        data[pixelIndex + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Draw 2°C contours and 20°C Isotherm
    const contourTemps = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];

    contourTemps.forEach((targetTemp) => {
      const is20C = targetTemp === 20;
      ctx.beginPath();
      let started = false;

      for (let x = 0; x < width; x += 4) {
        const pIdx = (x / (width - 1)) * (numPositions - 1);

        let foundDepth: number | null = null;
        for (let d = 0; d < numDepths - 1; d++) {
          const tTop = sampleTemp(pIdx, depths[d]);
          const tBot = sampleTemp(pIdx, depths[d + 1]);
          if (tTop !== null && tBot !== null) {
            if (
              (tTop >= targetTemp && tBot <= targetTemp) ||
              (tTop <= targetTemp && tBot >= targetTemp)
            ) {
              const diff = tBot - tTop;
              const f = Math.abs(diff) > 1e-4 ? (targetTemp - tTop) / diff : 0;
              foundDepth = depths[d] + f * (depths[d + 1] - depths[d]);
              break;
            }
          }
        }

        if (foundDepth !== null) {
          const y = (Math.sqrt(foundDepth) / Math.sqrt(maxDepth)) * (height - 1);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        } else {
          started = false;
        }
      }

      ctx.save();
      if (is20C) {
        ctx.strokeStyle = "#F6F1E2";
        ctx.lineWidth = 2.8;
        ctx.setLineDash([14, 6]);
        ctx.globalAlpha = 0.95;
      } else {
        ctx.strokeStyle = "#F6F1E2";
        ctx.lineWidth = 1.3;
        ctx.setLineDash([8, 8]);
        ctx.globalAlpha = 0.65;
      }
      ctx.stroke();
      ctx.restore();
    });

    setCanvasReady(true);
  }, [section, meta]);

  const captionText = useMemo(() => {
    if (!section) return null;
    const latStr =
      section.fixed?.lat !== undefined ? `${section.fixed.lat.toFixed(1)}°N` : "";
    const synthStr = section.synthetic ? " · Synthetic demo data" : "";
    return `Model prediction · ${section.date} · section at ${latStr}${synthStr}`;
  }, [section]);

  return (
    <div className={styles.oceanBandImage} style={{ position: "relative", width: "100%", height: "100%" }}>
      {/* 1. Underlying vector woodblock SVG (instant paint, zero CLS fallback) */}
      <ThermalBand />

      {/* 2. Real /section data canvas with Bayer dithering and isotherms */}
      <canvas
        ref={canvasRef}
        className={`${styles.oceanBandCanvas} ${canvasReady ? styles.oceanBandCanvasActive : ""}`}
        aria-hidden="true"
        data-testid="ocean-band-canvas"
      />

      {/* 3. Real data caption under the ruler */}
      {captionText && (
        <div className={styles.oceanBandCaption} data-testid="ocean-band-caption">
          {captionText}
        </div>
      )}
    </div>
  );
}
