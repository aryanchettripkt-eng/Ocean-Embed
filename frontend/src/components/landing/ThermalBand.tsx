import React from "react";
import styles from "./landing.module.css";

/**
 * Hand-crafted Woodblock Subsurface Thermal Band SVG (PRD_FRONTEND §0, §2, PLAN_LANDING §5).
 * Represents the 0 to 1,000m subsurface temperature field of the North Indian Ocean:
 * Warm surface mixed layer -> steep thermocline -> cold deep ocean abyss.
 * Pure vector SVG: 0 KB raster bloat, scales infinitely, 100 Lighthouse performance.
 */
export function ThermalBand() {
  return (
    <svg
      className={styles.oceanBandImage}
      viewBox="0 0 1600 500"
      preserveAspectRatio="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Subsurface ocean temperature thermal band showing thermocline and isotherms from surface to 1000 meters"
    >
      <defs>
        <linearGradient id="thermalFieldGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F2B66D" />
          <stop offset="12%" stopColor="#E59B56" />
          <stop offset="28%" stopColor="#CA774E" />
          <stop offset="45%" stopColor="#6F6D69" />
          <stop offset="65%" stopColor="#2F6B73" />
          <stop offset="85%" stopColor="#1F4E57" />
          <stop offset="100%" stopColor="#14353B" />
        </linearGradient>

        <linearGradient id="contourGlow" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#F6F1E2" stopOpacity="0.4" />
          <stop offset="50%" stopColor="#F6F1E2" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#F6F1E2" stopOpacity="0.4" />
        </linearGradient>
      </defs>

      {/* 1. Thermal Field Background Quad */}
      <rect width="1600" height="500" fill="url(#thermalFieldGrad)" />

      {/* 2. Subsurface Internal Waves & Contours */}
      <g fill="none" stroke="#F6F1E2" strokeLinecap="round">
        {/* 28°C Surface Contour */}
        <path
          d="M 0 50 Q 400 62 800 48 T 1600 54"
          strokeWidth="1.5"
          strokeDasharray="8 6"
          opacity="0.65"
        />

        {/* 24°C Upper Thermocline Contour */}
        <path
          d="M 0 110 Q 400 126 800 106 T 1600 118"
          strokeWidth="1.6"
          strokeDasharray="10 6"
          opacity="0.75"
        />

        {/* 20°C Isotherm (Primary indicator of thermocline depth) */}
        <path
          d="M 0 180 Q 400 210 800 172 T 1600 186"
          stroke="url(#contourGlow)"
          strokeWidth="2.8"
          strokeDasharray="14 6"
        />

        {/* 16°C Mid-depth Contour */}
        <path
          d="M 0 260 Q 400 278 800 252 T 1600 268"
          strokeWidth="1.5"
          strokeDasharray="8 8"
          opacity="0.65"
        />

        {/* 12°C Lower Thermocline Contour */}
        <path
          d="M 0 350 Q 400 362 800 344 T 1600 354"
          strokeWidth="1.4"
          strokeDasharray="6 8"
          opacity="0.55"
        />

        {/* 8°C Abyssal Contour */}
        <path
          d="M 0 435 Q 400 442 800 430 T 1600 436"
          strokeWidth="1.2"
          strokeDasharray="6 10"
          opacity="0.45"
        />
      </g>

      {/* 3. Equatorial Kelvin / Rossby Wave Displacements (Woodblock internal currents) */}
      <g fill="none" stroke="#E4D8BA" strokeWidth="1.4" opacity="0.4">
        <path d="M 180 160 Q 280 220 380 170 Q 480 130 580 175" />
        <path d="M 880 170 Q 1020 230 1180 180 Q 1320 140 1480 185" />
      </g>

      {/* 4. Fine Woodcut Isotherm Hatching */}
      <g stroke="#1E2A2B" strokeWidth="1" fill="none" opacity="0.15">
        <line x1="0" y1="20" x2="1600" y2="20" strokeDasharray="3 9" />
        <line x1="0" y1="80" x2="1600" y2="80" strokeDasharray="3 9" />
        <line x1="0" y1="145" x2="1600" y2="145" strokeDasharray="3 9" />
        <line x1="0" y1="220" x2="1600" y2="220" strokeDasharray="3 9" />
        <line x1="0" y1="305" x2="1600" y2="305" strokeDasharray="3 9" />
        <line x1="0" y1="395" x2="1600" y2="395" strokeDasharray="3 9" />
      </g>
    </svg>
  );
}
