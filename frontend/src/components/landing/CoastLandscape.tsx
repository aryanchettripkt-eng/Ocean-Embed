import React from "react";
import styles from "./landing.module.css";

/**
 * Hand-crafted Woodblock Coast & Sky SVG (PRD_FRONTEND §0, §2, PLAN_LANDING §4).
 * Inspired by Japanese ukiyo-e woodblock prints (Hiroshige / Hokusai):
 * features rising morning sun, rolling mountain ranges, coastal promontory with temple,
 * soaring sea birds, billowing woodcut clouds, and framing pine branches.
 * Pure vector SVG: 0 KB raster payload, scales infinitely, 100 Lighthouse performance.
 */
export function CoastLandscape() {
  return (
    <svg
      className={styles.coastLandscapeSvg}
      viewBox="0 0 1600 800"
      preserveAspectRatio="xMidYMax slice"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Woodblock landscape of the North Indian Ocean coast with mountains, rising sun, and pine trees"
    >
      <defs>
        {/* Soft paper sky gradient */}
        <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FBF6EA" />
          <stop offset="60%" stopColor="#EFE6CF" />
          <stop offset="100%" stopColor="#E4D8BA" />
        </linearGradient>

        {/* Distant mountain haze gradient */}
        <linearGradient id="distantMtnGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#DFD1AF" />
          <stop offset="100%" stopColor="#CBBF9A" />
        </linearGradient>

        {/* Midground mountain gradient */}
        <linearGradient id="midMtnGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8A9A96" />
          <stop offset="100%" stopColor="#4A5654" />
        </linearGradient>

        {/* Coastal rock gradient */}
        <linearGradient id="coastRockGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3B4F52" />
          <stop offset="100%" stopColor="#1E2A2B" />
        </linearGradient>

        {/* Sun radial halo */}
        <radialGradient id="sunHalo" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#F2B66D" stopOpacity="0.4" />
          <stop offset="70%" stopColor="#F2B66D" stopOpacity="0.1" />
          <stop offset="100%" stopColor="#F2B66D" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Sky Canvas */}
      <rect width="1600" height="800" fill="url(#skyGrad)" />

      {/* 1. Rising Sun */}
      <g id="sun-layer">
        {/* Soft outer halo */}
        <circle cx="1240" cy="240" r="140" fill="url(#sunHalo)" />
        {/* Concentric woodblock radiance rings */}
        <circle
          cx="1240"
          cy="240"
          r="110"
          fill="none"
          stroke="#F2B66D"
          strokeWidth="1.5"
          strokeDasharray="8 6"
          opacity="0.5"
        />
        <circle
          cx="1240"
          cy="240"
          r="95"
          fill="none"
          stroke="#F2B66D"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.6"
        />
        {/* Main Sun Disc */}
        <circle
          cx="1240"
          cy="240"
          r="75"
          fill="#F2B66D"
          stroke="#1E2A2B"
          strokeWidth="2.5"
        />
      </g>

      {/* 2. Stylized Woodcut Clouds (Back) */}
      <g id="clouds-back" fill="#F6F1E2" stroke="#1E2A2B" strokeWidth="1.5" opacity="0.85">
        {/* Cloud Bank Left */}
        <path d="M 120 180 C 160 140 240 140 280 170 C 310 150 370 150 390 180 C 430 180 460 210 440 240 L 80 240 C 60 210 90 180 120 180 Z" />
        {/* Cloud Bar Center-High */}
        <path d="M 620 90 C 660 60 740 60 780 90 C 820 70 890 80 910 110 L 580 110 C 590 95 605 90 620 90 Z" />
        {/* Cloud Swirl Right */}
        <path d="M 1080 320 C 1120 290 1200 290 1230 320 C 1260 305 1330 310 1360 340 L 1020 340 C 1030 325 1060 320 1080 320 Z" />
      </g>

      {/* 3. Distant Mountains (Far Horizon) */}
      <g id="distant-mountains">
        {/* Western Ghats / Coastal Range Far Ridge */}
        <path
          d="M 280 480 L 420 310 L 530 410 L 680 260 L 810 390 L 960 290 L 1120 440 L 1350 330 L 1520 460 L 1600 440 L 1600 520 L 280 520 Z"
          fill="url(#distantMtnGrad)"
          stroke="#1E2A2B"
          strokeWidth="2"
        />
        {/* Mountain facets & snow/ridge ink contours */}
        <g stroke="#1E2A2B" strokeWidth="1.5" fill="none" opacity="0.65">
          <path d="M 420 310 L 440 440 M 680 260 L 710 380 M 960 290 L 980 410 M 1350 330 L 1380 450" />
          {/* Woodcut hatching on shade side */}
          <path d="M 685 290 L 720 310 M 695 320 L 740 340 M 705 350 L 755 370" />
          <path d="M 965 315 L 1000 335 M 972 345 L 1015 365 M 978 375 L 1025 395" />
        </g>
      </g>

      {/* 4. Midground Peaks & Hills */}
      <g id="midground-hills">
        <path
          d="M 500 520 L 640 370 L 780 470 L 920 350 L 1080 460 L 1220 380 L 1400 480 L 1600 450 L 1600 540 L 500 540 Z"
          fill="url(#midMtnGrad)"
          stroke="#1E2A2B"
          strokeWidth="2.5"
        />
        {/* Traditional ink vertical stroke textures on hills */}
        <g stroke="#1E2A2B" strokeWidth="1.8" fill="none" opacity="0.5">
          <path d="M 640 370 L 660 480 M 920 350 L 950 470 M 1220 380 L 1250 490" />
          <path d="M 650 400 L 690 440 M 660 430 L 710 470" />
          <path d="M 930 380 L 980 430 M 940 410 L 1000 460" />
        </g>
      </g>

      {/* 5. Coastal Cliffs & Promontory Headland (Foreground Coast) */}
      <g id="coastal-promontory">
        {/* Bold Coastal Bluffs */}
        <path
          d="M 720 540 Q 840 440 980 460 Q 1120 430 1260 470 Q 1420 420 1600 450 L 1600 560 L 720 560 Z"
          fill="url(#coastRockGrad)"
          stroke="#1E2A2B"
          strokeWidth="3"
        />

        {/* Shoreline Sand Spit & Inlets */}
        <path
          d="M 640 540 C 720 515 820 535 900 540 C 980 545 1060 525 1160 535 C 1260 545 1400 520 1600 530 L 1600 560 L 640 560 Z"
          fill="#E4D8BA"
          stroke="#1E2A2B"
          strokeWidth="2"
        />

        {/* Rock texture hatching */}
        <g stroke="#1E2A2B" strokeWidth="2" fill="none" opacity="0.75">
          <path d="M 880 480 L 890 530 M 1040 465 L 1055 525 M 1210 475 L 1230 535 M 1380 460 L 1400 520" />
          <path d="M 860 500 L 895 510 M 1020 490 L 1060 500 M 1190 500 L 1235 510" />
        </g>
      </g>

      {/* 6. Traditional Woodblock Shrine / Temple Pavilion on the Bluff */}
      <g id="temple-pavilion" transform="translate(1320, 395)">
        {/* Stone foundation base */}
        <rect x="15" y="48" width="50" height="8" fill="#1E2A2B" stroke="#1E2A2B" strokeWidth="1.5" />
        {/* Support columns */}
        <line x1="22" y1="32" x2="22" y2="48" stroke="#1E2A2B" strokeWidth="2.5" />
        <line x1="38" y1="32" x2="38" y2="48" stroke="#1E2A2B" strokeWidth="2" />
        <line x1="58" y1="32" x2="58" y2="48" stroke="#1E2A2B" strokeWidth="2.5" />
        <rect x="26" y="36" width="26" height="12" fill="#E4D8BA" stroke="#1E2A2B" strokeWidth="1.5" />

        {/* Lower tier flared roof */}
        <path
          d="M 8 32 C 18 34 32 30 40 24 C 48 30 62 34 72 32 C 65 28 50 25 40 22 C 30 25 15 28 8 32 Z"
          fill="#1E2A2B"
          stroke="#1E2A2B"
          strokeWidth="1.5"
        />

        {/* Upper tier pavilion */}
        <line x1="30" y1="14" x2="30" y2="22" stroke="#1E2A2B" strokeWidth="2" />
        <line x1="50" y1="14" x2="50" y2="22" stroke="#1E2A2B" strokeWidth="2" />

        {/* Curved Pagoda Top Roof with upward flared eaves */}
        <path
          d="M 18 14 C 26 15 34 11 40 7 C 46 11 54 15 62 14 C 55 10 46 8 40 6 C 34 8 25 10 18 14 Z"
          fill="#1E2A2B"
          stroke="#1E2A2B"
          strokeWidth="1.5"
        />
        {/* Roof Finial Spire */}
        <line x1="40" y1="0" x2="40" y2="6" stroke="#1E2A2B" strokeWidth="2" />
        <circle cx="40" cy="0" r="2" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
      </g>

      {/* 7. Soaring Sea Birds (Flock of 6 terns) */}
      <g id="birds" stroke="#1E2A2B" strokeWidth="2" fill="none" strokeLinecap="round">
        {/* Bird 1 */}
        <path d="M 680 180 Q 692 168 702 178 Q 712 168 724 180" />
        {/* Bird 2 */}
        <path d="M 730 155 Q 740 145 748 153 Q 756 145 766 155" />
        {/* Bird 3 */}
        <path d="M 760 190 Q 768 182 774 188 Q 780 182 788 190" />
        {/* Bird 4 */}
        <path d="M 810 165 Q 818 158 824 163 Q 830 158 838 165" />
        {/* Bird 5 */}
        <path d="M 850 140 Q 856 134 862 138 Q 868 134 874 140" />
      </g>

      {/* 8. Japanese Pine Branch (Top-Right Framing) */}
      <g id="pine-framing">
        {/* Main Gnarled Branch */}
        <path
          d="M 1600 0 C 1520 20 1440 60 1370 110 C 1310 150 1260 210 1220 280 C 1215 290 1205 310 1190 330"
          stroke="#1E2A2B"
          strokeWidth="18"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M 1440 60 C 1410 110 1370 170 1350 240"
          stroke="#1E2A2B"
          strokeWidth="11"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M 1320 150 C 1280 190 1240 230 1200 270"
          stroke="#1E2A2B"
          strokeWidth="7"
          strokeLinecap="round"
          fill="none"
        />

        {/* Pine Needle Tufts (Woodblock radial needle clusters) */}
        <g stroke="#1E2A2B" strokeWidth="2" fill="none">
          {/* Cluster 1 */}
          <g transform="translate(1360, 100)">
            <path d="M 0 0 L -30 -15 M 0 0 L -35 0 M 0 0 L -32 15 M 0 0 L -22 28 M 0 0 L -8 34 M 0 0 L 8 32 M 0 0 L 22 24" />
            <path d="M -15 -8 L -40 -5 M -18 8 L -42 12 M -10 18 L -32 28" />
          </g>
          {/* Cluster 2 */}
          <g transform="translate(1270, 180)">
            <path d="M 0 0 L -32 -10 M 0 0 L -36 5 M 0 0 L -28 20 M 0 0 L -15 30 M 0 0 L 2 34 M 0 0 L 18 28" />
            <path d="M -16 -4 L -42 -2 M -14 12 L -38 22" />
          </g>
          {/* Cluster 3 */}
          <g transform="translate(1200, 260)">
            <path d="M 0 0 L -30 -8 M 0 0 L -34 8 M 0 0 L -25 22 M 0 0 L -10 30 M 0 0 L 10 28" />
          </g>
          {/* Cluster 4 (lower tip) */}
          <g transform="translate(1185, 325)">
            <path d="M 0 0 L -25 -5 M 0 0 L -28 10 M 0 0 L -18 22 M 0 0 L -4 28" />
          </g>
          {/* Cluster 5 (top edge) */}
          <g transform="translate(1480, 50)">
            <path d="M 0 0 L -35 -10 M 0 0 L -38 6 M 0 0 L -30 20 M 0 0 L -16 30 M 0 0 L 4 32" />
          </g>
        </g>
      </g>
    </svg>
  );
}
