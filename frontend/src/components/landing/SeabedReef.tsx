import React from "react";
import styles from "./landing.module.css";

/**
 * Hand-crafted Woodblock Seabed, Coral Reef & Pelagic Fish SVG (PRD_FRONTEND §0, §2, PLAN_LANDING §4).
 * Inspired by Japanese ukiyo-e woodcuts:
 * features deep rocky sea shelves, delicate sea fans and coral branches,
 * flowing sea kelp ribbons, and a dynamic school of ocean fish swimming eastward.
 * Pure vector SVG: 0 KB raster bloat, scales infinitely, 100 Lighthouse performance.
 */
export function SeabedReef() {
  return (
    <svg
      className={styles.seabedReefSvg}
      viewBox="0 0 1200 480"
      preserveAspectRatio="xMinYMax meet"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Woodblock seabed reef with coral fans, kelp, and a school of ocean fish"
    >
      <defs>
        {/* Deep abyss gradient for base shelf */}
        <linearGradient id="abyssShelfGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1E2A2B" />
          <stop offset="60%" stopColor="#14353B" />
          <stop offset="100%" stopColor="#0B1D20" />
        </linearGradient>

        {/* Coral gold gradient */}
        <linearGradient id="coralGoldGrad" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stopColor="#CA774E" />
          <stop offset="100%" stopColor="#F2B66D" />
        </linearGradient>

        {/* Kelp sea teal gradient */}
        <linearGradient id="kelpGrad" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stopColor="#1F4E57" />
          <stop offset="100%" stopColor="#2F6B73" />
        </linearGradient>

        {/* Fish body sheen */}
        <linearGradient id="fishSheenGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#2F6B73" />
          <stop offset="50%" stopColor="#E4D8BA" />
          <stop offset="100%" stopColor="#F6F1E2" />
        </linearGradient>
      </defs>

      {/* 1. Deep Oceanic Seabed Ledges (Rocky shelves) */}
      <g id="rocky-shelves">
        {/* Back rocky shelf */}
        <path
          d="M 0 480 L 0 310 Q 90 285 180 320 Q 300 270 420 340 Q 560 300 700 390 Q 840 360 980 430 L 1200 440 L 1200 480 Z"
          fill="#14353B"
          stroke="#1E2A2B"
          strokeWidth="3"
        />

        {/* Foreground rocky ledge */}
        <path
          d="M 0 480 L 0 360 Q 110 330 220 380 Q 340 330 460 405 Q 600 370 740 435 L 940 460 L 1200 470 L 1200 480 Z"
          fill="url(#abyssShelfGrad)"
          stroke="#1E2A2B"
          strokeWidth="3.5"
        />

        {/* Geological Strata & Woodcut Stippling */}
        <g stroke="#1E2A2B" strokeWidth="2" fill="none" opacity="0.7">
          <path d="M 40 380 Q 120 360 200 390" />
          <path d="M 80 410 Q 180 390 280 425" />
          <path d="M 260 370 Q 340 350 420 385" />
          <path d="M 380 420 Q 480 395 580 440" />

          {/* Stipple dot clusters on rock faces */}
          <circle cx="90" cy="350" r="1.5" fill="#1E2A2B" />
          <circle cx="110" cy="355" r="1.5" fill="#1E2A2B" />
          <circle cx="100" cy="365" r="1.5" fill="#1E2A2B" />
          <circle cx="210" cy="360" r="1.5" fill="#1E2A2B" />
          <circle cx="225" cy="368" r="1.5" fill="#1E2A2B" />
          <circle cx="330" cy="370" r="1.5" fill="#1E2A2B" />
          <circle cx="350" cy="380" r="1.5" fill="#1E2A2B" />
        </g>
      </g>

      {/* 2. Delicate Golden Sea Fans & Coral (Woodcut branches) */}
      <g id="sea-fans">
        {/* Coral Fan 1 (Left Mound) */}
        <g transform="translate(140, 330)" stroke="url(#coralGoldGrad)" strokeWidth="2.5" fill="none" strokeLinecap="round">
          <path d="M 0 0 C -20 -50 -50 -80 -80 -110 M 0 0 C -10 -60 -25 -110 -35 -140 M 0 0 C 15 -60 25 -100 30 -140 M 0 0 C 35 -50 65 -90 90 -115 M 0 0 C 55 -40 95 -65 125 -85" />
          {/* Secondary branching twigs */}
          <path d="M -40 -60 C -60 -85 -85 -95 -105 -105" strokeWidth="1.8" />
          <path d="M -18 -75 C -30 -105 -45 -125 -60 -135" strokeWidth="1.8" />
          <path d="M 20 -70 C 35 -95 55 -115 70 -125" strokeWidth="1.8" />
          <path d="M 50 -55 C 75 -75 105 -90 120 -95" strokeWidth="1.8" />
          {/* Coral polyp tips */}
          <circle cx="-80" cy="-110" r="3" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
          <circle cx="-35" cy="-140" r="3" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
          <circle cx="30" cy="-140" r="3" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
          <circle cx="90" cy="-115" r="3" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
          <circle cx="125" cy="-85" r="3" fill="#F2B66D" stroke="#1E2A2B" strokeWidth="1" />
        </g>

        {/* Coral Fan 2 (Mid-Left Ledge) */}
        <g transform="translate(420, 350)" stroke="#F2B66D" strokeWidth="2.2" fill="none" strokeLinecap="round" opacity="0.9">
          <path d="M 0 0 C -15 -40 -35 -70 -55 -90 M 0 0 C 5 -45 10 -80 15 -110 M 0 0 C 25 -40 45 -70 65 -90" />
          <path d="M -20 -45 C -40 -65 -60 -75 -75 -80" strokeWidth="1.5" />
          <path d="M 15 -50 C 30 -70 50 -85 65 -95" strokeWidth="1.5" />
          <circle cx="-55" cy="-90" r="2.5" fill="#F2B66D" />
          <circle cx="15" cy="-110" r="2.5" fill="#F2B66D" />
          <circle cx="65" cy="-90" r="2.5" fill="#F2B66D" />
        </g>
      </g>

      {/* 3. Swaying Sea Kelp & Ribbon Grass Blades */}
      <g id="sea-kelp" stroke="url(#kelpGrad)" strokeWidth="3.5" fill="none" strokeLinecap="round">
        {/* Kelp Cluster 1 */}
        <path d="M 270 370 C 250 300 300 230 270 160 C 250 110 275 60 260 20" />
        <path d="M 295 375 C 275 315 325 250 305 190 C 285 140 315 90 295 50" strokeWidth="3" />
        <path d="M 320 380 C 310 330 350 270 330 220 C 315 170 340 120 325 90" strokeWidth="2.5" />

        {/* Kelp Cluster 2 */}
        <path d="M 520 415 C 500 355 540 290 520 230 C 505 180 535 130 515 80" strokeWidth="3" />
        <path d="M 545 420 C 530 370 570 310 550 250 C 535 200 560 150 545 110" strokeWidth="2.5" />
      </g>

      {/* 4. School of Pelagic Fish (8 synchronized tuna/mackerel silhouettes) */}
      <g id="school-of-fish" transform="translate(180, 20)">
        {/* Fish Helper Definition:
            A woodblock streamlined fish swimming right:
            torpedo body, dorsal fin, ventral fins, forked tail, eye, gill slit */}
        
        {/* Fish 1 (Lead Scout - Top Right) */}
        <g transform="translate(680, 110)" stroke="#1E2A2B" strokeWidth="1.6" fill="url(#fishSheenGrad)">
          {/* Main Body */}
          <path d="M 0 12 C 28 2 54 8 72 12 C 54 16 28 22 0 12 Z" />
          {/* Forked Tail Fin */}
          <path d="M 0 12 L -14 3 L -9 12 L -14 21 Z" fill="#2F6B73" />
          {/* Dorsal Fin */}
          <path d="M 32 4 L 44 -3 L 42 6 Z" fill="#1F4E57" />
          {/* Pectoral Fin */}
          <path d="M 48 13 L 36 21 L 44 14 Z" fill="#1E2A2B" />
          {/* Eye & Gill Slit */}
          <circle cx="62" cy="10" r="1.5" fill="#1E2A2B" />
          <path d="M 54 7 C 52 11 52 13 54 17" fill="none" strokeWidth="1.2" />
        </g>

        {/* Fish 2 (Mid-High Lead) */}
        <g transform="translate(610, 75)" stroke="#1E2A2B" strokeWidth="1.5" fill="url(#fishSheenGrad)">
          <path d="M 0 11 C 26 2 50 7 66 11 C 50 15 26 20 0 11 Z" />
          <path d="M 0 11 L -13 3 L -8 11 L -13 19 Z" fill="#2F6B73" />
          <path d="M 30 4 L 40 -3 L 39 6 Z" fill="#1F4E57" />
          <path d="M 44 12 L 33 19 L 40 13 Z" fill="#1E2A2B" />
          <circle cx="56" cy="9" r="1.4" fill="#1E2A2B" />
          <path d="M 48 6 C 46 10 46 12 48 16" fill="none" strokeWidth="1.1" />
        </g>

        {/* Fish 3 (Center Major) */}
        <g transform="translate(540, 125)" stroke="#1E2A2B" strokeWidth="1.8" fill="url(#fishSheenGrad)">
          <path d="M 0 14 C 32 3 62 9 82 14 C 62 19 32 25 0 14 Z" />
          <path d="M 0 14 L -16 4 L -10 14 L -16 24 Z" fill="#2F6B73" />
          <path d="M 38 5 L 50 -4 L 48 7 Z" fill="#1F4E57" />
          <path d="M 54 15 L 40 24 L 50 16 Z" fill="#1E2A2B" />
          <circle cx="70" cy="12" r="1.6" fill="#1E2A2B" />
          <path d="M 60 8 C 58 13 58 15 60 20" fill="none" strokeWidth="1.3" />
        </g>

        {/* Fish 4 (Center Companion) */}
        <g transform="translate(470, 95)" stroke="#1E2A2B" strokeWidth="1.6" fill="url(#fishSheenGrad)">
          <path d="M 0 12 C 28 2 54 8 72 12 C 54 16 28 22 0 12 Z" />
          <path d="M 0 12 L -14 3 L -9 12 L -14 21 Z" fill="#2F6B73" />
          <path d="M 32 4 L 44 -3 L 42 6 Z" fill="#1F4E57" />
          <path d="M 48 13 L 36 21 L 44 14 Z" fill="#1E2A2B" />
          <circle cx="62" cy="10" r="1.5" fill="#1E2A2B" />
          <path d="M 54 7 C 52 11 52 13 54 17" fill="none" strokeWidth="1.2" />
        </g>

        {/* Fish 5 (Lower Center) */}
        <g transform="translate(490, 160)" stroke="#1E2A2B" strokeWidth="1.5" fill="url(#fishSheenGrad)">
          <path d="M 0 11 C 26 2 50 7 66 11 C 50 15 26 20 0 11 Z" />
          <path d="M 0 11 L -13 3 L -8 11 L -13 19 Z" fill="#2F6B73" />
          <path d="M 30 4 L 40 -3 L 39 6 Z" fill="#1F4E57" />
          <path d="M 44 12 L 33 19 L 40 13 Z" fill="#1E2A2B" />
          <circle cx="56" cy="9" r="1.4" fill="#1E2A2B" />
          <path d="M 48 6 C 46 10 46 12 48 16" fill="none" strokeWidth="1.1" />
        </g>

        {/* Fish 6 (Trailing Upper) */}
        <g transform="translate(400, 70)" stroke="#1E2A2B" strokeWidth="1.4" fill="url(#fishSheenGrad)">
          <path d="M 0 10 C 24 2 46 6 60 10 C 46 14 24 18 0 10 Z" />
          <path d="M 0 10 L -12 3 L -7 10 L -12 17 Z" fill="#2F6B73" />
          <path d="M 28 3 L 37 -3 L 36 5 Z" fill="#1F4E57" />
          <circle cx="51" cy="8" r="1.3" fill="#1E2A2B" />
        </g>

        {/* Fish 7 (Trailing Mid) */}
        <g transform="translate(360, 120)" stroke="#1E2A2B" strokeWidth="1.5" fill="url(#fishSheenGrad)">
          <path d="M 0 11 C 26 2 50 7 66 11 C 50 15 26 20 0 11 Z" />
          <path d="M 0 11 L -13 3 L -8 11 L -13 19 Z" fill="#2F6B73" />
          <path d="M 30 4 L 40 -3 L 39 6 Z" fill="#1F4E57" />
          <circle cx="56" cy="9" r="1.4" fill="#1E2A2B" />
        </g>

        {/* Fish 8 (Trailing Lower Rearguard) */}
        <g transform="translate(300, 150)" stroke="#1E2A2B" strokeWidth="1.4" fill="url(#fishSheenGrad)">
          <path d="M 0 10 C 24 2 46 6 60 10 C 46 14 24 18 0 10 Z" />
          <path d="M 0 10 L -12 3 L -7 10 L -12 17 Z" fill="#2F6B73" />
          <path d="M 28 3 L 37 -3 L 36 5 Z" fill="#1F4E57" />
          <circle cx="51" cy="8" r="1.3" fill="#1E2A2B" />
        </g>
      </g>
    </svg>
  );
}
