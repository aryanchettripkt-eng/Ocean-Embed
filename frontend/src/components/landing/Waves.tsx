import React from "react";
import styles from "./landing.module.css";

/**
 * Hand-crafted Japanese Woodblock Wave Component (PRD_FRONTEND §0, §2, PLAN_LANDING §4, §5).
 * Inspired by Hokusai's "Great Wave off Kanagawa":
 * Features rhythmic curling wave crests, woodblock claw foam fingers, and rhythmic swell lines.
 * Animated infinitely with pure CSS transforms (GPU hardware accelerated, zero JS runtime cost).
 */
export function Waves() {
  return (
    <div className={styles.wavesWrapper} aria-hidden="true">
      <div className={styles.waveTrack}>
        {/* Wave Layer 1 (Foreground Sea Surface Turquoise, Translating Left) */}
        <svg
          className={styles.wavePath1}
          viewBox="0 0 1600 50"
          preserveAspectRatio="none"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Main water body */}
          <path
            d="M 0 22 
               C 80 8 130 35 200 22 C 280 8 330 35 400 22 
               C 480 8 530 35 600 22 C 680 8 730 35 800 22 
               C 880 8 930 35 1000 22 C 1080 8 1130 35 1200 22 
               C 1280 8 1330 35 1400 22 C 1480 8 1530 35 1600 22 
               L 1600 50 L 0 50 Z"
            fill="#2F6B73"
          />

          {/* Under-crest shadow ink band */}
          <path
            d="M 0 24 C 80 12 130 37 200 24 C 280 12 330 37 400 24 C 480 12 530 37 600 24 C 680 12 730 37 800 24 C 880 12 930 37 1000 24 C 1080 12 1130 37 1200 24 C 1280 12 1330 37 1400 24 C 1480 12 1530 37 1600 24"
            stroke="#1E2A2B"
            strokeWidth="2"
            fill="none"
          />

          {/* Authentic Woodblock Foam Crest & Curling Foam Claws */}
          <path
            d="M 0 20 
               C 50 14 70 6 85 4 C 75 10 95 6 110 8 C 95 14 135 18 160 22 C 180 25 190 23 200 20
               C 250 14 270 6 285 4 C 275 10 295 6 310 8 C 295 14 335 18 360 22 C 380 25 390 23 400 20
               C 450 14 470 6 485 4 C 475 10 495 6 510 8 C 495 14 535 18 560 22 C 580 25 590 23 600 20
               C 650 14 670 6 685 4 C 675 10 695 6 710 8 C 695 14 735 18 760 22 C 780 25 790 23 800 20
               C 850 14 870 6 885 4 C 875 10 895 6 910 8 C 895 14 935 18 960 22 C 980 25 990 23 1000 20
               C 1050 14 1070 6 1085 4 C 1075 10 1095 6 1110 8 C 1095 14 1135 18 1160 22 C 1180 25 1190 23 1200 20
               C 1250 14 1270 6 1285 4 C 1275 10 1295 6 1310 8 C 1295 14 1335 18 1360 22 C 1380 25 1390 23 1400 20
               C 1450 14 1470 6 1485 4 C 1475 10 1495 6 1510 8 C 1495 14 1535 18 1560 22 C 1580 25 1590 23 1600 20"
            stroke="#F6F1E2"
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />

          {/* Dancing Spray Droplets */}
          <g fill="#F6F1E2">
            <circle cx="82" cy="1" r="1.5" />
            <circle cx="106" cy="3" r="1.2" />
            <circle cx="282" cy="1" r="1.5" />
            <circle cx="306" cy="3" r="1.2" />
            <circle cx="482" cy="1" r="1.5" />
            <circle cx="506" cy="3" r="1.2" />
            <circle cx="682" cy="1" r="1.5" />
            <circle cx="706" cy="3" r="1.2" />
            <circle cx="882" cy="1" r="1.5" />
            <circle cx="906" cy="3" r="1.2" />
            <circle cx="1082" cy="1" r="1.5" />
            <circle cx="1106" cy="3" r="1.2" />
            <circle cx="1282" cy="1" r="1.5" />
            <circle cx="1306" cy="3" r="1.2" />
            <circle cx="1482" cy="1" r="1.5" />
            <circle cx="1506" cy="3" r="1.2" />
          </g>
        </svg>

        {/* Wave Layer 2 (Background Deep Swell Navy, Translating Right) */}
        <svg
          className={styles.wavePath2}
          viewBox="0 0 1600 50"
          preserveAspectRatio="none"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M 0 28 
               Q 100 12 200 28 T 400 28 T 600 28 T 800 28 T 1000 28 T 1200 28 T 1400 28 T 1600 28 
               L 1600 50 L 0 50 Z"
            fill="#1F4E57"
            opacity="0.8"
          />
          <path
            d="M 0 28 
               Q 100 12 200 28 T 400 28 T 600 28 T 800 28 T 1000 28 T 1200 28 T 1400 28 T 1600 28"
            stroke="#F6F1E2"
            strokeWidth="2"
            strokeDasharray="24 10"
          />
        </svg>
      </div>
    </div>
  );
}
