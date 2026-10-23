import React from "react";
import styles from "./landing.module.css";

export function Waves() {
  return (
    <div className={styles.wavesWrapper} aria-hidden="true">
      <div className={styles.waveTrack}>
        {/* Wave Layer 1 (Translating Left, 18s) */}
        <svg
          className={styles.wavePath1}
          viewBox="0 0 1200 40"
          preserveAspectRatio="none"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Water body */}
          <path
            d="M 0 15 Q 75 0 150 15 T 300 15 T 450 15 T 600 15 T 750 15 T 900 15 T 1050 15 T 1200 15 L 1200 40 L 0 40 Z"
            fill="#2F6B73"
          />
          {/* Wave Foam Crest */}
          <path
            d="M 0 15 Q 75 0 150 15 T 300 15 T 450 15 T 600 15 T 750 15 T 900 15 T 1050 15 T 1200 15"
            stroke="#F6F1E2"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        </svg>

        {/* Wave Layer 2 (Translating Right, 26s) */}
        <svg
          className={styles.wavePath2}
          viewBox="0 0 1200 40"
          preserveAspectRatio="none"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M 0 20 Q 100 8 200 20 T 400 20 T 600 20 T 800 20 T 1000 20 T 1200 20 L 1200 40 L 0 40 Z"
            fill="#1F4E57"
            opacity="0.75"
          />
          <path
            d="M 0 20 Q 100 8 200 20 T 400 20 T 600 20 T 800 20 T 1000 20 T 1200 20"
            stroke="#F6F1E2"
            strokeWidth="1.8"
            strokeDasharray="16 8"
          />
        </svg>
      </div>
    </div>
  );
}
