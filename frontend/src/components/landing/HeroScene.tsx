import React from "react";
import Image from "next/image";
import { Callout } from "./Callout";
import { DepthRuler } from "./DepthRuler";
import styles from "./landing.module.css";

/**
 * Authentic Woodblock Hero Scene Plate (PRD_FRONTEND §0, §2, PLAN_LANDING_FIX §2.3, §2.4).
 * Single continuous woodblock plate with responsive callouts and depth ruler scaled in container units.
 */
export function HeroScene() {
  return (
    <div className={styles.stageArt} aria-hidden="false">
      {/* 1. Woodblock Plate Background Image */}
      <Image
        src="/landing/scene-2400.webp"
        alt="Woodblock-style illustration: a satellite above a coast, and a cutaway of the ocean below with an ARGO float and a warm layer near the surface"
        fill
        priority
        fetchPriority="high"
        sizes="100vw"
        className={styles.sceneImage}
      />

      {/* 2. Callout: Satellites Observe the Surface */}
      <Callout
        text={"SATELLITES\nOBSERVE\nTHE SURFACE"}
        variant="ink"
        className={styles.calloutSatellite}
      />

      {/* 3. Callout: Ocean Currents Move Heat (bare foam label) */}
      <Callout
        text={"OCEAN CURRENTS\nMOVE HEAT"}
        variant="bare"
        className={styles.calloutCurrents}
      />

      {/* 4. Callout: Argo Floats Provide Real Observations */}
      <Callout
        text={"ARGO FLOATS\nPROVIDE REAL\nOBSERVATIONS"}
        variant="foam"
        className={styles.calloutArgo}
      />

      {/* 5. Callout: Temperature at Depth */}
      <Callout
        text={"TEMPERATURE\nAT DEPTH"}
        variant="foam"
        className={styles.calloutTemp}
      />

      {/* 6. Inked Depth Ruler on right */}
      <DepthRuler />
    </div>
  );
}
