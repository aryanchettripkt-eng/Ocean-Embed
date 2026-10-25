import React from "react";
import Image from "next/image";
import { Callout } from "./Callout";
import { DepthRuler } from "./DepthRuler";
import styles from "./landing.module.css";

/**
 * Authentic Woodblock Hero Scene Plate (PLAN_LANDING_FIX §2, §3).
 * Sized to cover the hero section while maintaining 1673:940; anchored bottom-right.
 * Callouts and depth ruler stay inside stage in cqw units.
 */
export function HeroScene() {
  return (
    <div className={styles.stage} aria-hidden="false">
      {/* 1. Woodblock Plate Background Image */}
      <Image
        src="/landing/scene-2400.webp"
        alt="Woodblock-style illustration: a satellite above a coast, and a cutaway of the ocean below with an ARGO float and a warm layer near the surface"
        fill
        priority
        quality={85}
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
