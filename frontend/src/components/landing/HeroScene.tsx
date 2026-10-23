import React from "react";
import Image from "next/image";
import { CoastLandscape } from "./CoastLandscape";
import { Waves } from "./Waves";
import { ThermalBand } from "./ThermalBand";
import { SeabedReef } from "./SeabedReef";
import { DepthRuler } from "./DepthRuler";
import { Callout } from "./Callout";
import styles from "./landing.module.css";

/**
 * Woodblock Hero Scene (PRD_FRONTEND §0, §2, PLAN_LANDING §4, §5).
 * Composed entirely of hand-crafted Japanese ukiyo-e woodblock vector SVG layers:
 * - CoastLandscape: Sky, sun, mountains, coastal promontory, pagoda, and pine framing
 * - Satellite: Observation satellite with telemetry scanning beam
 * - Waves: Great Wave foam claws and surging sea surface
 * - ThermalBand: Subsurface temperature cross-section with DepthRuler
 * - SeabedReef: Abyssal rocky ledges, golden coral fans, kelp, and pelagic fish school
 * - ArgoFloat: Scientific profiling float bobbing in the water column
 */
export function HeroScene() {
  return (
    <div className={styles.heroSceneContainer} aria-hidden="false">
      {/* 1. Woodblock Sky & Coast Landscape (0 KB raster, pure vector SVG) */}
      <div className={styles.skyCoastLayer} aria-hidden="true">
        <CoastLandscape />
      </div>

      {/* 2. Orbiting Earth Observation Satellite & Sensor Scanning Beam */}
      <div className={styles.satelliteLayer} aria-hidden="true">
        <Image
          src="/landing/satellite.svg"
          alt=""
          width={240}
          height={180}
          priority
          className="w-full h-auto"
        />
      </div>

      <Callout
        text="Satellites observe the surface"
        style={{ top: "14%", right: "24%" }}
      />

      <Callout
        text="Ocean currents move heat"
        style={{ top: "42%", right: "12%" }}
      />

      {/* 3. Underwater Subsurface Region */}
      <div className={styles.waterArea} aria-hidden="true">
        {/* Japanese woodblock waves with Great Wave curling foam claws */}
        <Waves />

        {/* Subsurface thermal band cross-section with Depth Ruler */}
        <div className={styles.oceanBandContainer}>
          <ThermalBand />
          <DepthRuler />
        </div>

        {/* Rocky Seabed, Coral Reef & Pelagic Fish School (vector SVG) */}
        <div className={styles.seabedLayer}>
          <SeabedReef />
        </div>

        {/* Bobbing Scientific ARGO Profiling Float */}
        <div className={styles.argoFloatLayer}>
          <Image
            src="/landing/argo-float.svg"
            alt=""
            width={65}
            height={180}
            className="w-full h-auto"
          />
        </div>

        <Callout
          text="Temperature at depth"
          style={{ bottom: "28%", right: "14%" }}
        />

        <Callout
          text="Argo floats provide real observations"
          style={{ bottom: "10%", right: "26%" }}
        />
      </div>
    </div>
  );
}
