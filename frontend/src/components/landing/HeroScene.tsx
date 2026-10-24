import React from "react";
import Image from "next/image";
import { Waves } from "./Waves";
import { OceanBand } from "./OceanBand";
import { DepthRuler } from "./DepthRuler";
import { Callout } from "./Callout";
import styles from "./landing.module.css";

/**
 * Authentic Woodblock Hero Scene (PRD_FRONTEND §0, §2, PLAN_LANDING §4, §5, §7).
 * Renders layered raster artwork and vector physics overlays matching reference mockup:
 * - SkyCoast: Misty mountains, coastal promontory, pagoda on cliff, rising orange sun, birds
 * - Tree: Overhanging Japanese pine branches in top-right with scroll parallax
 * - Satellite & Dashed Orbit: Earth observation satellite traveling along dashed orbital trajectory
 * - Waterline Waves: Curling foam crests across the sea surface
 * - Ocean Currents: Flow trajectory arching into the thermocline ("OCEAN CURRENTS MOVE HEAT")
 * - Subsurface Thermal Band: Real /section data canvas with Bayer dithering, isotherms, DepthRuler
 * - Seabed Coral Reef: Abyssal rocks, coral fans, and pelagic fish school
 * - Argo Float: Scientific profiling float bobbing on the left side with observation callout
 */
export function HeroScene() {
  return (
    <div className={styles.heroSceneContainer} aria-hidden="false">
      {/* 1. Woodblock Sky & Coast Landscape Layer */}
      <div className={styles.skyCoastLayer} aria-hidden="true">
        <Image
          src="/landing/sky-coast.webp"
          alt="Woodblock coastal landscape of the North Indian Ocean with mountains and rising sun"
          fill
          priority
          sizes="(max-width: 1024px) 100vw, 75vw"
          className="object-cover object-bottom"
        />
      </div>

      {/* 2. Top-Right Overhanging Pine Tree Branches */}
      <div className={styles.treeLayer} aria-hidden="true">
        <Image
          src="/landing/tree.webp"
          alt=""
          width={623}
          height={450}
          priority
          className="w-full h-auto"
        />
      </div>

      {/* 3. Orbiting Earth Observation Satellite & Dashed Orbit Path */}
      <div className={styles.satelliteOrbitLayer} aria-hidden="true">
        <svg
          className={styles.orbitSvg}
          viewBox="0 0 1000 500"
          preserveAspectRatio="none"
          fill="none"
        >
          {/* Dashed elliptical orbital trajectory */}
          <path
            d="M 550 50 Q 720 140 880 380"
            stroke="#1E2A2B"
            strokeWidth="1.5"
            strokeDasharray="6 6"
            opacity="0.5"
          />
          {/* Orbital cross tick marker behind satellite */}
          <line x1="710" y1="140" x2="722" y2="140" stroke="#1E2A2B" strokeWidth="1.5" opacity="0.65" />
          <line x1="716" y1="134" x2="716" y2="146" stroke="#1E2A2B" strokeWidth="1.5" opacity="0.65" />
        </svg>

        <div className={styles.satelliteLayer}>
          <Image
            src="/landing/satellite.webp"
            alt=""
            width={280}
            height={200}
            priority
            className="w-full h-auto"
          />
        </div>
      </div>

      {/* Satellite Callout: Top-right below satellite */}
      <Callout
        text={"SATELLITES\nOBSERVE\nTHE SURFACE"}
        variant="ink"
        className={styles.satelliteCallout}
      />

      {/* 4. Underwater Subsurface Region */}
      <div className={styles.waterArea} aria-hidden="true">
        {/* Waves at the surface waterline */}
        <Waves />

        {/* Ocean Currents Flow Curve & Label */}
        <div className={styles.currentsFlowLayer}>
          <svg
            className={styles.currentsSvg}
            viewBox="0 0 500 200"
            fill="none"
            aria-hidden="true"
          >
            {/* Spiral current path curving down into water */}
            <path
              d="M 50 25 C 20 25, 10 55, 30 75 C 60 100, 160 80, 240 120 C 310 155, 380 170, 440 180"
              stroke="#F6F1E2"
              strokeWidth="1.5"
              strokeDasharray="5 5"
              opacity="0.75"
            />
            {/* Flow arrowhead */}
            <path
              d="M 432 173 L 442 181 L 434 187"
              stroke="#F6F1E2"
              strokeWidth="1.5"
              fill="none"
              opacity="0.8"
            />
          </svg>
          <div className={styles.currentsText}>
            <div>OCEAN CURRENTS</div>
            <div>MOVE HEAT</div>
          </div>
        </div>

        {/* Subsurface thermal band cross-section with Depth Ruler */}
        <div className={styles.oceanBandContainer}>
          <OceanBand />
          <DepthRuler />
        </div>

        {/* Rocky Seabed, Coral Reef & Pelagic Fish School */}
        <div className={styles.seabedLayer}>
          <Image
            src="/landing/seabed.webp"
            alt=""
            width={900}
            height={390}
            className="w-full h-auto"
          />
        </div>

        {/* Scientific ARGO Float Bobbing on Left Side */}
        <div className={styles.argoFloatLayer}>
          <Image
            src="/landing/argo-float.webp"
            alt=""
            width={65}
            height={220}
            className="w-full h-auto"
          />
        </div>

        {/* ARGO Float Callout: Beside float on the left */}
        <Callout
          text={"ARGO FLOATS\nPROVIDE REAL\nOBSERVATIONS"}
          variant="foam"
          className={styles.argoCallout}
        />

        {/* Temperature at Depth Callout: Beside depth ruler on right */}
        <Callout
          text={"TEMPERATURE\nAT DEPTH"}
          variant="foam"
          className={styles.tempCallout}
        />
      </div>
    </div>
  );
}
