import React from "react";
import Image from "next/image";
import { Waves } from "./Waves";
import { DepthRuler } from "./DepthRuler";
import { Callout } from "./Callout";
import styles from "./landing.module.css";

export function HeroScene() {
  return (
    <div className={styles.heroSceneContainer} aria-hidden="false">
      {/* 1. Sky & Coast background layer */}
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

      {/* 2. Top-right Pine branches */}
      <div className={styles.treeLayer} aria-hidden="true">
        <Image
          src="/landing/tree.webp"
          alt=""
          width={900}
          height={700}
          className="w-full h-auto"
        />
      </div>

      {/* 3. Orbiting Satellite & Callout */}
      <div className={styles.satelliteLayer} aria-hidden="true">
        <Image
          src="/landing/satellite.svg"
          alt=""
          width={160}
          height={120}
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

      {/* 4. Underwater Region */}
      <div className={styles.waterArea} aria-hidden="true">
        {/* Waves at the surface waterline */}
        <Waves />

        {/* Thermal band cross-section */}
        <div className={styles.oceanBandContainer}>
          <Image
            src="/landing/ocean-band-placeholder.webp"
            alt=""
            fill
            sizes="100vw"
            className={styles.oceanBandImage}
          />
          {/* Depth Ruler on the right face */}
          <DepthRuler />
        </div>

        {/* Seabed coral & reef bottom-left */}
        <div className={styles.seabedLayer}>
          <Image
            src="/landing/seabed.webp"
            alt=""
            width={1000}
            height={500}
            className="w-full h-auto"
          />
        </div>

        {/* Bobbing ARGO Float */}
        <div className={styles.argoFloatLayer}>
          <Image
            src="/landing/argo-float.svg"
            alt=""
            width={55}
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
