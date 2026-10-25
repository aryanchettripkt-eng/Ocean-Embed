import React from "react";
import Link from "next/link";
import styles from "./landing.module.css";

export function HowItWorks() {
  return (
    <section className={styles.howItWorksSection} aria-label="How OceanScope AI Works">
      <div className={styles.sectionHeader}>
        <p className={styles.sectionEyebrow}>How It Works</p>
        <h2 className={styles.sectionTitle}>Beneath The Surface</h2>
      </div>

      <div className={styles.cardsGrid}>
        {/* Step 1 */}
        <article className={styles.stepCard}>
          <span className={styles.stepNumber}>01</span>
          <h3 className={styles.stepHeading}>Satellites see the surface</h3>
          <p className={styles.stepBody}>
            Constellations of Earth observation satellites measure surface dynamics daily:
            sea surface temperature, salinity, height anomalies, and surface currents.
          </p>
        </article>

        {/* Step 2 */}
        <article className={styles.stepCard}>
          <span className={styles.stepNumber}>02</span>
          <h3 className={styles.stepHeading}>An AI model learns the ocean below</h3>
          <p className={styles.stepBody}>
            Deep learning models reconstruct 3D subsurface temperature
            structures from satellite surface observations and multi-decadal reanalysis.
          </p>
        </article>

        {/* Step 3 */}
        <article className={styles.stepCard}>
          <span className={styles.stepNumber}>03</span>
          <h3 className={styles.stepHeading}>You explore 15 depths, every day</h3>
          <p className={styles.stepBody}>
            Interactive 2D maps and 3D cutaway blocks reveal internal ocean heat and thermocline
            depths across the entire North Indian Ocean basin.
          </p>
        </article>
      </div>

      <div className={styles.teaserRow}>
        <Link href="/validation" className={styles.teaserLink}>
          <span>Explore model validation and accuracy metrics against independent ARGO floats</span>
          <span aria-hidden="true">→</span>
        </Link>
      </div>
    </section>
  );
}
