import React from "react";
import { ExploreLink } from "./ExploreLink";
import styles from "./landing.module.css";

export function HeroText() {
  return (
    <div className={styles.heroTextCol}>
      <h1 className={styles.h1Title}>OCEANSCOPE AI</h1>
      <p className={styles.tagline}>BENEATH THE SURFACE</p>
      <p className={styles.lede}>
        A quieter way to see the hidden temperature of the ocean.
      </p>
      <div className={styles.ctaContainer}>
        <ExploreLink href="/" />
      </div>
    </div>
  );
}
