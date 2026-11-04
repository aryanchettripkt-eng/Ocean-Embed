import React from "react";
import { ExploreLink } from "./ExploreLink";
import styles from "./landing.module.css";
export function HeroText() {
  return (
    <div className={styles.heroTextInner}>
      <h1 className={styles.h1Title} data-entrance>
        A New Depth of Intelligence —<br className={styles.desktopBreak} />{" "}
        Beneath the Surface
      </h1>
      <p className={styles.lede} data-entrance>
        A quieter way to understand the hidden temperature of the ocean,
        <br className={styles.desktopBreak} /> revealing how heat moves, shifts,
        and shapes our world.
      </p>
      <div className={styles.ctaContainer} data-entrance>
        <ExploreLink />
      </div>
    </div>
  );
}
