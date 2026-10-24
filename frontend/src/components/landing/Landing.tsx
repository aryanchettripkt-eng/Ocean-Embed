import React from "react";
import Link from "next/link";
import { Banner } from "../common/Banner";
import { HeroText } from "./HeroText";
import { HeroScene } from "./HeroScene";
import { HowItWorks } from "./HowItWorks";
import { LandingFooter } from "./LandingFooter";
import { DiveOverlay } from "./DiveOverlay";
import { pixelifySans, ibmPlexMono } from "./fonts";
import styles from "./landing.module.css";

export function Landing() {
  return (
    <div
      className={`${styles.landing} ${pixelifySans.variable} ${ibmPlexMono.variable}`}
    >
      {/* 1. Paper grain noise overlay (pointer-events: none) */}
      <div className={styles.grainOverlay} aria-hidden="true" />

      {/* 2. Synthetic data warning banner */}
      <div className={styles.bannerWrapper}>
        <Banner type="synthetic" />
      </div>

      <main>
        {/* 3. Hero Section: Centered fixed-aspect 1673:940 stage on desktop, stacked on mobile */}
        <section className={styles.heroSection} aria-label="Hero Introduction">
          <div className={styles.stage}>
            {/* Header: Logo and Nav inside stage */}
            <header className={styles.stageHeader} role="banner">
              <Link href="/" className={styles.brandLogo} aria-label="OceanScope AI Home">
                {/* Circular wave emblem */}
                <svg
                  className={styles.logoSvg}
                  viewBox="0 0 32 32"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  aria-hidden="true"
                >
                  <circle cx="16" cy="16" r="14.5" stroke="#1E2A2B" strokeWidth="2" fill="none" />
                  <path
                    d="M 6 18 C 9 18, 12 12, 16 12 C 20 12, 21 15, 19 17 C 17 19, 15 17, 16 15"
                    stroke="#1E2A2B"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                  />
                  <path
                    d="M 8 22 C 12 22, 16 18, 20 18 C 23 18, 25 20, 24 22"
                    stroke="#1E2A2B"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                  />
                </svg>
                <span className={styles.brandText}>OCEANSCOPE AI</span>
              </Link>

              {/* Desktop Nav: HOME · EXPLORE · DATA · LEARN · ABOUT */}
              <nav aria-label="Landing navigation" className={styles.stageNav}>
                <ul className={styles.navLinks}>
                  <li>
                    <Link href="/" className={`${styles.navLink} ${styles.navLinkActive}`}>
                      Home
                    </Link>
                  </li>
                  <li>
                    <Link href="/explore" className={styles.navLink}>
                      Explore
                    </Link>
                  </li>
                  <li>
                    <Link href="/validation" className={styles.navLink}>
                      Data
                    </Link>
                  </li>
                  <li>
                    <Link href="/about#method" className={styles.navLink}>
                      Learn
                    </Link>
                  </li>
                  <li>
                    <Link href="/about" className={styles.navLink}>
                      About
                    </Link>
                  </li>
                </ul>

                {/* Mobile Collapsible Details Menu (< 768 px) */}
                <details className={styles.mobileMenu}>
                  <summary className={styles.mobileMenuSummary} aria-label="Toggle navigation menu">
                    MENU ▾
                  </summary>
                  <div className={styles.mobileDropdown}>
                    <Link href="/">Home</Link>
                    <Link href="/explore">Explore</Link>
                    <Link href="/validation">Data</Link>
                    <Link href="/about#method">Learn</Link>
                    <Link href="/about">About</Link>
                  </div>
                </details>
              </nav>
            </header>

            {/* Hero Text Block */}
            <div className={styles.heroText}>
              <HeroText />
            </div>

            {/* Background scene art, callouts, and depth ruler */}
            <HeroScene />
          </div>
        </section>

        {/* 4. Below the Fold: How it works */}
        <HowItWorks />
      </main>

      {/* 5. Footer */}
      <LandingFooter />

      {/* 6. Dive Transition Overlay */}
      <DiveOverlay />
    </div>
  );
}
