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

      {/* 3. Header Row */}
      <header className={styles.headerRow} role="banner">
        <Link href="/" className={styles.brandLogo} aria-label="OceanScope AI Home">
          {/* Wave Logo Mark */}
          <svg
            width="28"
            height="20"
            viewBox="0 0 28 20"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <path
              d="M 2 12 Q 8 4 14 12 T 26 12"
              stroke="#1E2A2B"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <path
              d="M 2 16 Q 8 8 14 16 T 26 16"
              stroke="#2F6B73"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
          <span className={styles.brandText}>OCEANSCOPE AI</span>
        </Link>

        {/* Desktop Nav: HOME · EXPLORE · DATA · LEARN · ABOUT */}
        <nav aria-label="Landing navigation">
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

          {/* Mobile Collapsible Details Menu (No JS needed) */}
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

      {/* 4. Full-viewport Hero Section */}
      <main>
        <section className={styles.heroSection} aria-label="Hero Introduction">
          <div className={styles.heroContent}>
            <HeroText />
          </div>
          <HeroScene />
        </section>

        {/* 5. Below the Fold: How it works */}
        <HowItWorks />
      </main>

      {/* 6. Footer */}
      <LandingFooter />

      {/* 7. Dive Transition Overlay */}
      <DiveOverlay />
    </div>
  );
}
