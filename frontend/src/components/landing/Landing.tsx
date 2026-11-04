"use client";
import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Moon, Sun, Pause, Play } from "lucide-react";
import { Banner } from "../common/Banner";
import { useTheme } from "../../lib/use-theme";
import { HeroText } from "./HeroText";
import { HeroScene } from "./HeroScene";
import { HowItWorks } from "./HowItWorks";
import { LandingFooter } from "./LandingFooter";
import { DiveOverlay } from "./DiveOverlay";
import { manrope } from "./fonts";
import styles from "./landing.module.css";

const links = [
  ["Home", "/"],
  ["Explore", "/explore"],
  ["Data", "/validation"],
  ["Learn", "/about#method"],
  ["About", "/about"],
];
export function Landing() {
  const root = useRef<HTMLDivElement>(null);
  const bannerRef = useRef<HTMLDivElement>(null);
  const [bannerH, setBannerH] = useState(32);
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(false);
  const { theme, toggleTheme } = useTheme();
  useEffect(() => {
    if (!bannerRef.current) return;
    const update = () =>
      setBannerH(
        Math.round(bannerRef.current?.getBoundingClientRect().height ?? 0),
      );
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(bannerRef.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    let disposed = false;
    let pending = false;
    let cleanup: (() => void) | undefined;
    const enhance = () => {
      if (query.matches || pending || cleanup) return;
      pending = true;
      import("./landing-motion")
        .then(({ attachLandingMotion }) => {
          if (!disposed && root.current) cleanup = attachLandingMotion(root.current);
        })
        .catch(() => { /* HTML and CSS remain interactive without animation. */ })
        .finally(() => { pending = false; });
    };
    const sync = () => { setReduced(query.matches); enhance(); };
    setReduced(query.matches);
    query.addEventListener("change", sync);
    const timer = setTimeout(enhance, 200);
    return () => {
      disposed = true;
      clearTimeout(timer);
      cleanup?.();
      query.removeEventListener("change", sync);
    };
  }, []);
  return (
    <div
      ref={root}
      data-landing
      className={`${styles.landing} ${manrope.variable}`}
      style={{ "--banner-h": `${bannerH}px` } as React.CSSProperties}
    >
      <div ref={bannerRef} className={styles.bannerWrapper}>
        <Banner type="synthetic" />
      </div>
      <main>
        <section className={styles.heroSection} aria-label="Hero Introduction">
          <HeroScene paused={paused || reduced} theme={theme} />
          <header className={styles.heroHeader} data-landing-exit>
            <Link
              prefetch={false}
              href="/"
              className={styles.brandLogo}
              aria-label="OceanScope AI Home"
            >
              OCEANSCOPE AI
            </Link>
            <nav aria-label="Landing navigation" className={styles.heroNav}>
              <ul className={styles.navLinks}>
                {links.map(([label, href]) => (
                  <li key={label}>
                    <Link
                      prefetch={false}
                      data-micro="nav"
                      href={href}
                      className={styles.navLink}
                      aria-current={href === "/" ? "page" : undefined}
                    >
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
              <details className={styles.mobileMenu}>
                <summary aria-label="Toggle navigation menu">
                  Menu <span aria-hidden="true">＋</span>
                </summary>
                <div className={styles.mobileDropdown}>
                  {links.map(([label, href]) => (
                    <Link prefetch={false} key={label} href={href}>
                      {label}
                    </Link>
                  ))}
                </div>
              </details>
            </nav>
            <div className={styles.sceneControls}>
              <button
                className={styles.iconButton}
                onClick={toggleTheme}
                aria-label="Toggle dark/light theme"
                title={
                  theme === "dark"
                    ? "Switch to light mode"
                    : "Switch to dark mode"
                }
                data-micro="control"
              >
                <Sun className={styles.sunIcon} size={18} aria-hidden="true" />
                <Moon
                  className={styles.moonIcon}
                  size={18}
                  aria-hidden="true"
                />
              </button>
              <button
                className={styles.iconButton}
                onClick={() => setPaused((value) => !value)}
                disabled={reduced}
                aria-pressed={paused || reduced}
                aria-label={
                  paused ? "Resume ambient motion" : "Pause ambient motion"
                }
                title={
                  reduced
                    ? "Motion disabled by your system preference"
                    : paused
                      ? "Resume ambient motion"
                      : "Pause ambient motion"
                }
                data-micro="control"
              >
                {paused || reduced ? (
                  <Play size={16} aria-hidden="true" />
                ) : (
                  <Pause size={16} aria-hidden="true" />
                )}
              </button>
            </div>
          </header>
          <div className={styles.heroText} data-landing-exit>
            <HeroText />
          </div>
          <div className={styles.topicLabels} data-landing-exit>
            <span>Ocean observation</span>
            <i aria-hidden="true">•</i>
            <span>Temperature insights</span>
            <i aria-hidden="true">•</i>
            <span>A healthier ocean</span>
          </div>
        </section>
        <HowItWorks />
      </main>
      <LandingFooter />
      <DiveOverlay />
    </div>
  );
}
