import React from "react";
import Link from "next/link";
import styles from "./landing.module.css";

export function LandingFooter() {
  return (
    <footer className={styles.landingFooter} role="contentinfo">
      <div className={styles.footerContent}>
        <div className={styles.footerCredits}>
          <p>
            <strong>OceanScope AI</strong> is trained on GLORYS12 reanalysis and evaluated against
            independent in-situ ARGO float profiles. Satellite and atmospheric observations provided by
            Copernicus Marine Service (DUACS SLA, OSTIA SST, SSS), ECMWF (ERA5 surface heat fluxes & precipitation),
            PO.DAAC (OSCAR currents, CCMP winds), NOAA (WOA23 climatology, climate indices), and GEBCO bathymetry.
          </p>
          <p style={{ marginTop: "0.5rem", opacity: 0.8 }}>
            Built for oceanographic machine learning research and interactive subsurface exploration.
          </p>
        </div>

        <nav aria-label="Landing footer navigation">
          <ul className={styles.footerLinks}>
            <li>
              <Link href="/explore">Explorer</Link>
            </li>
            <li>
              <Link href="/validation">Validation</Link>
            </li>
            <li>
              <Link href="/about#method">Methodology</Link>
            </li>
            <li>
              <Link href="/about">About & Licenses</Link>
            </li>
            <li>
              <a
                href="https://github.com/aryanchettripkt-eng/Ocean-Embed"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </li>
          </ul>
        </nav>
      </div>
    </footer>
  );
}
