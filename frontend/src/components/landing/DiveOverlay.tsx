"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./landing.module.css";

export function DiveOverlay() {
  const router = useRouter();
  const [isActive, setIsActive] = useState(false);
  const [showSlowWarning, setShowSlowWarning] = useState(false);

  useEffect(() => {
    function handleDive(e: Event) {
      const customEvent = e as CustomEvent<{ href: string }>;
      const targetHref = customEvent.detail?.href ?? "/";

      setIsActive(true);
      setShowSlowWarning(false);

      // Timeout for waking up backend message if loading takes > 1.5s
      const warningTimer = setTimeout(() => {
        setShowSlowWarning(true);
      }, 1500);

      // Total transition: 450ms water rise + 350ms dither = 800ms
      const navTimer = setTimeout(() => {
        router.push(targetHref);
      }, 800);

      return () => {
        clearTimeout(warningTimer);
        clearTimeout(navTimer);
      };
    }

    window.addEventListener("oceanscope:dive", handleDive);
    return () => {
      window.removeEventListener("oceanscope:dive", handleDive);
    };
  }, [router]);

  if (!isActive) return null;

  return (
    <div
      className={`${styles.diveOverlay} ${styles.diveOverlayActive}`}
      role="status"
      aria-live="polite"
    >
      <div className={styles.diveWaterRise} />
      <div className={styles.diveDitherDissolve} />
      {showSlowWarning && (
        <div className={styles.diveStatusText}>Waking up the ocean…</div>
      )}
    </div>
  );
}
