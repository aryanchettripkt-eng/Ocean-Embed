"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import styles from "./landing.module.css";

export function DiveOverlay() {
  const router = useRouter();
  const [isActive, setIsActive] = useState(false);
  const [showSlowWarning, setShowSlowWarning] = useState(false);

  const warningTimerRef = useRef<NodeJS.Timeout | null>(null);
  const navTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    function handleDive(e: Event) {
      const customEvent = e as CustomEvent<{ href: string }>;
      const targetHref = customEvent.detail?.href ?? "/?view=3d";

      setIsActive(true);
      setShowSlowWarning(false);

      if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
      if (navTimerRef.current) clearTimeout(navTimerRef.current);

      // Timeout for waking up backend message if loading takes > 1.5s
      warningTimerRef.current = setTimeout(() => {
        setShowSlowWarning(true);
      }, 1500);

      // Total transition: 450ms water rise + 350ms dither = 800ms
      navTimerRef.current = setTimeout(() => {
        router.push(targetHref);
      }, 800);
    }

    window.addEventListener("oceanscope:dive", handleDive);
    return () => {
      window.removeEventListener("oceanscope:dive", handleDive);
      if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
      if (navTimerRef.current) clearTimeout(navTimerRef.current);
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
