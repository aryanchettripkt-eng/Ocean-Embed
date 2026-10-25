"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import styles from "./landing.module.css";

/**
 * Full Dive Transition Overlay (PLAN_LANDING §6, §8, PRD_FRONTEND §7).
 * - Phase 1 (0-400ms): Water rises from bottom to top, submerging the screen in sea water.
 * - Phase 2 (400-750ms): Dispatches document.startViewTransition() and dissolves water
 *   via 16-step Bayer dither mask, revealing the Explorer page beneath.
 * - Cross-route morph: Shared element view-transition-name: ocean-section morphs 2D section to 3D block.
 * - Focus: Automatically moves to Explorer's main heading (#explorer-heading).
 */
export function DiveOverlay() {
  const router = useRouter();
  const [isActive, setIsActive] = useState(false);
  const [isDithering, setIsDithering] = useState(false);
  const [showSlowWarning, setShowSlowWarning] = useState(false);

  const warningTimerRef = useRef<NodeJS.Timeout | null>(null);
  const phaseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cleanupTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    function handleDive(e: Event) {
      const customEvent = e as CustomEvent<{ href: string }>;
      const targetHref = customEvent.detail?.href ?? "/explore?view=3d";

      setIsActive(true);
      setIsDithering(false);
      setShowSlowWarning(false);

      if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
      if (phaseTimerRef.current) clearTimeout(phaseTimerRef.current);
      if (cleanupTimerRef.current) clearTimeout(cleanupTimerRef.current);

      // Warning timer if backend wake-up takes > 1.5s
      warningTimerRef.current = setTimeout(() => {
        setShowSlowWarning(true);
      }, 1500);

      // At 400ms: Water rise is complete. Trigger startViewTransition and begin dither dissolve
      phaseTimerRef.current = setTimeout(() => {
        setIsDithering(true);

        const performNavigation = () => {
          router.push(targetHref);
          // Move focus to Explorer main heading
          setTimeout(() => {
            const heading = document.getElementById("explorer-heading");
            if (heading) {
              heading.focus();
            }
          }, 150);
        };

        if (typeof document !== "undefined" && "startViewTransition" in document) {
          (document as unknown as { startViewTransition: (cb: () => void) => { finished: Promise<void> } })
            .startViewTransition(() => {
              performNavigation();
            });
        } else {
          performNavigation();
        }
      }, 400);

      // Total transition finishes at 800ms
      cleanupTimerRef.current = setTimeout(() => {
        setIsActive(false);
        setIsDithering(false);
      }, 850);
    }

    window.addEventListener("oceanscope:dive", handleDive);
    return () => {
      window.removeEventListener("oceanscope:dive", handleDive);
      if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
      if (phaseTimerRef.current) clearTimeout(phaseTimerRef.current);
      if (cleanupTimerRef.current) clearTimeout(cleanupTimerRef.current);
    };
  }, [router]);

  if (!isActive) return null;

  return (
    <div
      className={`${styles.diveOverlay} ${styles.diveOverlayActive}`}
      role="status"
      aria-live="polite"
    >
      {/* 1. Water layer that rises up, then dissolves with Bayer dither mask */}
      <div
        className={`${styles.diveWaterRise} ${isDithering ? styles.diveDitherDissolve : ""}`}
      />

      {/* 2. Slow loading status notice if Render cold-start takes > 1.5s */}
      {showSlowWarning && (
        <div className={styles.diveStatusText}>Waking up the ocean…</div>
      )}
    </div>
  );
}
