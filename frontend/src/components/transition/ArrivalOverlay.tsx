"use client";

/**
 * Full-screen layer for the landing → Explorer arrival (src/lib/arrival.ts). Mounted in Providers
 * so it survives the route change. It sweeps the water surface down over the landing page, holds
 * the colour of the map's water while the Explorer loads underneath, then fades into the live map.
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  MAP_WATER,
  arrivalCovered,
  arrivalMapReady,
  arrivalRevealed,
  finishArrival,
  useArrivalPhase,
} from "../../lib/arrival";

// GSAP is loaded when a transition starts, so it isn't in the shared bundle every page ships
const loadGsap = () => import("gsap").then((m) => m.gsap);
const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Sun-glitter highlights for the holding surface: fixed pseudo-random positions and sizes
 * (seeded, so server and client agree and every arrival looks the same). Decorative only.
 */
const GLINTS = (() => {
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  return Array.from({ length: 48 }, () => ({
    x: rand() * 100,
    y: rand() * 100,
    w: 16 + rand() * 48,
    h: 2 + rand() * 2.5,
    peak: 0.35 + rand() * 0.65,
  }));
})();

/** Show the cold-backend note after this long under the overlay. */
const WAKING_AFTER_MS = 4000;
/** Never hold the screen longer than this; the Explorer shows its own loading/error states. */
const MAX_COVER_MS = 15000;

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

export function ArrivalOverlay() {
  const phase = useArrivalPhase();
  const surfaceRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [waking, setWaking] = useState(false);
  const [dark, setDark] = useState(true);

  useIsoLayoutEffect(() => {
    if (phase === "sweeping") {
      setDark(document.documentElement.classList.contains("dark"));
      setWaking(false);
    }
  }, [phase]);

  // Sweep: the surface comes down from the top as we rise out of the water
  useIsoLayoutEffect(() => {
    if (phase !== "sweeping") return;
    const surface = surfaceRef.current;
    if (!surface || prefersReducedMotion()) {
      arrivalCovered();
      return;
    }
    let cancelled = false;
    let tween: { kill: () => void } | undefined;
    // Start off-screen until GSAP has loaded (no flash of a covered screen)
    surface.style.transform = "translateY(-100%)";
    void loadGsap()
      .then((gsap) => {
        if (cancelled) return;
        tween = gsap.fromTo(
          surface,
          // y: 0 explicitly, or GSAP keeps the pre-set translateY(-100%) as an extra pixel offset
          { y: 0, yPercent: -100 },
          { yPercent: 0, duration: 0.95, ease: "power2.in", onComplete: arrivalCovered }
        );
      })
      .catch(() => arrivalCovered());
    return () => {
      cancelled = true;
      tween?.kill();
    };
  }, [phase]);

  // Hold: the surface stays alive (sun glints twinkling on the water seen from above, drifting
  // slowly) so the wait reads as the sea, not an empty screen
  useEffect(() => {
    if (phase !== "covering" || prefersReducedMotion()) return;
    const field = rootRef.current?.querySelector<HTMLElement>("[data-glints]");
    if (!field) return;
    const glints = Array.from(field.children) as HTMLElement[];
    let cancelled = false;
    const tweens: { kill: () => void }[] = [];
    void loadGsap().then((gsap) => {
      if (cancelled) return;
      tweens.push(gsap.to(field, { x: 36, y: 12, duration: 6, ease: "sine.inOut", yoyo: true, repeat: -1 }));
      glints.forEach((glint, i) => {
        tweens.push(
          gsap.to(glint, {
            opacity: GLINTS[i]?.peak ?? 0.6,
            duration: gsap.utils.random(0.45, 1.1),
            ease: "sine.inOut",
            yoyo: true,
            repeat: -1,
            repeatDelay: gsap.utils.random(0.2, 1.4),
            delay: gsap.utils.random(0, 1.2),
          })
        );
      });
    });
    return () => {
      cancelled = true;
      tweens.forEach((t) => t.kill());
    };
  }, [phase]);

  // Hold: note a cold backend, and never trap the user behind the overlay
  useEffect(() => {
    if (phase !== "covering") return;
    const waking = setTimeout(() => setWaking(true), WAKING_AFTER_MS);
    const giveUp = setTimeout(() => {
      arrivalMapReady();
      finishArrival();
    }, MAX_COVER_MS);
    return () => {
      clearTimeout(waking);
      clearTimeout(giveUp);
    };
  }, [phase]);

  // Reveal: fade into the live map underneath
  useIsoLayoutEffect(() => {
    if (phase !== "revealing") return;
    const root = rootRef.current;
    if (!root || prefersReducedMotion()) {
      arrivalRevealed();
      return;
    }
    let cancelled = false;
    let tween: { kill: () => void } | undefined;
    void loadGsap()
      .then((gsap) => {
        if (cancelled) return;
        tween = gsap.to(root, { autoAlpha: 0, duration: 0.55, ease: "power1.out", onComplete: arrivalRevealed });
      })
      .catch(() => arrivalRevealed());
    return () => {
      cancelled = true;
      tween?.kill();
    };
  }, [phase]);

  if (phase === "idle") return null;

  const water = dark ? MAP_WATER.dark : MAP_WATER.light;
  const glint = dark ? "rgba(120, 220, 230, 0.35)" : "rgba(255, 255, 255, 0.7)";
  const sparkle = dark ? "rgba(150, 220, 230, 0.55)" : "rgba(255, 255, 255, 0.85)";

  return (
    <div
      ref={rootRef}
      role="status"
      aria-live="polite"
      aria-label="Entering the ocean explorer"
      className={`fixed inset-0 z-[100] overflow-hidden ${phase === "revealing" ? "pointer-events-none" : ""}`}
    >
      <div ref={surfaceRef} className="absolute inset-0" style={{ background: water }}>
        <div aria-hidden="true" data-glints className="absolute -inset-10">
          {GLINTS.map((g, i) => (
            <span
              key={i}
              className="absolute rounded-full opacity-0"
              style={{
                left: `${g.x}%`,
                top: `${g.y}%`,
                width: g.w,
                height: g.h,
                background: `radial-gradient(ellipse at center, ${sparkle}, transparent 70%)`,
              }}
            />
          ))}
        </div>
        {/* The waterline trails the surface: a bright meniscus fading into the water below it */}
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-full h-[30vh]"
          style={{
            background: `linear-gradient(to bottom, ${glint} 0, ${water.replace("rgb", "rgba").replace(")", ", 0.75)")} 3%, transparent 100%)`,
          }}
        />
      </div>
      {waking && phase === "covering" && (
        <div className="absolute inset-x-0 bottom-24 flex justify-center px-4">
          <div className="hud-glass rounded-full px-4 py-2 text-center text-xs">
            Waking up the ocean server… the first load after a quiet spell can take up to a minute.
          </div>
        </div>
      )}
    </div>
  );
}
