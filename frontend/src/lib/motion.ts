/**
 * Shared GSAP motion for the Explorer: one place for durations, easing, and the looping
 * animations used outside React (MapLibre DOM markers, the WebGL cutaway layer).
 * Every helper honours prefers-reduced-motion: loops don't start and transitions are instant.
 */
import { gsap } from "gsap";

export { gsap };

export const EASE_OUT = "power3.out";
export const EASE_IN = "power2.in";

export const DURATION = {
  enter: 0.34,
  exit: 0.2,
  collapse: 0.36,
  dive: 0.32,
} as const;

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

type Target = gsap.TweenTarget;

/** Continuous rotation (loading spinners). Returns null under reduced motion. */
export function loopSpin(target: Target, seconds = 0.9): gsap.core.Tween | null {
  if (prefersReducedMotion()) return null;
  return gsap.to(target, { rotation: "+=360", duration: seconds, ease: "none", repeat: -1, transformOrigin: "50% 50%" });
}

/** Opacity breathing (live/recording indicators). Returns null under reduced motion. */
export function loopPulse(target: Target, min = 0.35, seconds = 0.9): gsap.core.Tween | null {
  if (prefersReducedMotion()) return null;
  return gsap.fromTo(target, { opacity: 1 }, { opacity: min, duration: seconds, ease: "sine.inOut", yoyo: true, repeat: -1 });
}

/** Expanding, fading ring (map pins, live badge). Hidden under reduced motion. */
export function loopPing(target: Target, seconds = 1.1): gsap.core.Tween | null {
  if (prefersReducedMotion()) {
    gsap.set(target, { opacity: 0 });
    return null;
  }
  return gsap.fromTo(
    target,
    { scale: 1, opacity: 0.7 },
    { scale: 2, opacity: 0, duration: seconds, ease: "power1.out", repeat: -1, transformOrigin: "50% 50%" }
  );
}
