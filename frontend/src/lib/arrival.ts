"use client";

/**
 * Landing → Explorer arrival (PRD_FRONTEND §3.1, §4). A tiny shared state machine so the landing
 * page, the overlay that survives the route change (ArrivalOverlay, mounted in Providers) and the
 * Explorer can run one continuous transition:
 *
 *   idle → sweeping   landing: the water surface sweeps down the screen (we "rise out" of the sea)
 *        → covering   the screen is the colour of the map's water; route changes underneath
 *        → revealing  the Explorer's map has drawn: the overlay fades into the live map
 *        → idle       the Explorer zooms out/in and slides its panels in, then calls finishArrival()
 *
 * While an arrival is in progress <html data-arrival="pending"> hides the Explorer HUD
 * ([data-hud] elements) so nothing flashes before its entrance animation.
 */
import { useSyncExternalStore } from "react";

export type ArrivalPhase = "idle" | "sweeping" | "covering" | "revealing";

/** Basemap water colours (OpenFreeMap dark / positron `water` fill), so the hand-off is seamless. */
export const MAP_WATER = { dark: "rgb(27, 27, 29)", light: "rgb(194, 200, 202)" } as const;

let phase: ArrivalPhase = "idle";
let onCovered: (() => void) | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((fn) => fn());
}

function setPhase(next: ArrivalPhase) {
  if (phase === next) return;
  phase = next;
  emit();
}

export function getArrivalPhase(): ArrivalPhase {
  return phase;
}

export function subscribeArrival(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useArrivalPhase(): ArrivalPhase {
  return useSyncExternalStore(subscribeArrival, getArrivalPhase, () => "idle");
}

/** True from the landing click until the Explorer's entrance has finished. */
export function isArrivalPending(): boolean {
  return typeof document !== "undefined" && document.documentElement.dataset.arrival === "pending";
}

/**
 * Landing CTA: start the transition. `covered` runs once the screen is fully covered
 * (navigate there). Returns false if one is already running.
 */
export function beginArrival(covered: () => void): boolean {
  if (phase !== "idle") return false;
  onCovered = covered;
  document.documentElement.dataset.arrival = "pending";
  setPhase("sweeping");
  return true;
}

/** Overlay: the sweep finished and the screen is covered. */
export function arrivalCovered() {
  if (phase !== "sweeping") return;
  setPhase("covering");
  const cb = onCovered;
  onCovered = null;
  cb?.();
}

/** Explorer: the map has drawn under the overlay; fade it out. */
export function arrivalMapReady() {
  if (phase === "covering" || phase === "sweeping") setPhase("revealing");
}

/** Overlay: finished fading out. */
export function arrivalRevealed() {
  setPhase("idle");
}

/** Explorer: entrance finished (or skipped); show the HUD normally again. */
export function finishArrival() {
  if (typeof document !== "undefined") delete document.documentElement.dataset.arrival;
  if (phase !== "idle") setPhase("idle");
}
