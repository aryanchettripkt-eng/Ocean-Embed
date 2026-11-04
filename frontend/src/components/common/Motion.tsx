"use client";

/**
 * React wrappers around the shared GSAP motion (src/lib/motion.ts): panel enter/exit,
 * collapsing sections and the small looping indicators.
 */

import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { DURATION, EASE_IN, EASE_OUT, gsap, loopPing, loopPulse, loopSpin, prefersReducedMotion } from "../../lib/motion";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

type From = "top" | "bottom" | "left" | "right" | "fade";

const OFFSETS: Record<From, { x?: number; y?: number }> = {
  top: { y: -10 },
  bottom: { y: 12 },
  left: { x: -16 },
  right: { x: 16 },
  fade: {},
};

interface PresenceProps {
  show: boolean;
  /** Direction the panel slides in from (and back out to). */
  from?: From;
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

/**
 * Mounts `children` while `show` is true, animating in on open and out on close. While closing it
 * keeps rendering the last children it was shown with, so a panel doesn't re-render with cleared
 * state (e.g. the details panel after the selection is cleared) during its exit.
 */
export function Presence({ show, from = "bottom", className, style, children }: PresenceProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(show);
  const lastChildren = useRef(children);
  if (show) lastChildren.current = children;
  if (show && !mounted) setMounted(true);

  // When the wrapper's position classes change while it stays open (e.g. the details panel moving
  // aside for the guide), read where it was before this commit and glide from there (FLIP).
  const prevClassName = useRef(className);
  const fromRect = useRef<DOMRect | null>(null);
  if (className !== prevClassName.current) {
    fromRect.current = show && ref.current ? ref.current.getBoundingClientRect() : null;
    prevClassName.current = className;
  }

  useIsoLayoutEffect(() => {
    const el = ref.current;
    const before = fromRect.current;
    fromRect.current = null;
    if (!el || !before || !show || prefersReducedMotion()) return;
    const after = el.getBoundingClientRect();
    const dx = before.left - after.left;
    const dy = before.top - after.top;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
    gsap.fromTo(el, { x: dx, y: dy }, { x: 0, y: 0, duration: DURATION.enter, ease: EASE_OUT, clearProps: "transform" });
  }, [className]);

  useIsoLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    gsap.killTweensOf(el);
    const offset = OFFSETS[from];
    const reduce = prefersReducedMotion();

    if (show) {
      if (reduce) {
        gsap.set(el, { clearProps: "opacity,visibility,transform,pointerEvents" });
        return;
      }
      gsap.fromTo(
        el,
        { autoAlpha: 0, scale: 0.985, ...offset },
        {
          autoAlpha: 1,
          scale: 1,
          x: 0,
          y: 0,
          duration: DURATION.enter,
          ease: EASE_OUT,
          clearProps: "opacity,visibility,transform,pointerEvents",
        }
      );
    } else {
      if (reduce) {
        setMounted(false);
        return;
      }
      gsap.to(el, {
        autoAlpha: 0,
        scale: 0.985,
        ...offset,
        pointerEvents: "none",
        duration: DURATION.exit,
        ease: EASE_IN,
        onComplete: () => setMounted(false),
      });
    }
  }, [show]);

  if (!mounted && !show) return null;
  return (
    <div ref={ref} className={className} style={style}>
      {show ? children : lastChildren.current}
    </div>
  );
}

interface CollapseProps {
  open: boolean;
  id?: string;
  className?: string;
  children: React.ReactNode;
}

/** Animates its height between 0 and its content; hidden (out of the tab order) when closed. */
export function Collapse({ open, id, className, children }: CollapseProps) {
  const ref = useRef<HTMLDivElement>(null);
  const first = useRef(true);

  useIsoLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (first.current) {
      first.current = false;
      el.hidden = !open;
      return;
    }
    gsap.killTweensOf(el);
    const reduce = prefersReducedMotion();

    if (open) {
      el.hidden = false;
      if (reduce) return;
      gsap.fromTo(
        el,
        { height: 0, opacity: 0, overflow: "hidden" },
        { height: "auto", opacity: 1, duration: DURATION.collapse, ease: EASE_OUT, clearProps: "height,opacity,overflow" }
      );
    } else {
      if (reduce) {
        el.hidden = true;
        return;
      }
      gsap.fromTo(
        el,
        { height: el.offsetHeight, opacity: 1, overflow: "hidden" },
        {
          height: 0,
          opacity: 0,
          duration: DURATION.collapse * 0.8,
          ease: EASE_IN,
          onComplete: () => {
            el.hidden = true;
            gsap.set(el, { clearProps: "height,opacity,overflow" });
          },
        }
      );
    }
  }, [open]);

  return (
    <div ref={ref} id={id} className={className}>
      {children}
    </div>
  );
}

type LoopKind = "spin" | "pulse" | "ping";
const LOOPS: Record<LoopKind, (el: Element) => gsap.core.Tween | null> = {
  spin: (el) => loopSpin(el),
  pulse: (el) => loopPulse(el),
  ping: (el) => loopPing(el),
};

/** Runs a looping GSAP animation on `ref.current` while `active` is true. */
export function useLoop(ref: React.RefObject<Element | null>, kind: LoopKind, active = true) {
  useEffect(() => {
    const el = ref.current;
    if (!el || !active) return;
    const tween = LOOPS[kind](el);
    return () => {
      tween?.kill();
      gsap.set(el, { clearProps: "opacity,transform" });
    };
  }, [ref, kind, active]);
}

function LoopSpan({ kind, className, children }: { kind: LoopKind; className?: string; children?: React.ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  useLoop(ref, kind);
  return (
    <span ref={ref} className={className}>
      {children}
    </span>
  );
}

/** Rotating wrapper for a spinner icon. */
export function Spin({ className = "inline-flex", children }: { className?: string; children: React.ReactNode }) {
  return <LoopSpan kind="spin" className={className}>{children}</LoopSpan>;
}

/** Breathing opacity (e.g. a "loading" status dot). */
export function Pulse({ className, children }: { className?: string; children?: React.ReactNode }) {
  return <LoopSpan kind="pulse" className={className}>{children}</LoopSpan>;
}

/** Expanding ring; position it over the element it highlights. */
export function Ping({ className }: { className?: string }) {
  return <LoopSpan kind="ping" className={className} />;
}

/** Rotates a chevron between closed (0°) and open (180°). */
export function useFlip(ref: React.RefObject<Element | null>, open: boolean) {
  const first = useRef(true);
  useIsoLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const rotation = open ? 180 : 0;
    if (first.current || prefersReducedMotion()) {
      first.current = false;
      gsap.set(el, { rotation });
      return;
    }
    gsap.to(el, { rotation, duration: DURATION.collapse, ease: EASE_OUT, overwrite: "auto" });
  }, [open]);
}

/** Wraps an icon (e.g. a chevron) and rotates it 180° when `open`. */
export function FlipIcon({ open, className = "inline-flex", children }: { open: boolean; className?: string; children: React.ReactNode }) {
  const ref = useRef<HTMLSpanElement>(null);
  useFlip(ref, open);
  return (
    <span ref={ref} className={className}>
      {children}
    </span>
  );
}
