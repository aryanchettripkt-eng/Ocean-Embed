"use client";

import React, { useEffect, useRef } from "react";
import { gsap, prefersReducedMotion } from "../../lib/motion";

interface TypingDotsProps {
  /** Screen-reader text; the dots themselves are decorative. */
  label?: string;
  className?: string;
}

/** Three bouncing dots (chat-style "working on it"), driven by GSAP. Static under reduced motion. */
export function TypingDots({ label = "Loading", className = "" }: TypingDotsProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const dots = ref.current?.querySelectorAll<HTMLElement>(".hud-typing-dot");
    if (!dots || dots.length === 0) return;
    if (prefersReducedMotion()) {
      gsap.set(dots, { opacity: 0.6 });
      return;
    }
    const tl = gsap.timeline({ repeat: -1 });
    tl.fromTo(
      dots,
      { y: 0, opacity: 0.3 },
      { y: -3, opacity: 1, duration: 0.24, ease: "sine.out", stagger: 0.16, yoyo: true, repeat: 1 }
    ).to({}, { duration: 0.2 });
    return () => {
      tl.kill();
    };
  }, []);

  return (
    <span ref={ref} className={`inline-flex items-center gap-1 ${className}`} data-testid="typing-dots">
      {[0, 1, 2].map((i) => (
        <span key={i} aria-hidden="true" className="hud-typing-dot h-1.5 w-1.5 rounded-full bg-current opacity-30" />
      ))}
      <span className="sr-only">{label}</span>
    </span>
  );
}
