import React from "react";

interface TypingDotsProps {
  /** Screen-reader text; the dots themselves are decorative. */
  label?: string;
  className?: string;
}

/** Three bouncing dots (chat-style "working on it"). Static under prefers-reduced-motion. */
export function TypingDots({ label = "Loading", className = "" }: TypingDotsProps) {
  return (
    <span className={`inline-flex items-center gap-1 ${className}`} data-testid="typing-dots">
      {[0, 160, 320].map((delay) => (
        <span
          key={delay}
          aria-hidden="true"
          className="hud-typing-dot h-1.5 w-1.5 rounded-full bg-current"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
      <span className="sr-only">{label}</span>
    </span>
  );
}
