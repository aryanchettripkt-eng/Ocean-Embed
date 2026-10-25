import React from "react";
import styles from "./landing.module.css";

export interface CalloutProps {
  text: string;
  lines?: string[];
  variant?: "ink" | "foam" | "bare";
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Authentic woodblock bracket callout matching reference mockup.
 * Draws an open bracket on the left with the typography neatly indented to the right.
 * When variant is "bare", renders plain text with no bracket.
 */
export function Callout({
  text,
  lines,
  variant = "ink",
  className = "",
  style,
}: CalloutProps) {
  const textLines = lines ?? text.split("\n");
  const isFoam = variant === "foam" || variant === "bare";
  const strokeColor = isFoam ? "#F6F1E2" : "#1E2A2B";
  const textColor = isFoam ? "#F6F1E2" : "#1E2A2B";

  return (
    <figure
      className={`${styles.calloutWrapper} ${className}`}
      style={style}
    >
      <div className={styles.bracketContainer} style={{ color: textColor }}>
        {variant !== "bare" && (
          <svg
            className={styles.bracketSvg}
            width="8"
            height="100%"
            viewBox="0 0 8 36"
            preserveAspectRatio="none"
            aria-hidden="true"
            data-testid="callout-bracket-svg"
          >
            {/* Top horizontal tick */}
            <line x1="2" y1="2" x2="8" y2="2" stroke={strokeColor} strokeWidth="1.5" />
            {/* Vertical stem */}
            <line x1="2" y1="2" x2="2" y2="34" stroke={strokeColor} strokeWidth="1.5" />
            {/* Middle cross tick */}
            <line x1="0" y1="18" x2="4" y2="18" stroke={strokeColor} strokeWidth="1.5" />
            {/* Bottom horizontal tick */}
            <line x1="2" y1="34" x2="8" y2="34" stroke={strokeColor} strokeWidth="1.5" />
          </svg>
        )}

        <figcaption className={styles.bracketText}>
          {textLines.map((line, i) => (
            <span key={i} className={styles.bracketLine}>
              {line}
            </span>
          ))}
        </figcaption>
      </div>
    </figure>
  );
}
