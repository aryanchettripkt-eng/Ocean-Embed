import React from "react";
import styles from "./landing.module.css";

interface CalloutProps {
  text: string;
  lines?: string[];
  variant?: "ink" | "foam";
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Authentic woodblock bracket callout matching reference mockup.
 * Draws an open bracket on the left (top arm ┌, middle cross +, bottom arm └)
 * with the typography neatly indented to the right.
 */
export function Callout({
  text,
  lines,
  variant = "ink",
  className = "",
  style,
}: CalloutProps) {
  const textLines = lines ?? text.split("\n");
  const strokeColor = variant === "foam" ? "#F6F1E2" : "#1E2A2B";
  const textColor = variant === "foam" ? "#F6F1E2" : "#1E2A2B";

  return (
    <figure
      className={`${styles.calloutWrapper} ${className}`}
      style={style}
    >
      <div className={styles.bracketContainer} style={{ color: textColor }}>
        {/* Left-side bracket: top arm ┌, middle cross +, bottom arm └ */}
        <svg
          className={styles.bracketSvg}
          width="10"
          height="100%"
          viewBox="0 0 10 40"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {/* Top horizontal tick */}
          <line x1="2" y1="2" x2="9" y2="2" stroke={strokeColor} strokeWidth="1.5" />
          {/* Vertical stem */}
          <line x1="2" y1="2" x2="2" y2="38" stroke={strokeColor} strokeWidth="1.5" />
          {/* Middle cross tick */}
          <line x1="0" y1="20" x2="5" y2="20" stroke={strokeColor} strokeWidth="1.5" />
          {/* Bottom horizontal tick */}
          <line x1="2" y1="38" x2="9" y2="38" stroke={strokeColor} strokeWidth="1.5" />
        </svg>

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
