import React from "react";
import styles from "./landing.module.css";

interface CalloutProps {
  text: string;
  className?: string;
  style?: React.CSSProperties;
}

export function Callout({ text, className = "", style }: CalloutProps) {
  return (
    <figure className={`${styles.calloutWrapper} ${className}`} style={style}>
      <figcaption className={styles.bracketCallout}>{text}</figcaption>
    </figure>
  );
}
