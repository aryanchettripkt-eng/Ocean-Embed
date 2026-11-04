"use client";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./landing.module.css";
export function DiveOverlay() {
  const router = useRouter();
  const [active, setActive] = useState(false);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const dive = (event: Event) => {
      if (timer) return;
      const href =
        (event as CustomEvent<{ href: string }>).detail?.href ??
        "/explore?view=3d";
      setActive(true);
      timer = setTimeout(
        () => {
          router.push(href);
        },
        window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 300,
      );
    };
    window.addEventListener("oceanscope:dive", dive);
    return () => {
      window.removeEventListener("oceanscope:dive", dive);
      clearTimeout(timer);
    };
  }, [router]);
  return active ? (
    <div
      className={styles.diveOverlay}
      role="status"
      aria-label="Entering the ocean explorer"
    />
  ) : null;
}
