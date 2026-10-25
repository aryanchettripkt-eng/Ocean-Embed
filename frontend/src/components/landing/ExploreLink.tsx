"use client";

import React, { useEffect, useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../../lib/api/client";
import styles from "./landing.module.css";

interface ExploreLinkProps {
  href?: string;
  className?: string;
  children?: React.ReactNode;
}

export function ExploreLink({
  href,
  className = "",
  children,
}: ExploreLinkProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [targetHref, setTargetHref] = useState(href ?? "/explore?view=3d");

  // Determine target href based on desktop vs mobile (defaults to /explore?view=3d on desktop, /explore on mobile)
  useEffect(() => {
    if (href) {
      setTargetHref(href);
    } else if (typeof window !== "undefined") {
      setTargetHref(window.innerWidth < 768 ? "/explore" : "/explore?view=3d");
    }
  }, [href]);

  // 1. Warm backend on load: requestIdleCallback (fallback 1500ms)
  useEffect(() => {
    let cancelTimer: NodeJS.Timeout | null = null;
    let idleHandle: number | null = null;

    const warm = () => {
      queryClient.prefetchQuery({
        queryKey: ["meta"],
        queryFn: () => apiClient.getMeta(),
        staleTime: Infinity,
      });
    };

    if (typeof window !== "undefined") {
      if ("requestIdleCallback" in window) {
        idleHandle = window.requestIdleCallback(() => warm(), { timeout: 2000 });
      } else {
        cancelTimer = setTimeout(warm, 1500);
      }
    }

    return () => {
      if (cancelTimer) clearTimeout(cancelTimer);
      if (idleHandle !== null && "cancelIdleCallback" in window) {
        window.cancelIdleCallback(idleHandle);
      }
    };
  }, [queryClient]);

  // 2. Warm cache on hover/focus/touch: prefetch route + maplibre-gl
  const handleWarm = useCallback(() => {
    router.prefetch(targetHref);
    import("maplibre-gl").catch(() => {});
  }, [router, targetHref]);

  // 3. On click: check reduced-motion / startViewTransition
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    // Only intercept normal left clicks without modifiers
    if (e.button !== 0 || e.metaKey || e.altKey || e.ctrlKey || e.shiftKey) {
      return;
    }

    e.preventDefault();

    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const hasViewTransition =
      typeof document !== "undefined" && "startViewTransition" in document;

    if (prefersReducedMotion || !hasViewTransition) {
      router.push(targetHref);
      return;
    }

    // Trigger dive overlay animation
    window.dispatchEvent(
      new CustomEvent("oceanscope:dive", { detail: { href: targetHref } })
    );
  };

  return (
    <a
      href={targetHref}
      className={`${styles.ctaButton} ${className}`}
      onClick={handleClick}
      onMouseEnter={handleWarm}
      onFocus={handleWarm}
      onTouchStart={handleWarm}
    >
      {children ?? (
        <>
          <span>EXPLORE THE OCEAN</span>
          <span aria-hidden="true">→</span>
        </>
      )}
    </a>
  );
}
