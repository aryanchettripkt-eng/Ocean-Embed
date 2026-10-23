"use client";

import React from "react";
import { useMeta } from "../../lib/api/hooks";
import { formatDepth } from "../../lib/format";
import styles from "./landing.module.css";

const TARGET_DEPTHS = [0, 100, 300, 500, 1000];

export function DepthRuler() {
  const { data: meta, isLoading } = useMeta();

  const depths = meta?.depths_m;
  const maxDepth = depths && depths.length > 0 ? Math.max(...depths) : 1000;
  const sqrtMax = Math.sqrt(maxDepth);

  // If depths are available, pick nearest level to each target
  const displayTicks = React.useMemo(() => {
    if (!depths || depths.length === 0) {
      return TARGET_DEPTHS.map((target) => {
        const topPct = (Math.sqrt(target) / sqrtMax) * 100;
        return {
          depth: target,
          label: null,
          topPct: Math.min(Math.max(topPct, 0), 100),
        };
      });
    }

    // Deduplicate nearest depths
    const chosen = new Set<number>();
    const result: Array<{ depth: number; label: string; topPct: number }> = [];

    for (const target of TARGET_DEPTHS) {
      const nearest = depths.reduce((prev, curr) =>
        Math.abs(curr - target) < Math.abs(prev - target) ? curr : prev
      );
      if (!chosen.has(nearest)) {
        chosen.add(nearest);
        const topPct = (Math.sqrt(nearest) / sqrtMax) * 100;
        result.push({
          depth: nearest,
          label: formatDepth(nearest),
          topPct: Math.min(Math.max(topPct, 0), 100),
        });
      }
    }

    return result;
  }, [depths, sqrtMax]);

  return (
    <div
      className={styles.depthRuler}
      aria-label="Depth scale from ocean surface to depth"
      role="region"
    >
      {displayTicks.map((tick, i) => (
        <div
          key={`${tick.depth}-${i}`}
          className={styles.depthTick}
          style={{ top: `${tick.topPct}%` }}
        >
          <div className={styles.tickLine} />
          {!isLoading && tick.label && (
            <span className={styles.tickLabel} data-testid="depth-tick-label">
              {tick.label}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
