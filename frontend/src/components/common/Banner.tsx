"use client";

import React from "react";
import { AlertTriangle, Info } from "lucide-react";
import { useMeta } from "../../lib/api/hooks";

interface BannerProps {
  type?: "synthetic" | "caveat" | "live";
  message?: string;
  className?: string;
}

export function Banner({ type = "synthetic", message, className = "" }: BannerProps) {
  const { data: meta } = useMeta();

  // If type is synthetic, auto-check meta.synthetic if message not provided
  if (type === "synthetic") {
    const isSynthetic = meta?.synthetic ?? true;
    if (!isSynthetic) return null;

    const displayMessage =
      message ??
      meta?.caveats?.synthetic ??
      "This deployment is serving synthetic demo data, not model results.";

    return (
      <div
        role="alert"
        className={`flex w-full items-center justify-center gap-2 bg-amber-400 px-4 py-1 text-xs font-semibold text-neutral-950 ${className}`}
      >
        <AlertTriangle aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
        <span>{displayMessage}</span>
      </div>
    );
  }

  if (type === "live") {
    const displayMessage =
      message ??
      meta?.caveats?.live ??
      "Live maps are made about 3 days behind real time from near-real-time inputs. There is no GLORYS reference for these dates yet, so they have not been checked the way validation dates have.";

    return (
      <div
        className={`flex w-full items-start gap-2.5 rounded-xl bg-emerald-500/10 p-3 text-xs text-emerald-900 ring-1 ring-inset ring-emerald-500/25 dark:text-emerald-100 ${className}`}
      >
        <Info aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
        <p className="leading-relaxed">{displayMessage}</p>
      </div>
    );
  }

  // Generic caveat
  if (!message) return null;

  return (
    <div
      className={`flex w-full items-start gap-2.5 rounded-xl bg-black/[0.03] p-3 text-xs text-neutral-700 ring-1 ring-inset ring-black/[0.06] dark:bg-white/[0.04] dark:text-neutral-300 dark:ring-white/[0.08] ${className}`}
    >
      <Info aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-cyan-600 dark:text-cyan-400" />
      <p className="leading-relaxed">{message}</p>
    </div>
  );
}
