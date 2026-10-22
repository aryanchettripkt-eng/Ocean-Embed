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
        className={`w-full bg-amber-500/15 border-b border-amber-500/30 px-4 py-2 text-xs md:text-sm font-medium text-amber-300 flex items-center justify-center gap-2 backdrop-blur-sm ${className}`}
      >
        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
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
        className={`w-full bg-emerald-950/40 border border-emerald-800/40 rounded-lg p-3 text-xs text-emerald-200/90 flex items-start gap-2.5 backdrop-blur-sm ${className}`}
      >
        <Info className="h-4 w-4 shrink-0 text-emerald-400 mt-0.5" />
        <p className="leading-relaxed">{displayMessage}</p>
      </div>
    );
  }

  // Generic caveat
  if (!message) return null;

  return (
    <div
      className={`w-full bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-xs text-slate-300 flex items-start gap-2.5 backdrop-blur-sm ${className}`}
    >
      <Info className="h-4 w-4 shrink-0 text-cyan-400 mt-0.5" />
      <p className="leading-relaxed">{message}</p>
    </div>
  );
}
