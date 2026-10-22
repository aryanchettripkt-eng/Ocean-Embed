import React from "react";
import { formatSplit, type SplitVariant } from "../../lib/format";

interface SplitBadgeProps {
  split: string | null | undefined;
  className?: string;
}

const variantStyles: Record<SplitVariant, string> = {
  val: "bg-blue-950/80 text-blue-300 border-blue-700/60",
  test: "bg-purple-950/80 text-purple-300 border-purple-700/60",
  train: "bg-slate-900/80 text-slate-300 border-slate-700/60",
  live: "bg-emerald-950/80 text-emerald-300 border-emerald-600/60",
};

export function SplitBadge({ split, className = "" }: SplitBadgeProps) {
  const { label, variant } = formatSplit(split);
  const isLive = variant === "live";

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border shadow-sm backdrop-blur-sm ${variantStyles[variant]} ${className}`}
    >
      {isLive && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
        </span>
      )}
      {label}
    </span>
  );
}
