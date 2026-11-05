import React from "react";
import { formatSplit, type SplitVariant } from "../../lib/format";
import { Ping } from "./Motion";

interface SplitBadgeProps {
  split: string | null | undefined;
  className?: string;
}

const variantStyles: Record<SplitVariant, string> = {
  val: "bg-blue-500/10 text-blue-700 border-blue-500/30 dark:bg-blue-400/10 dark:text-blue-300 dark:border-blue-400/30",
  test: "bg-purple-500/10 text-purple-700 border-purple-500/30 dark:bg-purple-400/10 dark:text-purple-300 dark:border-purple-400/30",
  train: "bg-neutral-500/10 text-neutral-700 border-neutral-500/30 dark:bg-white/[0.06] dark:text-neutral-300 dark:border-white/15",
  live: "bg-emerald-500/10 text-emerald-700 border-emerald-500/30 dark:bg-emerald-400/10 dark:text-emerald-300 dark:border-emerald-400/30",
};

export function SplitBadge({ split, className = "" }: SplitBadgeProps) {
  const { label, variant } = formatSplit(split);
  const isLive = variant === "live";

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border ${variantStyles[variant]} ${className}`}
    >
      {isLive && (
        <span className="relative flex h-2 w-2">
          <Ping className="absolute inline-flex h-full w-full rounded-full bg-emerald-400" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
        </span>
      )}
      {label}
    </span>
  );
}
