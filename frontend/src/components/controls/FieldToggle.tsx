"use client";

import React from "react";
import { Layers } from "lucide-react";

export type FieldType = "pred" | "glorys" | "error";

interface FieldToggleProps {
  value: FieldType;
  onChange: (field: FieldType) => void;
  isLive?: boolean;
  className?: string;
}

export function FieldToggle({
  value,
  onChange,
  isLive = false,
  className = "",
}: FieldToggleProps) {
  const options: { id: FieldType; label: string; shortLabel: string; disabled?: boolean }[] = [
    { id: "pred", label: "Model prediction", shortLabel: "Prediction" },
    {
      id: "glorys",
      label: "GLORYS reanalysis",
      shortLabel: "GLORYS",
      disabled: isLive,
    },
    {
      id: "error",
      label: "Difference (model − GLORYS)",
      shortLabel: "Difference",
      disabled: isLive,
    },
  ];

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <label className="hud-label flex items-center gap-1.5">
        <Layers aria-hidden="true" className="h-3.5 w-3.5" />
        <span>Field</span>
      </label>
      <div className="grid grid-cols-3 gap-0.5 rounded-full bg-black/[0.04] p-0.5 dark:bg-white/[0.05]">
        {options.map((opt) => {
          const isSelected = value === opt.id;
          const isDisabled = opt.disabled;

          return (
            <button
              key={opt.id}
              type="button"
              disabled={isDisabled}
              aria-pressed={isSelected}
              aria-label={opt.label}
              onClick={() => onChange(opt.id)}
              title={isDisabled ? "No GLORYS reference yet for live dates" : opt.label}
              className={`h-7 truncate rounded-full px-2 text-xs font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400 ${
                isSelected
                  ? "bg-neutral-900 text-white shadow-sm dark:bg-white dark:text-neutral-950"
                  : isDisabled
                  ? "cursor-not-allowed text-neutral-400 opacity-50 dark:text-neutral-600"
                  : "text-neutral-600 hover:text-neutral-950 dark:text-neutral-400 dark:hover:text-white"
              }`}
            >
              {opt.shortLabel}
            </button>
          );
        })}
      </div>
      {isLive && (
        <p className="text-[11px] text-amber-700 dark:text-amber-400/90">
          * GLORYS &amp; Difference unavailable on Live dates
        </p>
      )}
    </div>
  );
}
