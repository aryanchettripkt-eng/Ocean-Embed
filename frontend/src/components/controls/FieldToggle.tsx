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
    <div className={`flex flex-col gap-1.5 ${className}`}>
      <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
        <Layers aria-hidden="true" className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
        <span>Field</span>
      </label>
      <div className="grid grid-cols-3 gap-1 bg-slate-100/90 dark:bg-white/[0.04] p-1 rounded-xl border border-slate-200/90 dark:border-white/10 shadow-inner">
        {options.map((opt) => {
          const isSelected = value === opt.id;
          const isDisabled = opt.disabled;

          return (
            <button
              key={opt.id}
              type="button"
              disabled={isDisabled}
              aria-pressed={isSelected}
              onClick={() => onChange(opt.id)}
              title={isDisabled ? "No GLORYS reference yet for live dates" : opt.label}
              className={`px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center truncate ${
                isSelected
                  ? "bg-cyan-500 text-slate-950 font-semibold shadow-sm shadow-cyan-500/20"
                  : isDisabled
                  ? "opacity-35 cursor-not-allowed text-slate-400 dark:text-slate-500"
                  : "text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:bg-slate-200/70 dark:hover:bg-white/10"
              }`}
            >
              <span className="hidden sm:inline">{opt.label}</span>
              <span className="sm:hidden">{opt.shortLabel}</span>
            </button>
          );
        })}
      </div>
      {isLive && (
        <p className="text-[11px] text-amber-600 dark:text-amber-400/90 italic flex items-center gap-1">
          * GLORYS & Difference unavailable on Live dates
        </p>
      )}
    </div>
  );
}
