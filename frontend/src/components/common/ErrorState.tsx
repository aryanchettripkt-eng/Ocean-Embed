"use client";

import React from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  title = "Failed to load data",
  message = "An error occurred while fetching information from the ocean server.",
  onRetry,
  className = "",
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={`flex flex-col items-center justify-center p-4 text-center ${className}`}
    >
      <div className="mb-2.5 flex h-9 w-9 items-center justify-center rounded-full bg-red-500/10 text-red-600 ring-1 ring-inset ring-red-500/30 dark:text-red-400">
        <AlertCircle aria-hidden="true" className="h-[18px] w-[18px]" />
      </div>
      <h3 className="mb-1 text-sm font-semibold text-neutral-900 dark:text-neutral-100">{title}</h3>
      <p className="mb-3 max-w-sm text-xs leading-relaxed text-neutral-500 dark:text-neutral-400">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          type="button"
          className="hud-btn border border-black/10 dark:border-white/10"
        >
          <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
}
