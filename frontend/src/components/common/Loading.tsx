"use client";

import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Spin } from "./Motion";

interface LoadingProps {
  label?: string;
  className?: string;
  fullScreen?: boolean;
}

export function Loading({ label = "Loading oceanographic data...", className = "", fullScreen = false }: LoadingProps) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isColdBackend = seconds >= 3;

  const content = (
    <div className={`flex flex-col items-center justify-center p-6 text-center ${className}`}>
      <Spin className="mb-3 inline-flex">
        <Loader2 aria-hidden="true" className="h-6 w-6 text-cyan-600 dark:text-cyan-300" />
      </Spin>
      <p className="text-sm font-medium text-neutral-700 dark:text-neutral-200">{label}</p>
      {isColdBackend && (
        <div className="hud-glass mt-3 max-w-xs rounded-xl p-3 text-xs text-neutral-500 dark:text-neutral-400">
          <p className="mb-0.5 font-medium text-neutral-900 dark:text-white">Waking up the ocean server...</p>
          <p>The backend may sleep after inactivity. Spinning up can take up to a minute ({seconds}s elapsed).</p>
        </div>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="flex min-h-[60vh] w-full items-center justify-center">
        {content}
      </div>
    );
  }

  return content;
}
