"use client";

import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

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
      <div className="relative mb-4">
        <Loader2 className="h-8 w-8 animate-spin text-cyan-400" />
        <div className="absolute inset-0 rounded-full blur-md bg-cyan-500/20 -z-10" />
      </div>
      <p className="text-sm font-medium text-slate-200 mb-1">{label}</p>
      {isColdBackend && (
        <div className="max-w-xs text-xs text-slate-400 mt-2 bg-slate-900/80 border border-slate-800 rounded-lg p-3">
          <p className="font-medium text-cyan-300 mb-0.5">Waking up the ocean server...</p>
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
