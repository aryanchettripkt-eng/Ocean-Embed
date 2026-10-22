"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Waves, BarChart3, Info, ExternalLink } from "lucide-react";
import { useMeta } from "../../lib/api/hooks";

export function Header() {
  const pathname = usePathname();
  const { data: meta } = useMeta();

  const navLinks = [
    { href: "/", label: "Explorer", icon: Waves },
    { href: "/validation", label: "Validation", icon: BarChart3 },
    { href: "/about", label: "About", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-500 p-1.5 flex items-center justify-center shadow-lg shadow-cyan-900/30 group-hover:shadow-cyan-600/40 transition-all">
              <Waves className="h-full w-full text-white" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-base tracking-tight text-white group-hover:text-cyan-300 transition-colors">
                OceanScope <span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] text-slate-400 -mt-1 hidden sm:inline">
                Subsurface Ocean Temperature
              </span>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-1 sm:gap-2">
            {navLinks.map(({ href, label, icon: Icon }) => {
              const isActive = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-slate-800/90 text-cyan-300 shadow-sm border border-slate-700/60"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Status / Model info */}
        <div className="flex items-center gap-3">
          {meta && (
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 bg-slate-900/60 border border-slate-800 px-2.5 py-1 rounded-full">
              <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
              <span>Run: {meta.run_id}</span>
            </div>
          )}
          <a
            href="https://github.com/aryanchettripkt-eng/Ocean-Embed"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 p-1.5 rounded-md hover:bg-slate-900 transition-colors"
            title="GitHub Repository"
          >
            <span className="hidden sm:inline">GitHub</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </header>
  );
}
