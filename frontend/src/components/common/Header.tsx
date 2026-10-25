"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Waves, BarChart3, Info, ExternalLink, Sun, Moon } from "lucide-react";
import { useMeta } from "../../lib/api/hooks";
import { Banner } from "./Banner";

export function Header() {
  const pathname = usePathname();
  const { data: meta } = useMeta();
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = typeof window !== "undefined" ? (localStorage.getItem("theme") as "dark" | "light" | null) : null;
    const initial = saved ?? "dark";
    setTheme(initial);
    if (initial === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("theme", next);
    if (next === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  };

  const navLinks = [
    { href: "/explore", label: "Explorer", icon: Waves },
    { href: "/validation", label: "Validation", icon: BarChart3 },
    { href: "/about", label: "About", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md transition-colors">
      <Banner type="synthetic" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 group" aria-label="OceanScope AI Home">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-500 p-1.5 flex items-center justify-center shadow-lg shadow-cyan-900/30 group-hover:shadow-cyan-600/40 transition-all">
              <Waves className="h-full w-full text-white" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-base tracking-tight text-slate-900 dark:text-white group-hover:text-cyan-600 dark:group-hover:text-cyan-300 transition-colors">
                OceanScope <span className="text-cyan-600 dark:text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 -mt-1 hidden sm:inline">
                Subsurface Ocean Temperature
              </span>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-1 sm:gap-2">
            {navLinks.map(({ href, label, icon: Icon }) => {
              const isActive =
                href === "/explore"
                  ? pathname === "/explore" || pathname.startsWith("/explore")
                  : pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-slate-100 dark:bg-slate-800/90 text-cyan-700 dark:text-cyan-300 shadow-sm border border-slate-200 dark:border-slate-700/60 font-semibold"
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-900/60"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Status / Model info & theme toggle */}
        <div className="flex items-center gap-3">
          {meta && (
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 px-2.5 py-1 rounded-full">
              <span className="h-2 w-2 rounded-full bg-cyan-500 dark:bg-cyan-400 animate-pulse" />
              <span>Run: {meta.run_id}</span>
            </div>
          )}

          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle dark/light theme"
            className="p-1.5 rounded-md text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-slate-700" />}
          </button>

          <a
            href="https://github.com/aryanchettripkt-eng/Ocean-Embed"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 flex items-center gap-1 p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors"
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

