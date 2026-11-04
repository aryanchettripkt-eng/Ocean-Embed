"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Waves, BarChart3, Info, ExternalLink, Sun, Moon, Compass } from "lucide-react";
import { useMeta } from "../../lib/api/hooks";
import { Banner } from "./Banner";

function readStoredTheme(): "dark" | "light" | null {
  try {
    if (typeof window !== "undefined" && typeof window.localStorage?.getItem === "function") {
      const val = window.localStorage.getItem("theme");
      return val === "dark" || val === "light" ? val : null;
    }
  } catch {
    // Ignore storage unavailability
  }
  return null;
}

function writeStoredTheme(next: "dark" | "light"): void {
  try {
    if (typeof window !== "undefined" && typeof window.localStorage?.setItem === "function") {
      window.localStorage.setItem("theme", next);
    }
  } catch {
    // Ignore storage unavailability
  }
}

export function Header() {
  const pathname = usePathname();
  const { data: meta } = useMeta();
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = readStoredTheme();
    const initial = saved ?? "dark";
    setTheme(initial);
    if (initial === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
    const metaScheme = document.querySelector('meta[name="color-scheme"]');
    if (metaScheme) metaScheme.setAttribute("content", initial);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    writeStoredTheme(next);
    if (next === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
    const metaScheme = document.querySelector('meta[name="color-scheme"]');
    if (metaScheme) metaScheme.setAttribute("content", next);
  };

  const navLinks = [
    { href: "/explore", label: "Explorer", icon: Waves, phoneOnly: false },
    { href: "/validation", label: "Validation", icon: BarChart3, phoneOnly: false },
    { href: "/about", label: "About", icon: Info, phoneOnly: false },
    // Beach mode is for phones only and only when its flag is on (PRD_FRONTEND §3.4f)
    ...(meta?.features?.beach ? [{ href: "/beach", label: "Beach", icon: Compass, phoneOnly: true }] : []),
  ];

  return (
    <header className="hud-divider sticky top-0 z-40 w-full border-b bg-white/80 backdrop-blur-xl backdrop-saturate-150 dark:bg-[#0a0c0d]/80">
      <Banner type="synthetic" />
      <div className="flex h-14 items-center justify-between gap-3 px-3 sm:px-5">
        {/* Brand + navigation */}
        <div className="flex min-w-0 items-center gap-3 sm:gap-5">
          <Link href="/" className="group flex shrink-0 items-center gap-2.5" aria-label="OceanScope AI Home">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 p-1.5 shadow-[0_4px_14px_rgba(8,145,178,0.35)]">
              <Waves aria-hidden="true" className="h-full w-full text-white" />
            </div>
            <div className="hidden flex-col leading-tight sm:flex">
              <span className="text-[15px] font-semibold tracking-tight text-neutral-950 dark:text-white">
                OceanScope <span className="text-cyan-600 dark:text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] text-neutral-500 dark:text-neutral-400">Subsurface Ocean Temperature</span>
            </div>
          </Link>

          <nav
            aria-label="Main"
            className="flex items-center gap-0.5 rounded-full bg-black/[0.04] p-1 dark:bg-white/[0.05]"
          >
            {navLinks.map(({ href, label, icon: Icon, phoneOnly }) => {
              const isActive =
                href === "/explore"
                  ? pathname === "/explore" || pathname.startsWith("/explore")
                  : pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  aria-label={label}
                  aria-current={isActive ? "page" : undefined}
                  className={`hud-btn h-7 ${phoneOnly ? "md:hidden" : ""} ${isActive ? "hud-btn-active" : ""}`}
                >
                  <Icon aria-hidden="true" className="h-3.5 w-3.5" />
                  {/* Icon-only below 640 px so the header fits a 360 px phone (PRD_FRONTEND §4) */}
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Run, theme, GitHub */}
        <div className="flex shrink-0 items-center gap-1.5">
          {meta && (
            <div className="mr-1 hidden items-center gap-2 rounded-full bg-black/[0.04] px-3 py-1 text-[11px] text-neutral-600 md:flex dark:bg-white/[0.05] dark:text-neutral-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.8)]" />
              <span>
                Run <span className="font-medium text-neutral-900 dark:text-neutral-100">{meta.run_id}</span>
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle dark/light theme"
            className="hud-icon-btn h-8 w-8"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? (
              <Sun aria-hidden="true" className="h-4 w-4" />
            ) : (
              <Moon aria-hidden="true" className="h-4 w-4" />
            )}
          </button>

          <a
            href="https://github.com/aryanchettripkt-eng/Ocean-Embed"
            target="_blank"
            rel="noopener noreferrer"
            className="hud-btn"
            title="GitHub Repository"
          >
            <span className="hidden sm:inline">GitHub</span>
            <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </header>
  );
}
