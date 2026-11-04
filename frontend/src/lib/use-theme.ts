"use client";

import { useEffect, useState } from "react";
export type Theme = "light" | "dark";

function savedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem("theme");
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.classList.toggle("light", theme === "light");
  document.documentElement.style.colorScheme = theme;
  document
    .querySelector('meta[name="color-scheme"]')
    ?.setAttribute("content", theme);
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    const system = window.matchMedia("(prefers-color-scheme: dark)");
    const sync = (event?: Event) => {
      const explicit =
        event instanceof CustomEvent ? (event.detail as Theme) : null;
      const next =
        explicit ?? savedTheme() ?? (system.matches ? "dark" : "light");
      applyTheme(next);
      setTheme(next);
    };
    sync();
    system.addEventListener("change", sync);
    window.addEventListener("storage", sync);
    window.addEventListener("oceanscope:theme", sync);
    return () => {
      system.removeEventListener("change", sync);
      window.removeEventListener("storage", sync);
      window.removeEventListener("oceanscope:theme", sync);
    };
  }, []);
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    try {
      window.localStorage.setItem("theme", next);
    } catch {
      /* Storage can be disabled. */
    }
    applyTheme(next);
    setTheme(next);
    window.dispatchEvent(new CustomEvent("oceanscope:theme", { detail: next }));
  };
  return { theme, toggleTheme };
}
