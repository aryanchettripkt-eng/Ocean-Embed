"use client";

import { useEffect, useState } from "react";

function readIsDark(): boolean {
  if (typeof document === "undefined") return true;
  return document.documentElement.classList.contains("dark");
}

/**
 * Tracks the `dark` class on <html> (set by the header's theme toggle). For code that needs
 * literal colours per theme, such as Recharts props, where Tailwind `dark:` classes can't reach.
 */
export function useIsDark(): boolean {
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    setIsDark(readIsDark());
    const observer = new MutationObserver(() => setIsDark(readIsDark()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return isDark;
}
