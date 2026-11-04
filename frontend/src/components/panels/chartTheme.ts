/**
 * Literal colours for the Recharts panels, per theme. Recharts takes colours as props, so
 * Tailwind `dark:` classes can't style them; pick with `chartTheme(useIsDark())`.
 * Series colours keep the same hues in both themes (cyan = model, violet = GLORYS,
 * amber = selected depth/date) and only shift lightness for contrast.
 */
export interface ChartTheme {
  grid: string;
  tick: string;
  axis: string;
  pred: string;
  predActive: string;
  glorys: string;
  glorysActive: string;
  marker: string;
}

const DARK: ChartTheme = {
  grid: "rgba(255, 255, 255, 0.07)",
  tick: "#8b9096",
  axis: "rgba(255, 255, 255, 0.14)",
  pred: "#22d3ee",
  predActive: "#67e8f9",
  glorys: "#a78bfa",
  glorysActive: "#c4b5fd",
  marker: "#f59e0b",
};

const LIGHT: ChartTheme = {
  grid: "rgba(15, 23, 42, 0.07)",
  tick: "#6b7280",
  axis: "rgba(15, 23, 42, 0.16)",
  pred: "#0891b2",
  predActive: "#0e7490",
  glorys: "#7c3aed",
  glorysActive: "#6d28d9",
  marker: "#d97706",
};

export function chartTheme(isDark: boolean): ChartTheme {
  return isDark ? DARK : LIGHT;
}

/** Shared tooltip shell classes (HUD glass, both themes). */
export const CHART_TOOLTIP_CLASS =
  "hud-glass min-w-[180px] space-y-1.5 rounded-xl px-3 py-2.5 text-xs";
