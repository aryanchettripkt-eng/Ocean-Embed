/**
 * Formatting utilities for oceanographic values (PRD_FRONTEND §4).
 */

/**
 * Format water temperature in °C.
 * Map tooltips use 1 decimal, tables use 2 decimals.
 */
export function formatTemp(val: number | null | undefined, decimals = 1): string {
  if (val === null || val === undefined || isNaN(val)) {
    return "—";
  }
  return `${val.toFixed(decimals)} °C`;
}

/**
 * Format ocean depth in meters according to PRD_FRONTEND §4:
 * "≥ 10 m rounded to the nearest metre, < 10 m one decimal"
 */
export function formatDepth(meters: number | null | undefined): string {
  if (meters === null || meters === undefined || isNaN(meters)) {
    return "—";
  }
  if (meters < 10) {
    return `${meters.toFixed(1)} m`;
  }
  return `${Math.round(meters)} m`;
}

/**
 * Format geographic coordinates: "12.375°N, 65.125°E" (PRD_FRONTEND §4).
 */
export function formatCoord(lat: number | null | undefined, lon: number | null | undefined): string {
  if (lat === null || lat === undefined || lon === null || lon === undefined || isNaN(lat) || isNaN(lon)) {
    return "—";
  }
  const latStr = `${Math.abs(lat).toFixed(3)}°${lat >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(lon).toFixed(3)}°${lon >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

/**
 * Format date YYYY-MM-DD into a readable label, e.g. "01 Jan 2020".
 */
export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const parts = dateStr.split("-");
  if (parts.length !== 3) return dateStr;
  const [year, month, day] = parts;
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mIdx = parseInt(month, 10) - 1;
  if (mIdx < 0 || mIdx > 11) return dateStr;
  return `${day} ${monthNames[mIdx]} ${year}`;
}

export type SplitVariant = "val" | "test" | "train" | "live";

/**
 * Format dataset split identifier to UI label and style variant.
 */
export function formatSplit(split: string | null | undefined): { label: string; variant: SplitVariant } {
  switch (split?.toLowerCase()) {
    case "val":
    case "validation":
      return { label: "Validation", variant: "val" };
    case "test":
      return { label: "Test", variant: "test" };
    case "train":
    case "training":
      return { label: "Training (in-sample)", variant: "train" };
    case "live":
      return { label: "Live", variant: "live" };
    default:
      return { label: split || "Unknown", variant: "val" };
  }
}
