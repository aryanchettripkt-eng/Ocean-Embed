export interface SwimRoute {
  /** Direction of travel: the source sprite faces left. */
  direction: -1 | 1;
  y: number;
  bend: number;
  phase: number;
}

/** Decorative scene coordinates. Endpoints are outside the frame and fade before wrapping. */
export function sampleSwimRoute(
  progress: number,
  time: number,
  route: SwimRoute,
) {
  const p = Math.max(0, Math.min(1, progress));
  const tangentY = Math.PI * Math.cos(p * Math.PI) * route.bend;
  return {
    x: route.direction * (p * 2.65 - 1.325),
    y:
      route.y +
      Math.sin(p * Math.PI) * route.bend +
      Math.sin(time * 0.45 + route.phase) * 0.012,
    rotation: Math.atan2(tangentY, 2.65) * route.direction,
    opacity: Math.max(0, Math.min(1, p / 0.09, (1 - p) / 0.09)),
  };
}
