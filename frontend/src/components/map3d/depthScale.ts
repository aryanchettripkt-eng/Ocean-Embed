/**
 * Math and geometric utilities for 3D Cutaway Explorer (PRD_FRONTEND §3.1b, §7).
 * Pure functions with zero GPU dependencies for robust unit testing.
 */

import { formatDepth } from "../../lib/format";

export type WallId = "south" | "north" | "west" | "east";

export interface DepthTick {
  depth_m: number;
  ratio: number; // 0 (surface) to 1 (max depth)
  label: string;
}

/**
 * Map depth in meters to vertical Z offset using sqrt(depth) scale.
 * Z is negative (depth increases downward into the earth).
 *
 * z(d) = - (sqrt(d) / sqrt(maxDepth)) * height
 */
export function depthToZ(depth_m: number, maxDepth_m: number, height: number): number {
  if (depth_m <= 0 || maxDepth_m <= 0 || height <= 0) return 0;
  const clampedDepth = Math.min(depth_m, maxDepth_m);
  const ratio = Math.sqrt(clampedDepth) / Math.sqrt(maxDepth_m);
  return -ratio * height;
}

/**
 * Inverse mapping: vertical Z offset to depth in meters.
 * d(z) = (-z / height)^2 * maxDepth
 */
export function zToDepth(z: number, maxDepth_m: number, height: number): number {
  if (z >= 0 || maxDepth_m <= 0 || height <= 0) return 0;
  const ratio = Math.min(1.0, Math.max(0.0, -z / height));
  return ratio * ratio * maxDepth_m;
}

/**
 * Compute depth ticks on sqrt(depth) scale from available metadata depths.
 * Targets nearest levels to [0.5, 50, 100, 200, 500, 1000, 2000].
 */
export function getDepthTicks(availableDepths: number[]): DepthTick[] {
  if (!availableDepths || availableDepths.length === 0) return [];

  const maxDepth = Math.max(...availableDepths);
  const targets = [0.5, 50, 100, 200, 500, 1000, 2000];
  const chosen = new Set<number>();

  // Always include surface and deepest level
  chosen.add(availableDepths[0]);
  chosen.add(maxDepth);

  for (const target of targets) {
    if (target > maxDepth) continue;
    let closest = availableDepths[0];
    let minDiff = Math.abs(target - closest);
    for (let i = 1; i < availableDepths.length; i++) {
      const diff = Math.abs(target - availableDepths[i]);
      if (diff < minDiff) {
        minDiff = diff;
        closest = availableDepths[i];
      }
    }
    chosen.add(closest);
  }

  const sorted = Array.from(chosen).sort((a, b) => a - b);
  return sorted.map((d) => {
    const ratio = maxDepth > 0 ? Math.sqrt(d) / Math.sqrt(maxDepth) : 0;
    return {
      depth_m: d,
      ratio,
      label: formatDepth(d),
    };
  });
}

/**
 * Compute column-wide color range for side walls (PRD_FRONTEND §3.1b):
 * vmin = min of all color_ranges[].vmin, vmax = max of all color_ranges[].vmax.
 */
export function getWallColorRange(
  colorRanges?: { depth_m: number; vmin: number; vmax: number }[]
): { vmin: number; vmax: number } {
  if (!colorRanges || colorRanges.length === 0) {
    return { vmin: 2.0, vmax: 32.0 };
  }
  let vmin = Infinity;
  let vmax = -Infinity;
  for (const cr of colorRanges) {
    if (cr.vmin < vmin) vmin = cr.vmin;
    if (cr.vmax > vmax) vmax = cr.vmax;
  }
  if (!isFinite(vmin) || !isFinite(vmax)) {
    return { vmin: 2.0, vmax: 32.0 };
  }
  return { vmin, vmax };
}

/**
 * Determine which two outer walls of the ocean block face the camera
 * given camera bearing in degrees (0° = North, 90° = East, 180° = South, 270° = West).
 *
 * Camera position relative to block center:
 * Cx = -sin(bearing)
 * Cy = -cos(bearing)
 *
 * Dot products with outward normals:
 * South: ( 0, -1) -> Cy * (-1) =  cos(bearing)
 * North: ( 0, +1) -> Cy * (+1) = -cos(bearing)
 * West:  (-1,  0) -> Cx * (-1) =  sin(bearing)
 * East:  (+1,  0) -> Cx * (+1) = -sin(bearing)
 */
export function getVisibleWalls(bearing: number): [WallId, WallId] {
  // Normalize bearing to [0, 360)
  const norm = ((bearing % 360) + 360) % 360;

  if (norm >= 0 && norm < 90) {
    // Looking North to East: camera is South-West
    return ["south", "west"];
  } else if (norm >= 90 && norm < 180) {
    // Looking East to South: camera is North-West
    return ["west", "north"];
  } else if (norm >= 180 && norm < 270) {
    // Looking South to West: camera is North-East
    return ["north", "east"];
  } else {
    // Looking West to North: camera is South-East (e.g. -20° / 340°)
    return ["east", "south"];
  }
}

/**
 * Find nearest coordinate in sorted array.
 */
export function findNearestCoord(val: number, arr: number[]): number {
  if (arr.length === 0) return val;
  let closest = arr[0];
  let minDiff = Math.abs(val - closest);
  for (let i = 1; i < arr.length; i++) {
    const diff = Math.abs(val - arr[i]);
    if (diff < minDiff) {
      minDiff = diff;
      closest = arr[i];
    }
  }
  return closest;
}

/**
 * Convert screen point to snapped grid cell on a plane at altitude Z.
 * Uses ray-plane intersection in Mercator world coordinates.
 */
export function screenPointToGridCell(
  screenX: number,
  screenY: number,
  cameraMatrix: number[], // 4x4 column-major projection-view matrix
  sunkZ: number,
  lats: number[],
  lons: number[],
  viewportWidth: number,
  viewportHeight: number
): { lat: number; lon: number } | null {
  // Invert 4x4 matrix
  const inv = invertMatrix4(cameraMatrix);
  if (!inv) return null;

  // Normalized device coordinates [-1, 1]
  const ndcX = (screenX / viewportWidth) * 2 - 1;
  const ndcY = 1 - (screenY / viewportHeight) * 2;

  // Ray near and far in clip space
  const nearPoint = transformVec4(inv, [ndcX, ndcY, -1, 1]);
  const farPoint = transformVec4(inv, [ndcX, ndcY, 1, 1]);

  if (!nearPoint || !farPoint) return null;

  // Ray origin and direction
  const p0 = [nearPoint[0] / nearPoint[3], nearPoint[1] / nearPoint[3], nearPoint[2] / nearPoint[3]];
  const p1 = [farPoint[0] / farPoint[3], farPoint[1] / farPoint[3], farPoint[2] / farPoint[3]];

  const dirZ = p1[2] - p0[2];
  if (Math.abs(dirZ) < 1e-7) return null; // Parallel to plane

  const t = (sunkZ - p0[2]) / dirZ;
  if (t < 0) return null; // Behind camera

  const mercX = p0[0] + t * (p1[0] - p0[0]);
  const mercY = p0[1] + t * (p1[1] - p0[1]);

  // Convert Mercator [0, 1] to lon, lat
  const lon = mercX * 360 - 180;
  const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * mercY)));
  const lat = (latRad * 180) / Math.PI;

  const snappedLat = findNearestCoord(lat, lats);
  const snappedLon = findNearestCoord(lon, lons);

  return { lat: snappedLat, lon: snappedLon };
}

/**
 * 4x4 matrix vector transform helper
 */
function transformVec4(m: number[], v: number[]): number[] {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

/**
 * Invert 4x4 column-major matrix
 */
export function invertMatrix4(m: number[]): number[] | null {
  const inv = new Array<number>(16);

  inv[0] =
    m[5] * m[10] * m[15] -
    m[5] * m[11] * m[14] -
    m[9] * m[6] * m[15] +
    m[9] * m[7] * m[14] +
    m[13] * m[6] * m[11] -
    m[13] * m[7] * m[10];

  inv[4] =
    -m[4] * m[10] * m[15] +
    m[4] * m[11] * m[14] +
    m[8] * m[6] * m[15] -
    m[8] * m[7] * m[14] -
    m[12] * m[6] * m[11] +
    m[12] * m[7] * m[10];

  inv[8] =
    m[4] * m[9] * m[15] -
    m[4] * m[11] * m[13] -
    m[8] * m[5] * m[15] +
    m[8] * m[7] * m[13] +
    m[12] * m[5] * m[11] -
    m[12] * m[7] * m[9];

  inv[12] =
    -m[4] * m[9] * m[14] +
    m[4] * m[10] * m[13] +
    m[8] * m[5] * m[14] -
    m[8] * m[6] * m[13] -
    m[12] * m[5] * m[10] +
    m[12] * m[6] * m[9];

  inv[1] =
    -m[1] * m[10] * m[15] +
    m[1] * m[11] * m[14] +
    m[9] * m[2] * m[15] -
    m[9] * m[3] * m[14] -
    m[13] * m[2] * m[11] +
    m[13] * m[3] * m[10];

  inv[5] =
    m[0] * m[10] * m[15] -
    m[0] * m[11] * m[14] -
    m[8] * m[2] * m[15] +
    m[8] * m[3] * m[14] +
    m[12] * m[2] * m[11] -
    m[12] * m[3] * m[10];

  inv[9] =
    -m[0] * m[9] * m[15] +
    m[0] * m[11] * m[13] +
    m[8] * m[1] * m[15] -
    m[8] * m[3] * m[13] -
    m[12] * m[1] * m[11] +
    m[12] * m[3] * m[9];

  inv[13] =
    m[0] * m[9] * m[14] -
    m[0] * m[10] * m[13] -
    m[8] * m[1] * m[14] +
    m[8] * m[2] * m[13] +
    m[12] * m[1] * m[10] -
    m[12] * m[2] * m[9];

  inv[2] =
    m[1] * m[6] * m[15] -
    m[1] * m[7] * m[14] -
    m[5] * m[2] * m[15] +
    m[5] * m[3] * m[14] +
    m[13] * m[2] * m[7] -
    m[13] * m[3] * m[6];

  inv[6] =
    -m[0] * m[6] * m[15] +
    m[0] * m[7] * m[14] +
    m[4] * m[2] * m[15] -
    m[4] * m[3] * m[14] -
    m[12] * m[2] * m[7] +
    m[12] * m[3] * m[6];

  inv[10] =
    m[0] * m[5] * m[15] -
    m[0] * m[7] * m[13] -
    m[4] * m[1] * m[15] +
    m[4] * m[3] * m[13] +
    m[12] * m[1] * m[7] -
    m[12] * m[3] * m[5];

  inv[14] =
    -m[0] * m[5] * m[14] +
    m[0] * m[6] * m[13] +
    m[4] * m[1] * m[14] -
    m[4] * m[2] * m[13] -
    m[12] * m[1] * m[6] +
    m[12] * m[2] * m[5];

  inv[3] =
    -m[1] * m[6] * m[11] +
    m[1] * m[7] * m[10] +
    m[5] * m[2] * m[11] -
    m[5] * m[3] * m[10] -
    m[9] * m[2] * m[7] +
    m[9] * m[3] * m[6];

  inv[7] =
    m[0] * m[6] * m[11] -
    m[0] * m[7] * m[10] -
    m[4] * m[2] * m[11] +
    m[4] * m[3] * m[10] +
    m[8] * m[2] * m[7] -
    m[8] * m[3] * m[6];

  inv[11] =
    -m[0] * m[5] * m[11] +
    m[0] * m[7] * m[9] +
    m[4] * m[1] * m[11] -
    m[4] * m[3] * m[9] -
    m[8] * m[1] * m[7] +
    m[8] * m[3] * m[5];

  inv[15] =
    m[0] * m[5] * m[10] -
    m[0] * m[6] * m[9] -
    m[4] * m[1] * m[10] +
    m[4] * m[2] * m[9] +
    m[8] * m[1] * m[6] -
    m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (Math.abs(det) < 1e-12) return null;

  det = 1.0 / det;
  for (let i = 0; i < 16; i++) {
    inv[i] *= det;
  }
  return inv;
}
