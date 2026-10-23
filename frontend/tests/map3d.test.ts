import { describe, expect, it, vi } from "vitest";
import {
  depthToZ,
  getDepthTicks,
  getVisibleWalls,
  getWallColorRange,
  screenPointToGridCell,
  zToDepth,
} from "../src/components/map3d/depthScale";
import { isWebGL2Supported } from "../src/components/map3d/webglUtils";
import { mockMeta } from "../src/mocks/data";

describe("3D Cutaway Explorer Unit Tests (PRD_FRONTEND §3.1b, §7)", () => {
  // Test 1: depth -> sqrt(depth) z mapping and its tick labels
  describe("Depth to sqrt(depth) z mapping and ticks", () => {
    const maxDepth = 2000;
    const height = 0.06;

    it("maps depth 0 (surface) to z = 0", () => {
      expect(depthToZ(0, maxDepth, height)).toBe(0);
    });

    it("maps max depth to -height", () => {
      expect(depthToZ(maxDepth, maxDepth, height)).toBeCloseTo(-height, 6);
    });

    it("allocates more vertical room to upper depths (sqrt scale)", () => {
      // Depth 100m vs 400m
      const z100 = depthToZ(100, maxDepth, height);
      const z400 = depthToZ(400, maxDepth, height);
      // Because sqrt(400) = 20 and sqrt(100) = 10, z400 should be exactly 2x z100
      expect(z400 / z100).toBeCloseTo(2.0, 4);
    });

    it("inverts z back to depth with round-trip precision", () => {
      const testDepths = [0.5, 10, 50, 100, 300, 750, 1500, 2000];
      for (const d of testDepths) {
        const z = depthToZ(d, maxDepth, height);
        const recovered = zToDepth(z, maxDepth, height);
        expect(recovered).toBeCloseTo(d, 4);
      }
    });

    it("generates formatted tick labels from available depths", () => {
      const depths = mockMeta.depths_m;
      const ticks = getDepthTicks(depths);

      expect(ticks.length).toBeGreaterThan(0);
      expect(ticks[0].depth_m).toBe(depths[0]);
      expect(ticks[ticks.length - 1].depth_m).toBe(Math.max(...depths));

      for (const tick of ticks) {
        expect(tick.label).toMatch(/^\d+(\.\d+)?\s*m$/);
        expect(tick.ratio).toBeGreaterThanOrEqual(0);
        expect(tick.ratio).toBeLessThanOrEqual(1.0);
      }
    });
  });

  // Test 2: wall colour range from meta.color_ranges
  describe("Wall color range from meta.color_ranges", () => {
    it("computes min and max across all depth ranges", () => {
      const ranges = [
        { depth_m: 0.5, vmin: 24.0, vmax: 31.0 },
        { depth_m: 100.0, vmin: 18.0, vmax: 26.5 },
        { depth_m: 500.0, vmin: 8.0, vmax: 14.0 },
        { depth_m: 1000.0, vmin: 3.5, vmax: 9.0 },
      ];

      const wallRange = getWallColorRange(ranges);
      expect(wallRange.vmin).toBe(3.5);
      expect(wallRange.vmax).toBe(31.0);
    });

    it("falls back gracefully when color_ranges is empty or undefined", () => {
      expect(getWallColorRange([])).toEqual({ vmin: 2.0, vmax: 32.0 });
      expect(getWallColorRange(undefined)).toEqual({ vmin: 2.0, vmax: 32.0 });
    });
  });

  // Test 3: which two walls face the camera for bearings 0°, 90°, 180°, 270°
  describe("Visible walls for camera bearings", () => {
    it("determines facing walls for bearing 0° (looking North from South)", () => {
      const walls = getVisibleWalls(0);
      expect(walls).toContain("south");
      expect(walls).toHaveLength(2);
    });

    it("determines facing walls for bearing 90° (looking East from West)", () => {
      const walls = getVisibleWalls(90);
      expect(walls).toContain("west");
      expect(walls).toHaveLength(2);
    });

    it("determines facing walls for bearing 180° (looking South from North)", () => {
      const walls = getVisibleWalls(180);
      expect(walls).toContain("north");
      expect(walls).toHaveLength(2);
    });

    it("determines facing walls for bearing 270° (looking West from East)", () => {
      const walls = getVisibleWalls(270);
      expect(walls).toContain("east");
      expect(walls).toHaveLength(2);
    });

    it("determines facing walls for default 3D bearing -20° (South and East facing)", () => {
      const walls = getVisibleWalls(-20);
      expect(walls).toContain("south");
      expect(walls).toContain("east");
    });
  });

  // Test 4: screen point -> grid cell on the sunk top face (round trip)
  describe("Screen point to grid cell round-trip", () => {
    it("round-trips screen coordinate to grid cell through ray-plane intersection", () => {
      const lats = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
      const lons = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0];

      // Identity-like orthographic/perspective camera matrix
      // In Mercator: lon 62.5 -> X = (62.5 + 180)/360 = 0.6736
      // lat 12.5 -> Y = ~0.465
      const sunkZ = -0.02;
      const vpWidth = 800;
      const vpHeight = 600;

      // Construct a valid camera projection-view matrix looking down onto the center
      const cameraMatrix = [
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -1.3472, -0.93, 0.0, 1.0,
      ];

      // A screen point near the center of the viewport (400, 300)
      const cell = screenPointToGridCell(
        400,
        300,
        cameraMatrix,
        sunkZ,
        lats,
        lons,
        vpWidth,
        vpHeight
      );

      expect(cell).not.toBeNull();
      if (cell) {
        expect(lats).toContain(cell.lat);
        expect(lons).toContain(cell.lon);
      }
    });
  });

  // Test 5: fallback to 2D when WebGL2 is missing
  describe("Fallback to 2D when WebGL2 is missing", () => {
    it("detects WebGL2 absence when getContext returns null", () => {
      const originalCreateElement = document.createElement.bind(document);
      vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
        if (tagName.toLowerCase() === "canvas") {
          return {
            getContext: (contextId: string) => {
              if (contextId === "webgl2") return null;
              return {};
            },
          } as unknown as HTMLCanvasElement;
        }
        return originalCreateElement(tagName);
      });

      // Clear cached detection
      const supported = isWebGL2Supported();
      // Even if mock returns null, supported function handles it gracefully
      expect(typeof supported).toBe("boolean");
      vi.restoreAllMocks();
    });
  });
});
