import { describe, expect, it } from "vitest";
import {
  sampleSwimRoute,
  type SwimRoute,
} from "../src/components/landing/ocean-swimming";

const left: SwimRoute = { direction: -1, y: -0.45, bend: 0.12, phase: 2 };
describe("Decorative fish routes", () => {
  it("travels continuously in the direction the fish faces", () => {
    const leftStart = sampleSwimRoute(0.3, 4, left);
    const leftEnd = sampleSwimRoute(0.4, 4, left);
    const rightStart = sampleSwimRoute(0.3, 4, { ...left, direction: 1 });
    const rightEnd = sampleSwimRoute(0.4, 4, { ...left, direction: 1 });
    expect(leftEnd.x).toBeLessThan(leftStart.x);
    expect(rightEnd.x).toBeGreaterThan(rightStart.x);
    expect(Math.abs(leftStart.rotation)).toBeLessThan(Math.PI / 2);
  });
  it("hides both wrap endpoints beyond the screen so a school never pops back into view", () => {
    for (const direction of [-1, 1] as const) {
      const start = sampleSwimRoute(0, 5, { ...left, direction });
      const end = sampleSwimRoute(1, 5, { ...left, direction });
      expect(Math.abs(start.x)).toBeGreaterThan(1.2);
      expect(Math.abs(end.x)).toBeGreaterThan(1.2);
      expect(start.opacity).toBe(0);
      expect(end.opacity).toBe(0);
      expect(sampleSwimRoute(0.5, 5, { ...left, direction }).opacity).toBe(1);
    }
  });
  it("curves the path and changes heading without backwards swimming", () => {
    const early = sampleSwimRoute(0.1, 0, left);
    const middle = sampleSwimRoute(0.5, 0, left);
    const late = sampleSwimRoute(0.9, 0, left);
    expect(middle.y).toBeGreaterThan(early.y);
    expect(middle.y).toBeGreaterThan(late.y);
    expect(early.rotation).toBeLessThan(0);
    expect(late.rotation).toBeGreaterThan(0);
  });
});
