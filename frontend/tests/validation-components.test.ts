import { describe, expect, it } from "vitest";
import { mockArgo, mockHeadline, mockMetrics } from "../src/mocks/data";

describe("Validation and Benchmark logic", () => {
  it("processes headline pre-registered criterion table correctly", () => {
    expect(mockHeadline.met).toBe(true);
    expect(mockHeadline.criterion).toContain("Bay of Bengal, 50–300 m");
    expect(mockHeadline.rows.length).toBe(6);

    const modelRow = mockHeadline.rows.find((r) => r.product === "model");
    expect(modelRow).toBeDefined();
    expect(modelRow?.rmse_c).toBe(0.28);
    expect(modelRow?.ci_low_c).toBe(0.24);
    expect(modelRow?.ci_high_c).toBe(0.32);

    const glorysRow = mockHeadline.rows.find((r) => r.product === "glorys12");
    expect(glorysRow).toBeDefined();
    expect(glorysRow?.diff_vs_model_c).toBe(-0.08);
  });

  it("extracts overall domain-wide validation row from metrics", () => {
    const overallRow = mockMetrics.rows.find(
      (r) => r.depth_m === "all" && r.model === "SYNTHETIC_DEMO"
    );

    expect(overallRow).toBeDefined();
    expect(overallRow?.n).toBe(108120);
    expect(overallRow?.rmse_c).toBe(0.2);
    expect(overallRow?.mae_c).toBe(0.19);
    expect(overallRow?.bias_c).toBe(0.15);
    expect(overallRow?.skill_vs_clim_pct).toBe(65.5);
  });

  it("verifies per-depth benchmark curve data", () => {
    const depthRows = mockMetrics.rows.filter(
      (r) => r.depth_m !== "all" && r.depth_m !== null
    );

    expect(depthRows.length).toBeGreaterThan(0);
    const depths = Array.from(new Set(depthRows.map((r) => r.depth_m)));
    expect(depths).toContain(0.5);
    expect(depths).toContain(100.0);
  });

  it("verifies ARGO float matchups and caveat presence", () => {
    expect(mockArgo.n_profiles_total).toBe(3);
    expect(mockArgo.n_obs_total).toBe(45);
    expect(mockArgo.caveat).toContain("assimilated into GLORYS");
    expect(mockArgo.rows.length).toBe(15);

    const surfaceRow = mockArgo.rows.find((r) => r.depth_m === 0.5);
    expect(surfaceRow).toBeDefined();
    expect(surfaceRow?.rmse_c).toBe(0.08);
    expect(surfaceRow?.glorys_rmse_c).toBe(0.08);
    expect(surfaceRow?.grep_rmse_c).toBe(0.08);
    expect(surfaceRow?.armor3d_rmse_c).toBe(0.14);
  });
});
