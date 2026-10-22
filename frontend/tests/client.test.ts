import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { apiClient, ApiError } from "../src/lib/api/client";
import { server } from "../src/mocks/server";

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("apiClient and MSW handlers", () => {
  it("fetches /health and /meta", async () => {
    const health = await apiClient.getHealth();
    expect(health.status).toBe("ok");
    expect(health.run_id).toBe("synthetic_v1");

    const meta = await apiClient.getMeta();
    expect(meta.synthetic).toBe(true);
    expect(meta.depths_m.length).toBe(15);
    expect(meta.available_dates.start).toBe("2020-01-01");
  });

  it("fetches 2D map slice", async () => {
    const map = await apiClient.getMap({ date: "2020-01-05", depth_m: 100.0, field: "pred" });
    expect(map.values.length).toBe(30);
    expect(map.values[0].length).toBe(60);
    expect(map.coverage.length).toBe(30);
    expect(map.stats.n_valid).toBeGreaterThan(0);
  });

  it("fetches vertical profile", async () => {
    const profile = await apiClient.getProfile({ lat: 12.5, lon: 65.5, date: "2020-01-05" });
    expect(profile.depths_m.length).toBe(15);
    expect(profile.pred.length).toBe(15);
    expect(profile.glorys.length).toBe(15);
  });

  it("fetches headline pre-registered criterion table", async () => {
    const headline = await apiClient.getHeadline();
    expect(headline.met).toBe(true);
    expect(headline.rows.length).toBe(6);
    expect(headline.criterion).toContain("Bay of Bengal, 50–300 m");
  });

  it("handles 404 reference_unavailable on live dates for glorys", async () => {
    await expect(
      apiClient.getMap({ date: "2020-01-09", field: "glorys" })
    ).rejects.toThrowError(ApiError);

    try {
      await apiClient.getMap({ date: "2020-01-09", field: "glorys" });
    } catch (err) {
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(404);
      expect(apiErr.code).toBe("reference_unavailable");
    }
  });

  it("handles 404 test_not_evaluated on sealed test split", async () => {
    try {
      await apiClient.getMetrics({ split: "test" });
      expect.fail("Should have thrown ApiError");
    } catch (err) {
      const apiErr = err as ApiError;
      expect(apiErr.status).toBe(404);
      expect(apiErr.code).toBe("test_not_evaluated");
    }
  });
});
