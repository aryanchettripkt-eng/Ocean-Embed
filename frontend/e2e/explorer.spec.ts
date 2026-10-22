import { test, expect } from "@playwright/test";

test.describe("OceanScope AI Explorer Flow (PRD_FRONTEND §7)", () => {
  test("loads explorer, displays synthetic banner, and inspects controls", async ({ page }) => {
    await page.goto("/");

    // 1. Synthetic banner should be visible
    const banner = page.getByRole("alert").filter({ hasText: /synthetic demo data/i });
    await expect(banner).toBeVisible({ timeout: 15000 });
    await expect(banner).toContainText(/synthetic demo data/i);

    // 2. Main title / header
    await expect(page.locator("header")).toContainText(/OceanScope AI/i);

    // 3. Controls panel presence
    await expect(page.locator("text=Field")).toBeVisible();
    await expect(page.locator("label").filter({ hasText: /Depth/i })).toBeVisible();
    await expect(page.getByRole("button", { name: "Model prediction" })).toBeVisible();
  });

  test("updates URL query parameters when changing depth and field", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("text=Field")).toBeVisible({ timeout: 15000 });

    // Click on a depth preset (e.g. 100 m)
    const depth100 = page.getByRole("button", { name: "100 m", exact: true });
    if (await depth100.isVisible()) {
      await depth100.click();
      await expect(page).toHaveURL(/depth=100/);
    }

    // Toggle field to GLORYS (if on validation date)
    const glorysBtn = page.getByRole("button", { name: "GLORYS reanalysis" });
    if (await glorysBtn.isEnabled()) {
      await glorysBtn.click();
      await expect(page).toHaveURL(/field=glorys/);
    }
  });

  test("restores state from shareable URL and opens details panel for cell", async ({ page }) => {
    // Navigate with complete state query params
    await page.goto("/?date=2020-01-05&depth=100&field=pred&lat=12.500&lon=65.500");

    // Details panel should be open with cell profile and time series
    await expect(page.locator("text=Vertical Temperature Profile")).toBeVisible({ timeout: 15000 });
    await expect(page.locator("text=Time Series at 100 m")).toBeVisible();

    // Close details panel
    const closeBtn = page.locator('button[aria-label="Close details panel"]');
    await closeBtn.click();
    await expect(page.locator("text=Vertical Temperature Profile")).not.toBeVisible();
  });

  test("navigates to validation benchmarks page and renders headline & metrics tables", async ({ page }) => {
    await page.goto("/validation");

    // Headline card
    const headlineCard = page.locator('[data-testid="headline-card"]');
    await expect(headlineCard).toBeVisible({ timeout: 15000 });
    await expect(page.locator('[data-testid="headline-verdict"]')).toContainText(/Verdict:/i);

    // Summary metric cards
    await expect(page.locator("text=Overall Val RMSE")).toBeVisible();

    // Metrics table and breakdown tabs
    await expect(page.locator("text=Validation Error Breakdown")).toBeVisible();
    await expect(page.locator("button", { hasText: "By season" })).toBeVisible();
    await expect(page.locator("button", { hasText: "By region" })).toBeVisible();
  });

  test("navigates to about page and displays documentation", async ({ page }) => {
    await page.goto("/about");

    await expect(page.locator("h1")).toContainText(/About OceanScope AI/i, { timeout: 15000 });
    await expect(page.locator("text=Active Deployment Parameters")).toBeVisible();
    await expect(page.locator("text=Data Sources & Acknowledgements")).toBeVisible();
  });
});
