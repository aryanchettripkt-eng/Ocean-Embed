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

  test("toggles theme between dark and light modes", async ({ page }) => {
    await page.goto("/");
    const themeBtn = page.locator('button[aria-label="Toggle dark/light theme"]');
    await expect(themeBtn).toBeVisible({ timeout: 15000 });

    // Initially dark
    await expect(page.locator("html")).toHaveClass(/dark/);

    // Click to switch to light mode
    await themeBtn.click();
    await expect(page.locator("html")).not.toHaveClass(/dark/);

    // Click again to switch back to dark mode
    await themeBtn.click();
    await expect(page.locator("html")).toHaveClass(/dark/);
  });

  test("updates URL query parameters when changing depth and field without skipping", async ({ page }) => {
    await page.goto("/?date=2020-01-05");
    await expect(page.locator("text=Field")).toBeVisible({ timeout: 15000 });

    // Click on depth preset 100 m
    const depth100 = page.getByRole("button", { name: "100 m", exact: true });
    await expect(depth100).toBeVisible({ timeout: 10000 });
    await depth100.click();
    await expect(page).toHaveURL(/depth=100/);

    // Click on GLORYS field toggle button
    const glorysBtn = page.getByRole("button", { name: "GLORYS reanalysis" });
    await expect(glorysBtn).toBeVisible({ timeout: 10000 });
    await expect(glorysBtn).toBeEnabled();
    await glorysBtn.click();
    await expect(page).toHaveURL(/field=glorys/);

    // Click on Difference field toggle button
    const diffBtn = page.getByRole("button", { name: /Difference/i });
    await expect(diffBtn).toBeVisible({ timeout: 10000 });
    await diffBtn.click();
    await expect(page).toHaveURL(/field=error/);
  });

  test("steps date using date picker buttons and updates URL", async ({ page }) => {
    await page.goto("/?date=2020-01-05");
    await expect(page.locator("text=Field")).toBeVisible({ timeout: 15000 });

    // Verify initial date input
    const dateInput = page.locator('input[type="date"]');
    await expect(dateInput).toHaveValue("2020-01-05");

    // Click Next day (+1d) button
    const nextDayBtn = page.getByTitle("Next day (+1d)");
    await expect(nextDayBtn).toBeVisible();
    await expect(nextDayBtn).toBeEnabled();
    await nextDayBtn.click();

    // Verify date updated to 2020-01-06 in input and URL
    await expect(dateInput).toHaveValue("2020-01-06");
    await expect(page).toHaveURL(/date=2020-01-06/);

    // Click Previous day (-1d) button
    const prevDayBtn = page.getByTitle("Previous day (-1d)");
    await expect(prevDayBtn).toBeVisible();
    await expect(prevDayBtn).toBeEnabled();
    await prevDayBtn.click();

    await expect(dateInput).toHaveValue("2020-01-05");
    await expect(page).toHaveURL(/date=2020-01-05/);
  });

  test("clicks on map canvas to select a cell and opens details panel", async ({ page }) => {
    await page.goto("/?date=2020-01-05&depth=100&field=pred");
    await expect(page.locator("text=Field")).toBeVisible({ timeout: 15000 });
    await expect(page.locator("text=Model Predicted Temperature")).toBeVisible({ timeout: 15000 });

    // Wait for the map canvas to load
    const mapCanvas = page.locator("canvas.maplibregl-canvas");
    await expect(mapCanvas).toBeVisible({ timeout: 15000 });

    const box = await mapCanvas.boundingBox();
    expect(box).not.toBeNull();
    const centerX = Math.round((box?.width ?? 600) / 2);
    const centerY = Math.round((box?.height ?? 400) / 2);

    await mapCanvas.click({ position: { x: centerX, y: centerY }, force: true });

    // Details panel should open with vertical temperature profile and time series
    await expect(page.locator("text=Vertical Temperature Profile")).toBeVisible({ timeout: 15000 });
    await expect(page.locator("text=Selected cell centre")).toBeVisible();
    await expect(page).toHaveURL(/lat=/);
    await expect(page).toHaveURL(/lon=/);

    // Close details panel
    const closeBtn = page.locator('button[aria-label="Close details panel"]');
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();
    await expect(page.locator("text=Vertical Temperature Profile")).not.toBeVisible();
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
