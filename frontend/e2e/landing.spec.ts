import { test, expect } from "@playwright/test";

test.describe("OceanScope AI Landing Page Flow (§10)", () => {
  test("/welcome loads with no console errors and renders hero", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto("/welcome");

    // 1. Synthetic banner should be visible (using .first() to handle Phase A dual render with root header)
    const banner = page.getByRole("alert").filter({ hasText: /synthetic demo data/i }).first();
    await expect(banner).toBeVisible({ timeout: 15000 });

    // 2. Main H1 heading and tagline
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toHaveText("OCEANSCOPE AI");
    await expect(page.getByText("BENEATH THE SURFACE", { exact: true })).toBeVisible();

    // 3. CTA button is visible
    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await expect(cta).toBeVisible();

    // 4. Verify no unexpected console errors occurred on page load
    expect(consoleErrors).toEqual([]);
  });

  test("CTA navigates to Explorer and map container appears", async ({ page }) => {
    await page.goto("/welcome");

    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await expect(cta).toBeVisible();
    await cta.click();

    // Explorer route is reached (with or without auto-appended query params like ?date=...&depth=...)
    await expect(page).toHaveURL(/\/(\?.*)?$/);

    // Explorer map container or controls should become visible
    await expect(page.locator("text=Field")).toBeVisible({ timeout: 15000 });
  });

  test("at 390 px mobile viewport there is no horizontal scroll", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/welcome");

    // Wait for initial render
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });

    const isHorizontalScrollable = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });

    expect(isHorizontalScrollable).toBe(false);
  });
});
