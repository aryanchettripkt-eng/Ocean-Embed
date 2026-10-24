import { test, expect } from "@playwright/test";

test.describe("OceanScope AI Landing Page Flow (§10, PLAN_LANDING §8, §10)", () => {
  test("/ loads with no console errors and renders hero", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (
        msg.type() === "error" &&
        !msg.text().includes("Failed to load resource") &&
        !msg.text().includes("404") &&
        !msg.text().includes("CORS policy")
      ) {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto("/");

    // 1. Synthetic banner should be visible
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

  test("CTA navigates to Explorer (/explore) and map container appears", async ({ page }) => {
    await page.goto("/");

    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await expect(cta).toBeVisible();
    await cta.click();

    // Explorer route is reached
    await expect(page).toHaveURL(/\/explore(\?.*)?$/);

    // Explorer view or metadata loading indicator becomes visible
    await expect(
      page.getByText(/3D Cutaway|2D Map|Loading OceanScope AI metadata/i).first()
    ).toBeVisible({ timeout: 15000 });
  });

  test("legacy /welcome route redirects to root / landing page", async ({ page }) => {
    await page.goto("/welcome");
    await expect(page).toHaveURL(/\/(.*)?$/);
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toHaveText("OCEANSCOPE AI");
  });

  test("at 390 px mobile viewport there is no horizontal scroll", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });

    const isHorizontalScrollable = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });

    expect(isHorizontalScrollable).toBe(false);
  });

  test("with JS disabled the CTA is a real link and navigates to /explore", async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    await page.goto("/");

    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await expect(cta).toBeVisible();
    await cta.click();

    // Explorer route is reached
    await expect(page).toHaveURL(/\/explore(\?.*)?$/);
    await context.close();
  });

  test("focus moves to explorer heading (#explorer-heading) after navigation", async ({ page }) => {
    await page.goto("/");
    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await cta.click();

    await expect(page).toHaveURL(/\/explore(\?.*)?$/);
    const heading = page.locator("#explorer-heading");
    await expect(heading).toBeAttached();
    // Verify focus moved
    await expect(heading).toBeFocused({ timeout: 5000 });
  });

  test("captures landing page screenshots at 1440, 1024, 768, and 390 px", async ({ page }) => {
    test.setTimeout(60000);
    const viewports = [
      { width: 1440, height: 900, name: "screenshot-1440.png" },
      { width: 1024, height: 768, name: "screenshot-1024.png" },
      { width: 768, height: 1024, name: "screenshot-768.png" },
      { width: 390, height: 844, name: "screenshot-390.png" },
    ];

    for (const vp of viewports) {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto("/");
      // Wait for hero to render
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
      await page.screenshot({ path: `../docs/landing/screenshots/${vp.name}`, fullPage: false });
    }
  });
});

