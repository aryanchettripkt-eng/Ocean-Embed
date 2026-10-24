import { test, expect } from "@playwright/test";

test.describe("Landing Visual Acceptance Suite (PLAN_LANDING_FIX Part 3)", () => {
  test("1440x900 hero fit: H1 and depth ruler within viewport and H1 on one line", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");

    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible({ timeout: 15000 });

    // 1. Assert H1 bounding box is fully inside viewport
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box) {
      expect(h1Box.y).toBeGreaterThanOrEqual(0);
      expect(h1Box.y + h1Box.height).toBeLessThanOrEqual(900);

      // Check H1 height is less than 1.3x font size (one line assertion)
      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    // 2. Assert DepthRuler bounding box is fully inside viewport
    const ruler = page.getByRole("region", { name: /Depth scale/i });
    await expect(ruler).toBeVisible({ timeout: 15000 });
    const rulerBox = await ruler.boundingBox();
    expect(rulerBox).not.toBeNull();
    if (rulerBox) {
      expect(rulerBox.y).toBeGreaterThanOrEqual(0);
      expect(rulerBox.y + rulerBox.height).toBeLessThanOrEqual(900);
    }

    // Capture visual screenshot at 1440x900
    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1440x900.png",
      fullPage: false,
    });
  });

  test("1673x972 reference aspect fit: captures screenshot", async ({ page }) => {
    await page.setViewportSize({ width: 1673, height: 972 }); // 940 + 32px banner
    await page.goto("/");

    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible({ timeout: 15000 });

    // H1 single-line check
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box) {
      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1673x972.png",
      fullPage: false,
    });
  });

  test("1920x1080 full HD fit: 1000 m label visible without scrolling", async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/");

    // Depth ruler is visible
    const ruler = page.getByRole("region", { name: /Depth scale/i });
    await expect(ruler).toBeVisible({ timeout: 15000 });
    const rulerBox = await ruler.boundingBox();
    expect(rulerBox).not.toBeNull();
    if (rulerBox) {
      expect(rulerBox.y + rulerBox.height).toBeLessThanOrEqual(1080);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1920x1080.png",
      fullPage: false,
    });
  });

  test("1024x768 landscape tablet viewport captures screenshot", async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1024x768.png",
      fullPage: false,
    });
  });

  test("768x1024 portrait tablet viewport captures screenshot", async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    await page.screenshot({
      path: "../docs/landing/screenshots/visual-768x1024.png",
      fullPage: false,
    });
  });

  test("390x844 mobile viewport: no horizontal scroll, H1 on one line, CTA visible", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    // 1. No horizontal scroll
    const isHorizontalScrollable = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(isHorizontalScrollable).toBe(false);

    // 2. H1 single line
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible({ timeout: 15000 });
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box) {
      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    // 3. CTA visible without scrolling
    const cta = page.getByRole("link", { name: /EXPLORE THE OCEAN/i });
    await expect(cta).toBeVisible({ timeout: 15000 });
    const ctaBox = await cta.boundingBox();
    expect(ctaBox).not.toBeNull();
    if (ctaBox) {
      expect(ctaBox.y + ctaBox.height).toBeLessThanOrEqual(844);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-390x844.png",
      fullPage: false,
    });
  });
});
