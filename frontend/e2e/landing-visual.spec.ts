import { test, expect, Page } from "@playwright/test";

async function waitForHeroImage(page: Page) {
  const img = page.locator('img[alt*="Woodblock-style illustration"]');
  await expect(img).toBeVisible({ timeout: 15000 });
  await img.evaluate(async (el: HTMLImageElement) => {
    if (!el.complete) {
      await new Promise((resolve) => {
        el.onload = resolve;
        el.onerror = resolve;
      });
    }
    if (el.decode) {
      await el.decode().catch(() => {});
    }
  });
}

test.describe("Landing Visual Acceptance Suite (PLAN_LANDING_FIX Fill-Screen)", () => {
  test("1920x1080 hero covers screen: heroSection equals viewport minus banner; H1 and depth ruler inside", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/");
    await waitForHeroImage(page);

    const heroSection = page.locator('section[aria-label="Hero Introduction"]');
    await expect(heroSection).toBeVisible({ timeout: 15000 });

    const banner = page.locator('[class*="bannerWrapper"]');
    const bannerBox = await banner.boundingBox();
    const heroBox = await heroSection.boundingBox();

    expect(heroBox).not.toBeNull();
    expect(bannerBox).not.toBeNull();

    if (heroBox && bannerBox) {
      // Hero bounding box equals viewport width
      expect(Math.round(heroBox.width)).toBe(1920);
      // Hero bounding box height equals viewport height minus banner height (within 1px rounding)
      expect(Math.abs(heroBox.height - (1080 - bannerBox.height))).toBeLessThanOrEqual(1.5);
    }

    // 1. Assert H1 inside heroSection and on one line
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible();
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box && heroBox) {
      expect(h1Box.y).toBeGreaterThanOrEqual(heroBox.y);
      expect(h1Box.y + h1Box.height).toBeLessThanOrEqual(heroBox.y + heroBox.height);

      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    // 2. Assert DepthRuler inside heroSection
    const ruler = page.getByRole("region", { name: /Depth scale/i });
    await expect(ruler).toBeVisible();
    const rulerBox = await ruler.boundingBox();
    expect(rulerBox).not.toBeNull();
    if (rulerBox && heroBox) {
      expect(rulerBox.y).toBeGreaterThanOrEqual(heroBox.y);
      expect(rulerBox.y + rulerBox.height).toBeLessThanOrEqual(heroBox.y + heroBox.height);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1920x1080.png",
      fullPage: false,
    });
  });

  test("1897x1000 screen fit: fills screen edge to edge without side gaps", async ({ page }) => {
    await page.setViewportSize({ width: 1897, height: 1000 });
    await page.goto("/");
    await waitForHeroImage(page);

    const heroSection = page.locator('section[aria-label="Hero Introduction"]');
    await expect(heroSection).toBeVisible({ timeout: 15000 });
    const heroBox = await heroSection.boundingBox();
    expect(heroBox).not.toBeNull();
    if (heroBox) {
      expect(Math.round(heroBox.width)).toBe(1897);
    }

    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible();
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box) {
      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1897x1000.png",
      fullPage: false,
    });
  });

  test("1440x900 hero covers screen: heroSection equals viewport minus banner; H1 and depth ruler inside", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");
    await waitForHeroImage(page);

    const heroSection = page.locator('section[aria-label="Hero Introduction"]');
    await expect(heroSection).toBeVisible({ timeout: 15000 });

    const banner = page.locator('[class*="bannerWrapper"]');
    const bannerBox = await banner.boundingBox();
    const heroBox = await heroSection.boundingBox();

    expect(heroBox).not.toBeNull();
    expect(bannerBox).not.toBeNull();

    if (heroBox && bannerBox) {
      expect(Math.round(heroBox.width)).toBe(1440);
      expect(Math.abs(heroBox.height - (900 - bannerBox.height))).toBeLessThanOrEqual(1.5);
    }

    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible();
    const h1Box = await h1.boundingBox();
    expect(h1Box).not.toBeNull();
    if (h1Box && heroBox) {
      expect(h1Box.y).toBeGreaterThanOrEqual(heroBox.y);
      expect(h1Box.y + h1Box.height).toBeLessThanOrEqual(heroBox.y + heroBox.height);

      const fontSizeStr = await h1.evaluate((el) => window.getComputedStyle(el).fontSize);
      const fontSize = parseFloat(fontSizeStr);
      expect(h1Box.height).toBeLessThan(1.3 * fontSize);
    }

    const ruler = page.getByRole("region", { name: /Depth scale/i });
    await expect(ruler).toBeVisible();
    const rulerBox = await ruler.boundingBox();
    expect(rulerBox).not.toBeNull();
    if (rulerBox && heroBox) {
      expect(rulerBox.y).toBeGreaterThanOrEqual(heroBox.y);
      expect(rulerBox.y + rulerBox.height).toBeLessThanOrEqual(heroBox.y + heroBox.height);
    }

    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1440x900.png",
      fullPage: false,
    });
  });

  test("1366x768 common laptop viewport captures screenshot", async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await page.goto("/");
    await waitForHeroImage(page);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1366x768.png",
      fullPage: false,
    });
  });

  test("1024x768 tablet viewport captures screenshot", async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto("/");
    await waitForHeroImage(page);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 15000 });
    await page.screenshot({
      path: "../docs/landing/screenshots/visual-1024x768.png",
      fullPage: false,
    });
  });

  test("390x844 mobile viewport: no horizontal scroll, H1 on one line, CTA visible", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await waitForHeroImage(page);

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
