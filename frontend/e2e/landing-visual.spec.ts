import { test, expect } from "@playwright/test";
import mockMeta from "../src/mocks/meta.json";

for (const theme of ["light", "dark"] as const) {
  for (const viewport of [
    { width: 1920, height: 1080 },
    { width: 1440, height: 900 },
    { width: 1024, height: 768 },
    { width: 768, height: 1024 },
    { width: 390, height: 844 },
    { width: 360, height: 800 },
  ]) {
    test(`${theme} ${viewport.width}: full-screen scene and reachable content`, async ({
      page,
    }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(
        (value) => localStorage.setItem("theme", value),
        theme,
      );
      await page.goto("/");
      await expect(page.getByRole("heading", { level: 1 })).toHaveText(
        "A New Depth of Intelligence — Beneath the Surface",
      );
      const hero = page.locator('section[aria-label="Hero Introduction"]');
      const box = await hero.boundingBox();
      const banner = await page
        .locator('[class*="bannerWrapper"]')
        .boundingBox();
      expect(box).not.toBeNull();
      expect(banner).not.toBeNull();
      expect(Math.abs(box!.width - viewport.width)).toBeLessThanOrEqual(1);
      expect(
        Math.abs(box!.height - (viewport.height - banner!.height)),
      ).toBeLessThanOrEqual(1.5);
      await expect(
        page.getByRole("link", { name: "Explore the Ocean", exact: true }),
      ).toBeInViewport();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.getByRole("button", { name: "Pause ambient motion" }).click();
      await page.screenshot({
        path: test.info().outputPath(`${theme}-${viewport.width}.png`),
        fullPage: true,
      });
    });
  }
}

test("theme follows system, toggles, persists and reaches Explorer", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/");
  await expect(page.locator("html")).toHaveClass("light");
  await page.getByRole("button", { name: "Toggle dark/light theme" }).click();
  await expect(page.locator("html")).toHaveClass("dark");
  await page.reload();
  await expect(page.locator("html")).toHaveClass("dark");
  await page
    .getByRole("link", { name: "Explore the Ocean", exact: true })
    .click();
  await expect(page).toHaveURL(/\/explore/);
  await expect(page.locator("html")).toHaveClass("dark");
  await expect(page.locator("#explorer-heading")).toBeFocused();
});

test("water, reflections and fish animate; pause freezes rendered pixels", async ({
  page,
}) => {
  await page.goto("/");
  const scene = page.locator('[data-scene-ready="true"]');
  await expect(scene).toBeVisible({ timeout: 20000 });
  const canvas = page.locator("canvas[data-ocean-canvas]");
  await expect(canvas).toHaveAttribute("data-running", "true");
  // Sample clear water, surface and fish regions independently, excluding HTML.
  const before = await canvas.screenshot();
  await page.waitForTimeout(1100);
  const after = await canvas.screenshot();
  expect(before.equals(after)).toBe(false);
  await page.getByRole("button", { name: "Pause ambient motion" }).click();
  await expect(canvas).toHaveAttribute("data-running", "false");
  await page.waitForTimeout(800);
  const frozen = await canvas.screenshot();
  await page.waitForTimeout(500);
  expect(frozen.equals(await canvas.screenshot())).toBe(true);
  await page.getByRole("button", { name: "Resume ambient motion" }).click();
  await expect(canvas).toHaveAttribute("data-running", "true");
});

test("reduced motion retains artwork and disables ambient animation", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Pause ambient motion" }),
  ).toBeDisabled();
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(0);
  await expect(page.locator('[class*="staticFish"] img')).toHaveCount(18);
  await page
    .getByRole("link", { name: "Explore the Ocean", exact: true })
    .click();
  await expect(page).toHaveURL(/\/explore/);
  await expect(page.locator("#explorer-heading")).toBeFocused();
});

test("changing reduced-motion preference disposes and recreates the scene", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
    timeout: 20000,
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(0);
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
    timeout: 20000,
  });
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(1);
});

test("unavailable WebGL retains static scene and usable CTA", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      ...args: Parameters<typeof original>
    ) {
      if (String(args[0]).includes("webgl")) return null;
      return original.apply(this, args);
    } as typeof original;
  });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.waitForTimeout(1200);
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(0);
  await expect(page.locator('[class*="staticFish"]')).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Explore the Ocean", exact: true }),
  ).toBeVisible();
});

test("context loss returns to the poster without losing navigation", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
    timeout: 20000,
  });
  await page
    .locator("canvas[data-ocean-canvas]")
    .evaluate((canvas) =>
      canvas.dispatchEvent(new Event("webglcontextlost", { cancelable: true })),
    );
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(0);
  await expect(page.locator('[class*="staticFish"]')).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Explore the Ocean", exact: true }),
  ).toBeVisible();
});

test("failed enhancement texture retains the loaded poster", async ({
  browser,
}) => {
  // Block SW interception so the deliberate network failure reaches TextureLoader.
  const context = await browser.newContext({ serviceWorkers: "block" });
  const page = await context.newPage();
  await page.route("**/api/v1/meta", (route) =>
    route.fulfill({ json: mockMeta }),
  );
  await page.emulateMedia({ colorScheme: "light" });
  await page.route("**/landing/ocean/dark-*.webp", (route) => route.abort());
  await page.goto("/");
  const poster = page.locator('img[src="/landing/ocean/light-1600.webp"]');
  await expect(poster).toBeVisible();
  await expect
    .poll(() =>
      poster.evaluate((image) => (image as HTMLImageElement).naturalWidth),
    )
    .toBeGreaterThan(0);
  await page.waitForTimeout(1200);
  await expect(page.locator("canvas[data-ocean-canvas]")).toHaveCount(0);
  await expect(
    page.getByRole("link", { name: "Explore the Ocean", exact: true }),
  ).toBeVisible();
  await context.close();
});

test("ambient rendering pauses outside the viewport and resumes on return", async ({
  page,
}) => {
  await page.goto("/");
  const canvas = page.locator("canvas[data-ocean-canvas]");
  await expect(canvas).toHaveAttribute("data-running", "true", {
    timeout: 20000,
  });
  await page.getByRole("contentinfo").scrollIntoViewIfNeeded();
  await expect(canvas).toHaveAttribute("data-running", "false");
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(canvas).toHaveAttribute("data-running", "true");
});
