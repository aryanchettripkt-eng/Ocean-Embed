# Landing page — implementation plan (for the coding agent)

**Status:** v1 (2026-10-24), written by Claude for Aryan. **Owner:** Aryan. **Implementer:** coding agent.
**Reference design:** `docs/landing/reference.png` (Aryan's mockup). It shows the look, not the final pixels: its text is baked into the image and its depth ruler is wrong (it says "500 m" twice).
**Governs:** everything below `frontend/src/components/landing/`, `frontend/src/app/welcome/` and `frontend/public/landing/`. Where this file and `docs/PRD_FRONTEND.md` disagree, PRD_FRONTEND wins; write the conflict in `docs/OPEN_QUESTIONS.md` and stop (AGENTS.md §2).

Read first: `AGENTS.md` (all hard rules, especially 1, 3, 5, 6, 7 and 10), `docs/PRD_FRONTEND.md` §0, §2, §4, §7, then this file.

---

## 0. Safety rules for this task (read before touching anything)

Two other things are running on Aryan's machine at the same time:
- **Another agent is finishing P2/P2b work in `E:\meta`** (frontend map code, backend, docs).
- **The ML data download is running** in its own terminal and writing into `E:\meta\ml\data\raw` (tens of GB, several hours).

So:
1. **Work only in a separate git worktree**, never in `E:\meta` itself. Aryan creates it (AGENTS rule 10: agents do not create branches):
   ```
   cd E:\meta
   git worktree add ..\meta-landing -b frontend/landing-page
   ```
   Everything below happens inside `E:\meta-landing\frontend`. Never run `git checkout`, `git switch`, `git stash`, `git reset`, `git clean` or `git worktree remove` in `E:\meta`; switching branches there would break the other agent's work.
2. **Do not touch** `ml/`, `backend/`, `fixtures/`, `.env` files, `envfiles/`, `.pnpm-store/`, or any running terminal. Do not run Python. Do not run `pnpm store prune`, global cleans, or anything that deletes caches outside `E:\meta-landing`.
3. **Run the dev server on port 3001** (`pnpm dev -p 3001`) with `NEXT_PUBLIC_USE_MOCKS=true`, so it never clashes with the other agent's server on 3000 and never calls the real backend while developing. For Playwright, set `PLAYWRIGHT_TEST_BASE_URL=http://localhost:3001` and start the server yourself; do not edit `playwright.config.ts`.
4. **Network and disk:** the download needs the bandwidth. `pnpm install` in the worktree is fine (pnpm hard-links from the existing store). No other large downloads. Keep new files in the worktree; the landing assets must total under 1 MB.
5. **Phase A creates new files only** (list in §3). It must not edit any existing file. Phases B and C edit shared files and wait for Aryan's go-ahead (see §8).
6. **Commit locally on `frontend/landing-page`. Never push, never open a PR.** Report to Aryan and ask (AGENTS rule 10).

---

## 1. Goal and success criteria

A landing page in the reference's art style (cream paper, ink lines, pixel display type, Japanese-woodblock coast, cutaway ocean) that explains OceanScope AI in one look and sends people into the Explorer with a "dive" transition.

Done when:
- It matches the reference's composition and mood at 1440, 1024, 768 and 390 px wide.
- **Performance** (Lighthouse, production build, mobile preset): Performance ≥ 90, Accessibility ≥ 95, Best Practices ≥ 95, SEO ≥ 95. LCP ≤ 2.0 s, CLS ≤ 0.02, TBT ≤ 150 ms.
- **Weight:** landing route's own JS ≤ 15 KB gzipped on top of the shared Next/React runtime (check `next build` output); no MapLibre, Recharts or chart code in the landing bundle; images ≤ 350 KB total on desktop, ≤ 180 KB on a phone; fonts ≤ 2 files, ≤ 60 KB total.
- Works without JavaScript (the CTA is a real link; all text is server-rendered HTML).
- No invented data anywhere (AGENTS rule 1): no temperatures, metrics or claims typed into the page.
- The synthetic banner shows when `/meta.synthetic` is true (PRD_FRONTEND §0 rule 4).

---

## 2. Design system for the landing page

Define as CSS custom properties on a `.landing` wrapper in `landing.module.css` (do not touch `globals.css` or `tailwind.config.ts` in Phase A):

| Token | Value | Use |
|---|---|---|
| `--paper` | `#EFE6CF` | page background |
| `--paper-shade` | `#E4D8BA` | cards, button hover |
| `--ink` | `#1E2A2B` | text, outlines (contrast on paper ≈ 12:1) |
| `--ink-soft` | `#4A5654` | secondary text (≥ 7:1 on paper) |
| `--sea-deep` | `#1F4E57` | water below the band |
| `--sea-surface` | `#2F6B73` | waves |
| `--foam` | `#F6F1E2` | wave crests |
| `--sun` | `#F2B66D` | sun disc only |

- **Type:** headline `Pixelify Sans` (600) for "OCEANSCOPE AI"; everything else `IBM Plex Mono` (400/500). Load both with `next/font/google` (self-hosted at build, no runtime request to Google), `display: "swap"`, Latin subset only, and `adjustFontFallback` on so swapping causes no layout shift.
- **Scale:** H1 `clamp(48px, 9vw, 128px)`, tagline `clamp(22px, 3.2vw, 44px)`, body 16–18 px, callouts 13–14 px uppercase with 0.06em letter spacing.
- **Paper grain:** one 128×128 noise PNG (≤ 6 KB) tiled with `mix-blend-mode: multiply` at 25% opacity over the whole page, `pointer-events: none`.
- **Ink details:** callouts use the bracket style from the reference (`⊤ + ⊥` drawn with CSS borders, not glyphs). The CTA is an ink-outlined box with a 3 px offset "print" shadow; on hover the shadow shrinks to 1 px and the box moves 2 px (a pressed-stamp feel).
- **Theme:** the landing page is always in the paper palette, in light and dark system mode (the art is the brand). The Explorer keeps its own theming. (Open question for Aryan in §9.)

---

## 3. Structure (Phase A: new files only)

```
frontend/
├── public/landing/
│   ├── sky-coast.avif / .webp        # sky, clouds, mountains, coast, temple, sun — no text, no sea
│   ├── tree.avif / .webp             # top-right branches, transparent background
│   ├── satellite.svg                 # or .webp with transparency
│   ├── argo-float.svg
│   ├── seabed.avif / .webp           # bottom-left coral and rocks, transparent
│   ├── grain.png                     # paper noise tile
│   ├── ocean-band-placeholder.webp   # illustrative band (Phase A only, see §5)
│   └── og.png                        # 1200×630 social preview
└── src/
    ├── app/welcome/page.tsx          # Phase A route; moves to "/" in Phase B
    ├── components/landing/
    │   ├── Landing.tsx               # server component: page layout
    │   ├── landing.module.css
    │   ├── HeroText.tsx              # H1, tagline, lede, CTA (server)
    │   ├── HeroScene.tsx             # layered art + waves + callouts (server)
    │   ├── Waves.tsx                 # inline SVG wave paths (server, CSS-animated)
    │   ├── Callout.tsx               # bracket callout (server)
    │   ├── DepthRuler.tsx            # depth ticks (client, reads /meta; see §5)
    │   ├── ExploreLink.tsx           # client: prefetch, warm-up, dive transition
    │   ├── DiveOverlay.tsx           # client: the water-rise + dither overlay
    │   ├── HowItWorks.tsx            # 3 short steps below the fold (server)
    │   └── LandingFooter.tsx         # data credits, links (server)
    └── tests/landing/                # vitest + Testing Library
frontend/e2e/landing.spec.ts           # Playwright
```

Rules: server components by default; only `DepthRuler`, `ExploreLink` and `DiveOverlay` are `"use client"`. Import existing code read-only (`lib/api/hooks.ts`, `lib/api/client.ts`, `lib/format.ts`, `components/common/Banner.tsx`); do not modify it.

---

## 4. Art assets (Aryan provides; the agent prepares)

The reference is a single flattened image with text baked in, so it cannot be used as-is. Aryan regenerates or cuts these layers from the same artwork (same palette, same light), each **without any text**:

| Layer | Size (desktop master) | Background | Notes |
|---|---|---|---|
| sky-coast | 2400×1100 | opaque paper | sky, birds, clouds, mountains, coast, temple, sun; leave the sea area empty (paper colour) |
| tree | 900×700 | transparent | branches top-right |
| seabed | 1000×500 | transparent | coral, rocks, fish bottom-left |
| satellite | 400×300 | transparent | SVG preferred |
| argo-float | 120×420 | transparent | SVG preferred |

Agent's job: convert with `sharp` (dev-only, one-off script outside `src/`, not committed as a dependency; or Squoosh by hand) to AVIF (q≈50) and WebP (q≈70) at 1x and 2x widths `[640, 1024, 1600, 2400]`, keep each desktop layer ≤ 120 KB AVIF, and serve through `next/image` with explicit `sizes`. If the layers are not ready, use labelled grey placeholder boxes and carry on; never ship the reference PNG (3 MB).

Licensing: Aryan confirms the artwork is his to use; credit the artist in the footer if it is not his own.

---

## 5. The hero, piece by piece

Layout: one full-viewport section (`min-height: 100svh`, `max-height: 1000px`), CSS grid with the text column on the left (desktop) and the scene filling the section behind it.

1. **Header row:** logo mark (inline SVG wave), "OCEANSCOPE AI", nav `HOME · EXPLORE · DATA · LEARN · ABOUT` as real links (`/`, explorer route, `/validation`, `/about#method`, `/about`). In Phase A the Explore link points to `/` (the current Explorer). Active item underlined. On phones it collapses to the logo plus a `<details>` menu (no JS).
2. **Text:** H1 "OCEANSCOPE AI", tagline "BENEATH THE SURFACE", lede "A quieter way to see the hidden temperature of the ocean.", CTA "EXPLORE THE OCEAN →" (`<a href>` via `ExploreLink`). The H1 is the LCP element: plain text, no image, no animation on first paint.
3. **Scene layers** (absolutely positioned, `aria-hidden="true"` except one `alt` summary on the sky layer): sky-coast (priority, `fetchPriority="high"`), tree, satellite with a dashed orbit (inline SVG path), waterline waves (`Waves.tsx`), the underwater band, seabed, ARGO float, fish.
4. **Callouts** as HTML (`<figure>`/`<figcaption>`), positioned with CSS so they stay sharp and readable at every size: "SATELLITES OBSERVE THE SURFACE", "OCEAN CURRENTS MOVE HEAT", "ARGO FLOATS PROVIDE REAL OBSERVATIONS", "TEMPERATURE AT DEPTH".
5. **Underwater band (Phase A: illustrative).** A painted warm-to-cool band with dashed contour lines, as in the reference. Because it is art, it carries **no numbers and no axis**, and no text calling it data. Phase C replaces it with the real `/section` field (§7).
6. **Depth ruler.** Ticks and labels come from `meta.depths_m` via `useMeta()`, formatted with `formatDepth`, showing a subset (≈ 0, 100, 300, 500, 1000 m: nearest levels to these targets). The vertical position uses the same √depth scale as the 3D view (PRD_FRONTEND §3.1b). While `/meta` loads, show the tick marks without labels; never hardcode depth values.
7. **Synthetic banner:** render `<Banner type="synthetic" />` at the very top of the page (it reads `/meta`). In Phase A the global header also renders it; that is fine on `/welcome`.

### Motion (CSS only, no animation library)
- Waves: two SVG paths translating horizontally in opposite directions (`transform` only), 18 s and 26 s loops.
- Satellite: travels along its dashed orbit with `offset-path`, 40 s loop; tiny rotation.
- ARGO float: bobs ±6 px, 6 s ease-in-out.
- Scroll parallax with `animation-timeline: scroll()` (sky slowest, tree and seabed fastest), wrapped in `@supports (animation-timeline: scroll())`; no JS scroll listeners.
- `@media (prefers-reduced-motion: reduce)`: all of the above off.
- Only animate `transform` and `opacity`; mark animated layers `will-change: transform` only while visible.

### Below the fold
- **How it works:** three steps in the same style: "Satellites see the surface" → "An AI model learns the ocean below" → "You explore 15 depths, every day". Copy only, no numbers.
- **Accuracy teaser (optional):** a single line that links to `/validation`. Any number shown must come from `/headline` or `/metrics` at runtime; if the call fails, show the link without a number.
- **Footer:** data credits (same sources and licence lines as `/about`), GitHub link, "Built for <hackathon name>" placeholder for Aryan to fill.

---

## 6. "Explore the ocean": warm-up, prefetch and the dive transition

`ExploreLink.tsx` (client) wraps a real `<a href>`:
1. **Warm the backend on page load:** in `requestIdleCallback` (fallback `setTimeout 1500`), call the existing `useMeta()` prefetch path (`queryClient.prefetchQuery(["meta"], apiClient.getMeta)`). This wakes the Render backend while people read the hero, hiding its cold start.
2. **On hover/focus/touchstart of the CTA:** `router.prefetch(explorerHref)`, and `import("maplibre-gl")` so the map library is in the cache. (Phase C adds prefetching `/map` for the default date and depth.)
3. **On click** (left click without modifier keys; otherwise let the browser handle it):
   - If `prefers-reduced-motion` or `!document.startViewTransition`: navigate normally.
   - Else run the dive: `DiveOverlay` animates water rising from the waterline to the top of the screen (clip-path on a `--sea-deep` layer, 450 ms, ease-in), then a **pixel-dither dissolve**: a mask sprite of 16 Bayer-matrix threshold steps (one 16-frame PNG strip, ≤ 4 KB, `image-rendering: pixelated`) stepped with `animation-timing-function: steps(16)` over 350 ms. Then `router.push(explorerHref)`.
   - Phase A: the Explorer appears after the overlay with a plain fade (no shared-element morph yet).
   - Phase B adds a real View Transition across routes (§8), so the dither dissolve reveals the Explorer instead of covering the swap.
4. Total time from click to Explorer visible: ≤ 1.1 s. If the route is not ready within 1.5 s, the overlay shows "Waking up the ocean…" in the landing style instead of freezing.
5. Keyboard: Enter on the CTA triggers the same path; focus moves to the Explorer's main heading after navigation.

---

## 7. Phase C: real data in the hero (after `/section` exists, PHASES P2b-B1)

- Replace the illustrative band with `OceanBand.tsx` (client, lazy, rendered after the H1 paints): fetch `/section?lat=<row nearest 15°N from meta>&field=pred` for the latest available date; draw it on a `<canvas>` with the existing `thermal` LUT (`lib/colormaps.ts`), using `meta.color_ranges`, ordered (Bayer 4×4) dithering to match the art, and the 2 °C contours as dashed foam-coloured lines.
- Label it small under the ruler: "Model prediction · {date} · section at {lat}°N" from the response, plus "Synthetic demo data" when `synthetic` is true.
- Until the data arrives (or if it fails), the illustrative band stays. No layout shift: the canvas sits exactly over it.
- This band is the same section the 3D view uses as a wall, so in Phase B/C the transition can morph it into the 3D block (`view-transition-name: ocean-section` on both).

---

## 8. Phases and when each may start

| Phase | What | Edits shared files? | Starts when |
|---|---|---|---|
| **A** | Everything above at `/welcome`: hero, assets, motion, how-it-works, footer, ExploreLink with the overlay dive, tests | No (new files only) | Now, in the worktree |
| **B** | Make the landing page `/`: route groups `app/(marketing)/page.tsx` (landing) and `app/(app)/explore/page.tsx` (current Explorer), `(app)/layout.tsx` holding `Header`; root `layout.tsx` keeps only `<html>`, fonts and `Providers`; Header "Explorer" link → `/explore`; redirect old shared links (`/?date=…` → `/explore?date=…`) in `middleware.ts`; update E2E paths; real cross-route View Transitions (React `unstable_ViewTransition` behind Next's `experimental.viewTransition`, or the `next-view-transitions` package, ~2 KB; pick one and say why in the commit) | Yes: `app/page.tsx` (move), `app/layout.tsx`, `Header.tsx`, `providers.tsx`, e2e, `next.config.mjs` | After the other agent's P2/P2b-F0 work is merged to `main` and Aryan says go; rebase `frontend/landing-page` on `main` first |
| **C** | Real `/section` band (§7) and shared-element morph into the 3D wall | Only landing files | After P2b-B1 (`/section`) and P2b-F1 (3D view) are merged |

**Phase B must also fix this bug in `providers.tsx`:** `mockReady` starts as `false` even when mocks are off, so the server-rendered HTML of every page is the "Initializing mock service…" spinner until JavaScript runs. That ruins LCP and SEO for the landing page. Start it as `process.env.NEXT_PUBLIC_USE_MOCKS !== "true"`.

---

## 9. Open questions for Aryan (answer before Phase B)
1. Landing page always in the paper palette, also in dark mode? (Recommended: yes.)
2. Hackathon name and team credits for the footer.
3. Who produces the art layers in §4, and by when? Phase A proceeds with placeholders until then.

---

## 10. Tests and checks (required before reporting)

- **Vitest:** H1 text renders; CTA is an `<a>` with the explorer href; nav links have hrefs; `DepthRuler` renders labels from mocked `meta.depths_m` and none while loading; synthetic banner appears when `synthetic: true` and not when false; `ExploreLink` skips the overlay when `prefers-reduced-motion` matches or `startViewTransition` is missing.
- **Playwright** (`e2e/landing.spec.ts`, mocks on, port 3001): `/welcome` loads with no console errors; the CTA navigates to the Explorer and the map container appears; with JS disabled the CTA still navigates; at 390 px no horizontal scroll.
- **Manual:** Lighthouse mobile on `pnpm build && pnpm start -p 3001`, attach the scores; `next build` route table showing the landing route's JS size; screenshots at 1440, 1024, 768, 390.
- **Area checks:** `pnpm typecheck && pnpm lint && pnpm test && pnpm build` (AGENTS §6), all green, outputs pasted in the report.
- Update `docs/PROGRESS.md` **in the worktree** with one row (branch `frontend/landing-page`). Do not edit `E:\meta\docs` directly.

## 11. Report format
What was built, files added, check outputs, Lighthouse scores, bundle sizes, screenshots, anything uncertain, and the question "OK to push `frontend/landing-page`?" (AGENTS rule 10).
