# Landing hero fix: analysis and implementation plan (for Gemini)

**Status:** v1, 2026-10-24, written by Claude for Aryan. **Owner:** Aryan. **Implementer:** Gemini (coding agent).
**Supersedes:** PLAN_LANDING.md §4 (art assets), §5 items 3–5 (scene layers, callouts, band) and §7 (real band in the hero). The rest of PLAN_LANDING.md still applies.
**Inputs:** `docs/landing/reference.png` (1673×940, the target) and Aryan's localhost screenshot of the current build at 1897×1000, taken 2026-10-24 13:05.

Read `AGENTS.md` first. The hard rules still hold: no invented data, don't touch `ml/`, `backend/`, `.env` files or running terminals, dev server on port 3001 with mocks on, commit locally and never push without Aryan's OK.

---

## Part 1. What's wrong: current build vs reference

Percentages are relative to the hero image (x from the left, y from the top).

### 1.1 Overall composition

| | Reference | Current build |
|---|---|---|
| Structure | One continuous painting. Sky, coast, sea surface, underwater and seabed flow into each other with no seams. | Five separate rectangles stacked on paper: a coast crop, a tree crop, a teal strip, a canvas band and a seabed crop. Hard edges are visible everywhere. |
| Fit | The whole scene, including the 1000 m ruler, fits on one 16:9 screen. | The hero is taller than the viewport. The band's bottom and the "1000 m" label fall below the fold at 1897×1000. |
| Frame | Thin dark ink border and a soft vignette, like a print. | None. |
| Balance | Text is top-left, the scene sweeps from the top-right down to the bottom-left, and the warm plume runs diagonally across the water. | Left 55 % of the sky is blank paper and everything sits in horizontal bands. Nothing is diagonal. |

### 1.2 Header

| | Reference | Current |
|---|---|---|
| Background | No bar. The header sits on the painting's paper. | A separate shaded bar (`--paper-shade`) with a bottom rule, boxed to 1440 px, so it doesn't line up with the full-width art. |
| Logo | Circular "wave" emblem about 54 px, at x 8 %. | Small two-line squiggle (28×20). |
| Nav | HOME, EXPLORE, DATA, LEARN, ABOUT spread from x 33 % to 65 %, so it stays in the left two-thirds, clear of the tree. | Pushed to the far right (x 68–86 %). |
| Tree | Starts at the very top edge (y 0) and hangs over the nav area. | Starts below the header bar (y ≈ 103 px), so it looks clipped. |
| Banner | None (it's a mockup). | Bright orange synthetic-data banner across the top. It's required by PRD_FRONTEND §0 rule 4 and stays, but it clashes with the palette (see 2.7). |

### 1.3 Text block

| | Reference | Current |
|---|---|---|
| H1 | One line, x 9.4–60.5 %, y 18–30 %, about 8 % of the image width (≈135 px at 1673). It runs over the sky. | **Two lines** ("OCEANSCOPE" / "AI"), capped at 96 px inside a 580 px column. |
| Tagline | About 3.2 % of width (≈54 px), y 32.5–37 %. | ≈36 px. Too small next to the H1. |
| Lede | Two lines, about 1.2 % of width (≈20 px), y 40–45 %. | One long 16 px line, light and hard to read. |
| CTA | At x 11–30 %, y 50–55.5 %, just above the waterline. Thin double-rule box. | Similar box with a heavier shadow, sitting well above the water because the water starts lower. |

### 1.4 Sky and coast

| | Reference | Current |
|---|---|---|
| Coast | Misty mountains from x 54 % fade into the paper on the left. The headland, pagoda and rocks are at the right. | The coast is a **rectangle** pasted at x 55–100 %, y 37–62 %. Its left and top edges are hard cuts, and pixel-cloud blocks are sliced flat. |
| Sun | A round disc at x 85 %, y 38 %, half hidden behind the hill. | **Two suns**: a clipped square block from the coast crop and a second one from the tree crop. |
| Pagoda | One. | **Two, stacked** (one from each crop). |
| Birds | Five, in a loose V by the H1. | Three, from the coast crop. |
| Satellite | One, about 14 % wide, at x 70 %, y 23 %, with a dashed orbit arcing down to the right past its callout. | **Two**: the new satellite image, plus a loose solar panel baked into `tree.webp`. The orbit is a short, mostly hidden curve. |
| Satellite callout | One, at x 73 %, y 33.5–41 %. | **Two, overlapping**: the HTML callout and the one baked into `tree.webp`. |

### 1.5 Sea surface

| | Reference | Current |
|---|---|---|
| Open sea | A strip of textured blue water with whitecaps from the horizon (y 56 %) down to the waterline (y ≈ 64 %), running across the full width behind the coast. | Missing. There's a 30 px flat teal strip with one dashed line, and the only real sea is inside the coast crop. |
| Waterline | Big woodblock wave crests with foam. | Thin flat line. |

### 1.6 Underwater

| | Reference | Current |
|---|---|---|
| Water | One continuous painted teal body. | Flat dark slate with no texture. |
| Temperature band | A painted, flowing warm plume (orange to amber to teal), diagonal from x 50 % at the surface to x 90 % at depth, edged with fine dashed contours. | Real-data canvas: navy-green-yellow-orange **horizontal stripes** (Explorer cmocean "thermal" colours), with straight dashed lines. It's broken into **two separate panels** by a dark gap at x 49–78 % (India is land in the 12.5° N mock section, and nulls are painted dark). The left third is covered by the seabed image's opaque teal. |
| Currents | Two dashed arrows sweeping down-right through the plume, with the label at x 43 %, y 70 %. | One short arrow near the top, pointing off into empty water. The label sits on the striped band. |
| ARGO float | Large (≈235 px tall, about 25 % of image height), at x 26 %, y 67–92 %, standing over the reef. Callout to its right. | Tiny (≈60 px), floating inside the striped band at y ≈ 78 %. |
| Depth ruler | Small and inked: a vertical foam line with five ticks at x 91–96 %, y 70–91 %, labels in plain foam text. "Temperature at depth" callout just left of it. | Full-height dashed line at the far right edge, labels in dark chips ("0.5 m", "100 m", "300 m", "500 m"). 1000 m is below the fold. |
| Seabed | Coral, rocks and fish fill the bottom-left third and blend into the water. | Blurred rectangle with a **white diagonal fade**, visible square patches where the float was erased, and a leftover float fragment. |

### 1.7 Root causes

1. **The art was cut out of `reference.png`.** PLAN_LANDING §4 says not to do this. The crops still carry baked text and objects (callout, satellite panel, sun, pagoda), have hard rectangular edges, and the "erased" areas have inpainting artefacts.
2. **The scene was built as stacked horizontal strips** with fixed pixel sizes and `flex` layout. The reference is one image with things placed diagonally across it. No amount of nudging strips will reproduce that.
3. **The real `/section` band replaced the painted plume.** A straight latitude line crosses land (India), the synthetic data is flat left to right, and the Explorer palette doesn't match the art. This was a flaw in PLAN_LANDING §7 itself, not only the code.
4. **The text is constrained to a column** (580 px, H1 capped at 96 px), so the H1 wraps.
5. **Nothing scales together.** Art, text and callouts use unrelated units (vw, px, %), so they drift apart at every width.

---

## Part 2. The fix

### 2.0 Approach

Treat the hero as **one illustrated plate with HTML on top**, like a poster:

- One clean, text-free version of the whole reference scene is the background (step 1).
- It sits in a fixed-aspect **stage** (1673:940) that always fits on one screen.
- Every piece of HTML (header, text, callouts, ruler) is positioned in **percentages of the stage** and sized in **container-query units** (`cqw`), so it scales exactly with the art.
- The real-data band leaves the hero.

Motion (separate satellite and float layers, parallax) is a later step, once the static page matches.

### 2.1 Art (Aryan, about 15 minutes, before Gemini starts step 2.3)

Make `scene-clean.png` at **2400×1349** (same 16:9 aspect as the reference) from `reference.png`, using Muse Image (edit mode), or any inpainting tool such as Photoshop Generative Fill or Gemini image edit. Prompt:

> Remove every piece of text and interface from this illustration, and fill those areas seamlessly with the surrounding art: the logo and "OCEANSCOPE AI" top-left, the navigation links, the large title "OCEANSCOPE AI", "BENEATH THE SURFACE", the sentence under it, the "EXPLORE THE OCEAN" button, all four bracket labels with their text ("SATELLITES OBSERVE THE SURFACE", "OCEAN CURRENTS MOVE HEAT", "ARGO FLOATS PROVIDE REAL OBSERVATIONS", "TEMPERATURE AT DEPTH"), and the depth scale on the right with its numbers and tick marks. Keep the satellite, its dashed orbit, the birds, sun, pagoda, coast, tree, waves, the ARGO float, the dashed current arrows, the warm temperature plume and its dashed contours, the fish and the coral exactly as they are. Same style, same colours, same resolution, no new objects.

Check it at 100 % zoom: no leftover letters, no smears where the title was, and the ruler area is plain water. Save it to `frontend/public/landing/src/scene-clean.png` (source, not served). If the image tool adds a border or changes the size, crop or pad back to 16:9.

Optional, later: a portrait crop for phones (see 2.6). Not needed for v1.

### 2.2 Clean-up (Gemini)

Delete from `public/landing/`: `sky-coast.webp`, `tree.webp`, `seabed.webp`, `satellite.webp`, `satellite.svg`, `argo-float.webp`, `argo-float.svg`, `ocean-band-placeholder.webp`.

Remove from the hero (`HeroScene.tsx`): the sky, tree, satellite, orbit SVG, `Waves`, the currents SVG, `OceanBand`, seabed and float layers.

Delete `Waves.tsx` and `ThermalBand.tsx`. Move `OceanBand.tsx` out of the hero (see 2.5).

Keep `Callout.tsx` (the left-bracket version is right), `DepthRuler.tsx` (logic is right; restyle it), `ExploreLink.tsx`, `DiveOverlay.tsx`, `HowItWorks.tsx`, `LandingFooter.tsx` and `fonts.ts`.

### 2.3 Assets (Gemini)

Using `scripts/generate-landing-assets.cjs` (dev-only `sharp`, not added as a dependency), produce `public/landing/scene-{640,1024,1600,2400}.{avif,webp}`. AVIF q≈50, WebP q≈72. The 2400 AVIF must be ≤ 300 KB; the 1024 one ≤ 120 KB. Render with `next/image`: `fill`, `priority`, `fetchPriority="high"`, `sizes="100vw"`, `alt="Woodblock-style illustration: a satellite above a coast, and a cutaway of the ocean below with an ARGO float and a warm layer near the surface"`.

### 2.4 Stage layout (Gemini)

Restructure the hero:

```
<section class="heroSection">       full width, paper background, centres the stage
  <div class="stage">               aspect-ratio 1673/940, container-type: inline-size
    <Image scene />                 fill, object-fit: cover
    <header class="stageHeader">    logo + nav, absolutely positioned
    <div class="heroText">          H1, tagline, lede, CTA
    <Callout × 4>
    <DepthRuler>
  </div>
</section>
```

CSS:

```css
.heroSection { display: flex; justify-content: center; background: var(--paper); }
.stage {
  position: relative;
  width: min(100vw, calc((100svh - var(--banner-h, 32px)) * 1673 / 940));
  aspect-ratio: 1673 / 940;
  container-type: inline-size;
  overflow: hidden;
  outline: 1.5px solid var(--ink);              /* print frame */
  box-shadow: inset 0 0 6cqw rgba(30,42,43,.18); /* vignette */
}
.stage > * { position: absolute; }
```

This makes the whole scene fit one screen at every desktop size. On very wide screens the stage is centred, with paper on both sides.

Remove `.headerRow`'s bar, background, border and 1440 px max-width. The landing header now lives **inside** the stage. The old full-width header row goes away on the landing route.

**Position and size table.** Copy these numbers; they're measured from `reference.png`. `left`/`top` are % of the stage; font sizes are `cqw` (1 cqw = 1 % of stage width).

| Element | left | top | width / size | Notes |
|---|---|---|---|---|
| Logo emblem (inline SVG, circle + wave) | 8 % | 3.2 % | 3.2 cqw square | Redraw as an ink circle with a curling wave inside, like the reference. |
| Brand text "OCEANSCOPE AI" | 12.4 % | 4.2 % | 1.45 cqw, Pixelify 600 | |
| Nav (5 links) | 33 % → 65 % | 4.8 % | 1.05 cqw, Plex Mono 500, uppercase, letter-spacing .06em | Spread evenly, `gap: 6.2cqw`. HOME underlined 2 px. |
| H1 | 9.4 % | 17 % | **8.2 cqw**, Pixelify 600, `white-space: nowrap`, line-height 1 | Must stay on one line. |
| Tagline | 10.6 % | 31.5 % | 3.2 cqw, Pixelify 600 (the reference's tagline is the pixel face, not Plex), letter-spacing .04em | |
| Lede | 10.6 % | 39.5 % | 1.25 cqw, Plex Mono 400, line-height 1.6, `max-width: 35cqw` | Breaks into two lines like the reference. |
| CTA | 10.8 % | 50.2 % | padding 1.1cqw 1.6cqw, 1.2 cqw text | Keep the ink box and 3 px print shadow. |
| Callout "Satellites observe the surface" | 72.8 % | 33.5 % | 0.95 cqw, ink | |
| Callout "Ocean currents move heat" | 43.2 % | 69.5 % | 0.95 cqw, foam, **no bracket** | The reference has a plain label here; add a `bare` variant to Callout. |
| Callout "Argo floats provide real observations" | 29.7 % | 78.3 % | 0.95 cqw, foam | |
| Callout "Temperature at depth" | 81.8 % | 68.6 % | 0.95 cqw, foam | |
| Depth ruler | 90.8 % | 69.8 % → 90.8 % (height 21 %) | line 1.5 px foam, ticks 1 cqw long, labels 0.9 cqw foam | See 2.5. |

Callouts: line-height 1.35, letter-spacing .06em, uppercase, the bracket 1.5 px in the text colour. **No background panels or chips** behind any text on the art.

### 2.5 Depth ruler and the real-data band

**Ruler.** Keep the logic in `DepthRuler.tsx` (levels from `meta.depths_m`, nearest to 0/100/300/500/1000, √depth positions, no labels while loading). Change only the styling to match the reference: a solid foam vertical line, short foam ticks to the left, plain foam labels to the right, no chips, no dashed line, and confined to the box in the table above. The first label will read "0.5 m" because that's the real first level. That's correct; don't hardcode "0 m".

Note that the ruler is decoration on a painting, not an axis for it. The plume in the art is illustrative, so the label stays "Temperature at depth" and no colour scale is shown.

**Real band.** Remove `OceanBand` from the hero. Phase C (PLAN_LANDING §7) is cancelled for the hero. If Aryan wants a real-data teaser later, it goes below the fold as its own section, and it must:
- clip to ocean columns (the longest run of non-null positions, so no land gaps);
- show a colour scale with units and the caption "Model prediction · date · section at lat °N", plus the synthetic note;
- not pretend to be the painted plume.

That's a separate task; don't do it now.

### 2.6 Smaller screens

- **≥ 1024 px:** the stage as above.
- **768–1023 px:** same stage. Check the callouts don't overlap at 768. If "Ocean currents" collides with the CTA, hide it below 900 px.
- **< 768 px:** the stage can't hold readable text. Stack instead:
  1. The header (logo + `<details>` menu) and the text block as normal flow on paper: H1 `clamp(40px, 12vw, 64px)` on one line, tagline, lede, CTA.
  2. Below them, the scene image full width at its own aspect ratio (`object-fit: cover`, `object-position: 70% 50%`), with only the four callouts (0.85 rem) and the ruler on top. Hide the baked header area by cropping the top 12 % (`object-position` / a wrapper with `overflow: hidden`, `aspect-ratio: 1673/830`).
- No horizontal scroll at 390 px.

### 2.7 Banner

The synthetic banner must stay (PRD_FRONTEND §0 rule 4). Don't edit `components/common/Banner.tsx`. In `landing.module.css`, wrap it and restyle it for the landing route only: `.bannerWrapper :global([role="alert"]) { background: var(--ink); color: var(--paper); font-family: var(--font-mono); font-size: 12px; }`. Measure its height into `--banner-h`, which the stage width uses; a fixed `32px` is fine if the banner is one line.

### 2.8 Footer fixes (from the first review)

- "Explorer" link → `/explore`.
- GitHub link: leave the placeholder as `#` with a `TODO(Aryan)` comment, not `https://github.com`.
- Replace "Created by Aryan Chettri & AI Pair Programming Team" with "Built for <hackathon name>" (placeholder; OPEN_QUESTIONS "Branding, credits text" is still open).
- Add ERA5 to the data credits so they match `/about`.

### 2.9 Out of scope for this fix

Separate animated satellite and float layers, scroll parallax, the cross-route View Transition and the real-data teaser. Leave `ExploreLink`/`DiveOverlay` as they are; they work.

---

## Part 3. Acceptance checks

**Visual (required, with screenshots in the report).** Take screenshots at 1673×940 + banner, 1440×900, 1920×1080, 1024×768, 768×1024 and 390×844, and put each next to `reference.png`. At 1673 width:
- [ ] The H1, tagline, lede, CTA, nav and callouts sit within about 1 % of the positions in the table.
- [ ] The H1 is on one line at every width ≥ 768.
- [ ] Exactly one satellite, one sun, one pagoda, and one copy of each callout. No baked text visible anywhere.
- [ ] No hard rectangular edges, white fades or patches in the art.
- [ ] The whole hero, including the "1000 m" ruler label, is visible without scrolling at 1440×900 and 1920×1080.
- [ ] No text has a background chip or panel.
- [ ] At 390 px: no horizontal scroll, H1 on one line, CTA visible without scrolling.

Add a Playwright test `e2e/landing-visual.spec.ts` that takes the 1440×900 screenshot and asserts that the bounding boxes of the H1 and the depth ruler are fully inside the viewport and that the H1's height is less than 1.3× its font size (one line).

**Code checks:** `pnpm typecheck && pnpm lint && pnpm test && pnpm build`, all green. Update the Vitest landing tests for the removed components (OceanBand caption test goes) and the new Callout `bare` variant. Landing images ≤ 350 KB desktop and ≤ 180 KB at 390 px, from the network panel.

**Report to Aryan:** the six screenshots side by side with the reference, check outputs, image sizes, anything that couldn't match. Ask "OK to commit/push?" and don't push before he answers.

## Part 4. Order of work

1. Aryan: `scene-clean.png` (2.1).
2. Gemini: 2.2 clean-up and 2.8 footer. These don't need the art and can start right away.
3. Gemini: 2.3 assets, 2.4 stage, 2.5 ruler, 2.7 banner.
4. Gemini: 2.6 small screens.
5. Gemini: Part 3 checks and the report.
