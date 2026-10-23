# PRD: Frontend Web App — OceanScope AI

**Status:** Draft v4 (2026-10-24): Meta features (§3.4: guide/navigator, voice, event time machine, postcards, lasso, beach mode); landing page in `docs/PLAN_LANDING.md`. v3 (2026-10-24): 3D cutaway view (§3.1b), map code fixes (§10). v2 (2026-10-22): headline card, live dates, competitor ARGO columns, updated sources  
**Owner:** Aryan  
**Implementer:** Coding agent  
**Lives in:** `frontend/` of the monorepo (see `AGENTS.md`).  
**Depends on:** `backend/openapi.json` (API contract, `PRD_BACKEND.md` Section 4).

---

## 0. Rules for the implementer (non-negotiable)

1. **The frontend never computes or invents scientific values.** It only displays what the API returns. No hardcoded temperatures, metrics, dates, depths or region bounds; read them from `/api/v1/meta`.
2. **Types come from the API contract.** Generate `src/lib/api/types.ts` from `backend/openapi.json` with `openapi-typescript`. Do not hand-write response types. If the API lacks something, write it in `docs/OPEN_QUESTIONS.md`; do not work around it with client-side math on other endpoints.
3. **Always label the source** of every number: *Model prediction*, *GLORYS reanalysis*, or *ARGO observation*. Never present predictions as observations.
4. **Show limitations.** The caveat strings from `/meta.caveats` are displayed verbatim where specified. If `synthetic` is true, a non-dismissible banner says so on every page.
5. **No secrets in the frontend.** Only `NEXT_PUBLIC_API_BASE_URL` (and `NEXT_PUBLIC_USE_MOCKS`). Every Meta model call (chat, transcribe, image, SAM) goes through our backend; the browser never calls the Meta API.
6. **TypeScript strict, no `any`, no `@ts-ignore`,** ESLint clean. Do not disable lint rules to pass CI.
7. **Do not add dependencies** beyond Section 2 without stating why in the commit message. No paid map tokens or services.
8. When unsure, write the question in `docs/OPEN_QUESTIONS.md` and stop that task.
9. **AI output is labelled and never a source of numbers.** Numbers shown next to AI text or images come from API fields, rendered as HTML; generated images never carry numbers; the frontend applies only the UI actions listed in PRD_BACKEND §6.2.

---

## 1. Goal

A public, fast, mobile-friendly web app where anyone can explore predicted subsurface ocean temperature in the North Indian Ocean by date, depth and location, see how accurate the model is, and (optionally) ask questions in plain language.

### Users
- Hackathon judges (5-minute tour, likely on a laptop, possibly a phone).
- Curious public / students.
- Ocean scientists checking the method and validation.

### Non-goals
Accounts, saving user data, uploads, editing, offline mode, 3D globe rendering, true volume rendering (ray-marched clouds). The regional 3D cutaway block in §3.1b is in scope.

---

## 2. Stack (fixed)

| Concern | Choice |
|---|---|
| Framework | Next.js (App Router, current stable 15.x), React, TypeScript strict |
| Styling | Tailwind CSS; components from shadcn/ui (copied in, not a runtime dep) |
| Data fetching | TanStack Query (React Query) |
| Map | MapLibre GL JS with a free, token-less basemap (e.g. OpenFreeMap or CARTO Positron/Dark Matter tiles; check terms) |
| Grid rendering | Client-side: draw the `/map` grid to a `<canvas>`, add it as a MapLibre `image` source over the region bbox; hover/click reads values from the in-memory array |
| 3D view | MapLibre `CustomLayerInterface` with raw WebGL2 (no three.js, deck.gl or other 3D library), in its own lazily loaded module (§3.1b) |
| Charts | Recharts (profile and time series) |
| Colormaps | cmocean `thermal` (temperature), `balance` (error), as precomputed 256-step lookup tables in `src/lib/colormaps.ts` |
| API types | `openapi-typescript` from `../backend/openapi.json` |
| Mocking | MSW (Mock Service Worker) serving `fixtures/sample_outputs`-derived JSON when `NEXT_PUBLIC_USE_MOCKS=true` |
| Tests | Vitest + Testing Library (units), Playwright (smoke e2e against mocks) |
| Hosting | Vercel |

---

## 3. Pages and features

### 3.1 `/` Explorer (main page)
Layout: map fills the viewport; a control panel (left sidebar on desktop ≥ 1024 px, bottom sheet on mobile); a details panel (right on desktop, sheet on mobile) when a point is selected.

**Controls**
- **Date:** date picker plus a slider with step buttons (±1 day), limited to `meta.available_dates`. Shows the split badge of the date (`Validation` / `Test` / `Training (in-sample)` / `Live`). A "Latest" button jumps to `meta.latest_live_date` when it is not null.
- **Depth:** a 15-step selector listing `meta.depths_m` as "0.5 m, 10 m, …" (formatted, not hardcoded).
- **Field:** segmented control: *Model prediction* | *GLORYS reanalysis* | *Difference (model − GLORYS)*. On `Live` dates the last two are disabled with the tooltip "No GLORYS reference yet for live dates", and `LIVE_CAVEAT` is shown under the map.
- **Play:** optional animation over dates at 2 frames/s, prefetching the next 3 days.

**Map**
- Grid overlay with the colormap; colour range from `meta.color_ranges` for the selected depth (fixed across dates so animation is comparable); `meta.error_range` for the difference field.
- Land / unsupported cells (coverage 0): transparent. Insufficient inputs (coverage 2): diagonal hatch pattern in grey.
- Colorbar legend with units (°C) and a coverage legend (valid / insufficient inputs / land).
- Hover tooltip: lat, lon, value with source label, or "No prediction (insufficient input data)".
- Click selects the cell (marker) and opens the details panel.
- Toggle: show ARGO profile locations for the date (`/argo/profiles`), as small markers.

**Details panel (selected cell)**
- Header: snapped lat/lon, date, split badge.
- **Vertical profile chart:** temperature (x) vs depth (y, increasing downward), two lines: model prediction and GLORYS (GLORYS line omitted on live dates); tooltip per depth.
- **Time series chart** at the selected depth: ±90 days around the date (`/timeseries`), both lines, vertical marker at the selected date.
- Short "How accurate is this depth?" line: val RMSE and bias at the selected depth from `/metrics`, linking to `/validation`.

**URL state:** `?date=&depth=&field=&lat=&lon=` kept in sync (shareable links; back/forward works). Invalid params fall back to defaults with no crash.

**Defaults:** latest available validation date, depth closest to 100 m, field = prediction, no selection.

### 3.1b 3D cutaway view (Explorer, desktop and tablet)
The Explorer gains a **View** switch: *Map* (the 2D map above, unchanged) | *3D* (a rotatable block of ocean). Reference mockup: the "OceanScope 3D map concept" artifact (illustrative data only; it shows the look and the interactions, not the code).

**What the user sees in *3D***
- The map tilts (pitch 50°, bearing −20°) and the region becomes a solid block hanging below the basemap. The **top face** shows the selected depth; the **side walls** show temperature from that depth down to the deepest level, along the region's edges.
- **Diving:** changing the depth (depth rail, slider, scroll over the rail, or `PageUp`/`PageDown` while the rail has focus) eases the top face down to the new depth over ~300 ms and slices the water above it away. A faint dashed outline marks the sea surface above the cut.
- **Contours** every 2 °C on the top face and the walls, and the **20 °C isotherm** drawn as a bright line labelled "20 °C isotherm (thermocline marker)". Both are drawn from the API values only; they are a way of drawing the field, not new data.
- **Same colour scale** as 2D: `meta.color_ranges` per depth and the `thermal` LUT for the top face. The walls span many depths, so they use one fixed range for the whole column: `vmin` = min of all `color_ranges[].vmin`, `vmax` = max of all `color_ranges[].vmax`. The colorbar shows both ranges ("Top face" / "Walls") when they differ.
- **Lighting:** walls shaded by angle to the camera, darkening slightly with depth, a soft shadow under the block and a thin light edge on the top face. Land cells (coverage 0) on the top face are drawn in the basemap land colour so the sunk face does not show the basemap through it.
- **Exploded layers** (secondary mode, toggle inside *3D*): 4 depths (0.5 m, ~100 m, ~300 m, deepest; picked as the nearest levels in `meta.depths_m`, plus the selected depth) floating as stacked sheets; the selected one is opaque and outlined, the rest at 60% opacity.
- Hover and click work as in 2D on the top face: tooltip with lat, lon, depth, value and source label; click selects the cell and opens the details panel. Values always come from the snapped grid cell, even when the drawing is smoothed.
- Pan, zoom and rotate as usual; the **vertical exaggeration** is fixed so the block is about one third as tall as it is wide.

**Data (no client-side science, §0 rule 1)**
- Top face: the existing `/map` for the selected depth (already cached and prefetched for adjacent depths).
- Walls: the new `/section` endpoint (PRD_BACKEND §4.11), one call per edge (south `lat = region.lat_min`, north `lat_max`, west `lon = region.lon_min`, east `lon_max`), fetched once per date and field and cached like `/map`. Only the two walls facing the camera are drawn.
- Exploded layers: `/map` for each shown depth.
- Do **not** build walls from 15 `/map` calls (about 15× the bytes and requests on a free-tier backend).

**Rendering rules**
- One MapLibre custom layer (`components/map3d/CutawayLayer.ts`) drawing the top face, walls, shadow and outlines with WebGL2: each face is a textured quad; field values go into a float texture and the colormap LUT into a 256×1 texture, so a depth or colour-range change is a uniform update, not a re-upload. Contours and the 20 °C line are computed in the fragment shader from the same texture.
- Display smoothing: linear texture filtering is allowed in 3D (labelled "Smoothed display" with a toggle to "Exact cells"); hatch for coverage 2 and transparency rules stay as in 2D.
- Vertical axis: depth is drawn on a square-root scale (`z ∝ √depth`) so the upper 300 m gets room; tick labels in metres from `meta.depths_m`.
- The 3D module is loaded with `import()` only when the user switches to *3D*; it must not grow the initial `/` bundle.
- After a theme switch (`map.setStyle`), re-add the custom layer on `styledata`, the same way the grid layer is re-added today.
- `prefers-reduced-motion`: no easing on dive and no auto-rotation.
- Auto-rotation: off by default; a "Tour" button (for the demo) slowly rotates until the user touches the map.

**Fallbacks**
- Width < 768 px, no WebGL2, or the 3D module fails to load: the *3D* option is hidden or disabled with a tooltip, and the 2D map is used. No blank map.
- `/section` error or 404: draw the top face only and show "Side walls unavailable for this date" in the map corner.
- Live dates: `field = glorys|error` is disabled exactly as in 2D.

**Section slice (optional, after the above works)**
- A "Slice" tool: click the map to place an east–west or north–south line; the block shows that section as an inner wall, and the details panel shows a depth × distance section chart (canvas, same colormap) from `/section`.

### 3.2 `/validation`
- **Headline card (top of page):** the pre-registered criterion text from `/headline.criterion`, then model vs GLORYS12 / GREP / ARMOR3D RMSE with 95% intervals as a small dot-and-whisker chart, and a plain verdict from `/headline.met`: "Met", "Not met" or "Not yet evaluated" (null or 404). Show the verdict exactly as returned, including "Not met". Label the period (validation or test).
- Summary cards: overall val RMSE, MAE, bias, skill vs climatology (from `/metrics?group_by=depth`, row `depth_m = all`).
- **Per-depth chart:** RMSE vs depth for the model and each baseline.
- Tables: by depth, by season, by region (sortable).
- ARGO section: per-depth errors for the model, GLORYS, GREP and ARMOR3D vs ARGO (columns with no data shown as "n/a"), counts, `ARGO_CAVEAT` verbatim.
- Test results section: shown only if `meta.test_evaluated`; otherwise the text "Final test-period evaluation not yet run."
- `GLORYS_CAVEAT` verbatim at the top.

### 3.3 `/about`
Static content (MDX allowed) explaining: the problem, data sources (DUACS SLA/ADT, OSTIA SST, MULTIOBS SSS, OSCAR, CCMP, ERA5, GEBCO, WOA23, climate indices from NOAA CPC and the Australian Bureau of Meteorology, GLORYS12, GREP, ARMOR3D, ARGO) with links and required acknowledgements/licence lines, the method in plain language (surface inputs with 7- and 60-day history, climate indices and recent ARGO floats → U-Net → 15 depth anomalies + climatology; trained on GLORYS and GREP, fine-tuned on ARGO), what "live" dates are, limitations (caveats verbatim from `/meta`), and team/credits. Numbers on this page must also come from the API, not be typed in.

### 3.4 Ocean guide and Meta features (core since v4; each part only if its `meta.features` flag is true)
Meta's models are how people move through the data: you ask, and the ocean moves. Every part is hidden (no dead UI) when its flag is false, and the rest of the site works without any of them. Each AI output is labelled: `AI_CAVEAT`, `POSTCARD_CAVEAT` or `SEGMENT_CAVEAT` from `/meta.caveats`, verbatim.

**a) Ocean guide = chat + navigator (`features.chat`)**
- A floating "Ask the ocean" button opens a side drawer (bottom sheet on phones). It sends the current explorer context (date, depth, field, view mode, selection, camera centre) with each message to `/chat`.
- Answers render as plain text/markdown (HTML disabled). Each answer has an expandable "Data used" section (tool calls and results) and the `provider_label` under the input.
- **Navigator:** the response's `ui_actions` are applied in order through the same URL-state setters the controls use (so Back undoes them), with a short camera animation (≤ 1.2 s, none under reduced motion). A small chip under the answer says what moved ("Moved to 15 Jan 2019 · 50 m · Bay of Bengal") with an "Undo" link. The frontend applies only the four action types in PRD_BACKEND §6.2 and ignores anything else.
- If `grounded` is false, show the fallback text exactly as returned with a neutral "Couldn't verify" badge.
- Three starter chips from the current context, e.g. "What's happening at 100 m here?", "Show me the warmest water today", "Take me to Cyclone Fani".
- Handles 429/503/502 with friendly messages; input disabled while waiting; 25 s client timeout.

**b) Voice dive (`features.voice`)**
- A hold-to-talk mic button in the guide's input (tap-to-toggle for accessibility). Records with `MediaRecorder` (Opus/WebM; MP4/AAC on Safari), max 30 s with a visible countdown, then posts to `/transcribe`.
- The transcript appears in the input for 1.5 s (editable; any key cancels auto-send) and is then sent to `/chat` as text.
- Microphone permission is requested only on the first press, with a one-line explanation first; if denied, the mic button explains how to re-enable it and typing still works. Audio is never stored or sent anywhere but our backend.

**c) Event time machine**
- An "Events" button in the Explorer opens a list from `/events` (title, dates, source link). Choosing one opens a compare strip: **Before / During / After** tabs (or a slider on desktop) that set the date to `before`, `start`…`end`, `after`, zoom to the event bbox and pick the first suggested depth; the 3D view works the same way.
- An "Explain" button asks the guide (`/chat`) to compare before and after for that bbox and depth; its UI actions can step through the phases.
- Every event shows its `source_name` and link. No event text is typed into the frontend.
- Events in the sealed test period never appear (the backend filters them; the frontend does not override that).

**d) Ocean postcards (`features.postcards`)**
- A "Make a postcard" button in the Explorer (and on an event) posts the current date, depth and event to `/postcard`. While waiting (up to 45 s), a "printing" animation in the landing-page paper style; cancel button.
- The result is shown as a card: the generated image plus an **HTML caption** overlaid from the response (`date`, depth, mean temperature, source label), and `POSTCARD_CAVEAT` under it. The picture itself never carries numbers.
- "Download" composes image + caption into one PNG on a client `<canvas>` (so the caption and caveat stay attached), and "Copy link" shares the explorer URL, not the image URL.

**e) Lasso (`features.segment`)**
- A "Lasso" tool in the map toolbar: tap a point on the 2D map or the 3D top face → `/segment` → draw the returned `outline` as an ink-style line with a soft fill; a panel shows the stats (mean/min/max/std, area) with the source label, the method ("Selected by SAM 3.1" or "Selected by similar temperature (±0.5 °C)") and `SEGMENT_CAVEAT`.
- "Ask about this area" sends the selection's bbox and stats as context to the guide.
- While the request runs (≤ 10 s), show a pulsing outline at the tapped point; on error, a one-line message and no stats.

**f) Beach mode (`features.beach`, phones only)**
- A "Beach mode" entry in the mobile menu opens `/beach`: one big "Point at the sea" button. On tap: request geolocation (and on iOS `DeviceOrientationEvent.requestPermission()`), read the compass heading, call `/nearest-ocean`, then show that cell's `/profile` as a big vertical temperature strip with the distance and direction ("62 km east of you"), plus the mic button to ask the guide about it.
- No compass → nearest ocean cell without a bearing. Permission denied → a manual "pick a spot on the map" fallback. Location is sent only on tap, never in the background, never stored in the browser beyond the session.
- Outside the region → "OceanScope covers the North Indian Ocean; here's the nearest point we cover", with a link to the map.

---

## 4. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Map redraw ≤ 150 ms after data arrives; date/depth change shows the new map within 1 s on a warm backend; JS for `/` ≤ 350 KB gzipped (MapLibre included); Lighthouse performance ≥ 80, accessibility ≥ 90 on `/` (desktop) |
| Cold backend | The backend may sleep. On first load, if `/meta` takes > 3 s, show "Waking up the ocean server… (can take up to a minute)" with a spinner; retry with backoff up to 90 s, then show an error with a Retry button |
| Caching | React Query `staleTime: Infinity` for map/profile/timeseries keyed by `run_id` + params; prefetch adjacent dates and depths on idle |
| Errors | Every fetch has loading, empty and error states; no blank screens; an error boundary per panel |
| Responsive | Usable at 360 px width; touch: tap = select cell; controls reachable with one thumb on mobile |
| Accessibility | Keyboard-operable controls; visible focus; colorbar has text min/max; charts have accessible summaries; colour is never the only signal (hatch pattern for coverage) |
| Theming | Light and dark mode (system default + toggle), basemap switches accordingly |
| Units/formatting | °C with 1 decimal on the map and tooltips, 2 decimals in tables; depths "123 m" (≥ 10 m rounded to the nearest metre, < 10 m one decimal); coordinates "12.375°N, 65.125°E" |
| Security | No `dangerouslySetInnerHTML` except sanitized MDX; chat text rendered as plain text/markdown with HTML disabled |
| SEO/sharing | Title, description, Open Graph image (static) for link previews |
| Meta features | All of §3.4 is lazily loaded (`import()` on first open); none of it adds to the initial `/` bundle. Permission prompts (mic, location, motion) only after a user tap with a one-line reason. Every AI request shows progress within 100 ms and can be cancelled. |
| 3D view | ≥ 45 fps while rotating on a mid-range laptop (integrated GPU) at 1080p; dive animation never blocks input; 3D module ≤ 25 KB gzipped and lazily loaded (not counted in the 350 KB `/` budget); switching Map ↔ 3D keeps date, depth, field and selection; URL state gains `view=map|3d` |

---

## 5. Structure

```
frontend/
├── src/
│   ├── app/                 # routes: page.tsx (explorer), validation/, about/, layout.tsx
│   ├── components/
│   │   ├── map/             # MapView, GridLayer (canvas), Colorbar, CoverageLegend, ArgoMarkers
│   │   ├── map3d/           # CutawayLayer (custom WebGL2 layer), shaders, ViewSwitch, DepthRail (lazily loaded)
│   │   ├── controls/        # DatePicker, DepthSelector, FieldToggle, PlayButton
│   │   ├── panels/          # DetailsPanel, ProfileChart, TimeSeriesChart
│   │   ├── validation/      # MetricCards, DepthErrorChart, MetricsTable, ArgoSection
│   │   ├── guide/           # GuideDrawer, MessageList, DataUsed, applyUiActions, MicButton (lazily loaded)
│   │   ├── events/          # EventList, PhaseCompare
│   │   ├── postcard/        # PostcardButton, PostcardCard, composeDownload
│   │   ├── lasso/           # LassoTool, SelectionPanel
│   │   ├── beach/           # BeachMode (route /beach)
│   │   └── common/          # Banner (synthetic/caveats), ErrorState, Loading, SplitBadge
│   ├── lib/
│   │   ├── api/             # client.ts (fetch wrapper, error parsing), types.ts (generated), hooks.ts (React Query)
│   │   ├── colormaps.ts
│   │   ├── format.ts
│   │   └── url-state.ts
│   └── mocks/               # MSW handlers + JSON generated from fixtures/sample_outputs
├── tests/ (vitest) and e2e/ (playwright)
├── .env.example
└── package.json
```
All API access goes through `lib/api/hooks.ts`; components never call `fetch` directly.

---

## 6. Environment variables
| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | e.g. `https://oceanscope-backend.onrender.com/api/v1` (backend is on Render, DECISIONS 2026-10-24) |
| `NEXT_PUBLIC_USE_MOCKS` | `true` to run fully on MSW mocks (default `false`) |

---

## 7. Tests (required)
- Meta features (mocked API): `applyUiActions` applies the four action types via URL state and ignores unknown ones; Undo restores the previous URL; a `grounded: false` answer shows the badge; the mic flow sends the edited transcript, not the raw one; postcard caption text comes from the response and the caveat is present; lasso shows the method and caveat; every feature's UI is absent when its flag is false; beach mode falls back when geolocation or orientation is denied.
- Unit: `format.ts`, `colormaps.ts` (endpoints and NaN handling), `url-state.ts` (round-trip, invalid input), API error parsing.
- Component: GridLayer renders correct pixel colour for a known value and transparent for null; DetailsPanel shows source labels; synthetic banner appears when `synthetic: true`; chat button absent when chat disabled; on a `live` date the GLORYS and Difference controls are disabled and `LIVE_CAVEAT` shows; the headline card renders "Met", "Not met" and "Not yet evaluated" correctly.
- 3D (unit, no GPU needed): the depth → √depth z mapping and its tick labels; wall colour range from `meta.color_ranges`; which two walls face the camera for bearings 0°, 90°, 180°, 270°; screen point → grid cell on the sunk top face (round trip); fallback to 2D when WebGL2 is missing (mock `getContext` returning null).
- E2E (Playwright, mocks on): load `/`, change depth and date, click a cell, see profile chart; open `/validation`, see tables; shareable URL restores state; switch to *3D* (`?view=3d`), change depth, hover the top face and see a tooltip; a screenshot of the 3D view is saved as a CI artifact (not a pixel-exact assertion).
- CI: `pnpm typecheck && pnpm lint && pnpm test && pnpm build && pnpm e2e` all green; generated types up to date with `backend/openapi.json`.

---

## 8. Milestones and acceptance

| M | Scope | Done when |
|---|---|---|
| F1 Skeleton | Next.js app, Tailwind, layout, theme, MSW mocks, generated types, CI, Vercel preview deploy | Preview URL loads with mocks; CI green |
| F2 Explorer map | MapView, GridLayer, controls, colorbar, coverage, URL state | Explorer works end-to-end on mocks; perf targets met locally |
| F3 Details | Profile + time series charts, accuracy line | E2E test passes |
| F4 Validation + About | `/validation`, `/about`, caveats | Pages render all metric groups from mocks |
| F5 Live backend | Point at the HF Space; cold-start UX | Production Vercel URL works against the live API with synthetic data, then real data |
| F6 Assistant (superseded by F8) | Chat drawer | Works against the backend with a mocked/real provider; hidden when disabled |
| F8 Meta features | §3.4 a–f, in PHASES P7 order | Tests pass on mocks; each part works against the deployed backend with the real Meta API; owner's 20-question test |
| F7 3D cutaway | §3.1b: view switch, custom layer (top face, walls, contours, 20 °C line, lighting), dive animation, depth rail, exploded layers, fallbacks; then the optional slice tool | 3D tests pass; fps and bundle targets met; works on the deployed site with synthetic data; Aryan signs off the look |

F1–F4 must be built against mocks and must not wait for the backend or the model.

---

## 9. Open questions for the owner
- Project name/branding on the site (keep "OceanScope AI"?), and any logo.
- Team/credits text for `/about`.
- Custom domain on Vercel, or the default `*.vercel.app`.

---

## 10. Required fixes to the current map code (review 2026-10-24, before F7)
These are breaches of §0 found in `components/map/MapView.tsx` and `lib/oceanMask.ts`. Fix them first (PHASES P2b-F0); the 3D layer builds on the same code.
1. **Region bounds are hardcoded** (`40, 100, 0, 30` in the image-source coordinates, hover and click handlers). Read them from `meta.region` and `meta.grid_resolution_deg`, and snap with the returned `lat`/`lon` arrays only (§0 rule 1).
2. **`isNorthIndianOcean` invents a land mask** and overrides the API's `coverage`. Land comes from `coverage = 0` in the API response, nothing else. Delete `lib/oceanMask.ts` and its uses. If the synthetic fixture's land mask looks wrong, fix the fixture generator (PRD_AI_ML §8.1), not the frontend.
3. **`canvas.toDataURL()` on every change** encodes a PNG on the main thread. Use a MapLibre `canvas` source (or `updateImage` with an object URL from `canvas.toBlob`) so redraws meet the 150 ms target.
4. **Hover handler sets React state on every mouse move.** Throttle to one update per animation frame and skip it when the cell has not changed.
5. **Two click paths:** a capture-phase DOM click listener and the MapLibre `click` handler both call `selectAtLngLat`, so one click selects twice. Keep one (the MapLibre handler) unless a test shows why both are needed.
6. **Pitch and rotation:** in *Map* view keep pitch 0 and disable drag-rotate so the 2D view stays flat; enable both only in *3D*.
