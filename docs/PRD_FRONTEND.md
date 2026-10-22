# PRD: Frontend Web App — OceanScope AI

**Status:** Draft v2 (2026-10-22): headline card, live dates, competitor ARGO columns, updated sources  
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
5. **No secrets in the frontend.** Only `NEXT_PUBLIC_API_BASE_URL` (and `NEXT_PUBLIC_USE_MOCKS`). LLM calls go through the backend's `/chat`.
6. **TypeScript strict, no `any`, no `@ts-ignore`,** ESLint clean. Do not disable lint rules to pass CI.
7. **Do not add dependencies** beyond Section 2 without stating why in the commit message. No paid map tokens or services.
8. When unsure, write the question in `docs/OPEN_QUESTIONS.md` and stop that task.

---

## 1. Goal

A public, fast, mobile-friendly web app where anyone can explore predicted subsurface ocean temperature in the North Indian Ocean by date, depth and location, see how accurate the model is, and (optionally) ask questions in plain language.

### Users
- Hackathon judges (5-minute tour, likely on a laptop, possibly a phone).
- Curious public / students.
- Ocean scientists checking the method and validation.

### Non-goals
Accounts, saving user data, uploads, editing, offline mode, 3D globe rendering.

---

## 2. Stack (fixed)

| Concern | Choice |
|---|---|
| Framework | Next.js (App Router, current stable 15.x), React, TypeScript strict |
| Styling | Tailwind CSS; components from shadcn/ui (copied in, not a runtime dep) |
| Data fetching | TanStack Query (React Query) |
| Map | MapLibre GL JS with a free, token-less basemap (e.g. OpenFreeMap or CARTO Positron/Dark Matter tiles; check terms) |
| Grid rendering | Client-side: draw the `/map` grid to a `<canvas>`, add it as a MapLibre `image` source over the region bbox; hover/click reads values from the in-memory array |
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

### 3.4 Assistant (only if `meta.chat.enabled`)
- Floating "Ask about the ocean" button → side drawer chat.
- Sends the current explorer context (date, depth, selected lat/lon) with each message.
- Shows each answer with an expandable "Data used" section listing the tool calls and their results (from the `/chat` response).
- Shows `provider_label` small under the input ("Answers by …, based on model outputs").
- Suggested starter questions (3 chips) derived from current context, e.g. "What's the temperature profile here on {date}?"
- Handles 429/503/502 with friendly messages; disables input while waiting; 25 s client timeout.
- If chat is disabled, no button is rendered (no dead UI).

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

---

## 5. Structure

```
frontend/
├── src/
│   ├── app/                 # routes: page.tsx (explorer), validation/, about/, layout.tsx
│   ├── components/
│   │   ├── map/             # MapView, GridLayer (canvas), Colorbar, CoverageLegend, ArgoMarkers
│   │   ├── controls/        # DatePicker, DepthSelector, FieldToggle, PlayButton
│   │   ├── panels/          # DetailsPanel, ProfileChart, TimeSeriesChart
│   │   ├── validation/      # MetricCards, DepthErrorChart, MetricsTable, ArgoSection
│   │   ├── chat/            # ChatDrawer, MessageList, DataUsed
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
| `NEXT_PUBLIC_API_BASE_URL` | e.g. `https://<hf-user>-oceanscope-api.hf.space/api/v1` |
| `NEXT_PUBLIC_USE_MOCKS` | `true` to run fully on MSW mocks (default `false`) |

---

## 7. Tests (required)
- Unit: `format.ts`, `colormaps.ts` (endpoints and NaN handling), `url-state.ts` (round-trip, invalid input), API error parsing.
- Component: GridLayer renders correct pixel colour for a known value and transparent for null; DetailsPanel shows source labels; synthetic banner appears when `synthetic: true`; chat button absent when chat disabled; on a `live` date the GLORYS and Difference controls are disabled and `LIVE_CAVEAT` shows; the headline card renders "Met", "Not met" and "Not yet evaluated" correctly.
- E2E (Playwright, mocks on): load `/`, change depth and date, click a cell, see profile chart; open `/validation`, see tables; shareable URL restores state.
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
| F6 Assistant | Chat drawer | Works against the backend with a mocked/real provider; hidden when disabled |

F1–F4 must be built against mocks and must not wait for the backend or the model.

---

## 9. Open questions for the owner
- Project name/branding on the site (keep "OceanScope AI"?), and any logo.
- Team/credits text for `/about`.
- Custom domain on Vercel, or the default `*.vercel.app`.
