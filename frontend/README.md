# oceanscope-frontend

Next.js 15 (App Router, React 19, TypeScript, Tailwind, MapLibre, TanStack Query) web app for OceanScope AI. Deployed to Vercel at https://ocean-embed-xi.vercel.app/ from `main`.

Before changing anything, read [`../AGENTS.md`](../AGENTS.md) (repo rules, which also bind coding agents) and [`../docs/PRD_FRONTEND.md`](../docs/PRD_FRONTEND.md) (what the frontend must do).

## Frontend quickstart

### 1. Tools

- **Node 22** (`>=22.13`, see `engines` in `package.json`; `.nvmrc` says 22, so `nvm use` works).
- **pnpm 11**, pinned by `packageManager` in `package.json`. The easiest way is Corepack, which ships with Node:

  ```bash
  corepack enable          # once per machine; then `pnpm` uses the pinned 11.x version
  pnpm --version           # should print 11.25.0
  ```

  Don't use npm or yarn: the lockfile is `pnpm-lock.yaml`.

### 2. Install

```bash
git clone https://github.com/aryanchettripkt-eng/Ocean-Embed.git
cd Ocean-Embed/frontend
pnpm install --frozen-lockfile
```

### 3. Environment

```bash
cp .env.example .env.local   # .env.local is git-ignored; never commit it
```

`.env.example` lists every variable the code reads. Pick one of three backends:

| Mode | Set in `.env.local` | Needs |
|---|---|---|
| **Deployed backend** (default in `.env.example`) | `NEXT_PUBLIC_API_BASE_URL=https://oceanscope-backend.onrender.com/api/v1` | Nothing. Real model output. The free Render instance can take about a minute to wake on the first request. |
| **Mocks, no backend** | `NEXT_PUBLIC_USE_MOCKS=true` | Nothing. MSW answers every API call in the browser with synthetic data (`src/mocks/`). Best for offline UI work. |
| **Local backend on fixtures** | `NEXT_PUBLIC_API_BASE_URL=http://localhost:7860/api/v1` | Python 3.11 and the backend running on `../fixtures/sample_outputs` (see the root README, "Run the site locally on synthetic data"). |

Synthetic data (mocks and fixtures) always shows the yellow "synthetic demo data" banner. That's intended. Note the fixtures only cover 2020-01-01 to 2020-01-10, so the time machine's `/events` call returns 503 against a local fixture backend; use the mocks or the deployed backend to work on events.

No secrets belong in the frontend. Every `NEXT_PUBLIC_*` value ends up in the browser bundle, and the Meta and Hugging Face keys live only in the backend's environment.

### 4. Run

```bash
pnpm dev        # http://localhost:3000
```

Pages: `/` (landing), `/explore` (2D map and 3D cutaway), `/validation`, `/about`, `/beach` (phone only, behind a flag). Restart `pnpm dev` after editing `.env.local`.

### 5. Checks (run all four before pushing; CI runs the same plus e2e)

```bash
pnpm typecheck
pnpm lint
pnpm test        # vitest unit tests (tests/)
pnpm build
```

End-to-end tests (Playwright, `e2e/`). They start `pnpm dev` with mocks on by themselves:

```bash
pnpm exec playwright install chromium   # once, or after a Playwright upgrade
pnpm e2e
```

If you change `../backend/openapi.json` (you shouldn't without Aryan's OK, it's a frozen contract), regenerate the API types with `pnpm generate-types` and commit `src/lib/api/types.ts`; CI fails if they differ.

## Meta feature flags

The "Talk to the ocean" features are switched on and off by the **backend**, not by frontend env vars. The frontend reads `GET /api/v1/meta` and shows each feature only when its flag is true:

| `meta.features.*` | UI | Backend env var that turns it on |
|---|---|---|
| `chat` | "Ask the ocean" guide drawer | `CHAT_ENABLED` (plus a configured Meta client and budget, or `META_CLIENT=mock`) |
| `voice` | Mic button in the guide | `VOICE_ENABLED` (same conditions) |
| `postcards` | "Make a postcard" | `POSTCARD_ENABLED` (same conditions) |
| `segment`, `segment_method` | Lasso (`sam` or `threshold`) | `SEGMENT_ENABLED`, `SEGMENT_PROVIDER` |
| `beach` | `/beach` link in the phone header | `BEACH_ENABLED` |
| `provider_label` | Model name shown in the guide | `PROVIDER_LABEL` / `META_CLIENT` |

The Events / time-machine button appears whenever `/events` returns at least one event; it has no flag. All flags default to off. See `../backend/app/meta/config.py` and `../backend/.env.example` for the exact rules.

To build these UIs without a backend or any Meta key, run on mocks with every flag on:

```bash
NEXT_PUBLIC_USE_MOCKS=true NEXT_PUBLIC_MOCK_META_FEATURES=true pnpm dev
```

Or run the local backend with `META_CLIENT=mock` and the `*_ENABLED` flags set to `true`, which uses the backend's offline mock guide. Tests must never call the real Meta API (AGENTS.md hard rule 11).

## Branches and pull requests

- `main` is protected by convention: never commit or force-push to it directly.
- Branch from an up-to-date `main`, named `frontend/<short-task>` (for example `frontend/profile-chart-tooltip`):

  ```bash
  git switch main && git pull
  git switch -c frontend/<short-task>
  ```

- Keep each PR to one task, stay inside `frontend/`, and don't change the API contract (`backend/openapi.json`) or add dependencies without asking Aryan first (AGENTS.md §4, §5).
- Commit messages: `frontend: <what> (<PRD section>)`.
- Push and open a PR into `main`. CI (`.github/workflows/ci.yml`) runs type generation check, typecheck, lint, unit tests, e2e and build; it must be green. If Vercel's GitHub integration is on for the repo, it also posts a preview deployment link on the PR.
- Aryan reviews and merges. A merge to `main` deploys to production on Vercel.
