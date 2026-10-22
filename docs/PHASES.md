# Project phases — OceanScope AI

**Status:** v1 (2026-10-22). This is the execution order for the whole project. Each phase has an **exit gate**; do not start a phase whose "Needs" are not met. Task IDs (e.g. `P1-B2`) are used in branch names, commits and `docs/PROGRESS.md`.

Who: **Aryan** = owner (accounts, decisions, Colab/Kaggle runs, approvals) · **Gemini** = implementer · **Claude** = planning, task breakdown, review (can implement).

```
P0 Setup ──► P1 Build on synthetic data ──► P2 Live synthetic site ─────────────────────────┐
   │              (backend ∥ frontend ∥ ML data M1)                                           │
   └─► accounts ─► P3 Real data + baselines ──► P4 U-Net training ──► P5 Evaluation ──► P6 Real outputs live ──► P8 Polish + demo
                                                                                       P7 Assistant (optional, any time after P2)
```
Backend/frontend work never waits for the model: P1 and P2 run on synthetic sample outputs while P3–P5 run on Colab/Kaggle.

---

## P0 — Setup (½ day)
**Goal:** an empty but correctly structured repo with CI, and accounts in place.

| ID | Task | Who | Ref |
|---|---|---|---|
| P0-1 | Create GitHub repo; push current `E:\meta` contents; protect `main` | Aryan (+ Claude) | AGENTS §3 |
| P0-2 | Scaffold `ml/`, `backend/`, `frontend/`, `fixtures/` with package manifests, linters, empty test suites | Gemini | AGENTS §3 |
| P0-3 | CI workflow: ml, backend, frontend jobs (lint, type, test); path filters | Gemini | AGENTS §6 |
| P0-4 | Create accounts: Copernicus Marine, NASA Earthdata, Hugging Face, Vercel; store tokens as Colab/Kaggle secrets | Aryan | OPEN_QUESTIONS |
| P0-5 | `manifest.schema.json` (ML + backend copies) and the sample-outputs generator → commit `fixtures/sample_outputs/` | Gemini | PRD_AI_ML §6, §8.1 |

**Exit gate:** CI green on `main` with empty suites; `fixtures/sample_outputs/` committed and valid against the schema; accounts exist.

---

## P1 — Build on synthetic data (3 parallel tracks)
**Needs:** P0.

### Track B — Backend (PRD_BACKEND)
| ID | Task | Ref |
|---|---|---|
| P1-B1 | Settings, loader, manifest validation, `/health`, `/meta`, Dockerfile, `export_openapi` | §3, §4.1–4.2, M B1 |
| P1-B2 | `/map`, `/profile`, `/timeseries`, `/region-summary` + grid snapping | §4.3–4.6, M B2 |
| P1-B3 | `/metrics`, `/argo`, `/argo/profiles`, caveat strings | §4.7–4.9, §7, M B3 |

### Track F — Frontend (PRD_FRONTEND)
| ID | Task | Ref |
|---|---|---|
| P1-F1 | Next.js skeleton, Tailwind, theme, layout, generated API types, MSW mocks, CI | M F1 |
| P1-F2 | Explorer map: MapLibre, canvas grid layer, controls, colorbar, coverage legend, URL state | §3.1, M F2 |
| P1-F3 | Details panel: profile + time-series charts, accuracy line | §3.1, M F3 |
| P1-F4 | `/validation` and `/about` pages, synthetic banner, caveats | §3.2–3.3, M F4 |

F starts on mocks generated from the committed `openapi.json`; it may begin once P1-B1 has produced `openapi.json` (the full schema can be stubbed ahead of implementation).

### Track M — ML data, first month (PRD_AI_ML M1, part 1)
| ID | Task | Who | Ref |
|---|---|---|---|
| P1-M1 | `download.py` with `--dry-run`; download **one month** (Jan 2019) of all products | Gemini, run by Aryan | §4.1 |
| P1-M2 | `inspect.py` → `data_inspection.md` | Gemini | §4 |
| P1-M3 | **Owner decision:** confirm region, 15 depths, period, splits in `experiment.yaml`; log in DECISIONS.md | Aryan (Claude proposes) | §2 |

**Exit gate:** backend tests 1–6, 8, 9 pass; frontend E2E passes on mocks; inspection report reviewed and all `[CONFIRM]` values confirmed.

---

## P2 — Live synthetic site
**Needs:** P1-B*, P1-F*.

| ID | Task | Who |
|---|---|---|
| P2-1 | Upload `fixtures/sample_outputs` to an HF dataset repo; deploy backend to HF Spaces | Gemini + Aryan (tokens) |
| P2-2 | Deploy frontend to Vercel against the live API; CORS; cold-start "waking up" UX | Gemini + Aryan |
| P2-3 | Smoke test on phone and laptop; fix issues | Aryan + Gemini |

**Exit gate:** public Vercel URL works end-to-end with the SYNTHETIC banner visible. From here on the site always works; later phases only swap the data.

---

## P3 — Real data and baselines
**Needs:** P1-M3 (decisions confirmed). Runs in parallel with P2.

| ID | Task | Ref |
|---|---|---|
| P3-1 | `preprocess.py` + climatology on the one-month sample; sample-day figure; tests 3–5 | M1 |
| P3-2 | Full-period download (server-side subset); full preprocessing; upload processed store to Kaggle Dataset / Drive | §4.1, §11 |
| P3-3 | Baselines: climatology + ridge (LightGBM optional) → `baseline_*_val.csv` | §5.6 |

**Exit gate:** processed store passes schema check and loads in a Colab and a Kaggle notebook; baseline validation metrics exist.

---

## P4 — U-Net training (Colab/Kaggle)
**Needs:** P3.

| ID | Task | Ref |
|---|---|---|
| P4-1 | `dataset.py`, `unet.py`, `train.py` with AMP, resume, run logging; tests 1, 2, 6, 8 | §5 |
| P4-2 | Thin Colab + Kaggle notebooks per runbook | §11 |
| P4-3 | First full training run; verify resume across a session restart | M2 |
| P4-4 | Up to ~5 tuning runs on **validation only** (lr, channels, extra inputs); pick best | M2 |

**Exit gate:** U-Net validation RMSE beats climatology at ≥ 10 of 15 depths. If not, stop and review with Claude/Aryan before more runs (no test-set peeking).

---

## P5 — Evaluation
**Needs:** P4.

| ID | Task | Ref |
|---|---|---|
| P5-1 | `evaluate.py`: per depth / season / region CSVs | §6.3, M3 |
| P5-2 | ARGO download (date range of published splits) + `argo_validation.py` | §7 |
| P5-3 | `evaluation_<run_id>.md` generated by code, with the leakage caveats | M3 |
| P5-4 | Claude reviews the report for leakage and overclaiming | Claude |

**Exit gate:** all §6.3 metric files exist for the best run; report reviewed.

---

## P6 — Real outputs live
**Needs:** P2, P5.

| ID | Task | Ref |
|---|---|---|
| P6-1 | `predict.py` for published splits (float16); `manifest.json`; contract test | M4 |
| P6-2 | `publish.py` → HF dataset repo; restart Space; verify `/meta.synthetic = false` | §6.4 |
| P6-3 | Frontend check with real data: colour ranges, coverage, performance | PRD_FRONTEND §4 |

**Exit gate:** public site serves real predictions; banner gone; validation page shows real metrics.

---

## P7 — Assistant (optional)
**Needs:** P2 (can be built on synthetic data); a provider with tool calling.

| ID | Task | Ref |
|---|---|---|
| P7-0 | **Owner decision:** provider (Meta if available, else OpenAI-compatible) or skip | Aryan |
| P7-1 | Backend `/chat`: provider, tools, prompt, grounding guard, limits; test 7 | PRD_BACKEND §6, M B5 |
| P7-2 | Frontend chat drawer with "Data used" | PRD_FRONTEND §3.4, M F6 |
| P7-3 | Owner tests 20 questions incl. trick ones; enable in prod | Aryan |

**Exit gate:** grounding guard catches fabricated numbers in tests; owner approves enabling.

---

## P8 — Polish, final test, demo
**Needs:** P6 (and P7 if doing it).

| ID | Task | Who |
|---|---|---|
| P8-1 | **Final test run** (`final_test.py`), once, on owner instruction; republish with `test_evaluated = true` | Aryan says go; Gemini runs |
| P8-2 | README: what it is, live URL, architecture diagram, how to reproduce, limitations, data credits | Gemini, Claude reviews |
| P8-3 | Lighthouse/accessibility pass, mobile check, OG image, `/about` text | Gemini |
| P8-4 | Demo script (3–5 min) + backup screen recording in case the Space is asleep or down | Aryan + Claude |
| P8-5 | Freeze: tag release, pin `HF_DATASET_REVISION`, no deploys in the last 12 h before judging | Aryan |

**Exit gate:** Definition of Done in the handoff §12 all ticked; demo rehearsed.

---

## P9 — Stretch (only if time remains, in this order)
SST input → multi-day input window → uncertainty (`temp_pred_std`, needs a contract change) → SAM region selection.

---

## Critical path and owner checkpoints
Critical path: **P0 → P1-M → P3 → P4 → P5 → P6 → P8**. Data download and training are the slowest and riskiest steps, so start P1-M on day one.

Aryan must act at: P0-1, P0-4, P1-M3 (confirm config), P2 tokens, P3/P4 notebook runs, P4 exit gate review, P7-0, P7-3, P8-1, P8-5.
