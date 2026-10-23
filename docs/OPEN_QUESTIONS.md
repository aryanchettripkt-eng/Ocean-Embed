# Open questions

Format: `- [YYYY-MM-DD] (area, asked by) question — status`

- [2026-10-22] (ml, Claude) Create Copernicus Marine, NASA Earthdata, Copernicus CDS and Hugging Face accounts. — open
- [2026-10-22] (ml, Claude) Confirm region bbox, 15 GLORYS depths, time period (1993 start) and splits after the M1 data inspection. — open
- [2026-10-22] (ml, Claude) ARGO source: Argo GDAC or Copernicus INSITU? — open
- [2026-10-22] (ml, Claude) GREP: confirm end date (catalogue says 2020-12-15, docs say "regularly updated") and members at M1. If it ends in 2020, the §1.1 test-year comparison is GLORYS12 + ARMOR3D only. — open
- [2026-10-22] (ml, Claude) Storage: size of 1993–2021 inputs (~143 channels) vs Kaggle Dataset limits and Drive quota; decide after the one-month download. — answered 2026-10-23: stream and reduce, store ≤ 20 GB on a Kaggle Dataset (DECISIONS 2026-10-23, PRD_AI_ML §4.2)
- [2026-10-22] (backend, Claude) Chat provider: Meta (Muse Spark / Llama API) if access is granted; otherwise which OpenAI-compatible provider, or keep chat off? — open
- [2026-10-22] (frontend, Claude) Branding, credits text, custom domain? — open
- [2026-10-23] (ml, Claude) CCMP winds source for the full period: the RSS server ran at ~30 KB/s. Options: a faster CCMP mirror or OPeNDAP endpoint (same product, preferred), or switch to Copernicus `WIND_GLO_PHY_L4_MY_012_006` (different product, needs a DECISIONS entry). — open
- [2026-10-23] (ml, Claude) Start year: keep 1993 (more history) or start in 2004 (ARGO era, about 40% less data and build time)? To be proposed with the M1 config. — open
- [2026-10-24] (frontend, Claude) 3D view on the landing page: open the Explorer in *3D* on desktop for the demo (recommended), or in *Map* with a visible "3D" switch? — answered 2026-10-24: open in *3D* on desktop (Aryan)
- [2026-10-24] (ml, Claude) Synthetic fixtures are at 1° spacing, so the 3D walls look coarse until real 0.25° outputs exist. Regenerate the fixtures at 0.25° for the demo period (still < 5 MB if only a few days), or accept the coarse look until P6? — answered 2026-10-24: regenerate at 0.25° (Aryan)
