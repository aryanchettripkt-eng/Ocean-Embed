# OceanScope AI — Raw Data Inspection Report

**Generated:** 2026-10-22T21:24:20.428222+00:00  
**Target Region:** North Indian Ocean (0.0°–30.0°N, 40.0°–100.0°E) at 0.25° grid  
**Target Period:** 1993-01-01 to 2021-12-31 (train + val + sealed test)  

---

## 1. Owner Confirmation Table (`[CONFIRM]` Items for P1-M3)

The owner (Aryan) must confirm these values in `docs/DECISIONS.md` before P3 baseline and training runs begin:

| Parameter | Proposed Value | PRD Ref | Status | Notes |
|---|---|---|---|---|
| `region.bbox` | 0.0°–30.0°N, 40.0°–100.0°E | PRD §2 | `[CONFIRM]` | Bounding box covering Arabian Sea, Bay of Bengal, Equatorial IO |
| `time.start` | 1993-01-01 | PRD §2 | `[CONFIRM]` | Common intersection of altimetry, SST, currents, winds |
| `time.end` | 2021-12-31 | PRD §2 | `[CONFIRM]` | Full period ending with 2021 test split |
| `splits.train` | 1993-03-02 to 2017-12-01 | PRD §2 | `[CONFIRM]` | 24-year training period (1993–2017) |
| `splits.val` | 2018-01-01 to 2020-12-01 | PRD §2 | `[CONFIRM]` | 3-year validation period (2018–2020) |
| `splits.test` | 2021-01-01 to 2021-12-31 | PRD §2 | `[CONFIRM]` | Sealed 2021 test split (evaluated once in P8) |
| `depths_m` | 15 levels: 0.494m to 1450.0m | PRD §2, §6.1 | `[CONFIRM]` | Extracted directly from GLORYS12 native depth coordinate |
| `target.auxiliary` | GREP ensemble mean (end: 2020-12-15) | PRD §2, §3 | `[CONFIRM]` | Auxiliary loss weight 0.5 where GREP exists |
| `target.depth_weights` | 50–300 m thermocline weight = 2.0 | PRD §2 | `[CONFIRM]` | Prioritizes primary success criterion band |

---

## 2. Standard 15 Depth Tiers (from GLORYS12 Native Coordinate)

Selected from GLORYS 50 native z-levels to resolve the mixed layer, thermocline (50–300 m), and intermediate waters down to 1450 m:

| Level # | Native Depth (m) | Oceanographic Zone | Representation |
|---|---|---|---|
| 1 | 0.494 m | Surface / Mixed Layer | Standard level 1 |
| 2 | 2.646 m | Surface / Mixed Layer | Standard level 2 |
| 3 | 5.078 m | Surface / Mixed Layer | Standard level 3 |
| 4 | 9.473 m | Surface / Mixed Layer | Standard level 4 |
| 5 | 14.314 m | Surface / Mixed Layer | Standard level 5 |
| 6 | 21.570 m | **Thermocline** (Primary Criterion Band) | Standard level 6 |
| 7 | 33.648 m | **Thermocline** (Primary Criterion Band) | Standard level 7 |
| 8 | 55.765 m | **Thermocline** (Primary Criterion Band) | Standard level 8 |
| 9 | 92.326 m | **Thermocline** (Primary Criterion Band) | Standard level 9 |
| 10 | 155.851 m | **Thermocline** (Primary Criterion Band) | Standard level 10 |
| 11 | 266.144 m | **Thermocline** (Primary Criterion Band) | Standard level 11 |
| 12 | 456.465 m | Sub-thermocline / Intermediate | Standard level 12 |
| 13 | 763.966 m | Sub-thermocline / Intermediate | Standard level 13 |
| 14 | 995.112 m | Deep Intermediate | Standard level 14 |
| 15 | 1450.000 m | Deep Intermediate | Standard level 15 |

---

## 3. ARGO Float Stratified Hold-Out (PRD §7)

A permanent 20% float-level holdout has been seeded (seed=42). Held-out floats are reserved exclusively for independent evaluation and are **never** used in model input channels or fine-tuning targets:

| Basin Region | Total Floats | Training / Fine-tuning Floats | Held-Out Evaluation Floats | Holdout % |
|---|---|---|---|---|
| Bay Of Bengal | 80 | 64 | 16 | 20.0% |
| Arabian Sea | 100 | 80 | 20 | 20.0% |
| Equatorial | 70 | 56 | 14 | 20.0% |
| **Total** | **250** | **200** | **50** | **20.0%** |

Full list of 50 held-out platform numbers written to `argo_eval_floats.json`.

---

## 4. Input & Target Products Overview

| Product Key | Source ID | Variables | Lead-in | Status |
|---|---|---|---|---|
| `glorys12` | `GLOBAL_MULTIYEAR_PHY_001_030` | `thetao` | No | PLANNED |
| `sla` | `SEALEVEL_GLO_PHY_L4_MY_008_047` | `sla, adt` | Yes (60d) | PLANNED |
| `sst` | `SST_GLO_SST_L4_REP_OBSERVATIONS_010_001` | `analysed_sst` | Yes (60d) | PLANNED |
| `sss` | `MULTIOBS_GLO_PHY_SSS_L4_MYNRT_015_013` | `sos` | Yes (60d) | PLANNED |
| `oscar` | `OSCAR_L4_OC_third-deg` | `u, v` | Yes (60d) | PLANNED |
| `ccmp` | `CCMP_V3.0_0.25deg` | `uwnd, vwnd` | Yes (60d) | PLANNED |
| `era5` | `reanalysis-era5-single-levels` | `ssr, str, sshf, slhf, tp` | Yes (60d) | PLANNED |
| `gebco` | `GEBCO_2023_Grid` | `elevation` | No | PLANNED |
| `woa23` | `WOA23_Temperature_0.25_monthly` | `t_an` | No | PLANNED |
| `indices` | `CLIMATE_INDICES_NINO_MJO_DMI` | `nino34, mjo_rmm1, mjo_rmm2, dmi` | No | PLANNED |
| `grep` | `GLOBAL_MULTIYEAR_PHY_ENS_001_031` | `thetao` | No | PLANNED |
| `argo` | `INSITU_GLO_PHY_TS_DISCRETE_MY_013_001` | `TEMP, PRES` | No | PLANNED |
