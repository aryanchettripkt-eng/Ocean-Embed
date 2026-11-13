# Hierarchical error bars: the rule (written 2026-11-13, before any fit)

The question: the global last layer leaves the Bay of Bengal 50–300 m error bar almost flat. Is that because the features carry no information about the error size there (a feature limit), or because the Gaussian fit (BaM) gets a hierarchical posterior wrong (an inference limit)?

**2021 is not read.** Everything below is fitted and scored on the 2018–2020 held-out floats only (`uncert_val.parquet`, 51 floats). 2021 has been viewed four times already (see [README](README.md)).

## Data and split
- Rows: the 2018–2020 match-ups of the 51 held-out floats. One row per profile and level (`thin`), as in the current fit.
- **5-fold cross-validation over floats**, replacing the dev/final halves. Each float has a home region: the region of its median position, as in `holdout_halves`. Folds are assigned with seed 2026, stratified by home region, so the 15 Bay of Bengal floats split 3 per fold. Fit on four folds, score the fifth, pool the five scored folds.
- The whitening, the per-level scale s_d and every fit are refitted inside each training fold. Nothing from the scored fold is used.
- Fixed settings, from SELECTION.md: 8 whitened directions, clipped to the fit range, mean correction on, error size capped at 3× s_d.

## Regions
- Bay of Bengal (lon ≥ 78°E, lat ≥ 5°N), Arabian Sea (lon < 78°E, lat ≥ 5°N), and equatorial (lat < 5°N). These are the repo's §6.3 regions, made disjoint.
- There is no "south" region, because no 2018–2020 held-out profile lies south of 0°. So R = 3.
- A row's region is the region of its profile, so a float that crosses a boundary contributes to both regions.

## Step 2: is there signal in the Bay of Bengal? (done before any hierarchy)
- **Rows:** Bay of Bengal, 50–300 m, from the scored folds. The target is the scaled error r = (obs − pred) / s_d.
- **Candidates:**
  1. the global layer's predicted log sd, out of fold;
  2. each of the 8 whitened directions;
  3. monsoon phase: Dec–Feb winter monsoon, Mar–May spring inter-monsoon, Jun–Sep summer monsoon, Oct–Nov autumn inter-monsoon;
  4. the U-Net's own upper-ocean stratification at the profile: predicted T at about 5 m minus predicted T at the row's level. This is available at inference time, and it is the closest thing in the table to the barrier-layer and stratification signal.
- **Sea-surface salinity:** it is an input channel of the store, but it is not in the match-up table. Scoring it needs one more Kaggle collect that samples SSS at each profile. Until then it is reported as not tested, not as "no signal".
- **Measures:**
  - Spearman rank correlation of each candidate with |r|, with a float-bootstrap 95% interval;
  - 5-fold (by float) cross-validated R² of a linear fit of log(r²) on the candidate. Monsoon phase enters as one-hot.
  - R² is low even for a perfect error-size model, because log χ²₁ has variance π²/2 ≈ 4.9. So the comparison is between candidates, not against 1.
- **Signal:** a candidate has signal if its Spearman interval excludes 0 and its CV R² is above 0.

## Step 3: the hierarchical layer
Per depth, with x the 9 design columns (8 whitened directions and a constant):
- r ~ N(β·x, exp(γ_g(i)·x)²), where g(i) is the row's region;
- β is global, with β ~ N(0, 0.5²), as now;
- γ_g = γ + δ_g, with γ ~ N(0, 0.5²) and δ_g ~ N(0, τ² I) for each of the 3 regions;
- log τ ~ N(log 0.2, 1²).

The dimension is 9 + 9 + 27 + 1 = 46 per depth. Two parameterisations of the same posterior:
- **Centred:** δ_g is a free parameter. The log prior on (δ, log τ) is −K log τ − Σδ² / (2τ²) + log N(log τ), with K = 27.
- **Non-centred:** δ_g = τ η_g with η_g ~ N(0, I). The log prior is −Σ η² / 2 + log N(log τ).

Each parameterisation gets an analytic gradient, checked against finite differences in `test_uncertainty.py`.

Three fits of the hierarchical layer:
- **Laplace:** at the MAP of the non-centred posterior. The joint MAP of the centred version can collapse to τ → 0, which is part of the funnel problem.
- **BaM, centred:** starts from the Laplace fit mapped to centred coordinates, with the same schedule as now (reg0 = 10, 100 iterations).
- **BaM, non-centred:** same start and schedule, in non-centred coordinates.

## Step 4: the reference posterior
- **NUTS (numpyro)** on the same model, non-centred, at three depths: 92.3 m, 186.1 m and 380.2 m. These are the levels in the table nearest 100, 200 and 500 m.
- It is fitted on all 2018–2020 rows (thinned), not cross-validated, because it is a reference for the posterior, not a forecast.
- Settings: 4 chains, 1000 warm-up and 1000 draws each. The reference is used only if every R-hat is below 1.01 and there are no divergences. Centred NUTS is also run, and its divergence count is reported as a funnel diagnostic.
- The comparison is made on the same rows and the same whitening for all four methods: Laplace, BaM-centred, BaM-non-centred and NUTS.
  - a scatter plot of (log τ, δ_BoB on the constant column) from the NUTS draws;
  - sd(log τ) for each method divided by the NUTS value, and the same for δ_BoB on the constant column;
  - whether centred BaM converges: its distance to the MAP, as now, and whether its estimate of log τ is stable over the last 20 iterations.
- **Funnel verdict:** BaM's sd(log τ) is under 0.7× the NUTS sd(log τ). This is reported per depth and per parameterisation.

## Step 5: scoring
- **Methods:**
  - the constant bar;
  - the global layer (BaM, as now);
  - the hierarchical layer with Laplace;
  - the hierarchical layer with BaM-centred;
  - the hierarchical layer with BaM-non-centred.
- Every method is scored on the pooled out-of-fold rows on 68% and 95% coverage, CRPS, NLL and mean sd. Each score gets a float-bootstrap 95% interval.
- **Groups:** Bay of Bengal 50–300 m, each depth band, and all rows.
- **"Sharpens" verdict:** in the Bay of Bengal 50–300 m, a hierarchical version sharpens if all three of these hold:
  1. its mean sd is below the constant bar's;
  2. its 95% coverage is 0.93 or above;
  3. its CRPS is below the global layer's, with the paired float-bootstrap interval of the difference excluding 0.

  If the point estimate is lower but the interval includes 0, that is reported as "not shown".

## Step 6: which question to ask
The question is chosen from the findings, using the table in Aryan's plan of 2026-11-13. It is not chosen in advance.

---

# Results (2026-11-13, 2018–2020 only; 2021 not read)

Files are in [hierarchy/](hierarchy/). There are 51 held-out floats and 1,219,070 rows in total. The Bay of Bengal 50–300 m group has 15 floats and 92,458 rows.

## Step 1: the global layer under 5-fold CV
The baseline is the global layer on the new split. In the Bay of Bengal 50–300 m:
- coverage is 0.76 at 68% and 0.97 at 95%;
- the mean sd is 0.823 °C, against 0.834 °C for the constant bar;
- CRPS is 3.0% better than the constant bar (0.9–4.3%).

Over all rows, CRPS is 5.7% better (4.0–7.1%), with coverage 0.71 / 0.94. These numbers are close to the earlier halves check, so the new split doesn't change the picture. Files: `cv_scores_global_only.csv`.

## Step 2: is there signal in the Bay of Bengal?
The test is on out-of-fold Bay of Bengal 50–300 m rows, against |r| and log r². Files: `signal_bob.csv` and `signal_within_depth.txt`.

| Candidate | Spearman with \|r\| (95% CI) | CV R² on log r² |
|---|---|---|
| Global layer's predicted log sd | 0.14 (0.08 to 0.25) | 0.010 |
| U-Net predicted stratification, T(5 m) − T(z) | −0.15 (−0.27 to −0.10) | 0.009 |
| Depth (level) alone | −0.12 | 0.007 |
| Each single whitened direction | −0.06 to 0.01 (one interval excludes 0) | all below 0 |
| All 8 directions together | | −0.004 |
| Monsoon phase | | −0.013 |
| Sea-surface salinity | not tested: not in the table, needs one Kaggle collect | |

- **Monsoon phase:** no signal. Summer-monsoon errors are a little larger (mean |r| 0.72 against 0.63–0.65), but this does not carry over to held-out floats.
- **Stratification:** some signal, but much of it is depth. Within each level it changes sign:
  - more stratified profiles have larger errors at 66 m (ρ = +0.17);
  - they have smaller errors at 130–266 m (ρ −0.11 to −0.26).

  On top of a per-depth offset it raises CV R² from 0.010 to 0.013. The global layer's log sd raises it to 0.018.
- **Overall:** the signal is real but small. R² of 0.018 corresponds to the predictable part of log sd varying by about ±15%. The individual features carry almost nothing on their own. What the global layer gets right is mostly a Bay of Bengal-specific error size per depth, and that is exactly what a per-region term can add.

## Step 4: the reference posterior and the funnel
Files: `reference_compare.csv`, `funnel_nuts.png`, `nuts.log`, `diag_nc.log`.

The NUTS run was clean:
- non-centred, there were 0 divergences at all three depths and R-hat ≤ 1.002;
- centred, NUTS is fine at 92 m and 186 m, but at 380 m it has 252 divergences and R-hat 42.7. Centred NUTS fails there.

| Depth | NUTS log τ | Laplace | BaM centred | BaM non-centred |
|---|---|---|---|---|
| 92 m | −2.33 ± 0.23 | −0.57 ± 0.57 | −2.36 ± 0.15 (0.62×) | −0.92 ± 0.015 (0.06×) |
| 186 m | −3.09 ± 0.27 | −0.82 ± 0.67 | −3.04 ± 0.14 (0.53×) | −1.34 ± 0.022 (0.08×) |
| 380 m | −2.32 ± 0.20 | −0.72 ± 0.66 | −2.34 ± 0.14 (0.68×) | −0.74 ± 0.008 (0.04×) |

Each cell is the mean ± sd of log τ. The ratio in brackets is sd(log τ) relative to NUTS.

- **The posterior itself is not a funnel here.** The data identify τ well (τ ≈ 0.05–0.10), and the NUTS draws of (log τ, δ_BoB) form a plain ellipse. The Bay of Bengal offset is real at 380 m (δ_BoB ≈ −0.22, a 20% smaller error size there) and small at 92 m and 186 m.
- **BaM centred** gets the location of log τ right from the default start. Its spread is 0.53–0.68× the NUTS value, below the 0.7 line at all three depths, so by the rule this is the **funnel verdict**. Its sd for δ_BoB matches NUTS (0.91–0.98×).
  - With larger batches, started at the NUTS mean, its mean slides into the narrow end: −3.13 at batch 1024, against −2.33 for NUTS. The sd stays at about 0.6×.
- **BaM non-centred** is much worse:
  - from the Laplace start it settles in the wrong place;
  - started at the NUTS answer, it keeps the location but its sd for log τ collapses to about 0.1× NUTS, whatever the batch size (64, 256 or 1024) or the number of iterations;
  - in CV it is unstable: at one shallow level in one fold the mean correction runs off, and its overall NLL explodes.

  Non-centring turns well-identified group effects into a curved ridge between η and log τ (η ∝ 1/τ for a fixed δ). A Gaussian fitted by score matching seems to squeeze that ridge.
- **Laplace** (at the non-centred joint mode) puts log τ at about −0.7. The joint mode in (η, log τ) is not the marginal mode, which is a known failure of that parameterisation.

## Step 5: scores under 5-fold CV
Bay of Bengal 50–300 m: 15 floats, 92,458 rows. File: `cv_scores.csv`.

| Method | Cover 68 | Cover 95 (CI) | Mean sd °C | CRPS skill vs constant | NLL |
|---|---|---|---|---|---|
| Constant bar | 0.77 | 0.97 (0.95–0.98) | 0.834 | 0 | 1.060 |
| Global layer (BaM) | 0.76 | 0.97 (0.95–0.98) | 0.823 | 3.0% (0.9–4.3) | 1.004 |
| Hierarchical, Laplace | 0.70 | 0.97 (0.95–0.98) | 0.853 | 0.9% (−0.8–3.0) | 1.003 |
| **Hierarchical, BaM centred** | 0.71 | 0.95 (0.92–0.96) | **0.746** | **3.7% (1.3–4.8)** | 0.988 |
| Hierarchical, BaM non-centred | 0.69 | 0.94 (0.90–0.95) | 0.767 | 0.8% (−6.7–3.3) | 1.390 |

Paired CRPS difference against the global layer, Bay of Bengal 50–300 m:
- centred BaM: −0.0026 °C (−0.0045 to −0.00005);
- Laplace: +0.0088 (+0.0031 to +0.0110);
- non-centred BaM: +0.0094 (−0.0010 to +0.0284).

**Sharpens verdict:** the centred BaM hierarchy meets all three criteria in the Bay of Bengal 50–300 m:
1. its mean sd is 0.746, below the constant bar's 0.834;
2. its 95% coverage is 0.953, at least 0.93;
3. its CRPS is lower than the global layer's, and the interval excludes 0.

The gain is small: the bars are about 9% narrower than the global layer's, and CRPS skill goes from 3.0% to 3.7%. The interval only just excludes 0.

Over all rows, the centred hierarchy is level with the global layer: CRPS skill 6.0% against 5.7%, and 95% coverage 0.94. Laplace and non-centred BaM do not sharpen.

## What it means (Step 6)
- **Features:** they carry a little signal about the Bay of Bengal error size, and season carries none. Salinity is still untested. A per-region error size helps a little, through a Bay of Bengal-specific level per depth, not through new information in the features.
- **Inference:** the BaM result depends on the parameterisation, and in the opposite direction to the usual HMC advice:
  - centred BaM is close: it gets the right location, with the spread 0.5–0.7× too narrow;
  - non-centred BaM collapses the log τ spread about 10× and can be unstable;
  - NUTS is fine non-centred and breaks centred at 380 m.
- **The email question** (from the plan's table, "Non-centring fixes BaM" turned around):
  > With informative data (τ well identified), BaM fitted in the non-centred parameterisation collapsed the posterior sd of log τ to about 0.1× the NUTS value, even when started at the NUTS answer. The centred version kept the right location but was 0.5–0.7× too narrow. Since the score-based divergence is not reparameterisation-invariant, is there a known rule for which parameterisation BaM should use in hierarchical models? Or is a curved (η, log τ) ridge where a richer family such as EigenVI is needed?
- **Not done:** sea-surface salinity in Step 2. It needs one Kaggle collect that samples the SSS input channel at each profile.

## Cross-check at 380 m: official GSM-VI code and the divergence (added 2026-11-13)
Script: `ml/scripts/hier_gsmvi_check.py`. Log: `hierarchy/gsmvi_check_380.log`.

The non-centred posterior was refitted with the official `gsmvi.bam.BaM`: the paper's `linear` schedule (reg0 = 10), batch 64, float64. A fresh NUTS run gives log τ −2.31 ± 0.198 with 0 divergences.

| Start | 100 iterations | 1000 iterations |
|---|---|---|
| This repo's Laplace fit | −1.08 ± 0.017 | −1.18 ± 0.019 |
| NUTS-matched Gaussian | −2.35 ± 0.051 | −2.35 ± 0.048 |
| N(0, I) | −0.48 ± 0.012 | −0.50 ± 0.014 |

The official code shows the same collapse as this repo's implementation: the sd of log τ is 0.06–0.25× the NUTS value. Started at the NUTS answer, it keeps the location and shrinks the spread. From the other starts it also settles in the wrong place. Official centred BaM from the Laplace start gives −2.33 ± 0.13 (0.66×), in line with this repo's centred result.

**The score-based divergence** BaM minimises is D(q; p) = E_q ‖Cov(q)^½ (∇log q − ∇log p)‖². It was estimated by Monte Carlo with 20,000 draws in non-centred coordinates:
- the collapsed BaM solutions score 2.6–7.3, which includes this repo's 100-iteration fit (7.3);
- the NUTS-matched Gaussian scores 491;
- Laplace scores 2.6 × 10¹⁷.

The divergence is about 100× lower at the collapsed solutions than at the Gaussian with the NUTS mean and covariance. So the collapse comes from the objective, not from BaM failing to optimise it. The divergence is also nearly flat across collapsed solutions at quite different log τ (−1.1 and −2.35 both score about 3), which is why the end point depends on the start.
