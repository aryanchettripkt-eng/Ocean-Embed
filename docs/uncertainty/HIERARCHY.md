# Hierarchical error bars: the rule (written 2026-11-13, before any fit)

The question: the global last layer leaves the Bay of Bengal 50–300 m error bar almost flat. Is that because the features carry no information about the error size there (a feature limit), or because the Gaussian fit (BaM) gets a hierarchical posterior wrong (an inference limit)?

**2021 is not read.** Everything below is fitted and scored on the 2018–2020 held-out floats only (`uncert_val.parquet`, 51 floats). 2021 has been viewed four times already (see [README](README.md)).

## Data and split
- Rows: the 2018–2020 match-ups of the 51 held-out floats. One row per profile and level (`thin`), as in the current fit.
- **5-fold cross-validation over floats**, replacing the dev/final halves. Each float has a home region: the region of most of its profiles. Folds are assigned with seed 2026, stratified by home region, so the 15 Bay of Bengal floats split 3 per fold. Fit on four folds, score the fifth, pool the five scored folds.
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
