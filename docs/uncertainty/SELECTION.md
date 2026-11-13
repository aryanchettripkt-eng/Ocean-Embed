# Selection rule for the error-bar fix (written 2026-11-13 before running the grid)

- Data: 2018–2020 held-out floats only. Fit on the dev half, score the final half (`holdout_halves`, seed 2026). 2021 is not read.
- Grid: n_dirs ∈ {4, 8, 16, all}, clip whitened features to the fit range ∈ {no, yes}, mean correction ∈ {yes, no}.
- Rule: among settings whose full-posterior (bll_bam) blow-up share is below 0.5% of final-half rows (error bar > 5× the constant one), pick the lowest mean CRPS over all final-half rows.
- The chosen setting is then run once on 2021 and reported as it comes out.

## Round 1 result (2018–2020 only) and amendment, written before round 2
- None of the 16 settings met the blow-up limit. The lowest blow-up share was 0.62%, at n_dirs 4, and it came with wide bars (CRPS 0.34 vs 0.30 for the constant bar).
- On blown-up rows the real RMS error is 1.6–2.4× the constant bar's sd, but the predicted sd is more than 5×, and 95% coverage there is 100%. The bars point at the right places but overshoot. Log: [selection_round1.log](selection_round1.log).
- Amendment: add a cap on the predicted error size (Aryan's "capped extrapolation"). Options:
  - **fit range**: the log error size per level is kept within the range it took on the fitting rows;
  - **3×**: the error size is kept at or below 3× the constant bar.
- Round 2 grid: n_dirs ∈ {4, 8, 16}, features clipped to the fit range, mean correction ∈ {yes, no}, cap ∈ {fit range, 3×}.
- Rule unchanged: blow-up share below 0.5%, then lowest CRPS on all final-half rows. Tie-break: within 1% of the best CRPS, prefer no mean correction (the bar is then centred on the U-Net's own prediction).

## Round 2 result and choice (2018–2020 only; 2021 not yet read)
- All "fit range" caps still exceed 0.5% blow-ups (0.62–1.34%). Every 3× cap passes (0%, by construction). One run, n_dirs 16 with fit range and mean correction, did not finish; it fails anyway, since all fit-range settings do.
- Lowest CRPS among the passing settings: **n_dirs 8, features clipped, 3× cap, mean correction on**, at 0.2734. That's 7.8% below the constant bar (0.2964), with 68% / 95% coverage of 0.740 / 0.949.
- Tie-break: the best setting without mean correction is 0.2806, which is 2.6% worse, outside the 1% margin. So mean correction stays on.
- The rows hitting the cap have real RMS errors about 2× the constant sd, so a 3× cap still leaves room above their actual errors.
- Log: [selection_round2.log](selection_round2.log).
