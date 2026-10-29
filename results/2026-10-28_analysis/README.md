# I11 full-arm analysis, 2026-10-28 (owner analysis)

Project `/project/home/p201509/work/cvri/simulation`. Governing text: `final/AMENDMENT_2026-10-28.md` over
`final/STUDY_PLAN.md` section 4. No pre-registered line or Amendment 1 setting moved. Every number is in `RESULTS.md`
(tables per estimand, generated) and `results.json`; nothing here is hand-typed except this summary.

## Regenerate
`sbatch --qos=default --time=01:00:00 code/full/analysis/analysis_job.sh 2000 results/2026-10-28_analysis figures/drafts`
(script `code/full/analysis/run_analysis.py`, with `write_results_md.py` and `figures_draft.py`; SHA-256 in
`code_sha256.txt`). Script runtime 220 s (job 5395517; 397 s wall including Python imports on Lustre).

## Jobs (account p201509, partition cpu, 1 node)
| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5395475 | test | test; FAILED at start (/usr/bin/time absent) | 00:00:07 | 0.00 |
| 5395478 | test | nboot 200, all outputs to test/ (now `test_job5395478.tar.gz`) | 00:04:55 | 0.08 |
| 5395486 | default | full, nboot 2000 | 00:05:41 | 0.09 |
| 5395517 | default | full rerun after a figure c legend fix; numbers identical to 5395486 | 00:06:42 | 0.11 |
Total 0.28 CPU node-hours (cap 5). Logs: `logs_analysis_2026-10-28.tar.gz`.

## Inputs (upstream, read only)
`results/2026-10-28_armC/{estimates,constructions,noise_estimates}.csv` (calibC), `results/2026-10-28_G3G4_ministudy/estimates.csv`
(one-voxel trio), `results/2026-10-28_armT/estimates.csv` + `analysis.json`, `results/2026-10-28_armF/estimates_armF.csv` +
`summary_armF.json`, `data/processed/acdc_mesh_v2/MANIFEST.tsv` (exclusions). No external sources.

## Software and seeds
Env `cvri-sim`: Python 3.11.17, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, statsmodels 0.15.0 (check only), matplotlib 3.11.2,
patsy (statsmodels dependency). Font Nimbus Sans via `code/figures/vendor/figstyle.py`. Every bootstrap: 2,000
cluster resamples over hearts, fresh `numpy.random.default_rng(20261028)` per cell. Nothing installed.

## Definitions chosen here (plan or A3 silent): see the docstring of run_analysis.py
1. CV per heart = median over θ* of SD/mean across constructions (reference included), then median over hearts.
2. LMM shares: REML random-intercept model, sum-to-zero coding, share = term fixed-effect variance (or variance
   component) over the sum of all six. Fitted with own closed-form REML (agrees with statsmodels MixedLM to 4e-7 in
   variance components, 8e-10 in β). The plan's model has no heart × source term, so heart-specific network effects
   go into the residual; the gates-preview per-heart sums-of-squares split (E2b) is reported beside it.
3. Break-even: median over hearts of SD_noise against median over hearts of SD_cons, log-linear in severity
   (−dB, or ln level); "beyond grid", never extrapolated. Censored noise replicates kept at the bound (A1 rule), dropped in
   a sensitivity. This answers calibC open question 2.
4. Three-length rule: gates `g4_block` logic; flag on in-sample R² ≥ 0.9 for both (plan wording); LOHO also reported.
5. Arm T: recomputed from estimates.csv (round 1), identical to the owner's point values (max difference 0). Arm F: per-heart
   mean log ratio, geometric mean over hearts; matches the owner's 0.9706 and 0.9293.

## Verification
Independent login-node recomputation (separate code): primary CV 2.576% (v_endo), 16.084% (v_t); displacement break-even
v_endo 19.77 mm. Both are identical to RESULTS.md.

## Figures (DRAFT, `figures/drafts/`)
Files: `draft_fig_a_construction_cv`, `draft_fig_b_variance_shares`, `draft_fig_c_noise_vs_construction` (.pdf, 600 dpi .png,
_grey.png, _actualsize.png, raw CSV). 183 mm wide. Checked at print size and in greyscale. Residual defects: in fig a
the five sources differ only by colour, so they cannot be told apart in greyscale. figqa reports missing labels on shared
axes (by design) and error-bar caps at the axes edge in fig b. Some fig b point estimates lie outside their percentile
CI (skewed bootstrap), and those bars are clipped at 0.

## Not done or not verified
- The variance-share bootstrap and the free-rule R² were not recomputed independently.
- The RV wall 2 mm set at 1.5 mm cannot be estimated (1 construction). The 2 mm comparison uses the 1.0 mm extra set, which was not pre-registered.
- The trio heart 104 A2 v_endo CV is 4.6% here against 4.7% in the calibC README. I did not trace the difference.

## Correction 2026-10-28 (follow-up round, owner analysis)

No pre-registered definition or Amendment 1 setting changed. `run_analysis.py` was rerun end to end (nboot 2000, seed
20261028; jobs below) and all 3,711 numeric values of the earlier `results.json` (job 5395517) are identical; the only
changed fields are the two E6 effect labels. New items are additive and labelled. Code: docstring block "Correction
2026-10-28" in `code/full/analysis/run_analysis.py`; tables in `RESULTS.md` (section "Correction 2026-10-28").

| item | old value | new value | reason |
|---|---|---|---|
| E1 reference construction | included, membership not stated | stated as a member of S x M (STUDY_PLAN section 3: expert labels are a level of S; section 4 item 1: CV across S x M); primary kept. Sensitivity without it: v_endo 2.6% [2.2%, 3.0%], v_t 15.0% [12.5%, 19.4%]; paired difference v_endo 0.0% [-0.2%, 0.2%], v_t -1.1% [-1.5%, 1.0%] | reference θ̂/θ* is 1.000 by construction |
| E2 term "heart" | label "heart" | "heart-level mean construction offset plus between-heart variation" (values unchanged); descriptive per-heart sequential SS on non-reference rows added (E2c) | reference rows sit at log y = 0, so the heart intercept absorbs each heart's mean construction offset |
| E6 Kruskal-Wallis effect | "epsilon^2" = (H - k + 1)/(n - k): v_endo 0.005, v_t 0.402 | "eta^2_H", same values; epsilon^2_R = H/(n - 1) added: v_endo 0.086 [0.042, 0.344], v_t 0.451 [0.347, 0.657] | formula is eta^2_H |
| Root-shift flag denominator | 3 of 197 (1.5 mm), 4 of 247 (1.0 mm), primary set, read as the A4 count | A4 literal: 8 of 296 (1.5 mm), 9 of 395 (1.0 mm); primary set 3 of 197, 4 of 247 kept beside it | A4 denominator is all non-reference constructions of that resolution |
| statsmodels warning | printed once at run_analysis.py:329, fit not named | recorded per parameter: v_endo only | report, do not hide |
| E3 break-even | "beyond grid" | unchanged; extrapolated descriptive (E3x) added, labelled not pre-registered | requested descriptive |

Root-flag reconciliation: mesh2 README 8 of 245 and 9 of 295 counted RV wall 3 mm constructions only; calibC README
8 of 294 and 9 of 393 counted calibrated constructions; the A4 count (296, 395) adds the RV wall sensitivity sets (51 at
1.5 mm, 100 at 1.0 mm) to the mesh2 count, and differs from calibC by four constructions with a mapped root that were not
calibrated (G1 mask not usable: heart 129 CineMA and heart 133 nnU-Net UKB SAX, both resolutions). All flags lie in
hearts 114 and 133 (primary) and ukbb_cardiac (109, 113, 118, 131, 136).

ConvergenceWarning: raised by the statsmodels MixedLM cross-check for v_endo only. statsmodels 0.15.0 emits it whenever an
estimated random-effect variance is below a fixed 0.01; v_endo s2 is 0.00456. The fit reports converged, the own REML fit
that produces the shares agrees to 8.5e-10 in β, and the variance is not on the boundary.

E3x (extrapolated, descriptive, not pre-registered; OLS of log median noise SD on severity over all tested levels,
correlation loss): white noise v_endo -8.9 dB [-11.3, -7.7], v_t -12.6 dB [-13.9, -10.7]; lp40 2.8 dB [0.5, 3.7], 0.7 dB
[-0.9, 2.4]; baseline wander 60.3% [46.5, 82.6], 46.4% [38.6, 67.6]; displacement 20.0 mm [16.2, 28.0] (inside the grid;
interpolated primary 19.8 mm), 32.4 mm [22.9, 43.2]. Each lies far outside the tested range except displacement v_endo,
and depends on log-linearity continuing beyond it.

Open for a later decision: a heart x source variance-component model (log y ~ S * M + θ* + (1 | heart) + (1 | heart:source)) as
a candidate replacement for reading the "heart" share; not fitted.

Correction jobs (account p201509, qos test): 5395684 test nboot 200 into notes/scratch (00:06:26, 0.11); 5395728 full
(00:06:30, 0.11); 5395744 full after a figure b label fix (00:06:32, 0.11; final outputs). Total 0.33 CPU node-hours.

## Correction/Addition (Amendment 2), 2026-10-28 (owner varmodel)

Governing text: `final/AMENDMENT_2_2026-10-28.md` B2 and B3, declared before these analyses were run and added after the
full-arm data were seen. No pre-registered estimand, line or primary value changed. E2 stays the pre-registered model.
Code: new labelled section "E2d" and a B3 block in `code/full/analysis/run_analysis.py` (docstring block "Amendment 2"),
rendering in `write_results_md.py`. Tables: `RESULTS.md` sections "E2d", the B3 row and sentence in "E7", and "Addition
(Amendment 2)". JSON keys added: `E2d_secondary_heart_source`, `E7_B3_rv2_lower_bracket`.

Unchanged-value check: results.json backed up first (`notes/scratch/2026-10-28-varmodel-results.pre-B2.json`, output of
job 5395798). After job 5395848, all 3,325 numeric and 853 non-numeric leaves of the old file are identical (runtime keys
excluded). `tables/*.csv` and every figure PNG and CSV in `figures/drafts/` are byte-identical; the three figure PDFs
differ in embedded metadata only. In RESULTS.md the only changed earlier line is the runtime.

B2 (E2d, secondary model). log(theta_hat/theta*) ~ S x M + theta* + (1 | heart) + (1 | heart:source), REML, primary
non-reference rows (50 hearts, 444 constructions, 247 heart:source groups, 1,332 estimates per loss), both losses.
Within-heart construction share = (V_S + V_M + V_SxM + s2_heart:source) / (same + s2_resid), with 95% cluster-bootstrap CI:

| loss | v_endo | v_t |
|---|---|---|
| correlation (primary) | 22.0% [4.9, 32.9] | 58.3% [46.9, 67.7] |
| amplitude (secondary) | 53.1% [44.5, 66.4] | 58.1% [47.0, 67.6] |

For comparison, E2 (pre-registered, no heart:source term, reference rows included) gives 2.7% and 19.3%. Most of the
added construction share is the heart:source component (correlation loss: 87.4% of construction variance for v_endo,
69.1% for v_t), i.e. network effects that differ between hearts. Variance components and all shares are in RESULTS.md.
- Fitting: own two-component REML (relative SDs bounded at 0 by L-BFGS-B, 4 starts). Cross-check against statsmodels
  0.15.0 MixedLM (`vc_formula` for heart:source) on all four point fits: variance components agree to at most 1.6e-4
  relative, fixed effects to at most 1e-7; the own REML criterion is equal or lower at its optimum.
- statsmodels "MLE may be on the boundary" ConvergenceWarning for corr v_endo and amp v_endo (its fixed 0.01 absolute
  threshold; all variances there are below 0.01). No point fit is on the boundary in the own REML.
- Bootstrap counts out of 2,000 per cell (corr v_endo, corr v_t, amp v_endo, amp v_t): heart:source on the boundary 63, 0,
  0, 0; heart on the boundary 0 in all; best start not converged 12, 15, 17, 21; any of 4 starts not converged 52, 68, 73,
  90; rank-deficient 0; errors 0. Flagged fits are kept in the CIs.
- The expert S x M column is dropped (no expert 1.5 mm row in non-reference data). The split of fixed construction
  variance among S, M and S x M depends on this coding, and so, slightly, does the within-heart share (corr v_endo 21.7%
  with the task114 S x M column dropped instead, against 22.0%; login-node check). The random-effect variance components
  and fitted values do not depend on it. Fixed-term variances are
  biased upward in resamples, so some point estimates lie below their percentile CI.

B3 (E7). The 1.0 mm RV wall 2 mm set (10 hearts) is now a labelled E7 row: "not pre-registered; 1.5 mm level not
estimable (49 of 50 fail G2 F3)", the count computed from the mesh MANIFEST (50 constructions at 1.5 mm: 49 fail G2 F3).
Values repeat the existing 1.0 mm row: v_t 12.5% [8.4, 20.2], difference against RV wall 3 mm at 1.0 mm -1.5 points
[-11.0, 1.2]; v_endo 2.5% [1.3, 4.3], difference -0.0 points [-1.3, 1.2]. The 4 mm level at 1.0 mm is listed beside it as
the upper bracket. No new runs.

Seeds: every E2d bootstrap cell uses a fresh `numpy.random.default_rng(20261028)` and `rng.choice` over hearts per
resample (as E2); B3 reuses the E7 resamples (seed 20261028). No other stochastic step.

Job (account p201509, partition cpu, qos test, 1 node): 5395848, nboot 2000, end to end, 00:06:21, 0.11 CPU node-hours.
Log: `logs_varmodel_2026-10-28.tar.gz`. Nothing installed. After the job, one generated sentence (the E2d coding note) was corrected in `write_results_md.py` and RESULTS.md was re-rendered from the job's results.json on the login node (no recomputation; diff confirmed that only that line changed); `code_sha256.txt` was refreshed.

Assumptions (no user input possible): (1) theta* enters as the 3-level index factor, as E2; (2) "within-heart
construction share (heart:source + residual attributable to S and M)" is read as the E2 form with heart:source added to
the construction part; the fixed-only variant and the source part are reported beside it; (3) boundary = relative SD below
1e-4; (4) the B3 row relabels existing values rather than recomputing them.

## Addition (Amendment 2 B1 and B4), 2026-10-28 (owner integration)

Governing text: `final/AMENDMENT_2_2026-10-28.md` B1 and B4 (declared before these runs, added after the full-arm data were
seen). No pre-registered estimand, line or primary value changed; the primary estimand stays on the A2 layer and E3 on the
A3 grid. Code: new labelled sections "B1" and "B4" in `code/full/analysis/run_analysis.py` (docstring lines B1, B4) and
`b1_lines`, `b4_lines` in `write_results_md.py`. `COMP` (A3 grid, used by E3, E3x and the figures) is unchanged; the
extended grid is a separate `COMP_EXT`. JSON keys added: `B1_smooth_layer_SENSITIVITY`, `B4_E3_extended_grid`. New
tables: `tables/B1_per_heart_cv_sdf_corr.csv`, `tables/B4_noise_sd_per_heart_corr_newlevels.csv`. RESULTS.md sections
"B1" and "B4".

Unchanged-value check: results.json, RESULTS.md, README.md and both scripts backed up first
(`notes/scratch/2026-10-28-integration-*.pre-B1B4.*`). After job 5396070, NaN-aware comparison: all 3,703 numeric and
942 non-numeric leaves of the previous results.json are identical (runtime keys excluded). The 9 earlier `tables/*.csv`
and every figure PNG and CSV are byte-identical; the three figure PDFs differ (embedded metadata, as in earlier reruns).
In RESULTS.md two earlier lines changed: the runtime line, and the "Addition (Amendment 2)" sentence, which said that B1
and B4 were not in the file and now points to the B1 and B4 sections.

B1 (`results/2026-10-28_layer_sdf/`). E1 CV of theta_hat/theta*, median over 50 hearts [95% CI]; difference B1 minus A2
in percentage points, hearts resampled jointly (2,000, `default_rng(20261028)`):

| loss | parameter | B1 | A2 | B1 minus A2 | median per-heart difference |
|---|---|---|---|---|---|
| corr | v_endo | 3.1% [2.8, 3.7] | 2.6% [2.2, 2.9] | +0.5 [+0.2, +1.0] | +0.6 [+0.3, +0.8] |
| corr | v_t | 13.0% [12.0, 16.2] | 16.1% [12.5, 18.7] | -3.1 [-4.9, -0.3] | -2.9 [-3.6, -1.4] |
| amp | v_endo | 3.6% [2.9, 4.3] | 3.1% [2.5, 3.6] | +0.5 [+0.0, +0.9] | +0.5 [+0.3, +0.6] |
| amp | v_t | 11.6% [10.2, 14.1] | 12.8% [11.3, 15.6] | -1.2 [-2.2, +0.2] | -1.6 [-2.1, -0.8] |

Layer share of myocardial volume (median, 2.5th to 97.5th percentile): B1 29.0% (18.7 to 44.2) at 1.5 mm and 29.5% (19.0
to 44.0) at 1.0 mm, against A2 43.6% (28.0 to 66.4) and 38.2% (24.6 to 57.9). B1-only volume: median 0, maximum 1.6e-5 of
myocardial volume. Unit test against its lines: PASS (16 of 16 ellipsoid poses within +-15%; B1 error -5.4% to -4.9% at
1.5 mm, -3.1% to -2.9% at 1.0 mm; ratio 1.0/1.5 mm 1.023). QC: 0 NaN, theta* identical to arm C, 0 unmatched keys,
censored corr 26 (A2 52), amp 7 (A2 30).

B4 (extended grid, Amendment 2 B4). Measured break-even [95% CI], corr / amp:
white v_endo -9.1 dB [-11.1, -8.2] / -3.3 [-4.7, -2.0], v_t -11.0 [-12.2, -9.5] / -2.6 [-4.1, -0.7]; lp40 v_endo 3.6
[1.7, 4.3] / 9.3 [7.8, 10.3], v_t 1.7 [0.8, 3.1] / 10.7 [8.6, 12.3]; wander v_endo 63.7% [50.2, beyond grid > 80] / 24.0
[19.4, 31.2], v_t 50.5 [39.6, 66.1] / 14.9 [11.8, 20.5]; displacement v_endo 19.8 mm [16.2, 31.2] / 11.6 [9.1, 14.1], v_t
32.6 [23.8, beyond grid > 40] / 10.3 [8.5, 13.7]. Measured minus E3x lies between -1.0 and +4.2 native units in all 16
cells. The 95% CI of the difference excludes 0 for corr lp40 (v_endo +0.8 dB [+0.1, +1.6], v_t +1.0 [+0.4, +2.1]), has its
lower end at +0.0 for corr white v_t (+1.6 [+0.0, +2.6]) and amp lp40 v_t (+0.4 [+0.0, +0.7]), and includes 0 in the other
12 cells (table in RESULTS.md). A censored-dropped sensitivity (corr) is reported beside it; at lp40 of -5 dB or
worse and at wander 80% many replicates sit at a prior bound, so noise SD there is a lower bound.

Definitions chosen here (Amendment 2 silent): (1) the paired difference is the difference of medians, hearts resampled
jointly (as E7); the median of per-heart differences is given beside it. (2) Measured minus E3x is resampled jointly, which
is exact because E3, E3x and B4 each draw the same `rng.integers` sequence from a fresh `default_rng(20261028)` over the
same 50 hearts; resamples where either value is beyond grid are left out and counted. (3) A self-check confirms the
generalised break-even block reproduces E3 on the A3 grid exactly. Seeds: every B1 and B4 bootstrap cell uses
`default_rng(20261028)`; no other stochastic step.

Job (account p201509, partition cpu, qos test, 1 node): 5396070, nboot 2000, end to end, 00:06:52, 0.11 CPU node-hours.
Log: `logs_integration_2026-10-28.tar.gz`.
