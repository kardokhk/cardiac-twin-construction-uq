# Arm T (torso and electrode geometry), I11 full-arm run, 2026-10-28

Project root `/project/home/p201509/work/cvri/simulation`. Owner armT. Plan: `final/STUDY_PLAN.md` section 3 arm T and
section 4 item 5; settings from `final/AMENDMENT_2026-10-28.md` (A2, A4), which governs. In silico, same-model,
noise-free. Cross-heart comparisons are associations. Every number below is in `analysis.json` / `tables.md`
(made by `code/full/armT/t05_analyse.py`) unless another file is named.

**Correction 2026-10-28 below.** The refinement design of the original run pooled all arms (including the own-torso arm) into one refinement set per heart; the follow-up rerun (per-arm refinement, two rounds) and the old and new values are in the "Correction 2026-10-28" section at the end. Original text is kept unchanged above it.

## Design as run

- Hearts: 17 Malik 2026 HCM (Zenodo 18890229, I13 pilot conversion, read only from `data/processed/pilot_I13`) and
  5 KCL healthy (Zenodo 7253863, CC BY 4.0, downloaded 2026-10-28 to `data/raw/kcl_torso/`, md5 and SHA-256 in its
  README; converted by `t01_convert_kcl.py`). KCL axes were found from anatomy and were already LPS in all 5
  (`data/kclK_convert.json`).
- Construction (all 22 identical): 1.5 mm voxelisation of the deposited ventricles (pilot c0ref algebra; Malik labels
  identical to the pilot c0ref, agreement 1.0 in all 17), Bayer fibres with numpy global seed 20261028 (seeded pyamg),
  A2 fast layer (element centroid within 1.5 mm of the LV or RV cavity surface; share 0.147 to 0.388 of elements, against
  0.135 to 0.368 for one voxel; `hearts.csv`), four rule roots mapped (maximum shift 0.73 to 1.32 mm), fibre velocity
  0.6 m/s, openCARP v20.0 (GIT ffce1951), renumbering 0, tend 1000 ms, one process per core, own scratch dir per run.
- Own torso: electrode rule v2. KCL has no spine tag; the spine proxy is bone nodes within 35 mm of the mid-sagittal x
  and posterior to the heart (assumption). Lead fields homogeneous 0.2 S/m.
- References: Malik 09 and 17; a heart is never its own reference (HCM n = 16 per reference, healthy 5, pooled 21).
  Registrations: `icp` (frame initialisation, then ICP onto the reference construction surface) and `trans`
  (orientation kept, centroid moved to the ICP centroid; PRIMARY). Identity check (09 into 09, 17 into 17):
  |W - W_own|/|W_own| = 3.0e-8 and 3.3e-8.
- theta*: 3 per heart, A4 Latin hypercube (v_endo 0.8 to 2.6, v_t 0.15 to 0.45 m/s), seed 20261028 + id (A4 governs over
  the I13 seed 13013); ids Malik 1 to 17, KCL 201 to 205 (chosen to avoid ACDC ids 101 to 150). Prior 0.5 to 4.0 and
  0.1 to 0.8 m/s. Held-out runs seed [20261028, id, 1].
- Targets in the own torso; calibration in the reference torso; losses corr (primary) and amp. Emulator 9 x 9 log grid,
  round 0, one refinement run at every distinct round-0 estimate (554 runs), round 1 reported.
- Bias per heart = mean over theta* of theta_hat/theta* - 1 (signed) or of its absolute value; mean over hearts with
  cluster bootstrap 95% CI (2,000 resamples, seed 20261028). Bound estimates kept at the bound (primary).

## Results (pooled = HCM-heavy, 16 HCM + 5 healthy)

Known answer: own-torso recovery max |error| 3.1e-5 (v_endo) and 5.0e-5 (v_t), 0 bound hits, 132 calibrations.

Primary, correlation loss, translation only:

| ref | set | v_endo signed | v_endo abs | v_t signed | v_t abs | bound hits | fit loss median [IQR] |
|---|---|---|---|---|---|---|---|
| 09 | HCM 16 | -5.3% [-10.0, -0.8] | 9.2% [6.5, 12.2] | -1.8% [-11.3, +7.7] | 17.9% [13.2, 23.5] | 0/48 | 0.086 [0.051, 0.176] |
| 09 | healthy 5 | -1.0% [-4.9, +3.6] | 4.7% [3.0, 6.2] | -1.5% [-11.9, +9.1] | 9.4% [3.6, 15.5] | 0/15 | 0.222 [0.109, 0.270] |
| 09 | pooled 21 | -4.3% [-7.7, -0.6] | 8.1% [5.9, 10.3] | -1.8% [-9.4, +5.7] | 15.9% [11.7, 20.5] | 0/63 | 0.105 [0.052, 0.226] |
| 17 | HCM 16 | -0.9% [-3.3, +1.7] | 4.6% [3.1, 6.2] | +1.4% [-2.7, +5.3] | 8.6% [6.5, 10.7] | 0/48 | 0.036 [0.021, 0.073] |
| 17 | healthy 5 | -0.7% [-2.4, +0.6] | 1.6% [0.8, 2.8] | +7.1% [-0.6, +18.0] | 10.2% [3.9, 19.8] | 0/15 | 0.045 [0.026, 0.055] |
| 17 | pooled 21 | -0.9% [-2.7, +1.1] | 3.9% [2.7, 5.2] | +2.7% [-1.1, +6.9] | 8.9% [6.7, 11.9] | 0/63 | 0.039 [0.021, 0.070] |

Full ICP, correlation loss, pooled: ref 09 v_endo abs 15.5% [10.9, 20.2], v_t abs 26.2% [19.3, 34.2], 4/63 bound, fit
0.274; ref 17 13.3% [9.7, 17.7] and 30.5% [20.3, 42.4], 3/63, fit 0.137. Amplitude loss and every HCM/healthy split:
`tables.md` (amplitude loss has 4 to 7 of 63 bound hits and fit loss 0.23 to 0.65).

Forward check (same activation, own vs reference ECG, median [IQR] over 21 hearts of the per-heart median mean-lead
r): trans 0.86 [0.75, 0.94] (09), 0.96 [0.92, 0.98] (17); icp 0.65 [0.56, 0.87] (09), 0.84 [0.71, 0.92] (17). ICP
rotation median 21.6 deg (10.7 to 38.9) into 09 and 20.8 deg (5.6 to 41.0) into 17.

S5 (exploratory): paired ICP minus translation-only |bias|, correlation loss: ref 09 +7.4 [+4.2, +10.6] points
(v_endo), +10.3 [+5.7, +15.5] (v_t); ref 17 +9.5 [+5.6, +13.8] and +21.6 [+12.2, +32.6]. Spearman rho, rotation vs ICP
lead r: -0.86 [-0.96, -0.60] (09), -0.62 [-0.85, -0.26] (17); rotation vs ICP |bias| (corr): 09 +0.67 [+0.30, +0.88]
(v_endo), +0.55 [+0.13, +0.84] (v_t); 17 +0.39 [-0.06, +0.74], +0.28 [-0.23, +0.73]. Under translation only (correlation loss), rotation
vs |bias| is +0.08 to +0.54 with CIs that include 0 except v_endo, ref 09 (+0.54 [+0.10, +0.78]).

## Defects, deviations, unverified

- Malik torsos: 4,087 to 7,375 construction nodes (about 5%) lie outside the 12 candidate torso tets and take an
  extrapolated lead field (KCL 342 to 472); sensitivity to more candidates not run here.
- Malik 04: own V6 is 48 mm lateral of the heart (pilot note); its own-torso V6 weight dominates W_own, so its amplitude
  loss mostly fits V6 and |W_ref - W_own|/|W_own| = 1.000 in all four arms.
- One refinement round, not two as in the gates: 17 of 504 reference estimates moved by more than 5 points between
  rounds (10 amp, 7 corr), 3 by more than 20 (all amp, near-flat loss).
- Test job 5393446: ICP on identical point sets returned NaN in the identity check; fixed (identity without ICP).
- Healthy n = 5: bootstrap CIs are unreliable at this size.

## Jobs (account p201509, partition cpu, 1 node)

| job | qos | purpose | elapsed | node-h |
|---|---|---|---|---|
| 5393446 | test | test: K3 convert, prep 09/17/K3, registration, 270 runs | 00:11:23 | 0.19 |
| 5393579 | test | test 2: fixed identity check, calibration rounds | 00:04:55 | 0.08 |
| 5393781 | default | full: 5 conversions, 22 preps, 84 registrations, 1,980 + 554 runs, calibration | 00:26:56 | 0.45 |

Total 0.72 CPU node-hours (cap 15).

## Reproduce

`sbatch --qos=default --time=02:00:00 code/full/armT/armT_job.sh` (MODE=full), then on the login node
`/project/home/p201509/envs/cvri-sim/bin/python -I code/full/armT/t05_analyse.py` (about 1 min).
Env `cvri-sim`: Python 3.11.17, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, h5py 3.16.0, pyamg 5.3.0. OMP_NUM_THREADS=1.
Code copied, not edited: `eplib.py` (pilot I13, SHA-256 035c9368...), `carpio.py` (gates mesh, 032f2524...),
`bayer_i09.py` (gates ka, c61bd95d...; reads `code/pilot/I09/02_case.py`, ec6f4823...). Known-answer tests:
`test_armt.py` (closest point on triangle, A4 Latin hypercube strata, bound rule), run at the start of every job.

## Correction 2026-10-28

Follow-up after an independent check. Code: `code/full/armT/t06_rerun_perarm.py` (rerun),
`t07_checks.py` (identity, ICP trim, KCL axes), `t08_compare.py` (old versus new), `t05_analyse.py` gained
`--est/--round/--suffix` (defaults reproduce `tables.md`, `hearts.csv`, `registration_forward.csv` byte for byte and
every original key of `analysis.json`; three keys were added to it: `own_by_round`, `report_round`, `estimates_file`).
Jobs 5395663 (qos test, 3 hearts) and 5395720 (qos short, 22 hearts). Full old-versus-new tables: `compare_v2.md`;
new tables: `tables_v2.md`, `analysis_v2.json`, `estimates_v2.csv` (rounds 0, 1, 2), `refinement_v2.csv`,
`checks_v2.json`. The original `estimates.csv`, `analysis.json` values, `tables.md` and HDF5 groups `runs/ref_*`,
`calib/*` are kept; the rerun writes HDF5 groups `refv2/<arm>` and `calib_v2/<arm>`.

### C1. Refinement design (superseded)

- Old: one refinement set per heart, built from the pooled round-0 estimates of every arm of that heart, own-torso arm
  included (`t04_runs.py` lines 186-196), 554 runs, one refinement round, round 1 reported. Because the own-torso
  estimate sits at theta*, every reference arm calibrated on an emulator with a kriged point (nugget 1e-6) within about
  0.004 log units of theta*.
- New: a separate refinement set per heart-arm, filled only from that arm's own estimates; two refinement rounds with
  the gates rule (round 0 both losses, round 1 correlation loss, merge within 0.01 log units); round 2 reported.
  569 + 16 = 585 runs. No reference arm received a point it did not propose (`proposed_by` = arm for all 585 rows);
  190 of 519 reference-arm points lie within one grid half-step (0.13 log units) of a theta*, all proposed by that arm.
- Reason: the pooled design leaked a truth-adjacent point into the reference-torso calibrations and made the
  own-torso check near-circular.
- Round 0 of the rerun reproduces round 0 of the original exactly (636 of 636 estimates, difference 0).

### C2. Own-torso known answer (max |theta_hat/theta* - 1| over 66 estimates per loss)

| version | corr v_endo | corr v_t | amp v_endo | amp v_t |
|---|---|---|---|---|
| old, pooled refinement, round 1 | 3.1e-5 | 5.0e-5 | 2.3e-5 | 4.1e-5 |
| new, no refinement (round 0) | 1.2e-3 | 3.9e-3 | 1.3e-3 | 4.3e-3 |
| new, own refinement only (round 2) | 2.2e-5 | 1.0e-4 | 3.2e-5 | 1.0e-4 |

The old README value "3.1e-5 (v_endo) and 5.0e-5 (v_t)" is replaced by: without refinement 0.12% and 0.39% (corr; 0
bound hits), with own-arm refinement 2.2e-5 and 1.0e-4. The own-arm refinement point is still the own arm's own
estimate and therefore near theta* by construction; the no-refinement error (below 0.5%, inside the 2% criterion) is
the informative figure for the emulator, and it shows the near-zero error of the original was not produced by the
leak alone.

### C3. Headline cells, correlation loss (old -> new; mean over hearts, heart-cluster bootstrap 95% CI)

| ref | reg | set | v_endo abs | v_t signed | v_t abs | bound hits |
|---|---|---|---|---|---|---|
| 09 | trans | HCM 16 | 9.2% [6.5, 12.2] -> 9.2% [6.6, 12.2] | -1.8% [-11.3, +7.7] -> -1.4% [-11.1, +8.7] | 17.9% [13.2, 23.5] -> 18.2% [13.2, 24.3] | 0/48 -> 0/48 |
| 09 | trans | pooled 21 | 8.1% [5.9, 10.3] -> 8.1% [5.9, 10.3] | -1.8% [-9.4, +5.7] -> -1.4% [-9.2, +6.4] | 15.9% [11.7, 20.5] -> 16.1% [11.7, 21.0] | 0/63 -> 0/63 |
| 17 | trans | pooled 21 | 3.9% [2.7, 5.2] -> 3.9% [2.7, 5.2] | +2.7% [-1.1, +6.9] -> +2.8% [-1.2, +6.9] | 8.9% [6.7, 11.9] -> 9.0% [6.7, 11.9] | 0/63 -> 0/63 |
| 09 | icp | pooled 21 | 15.5% [10.9, 20.2] -> 15.5% [10.9, 20.1] | -4.8% [-16.4, +7.2] -> -5.0% [-16.6, +6.9] | 26.2% [19.3, 34.2] -> 26.7% [19.5, 35.1] | 4/63 -> 4/63 |
| 17 | icp | pooled 21 | 13.3% [9.7, 17.7] -> 13.3% [9.7, 17.6] | +14.4% [+0.9, +29.8] -> +14.4% [+0.7, +30.0] | 30.5% [20.3, 42.4] -> 30.5% [20.1, 42.7] | 3/63 -> 3/63 |

Across all 24 correlation-loss cells (both references and registrations, HCM, healthy and pooled sets) point
estimates changed by at most 0.8 points (largest: ref 09, ICP, HCM 16, v_t abs 26.6% [18.4, 37.0] -> 27.3% [18.5,
38.6]) and CI bounds by at most 1.7 points; bound hits were unchanged; fit loss medians changed by at most 0.001.
Every cell, both losses: `compare_v2.md`. Largest change in any cell: amplitude loss, ref 09, ICP, HCM 16, v_t abs
52.0% [27.7, 82.0] -> 49.7% [27.3, 77.6] (pooled 43.1% [24.4, 66.0] -> 41.4% [24.0, 62.5]), bound hits 7/48 -> 5/48;
amplitude loss ref 17 translation bound hits 5/63 -> 6/63. Per estimate (504 reference-torso estimates), old round 1
versus new round 2 differ by more than 1 point in 79, more than 5 points in 13, maximum 86 points (heart 04, ref 09, ICP,
amplitude loss, near-flat loss surface; fit loss 0.485 -> 0.490).

The old defects bullet "One refinement round, not two as in the gates: 17 of 504 reference estimates moved by more
than 5 points between rounds" is replaced by: with per-arm refinement, 4 of 504 moved by more than 5 points from round 0
to 1 (3 amplitude loss; 2 by more than 20) and 1 of 504 from round 1 to 2 (none by more than 20).

### C4. S5 (exploratory; old -> new)

Paired ICP minus translation-only |bias|, correlation loss: ref 09 v_endo +7.4 [+4.2, +10.6] -> +7.4 [+4.2, +10.6]
points, v_t +10.3 [+5.7, +15.5] -> +10.6 [+5.9, +16.1]; ref 17 +9.5 [+5.6, +13.8] -> +9.4 [+5.6, +13.7], +21.6 [+12.2,
+32.6] -> +21.6 [+12.1, +32.7]. Spearman rho, rotation vs ICP |bias| (corr): ref 09 v_endo +0.67 -> +0.66 [+0.28, +0.88]
(the old README printed +0.67; `analysis.json` holds +0.665, so +0.66 or +0.67 by rounding), v_t +0.55 [+0.13, +0.84]
-> +0.55 [+0.11, +0.85]; ref 17 +0.39 [-0.06, +0.74] -> +0.40 [-0.05, +0.74], +0.28 [-0.23, +0.73] -> +0.26 [-0.23,
+0.72]. Translation only, rotation vs |bias|, ref 09 v_endo +0.54 [+0.10, +0.78] -> +0.54 [+0.10, +0.79]. Rotation vs
lead r is unchanged (forward model, no calibration). Largest S5 change: ref 17, ICP, amplitude loss, v_endo rho +0.28
[-0.17, +0.65] -> +0.16 [-0.29, +0.61]. The interpretation is unchanged: larger ICP rotation is associated with larger
|bias| and lower lead r; associations, not effects.

### C5. Identity check through the ICP and translation-only paths

- Old: "|W - W_own|/|W_own| = 3.0e-8 and 3.3e-8", computed with R = I, t = 0 and no ICP call (`t03_register.py`
  lines 46-48), and "ICP on identical point sets returned NaN" (test job 5393446).
- New (`t07_checks.py identity`, registration and weight code copied from `t03_register.py`): from the frame
  initialisation (R0 - I below 3e-16), unchanged `eplib.icp` returned a finite identity (rotation 0, |t| below 1e-13
  mm) for both hearts: ICP path |W - W_own|/|W_own| 7.3e-6 (09) and 3.3e-8 (17); translation path 3.0e-8 and 3.3e-8.
  The NaN of job 5393446 did not reproduce on the stored constructions; its cause is not established. A variant with
  the trim written `d <= p90` gives the same results (5.7e-6 and 3.3e-8) and, on the 84 real registrations, differs
  from `eplib.icp` by at most 0.014 deg and 0.048 mm; `eplib.icp` reproduces the stored R and t exactly (difference 0).
- From a perturbed start (10 deg about (1,1,1), shift (3, -2, 4) mm), ICP did not return to the identity: residual
  rotation 2.5 deg (09) and 1.7 deg (17), surface RMS 0.86 and 0.71 mm, centroid offset 0.50 and 0.68 mm; ICP-path
  |W - W_own|/|W_own| 8.0% and 4.7%, translation path 2.6% and 2.9%. ICP therefore carries a residual of about 2 deg on
  these voxel surfaces; part of the ICP-arm bias may be registration error rather than torso geometry (not separated).

### C6. KCL axis assumption

Assumption: the KCL coordinates, read as (x, y, z) = patient left, posterior, superior (LPS), match the Malik LPS
frame. Evidence from conversion (`data/kclK_convert.json`): the anatomy-derived world-to-LPS matrix is the identity in
all 5 hearts, heart 20 to 39 mm anterior of the body centroid, LV blood pool 22 to 30 mm left of the RV blood pool. Test
(`t07_checks.py axes`): translation-only lead r (own-torso ECG vs reference-torso ECG at the 3 theta*, median), heart
reoriented about its centroid:

| heart | ref | LPS (as used) | RAS | LAS (mirror) | RPS (mirror) |
|---|---|---|---|---|---|
| K1 | 09 / 17 | +0.95 / +0.93 | -0.17 / -0.39 | -0.06 / -0.04 | +0.37 / +0.32 |
| K2 | 09 / 17 | +0.88 / +0.98 | -0.68 / -0.73 | +0.44 / +0.32 | -0.35 / -0.28 |
| K3 | 09 / 17 | +0.63 / +0.95 | -0.50 / -0.60 | +0.51 / +0.72 | -0.39 / -0.62 |
| K4 | 09 / 17 | +0.72 / +0.92 | -0.26 / -0.22 | +0.22 / +0.20 | -0.19 / -0.49 |
| K5 | 09 / 17 | +0.65 / +0.96 | -0.42 / -0.49 | +0.57 / +0.43 | -0.13 / -0.18 |

LPS is the consistent convention: it gives the highest lead r for every heart and reference (0.63 to 0.98, within the
Malik HCM range 0.67 to 0.99), RAS gives negative r in all 10 (-0.73 to -0.17) and would place the heart posterior of
the body centroid with the LV right of the RV; the mirror readings give -0.62 to +0.72. The LPS run reproduces the
stored translation-only W (relative difference at most 2.1e-12). Not tested: other permutations of axes.

`MANIFEST.tsv` regenerated 2026-10-28: the 22 HDF5 files and `analysis.json` changed hash (added groups and keys
only; old `analysis.json` SHA-256 c1dd3a15...), 10 follow-up files added.

### Jobs (follow-up)

| job | qos | purpose | elapsed | node-h |
|---|---|---|---|---|
| 5395663 | test | follow-up test: hearts 09, 17, K3 | 00:14:06 | 0.24 |
| 5395720 | short | follow-up full: 22 hearts, 585 refinement runs, checks | 00:22:26 | 0.37 |

Follow-up total 0.61 CPU node-hours (cap 3); arm T total 1.33.
