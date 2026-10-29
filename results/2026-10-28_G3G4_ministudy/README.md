# Gates G3 and G4: mini-study on the G1 trio (owner calib, 2026-10-28)

Project root `/project/home/p201509/work/cvri/simulation`. Hearts: ACDC test patients 101 (DCM), 102 (NOR) and
104 (HCM), at ED. Each heart has 12 constructions: 6 segmentation sources (expert plus 5 networks) at 2 voxel sizes
(1.5 and 1.0 mm). All 36 constructions were built by owner mesh (`data/processed/acdc_mesh/MANIFEST.tsv`). The
reference construction is expert labels at 1.5 mm. All numbers below are in `gates.md` and `gates.json`.

## Verdict against the pre-registered lines (STUDY_PLAN section 7; lines not moved)

- **G3: PASS.** Primary analysis: correlation loss, white noise, all sources. The between-construction SD of
  theta-hat/theta* was at least 2 x the 20 dB noise-arm SD in 3 of 3 hearts for both parameters. The ratio
  SD_cons / SD_noise(20 dB) ran from 32.0 (95% CI 23.9 to 41.5; 102, v_endo) to 197.7 (78.4 to 281.9; 101, v_endo).
  Every sensitivity variant also passed in 3 of 3 hearts for both parameters: amplitude loss, 40 Hz low-pass noise,
  excluding ukbb_cardiac, and dropping censored estimates. The smallest ratio was 3.21 (1.71 to 5.16; amplitude
  loss, low-pass noise, 101, v_t).
- **Break-even SNR.** Under the primary analysis it was below 10 dB in every heart and parameter. The values were
  extrapolated beyond the measured 10 to 30 dB range (-24.5 to -9.8 dB), so read them only as "< 10 dB". Only for
  amplitude loss with low-pass noise did v_t break even inside the range: 11.1 dB (8.3 to 14.7; heart 101) and
  10.3 dB (7.6 to 14.7; heart 102).
- **G4: reframe trigger NOT met.** For the three-length rule under the correlation loss, all sources (n = 33
  non-reference constructions), in-sample R² was 0.47 (cluster-bootstrap 95% CI 0.30 to 0.90) for v_endo and 0.62
  (0.49 to 0.85) for v_t. Leave-one-heart-out R² was -28.0 and -22.9. Under the amplitude loss, in-sample R² was 0.47
  and 0.12. Excluding ukbb_cardiac, it was 0.38 and 0.56 (correlation loss) and 0.42 and 0.65 (amplitude loss).
  With 3 clusters the bootstrap CIs are close to uninformative, and the LOHO model extrapolates across hearts.

### Why the G3 line looks easy (reported separately; the line is unchanged)

The noise arm calibrates the same model on the same emulator, so noise is the only error source. White noise
sampled every 0.5 ms is mostly averaged out by a 2-parameter fit. At 20 dB, SD(theta-hat/theta*) was 0.08% to 0.15%
for v_endo and 0.31% to 0.40% for v_t. The 40 Hz band-limited noise is 3 to 5 times larger and the gate still
passes. The 20 dB white-noise line therefore does not test whether construction variance exceeds realistic
clinical ECG error, which also includes baseline wander, electrode placement and model form. Arm F (model form)
and arm T (torso) remain the relevant comparators.

## Primary-estimand preview (descriptive, 3 hearts)

Between-construction CV of theta-hat/theta* over the 12 constructions. Each cell gives the median over the 3 theta*
of each heart, with the range in brackets.

| loss | heart | v_endo | v_t |
|---|---|---|---|
| corr | 101 | 14.5% (4.1 to 18.5) | 30.9% (25.4 to 31.9) |
| corr | 102 | 2.6% (1.5 to 4.5) | 22.9% (20.0 to 23.2) |
| corr | 104 | 15.5% (3.5 to 27.0) | 23.8% (20.1 to 29.4) |
| amp | 101 | 5.8% (3.1 to 6.2) | 14.9% (14.4 to 34.6) |
| amp | 102 | 4.1% (3.4 to 4.5) | 11.8% (8.3 to 14.9) |
| amp | 104 | 13.5% (4.0 to 18.5) | 50.5% (22.3 to 84.2) |

The median over hearts under the correlation loss is 14.5% for v_endo and 23.8% for v_t. These spreads are larger
than the pilot's 1.4% and 6.6%.

Two-way sums of squares of log(theta-hat/theta*) were computed after removing the theta* mean. Each cell gives the
shares in the order segmentation / mesh / interaction / residual, under the correlation loss:

| heart | v_endo | v_t |
|---|---|---|
| 101 | 0.31 / 0.39 / 0.09 / 0.21 | 0.82 / 0.00 / 0.07 / 0.11 |
| 102 | 0.02 / 0.69 / 0.05 / 0.24 | 0.91 / 0.03 / 0.02 / 0.03 |
| 104 | 0.22 / 0.28 / 0.03 / 0.47 | 0.23 / 0.02 / 0.02 / 0.73 |

**Mesh resolution shifts estimates even with expert labels.** On expert labels at 1.0 mm, the median ratio against
theta* was 1.09 for v_endo and 1.05 for v_t (correlation loss); under the amplitude loss it was 1.08 and 1.11.
One likely mechanism (inference, not tested) is the one-voxel fast endocardial layer, which is thinner at 1.0 mm.
If so, part of the "mesh" factor is a model-definition artefact rather than discretisation error.

**Fit quality is associated with spread.** The final correlation loss (1 - mean r) had a median of 0.056 (2.5th to
97.5th percentile 7e-11 to 0.25; maximum 0.28); amplitude loss median 0.037 (to 0.11; max 0.12). By source, the
correlation-loss median was 0.007 for expert, 0.039 to 0.057 for the four strong networks and 0.160 for
ukbb_cardiac. The Spearman correlation between final loss and |log(v_t ratio)| was 0.53 (correlation loss) and 0.58
(amplitude loss). Constructions that cannot reproduce the target ECG give the most displaced velocities.

**Censoring.** Under the widened prior, 10 of 108 correlation-loss estimates sat on a bound: 8 at v_t = 0.1 and 2
at v_endo = 4.0. Eight of the 10 came from ukbb_cardiac. Under the amplitude loss, 3 of 108 sat at v_t = 0.8 (above
v_fib; flagged). In the noise arm, 2 of 1,800 amplitude-loss estimates and 0 correlation-loss estimates were
censored. Censored estimates are kept at the bound in the primary numbers, with a dropped-censored sensitivity
alongside (`corr|white|all|no_censored`, also 3 of 3).

## Design as run (fixed before the full job; see `code/gates/calib/calib_lib.py` docstring)

- **Prior:** v_endo 0.5 to 4.0 m/s and v_t 0.1 to 0.8 m/s, as a log box (pilot: 0.8 to 3.0 and 0.15 to 0.5; every
  pilot bound hit was at v_t = 0.5, except one amplitude-loss hit at v_endo = 3.0). v_fib = 0.6 m/s fixed.
- **theta\*:** 3 per heart by Latin hypercube in log space over v_endo 0.8 to 2.6 and v_t 0.15 to 0.45; seed
  20261028 + patient id. Values are in each HDF5 under `targets/theta_star`. Targets are direct openCARP runs on the
  reference construction.
- **Emulator and calibration:** 9 x 9 log grid with pilot cubic-spline emulator, calibrated with
  `eplib.calibrate` (unchanged). Refinement used simple kriging of the nodal residual at refinement runs: round 1
  ran once at each distinct round-0 theta-hat (both losses), and round 2 at each primary-loss round-1 theta-hat.
  There were 212 refinement runs, at most 7 per construction (cap 9); round 2 needed only 5 runs, because the other
  estimates had moved less than 0.01 log units. Emulator error at the refinement points before correction: nodal
  RMS median 0.034 ms (97.5th percentile 0.073; maximum 0.71 ms); maximum absolute error 3.7 ms. The largest change
  from round 0 to the final estimate was 3.5% (v_endo) and 6.8% (v_t) under the correlation loss. Recovery on the
  reference construction was within 4.1e-5 relative error.
- **Noise (arm N):** SNR_dB per lead = 10 log10(P_l / sigma_l²), where P_l is the mean of x_l(t)² over the QRS
  window [0, max target activation + 10 ms] at 0.5 ms sampling. Noise was added to every sample. Primary noise is
  white; the sensitivity is 4th-order Butterworth 40 Hz zero-phase low-pass noise rescaled to the same variance.
  Settings: SNR 10, 15, 20, 25 and 30 dB, 20 replicates each, both losses on the same draws. Seed per cell:
  `default_rng([20261028, patient, istar, snr_db, noise_type_index])`. Calibration used the reference construction's
  final corrected emulator.
- **G3 statistic:** SD = sqrt(mean over theta* of variance, ddof 1) of theta-hat/theta*. SD_cons is taken over the
  12 constructions (reference included); SD_noise over the 20 replicates. Break-even interpolates log10 SD_noise
  linearly against SNR. Outside the measured range it extrapolates with the least-squares slope over all 5 SNRs.
  CIs come from 2,000 bootstrap resamples of constructions within the heart and replicates within cell (seed
  20261028).
- **G4:** r_c = mean over theta* of theta-hat_c / theta-hat_ref. Lengths are L_scale, L_wall and L_base, using the
  pilot definitions (`sax_route.lengths`, identical to `code/pilot/I11/09_analyse.py` inputs) computed from each
  construction's `lab`. Fit is OLS. The cluster bootstrap resamples hearts 2,000 times (seed 20261028). Length-ratio
  SDs were 2.7% (scale), 8.7% (wall) and 4.7% (base). L_wall shows a voxel-size step of about +12% at 1.5 mm against
  1.0 mm, which is a discretisation effect of the distance-transform definition.

## Files

- `heart_patient{101,102,104}.h5`: one per heart. Each holds targets (theta_star, T, ecg); grid (G1, G2, theta);
  and, per construction under `cons/<name>/`, grid_T (81 x nodes), grid_wall_s, ref_theta, ref_T, ref_Temu0,
  ref_round, ref_wall_s, calib (round x theta* x loss x [v_endo, v_t, fval]) and attributes (lengths, n_nodes).
  `noise/theta_hat` holds theta* x SNR x noise type x replicate x loss x [v_endo, v_t, fval]. Sizes: 487, 278 and
  510 MB.
- `estimates.csv` (arm C; all rounds, use round 2), `noise_estimates.csv` (arm N), `refinement.csv`,
  `constructions.csv`: tidy tables.
- `gates.json` and `gates.md` hold every gate number. `logs_2026-10-28.tar.gz` holds the job logs.
  `test_run_5392570.tar.gz` holds the qos=test run outputs (heart 102, 3 replicates; HDF5 not kept).

## Regeneration

- From saved outputs: `python -I code/gates/calib/csv_export.py <dir> 101 102 104` (55 s) and then
  `python -I code/gates/calib/analyse_gates.py <dir>` (78 s on a compute node, 102 s wall on the login node). Both
  use `/project/home/p201509/envs/cvri-sim`. A rerun into a scratch directory from the HDF5 reproduced all four CSVs
  byte-identically and `gates.md` identically, apart from the runtime line.
- Full pipeline: `sbatch --export=ALL,MODE=full code/gates/calib/calib_job.sh` (34 min on one node).
- Tests: `code/gates/calib/test_calib.py` (LHS strata, kriging exactness, realised SNR, bound flags, break-even known
  answer); they pass.
- Script SHA-256 prefixes: analyse_gates.py b04301fe, calib_lib.py bf9d937a, csv_export.py e05aeaf3,
  run_ministudy.py 2cf3d3a4, test_calib.py 843cd8c8, calib_job.sh fa9ce059.

## Jobs and software

| job | purpose | walltime | CPU node-hours |
|---|---|---|---|
| 5392570 | qos test, heart 102, 3 noise replicates | 00:09:37 | 0.16 |
| 5392578 | full mini-study and analysis | 00:34:15 | 0.57 |

Software: openCARP v20.0 AppImage (`-renumbering 0`, pure eikonal, tend 1000 ms) and Python 3.11.17 in env cvri-sim
(numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pandas 3.0.6). Nothing was installed. Code: `code/gates/calib/`. Inputs:
construction HDF5 files from owner mesh (`code/gates/mesh/acdc_lib.py`, seeded Bayer fibres, job 5392520); pilot
`code/pilot/I13/eplib.py` (unchanged). The 0.5 ms ECG sampling and the tau 1 ms template follow the pilot.
Timing: the median grid run took 42 to 50 s at 1.0 mm and 5 to 13 s at 1.5 mm, packed 126 per node.

## Open issues

1. The widened prior still censors ukbb_cardiac (8 of its 36 correlation-loss estimates). Whether to keep this network was decided
   in Amendment 1.
2. With 3 hearts, the G4 CIs and the LOHO R² are not informative. The pre-registered statistic (in-sample R²) is
   reported as the gate value.
3. The voxel-size bias on expert labels (+9% v_endo) needs to be resolved before arm C. One option is to fix the
   fast-layer thickness in mm rather than in voxels; this is a design question, not changed here.
4. Constructions that fit the target poorly (1 - r up to 0.28) carry much of the spread. The plan does not say
   whether fit quality should gate inclusion in arm C.
