# Full arms C and N, A2 versus one-voxel trio comparison (owner calibC, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Governing text: `final/AMENDMENT_2026-10-28.md` (A1 to A4) over
`final/STUDY_PLAN.md` sections 3 and 4. No pre-registered line or Amendment 1 setting was moved. Scope:
`code/full/calib/` and this directory. QC only; no estimands (analysis owner).

## Jobs (account p201509, partition cpu, 1 node each)

| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5394180_0 | test | heart 102, 27 constructions, nrep 3; HDF5 complete, TIMEOUT in CSV export | 00:30:32 | 0.51 |
| 5394406 | test | driver rerun after I/O patch (102, 3 constructions, nrep 2), export, QC; FAILED only from a broken pipe in a log print | 00:22:26 | 0.37 |
| 5394519_0..4 | default | full run, 5 shards of 10 hearts (`shards.json`) | 02:12:21, 02:21:01, 02:10:13, 02:07:49, 02:11:21 | 11.05 |
| 5394520 | default | CSV export and QC | 00:04:49 | 0.08 |

Total 12.01 CPU node-hours (cap 40). Logs: `logs_armC_2026-10-28.tar.gz`. Code SHA-256: `code_sha256.txt`.
Rerun: `mkdir -p logs; sbatch --array=0-4 code/full/calib/armC_job.sh`, then `sbatch code/full/calib/export_job.sh`.

## Software, inputs, seeds

openCARP v20.0 AppImage, `-renumbering 0`, one eikonal per core, `OMP_NUM_THREADS=1`, 126 workers per node, own
`/dev/shm/calibC_<job>_<task>` scratch per array task. Env `cvri-sim` (Python 3.11.17, numpy 2.4.6, scipy 1.17.1, h5py
3.16.0, pandas 3.0.6). Nothing installed. Code copied from `code/gates/calib/` (gates logic unchanged: 9 x 9 log grid,
cubic-spline emulator with kriged refinement, up to 9 refinement runs, 25 x 25 grid then Nelder-Mead, both losses,
bound rule rtol 1e-3); forward model `code/full/mesh/acdc_lib.py` (A2 `etag`). Constructions:
`data/processed/acdc_mesh_v2/MANIFEST.tsv`, every row with `include_primary` or `include_sensitivity` (737). Reference:
expert, 1.5 mm, RV wall 3 mm. θ*: Latin hypercube, seed 20261028 + patient id (trio θ* identical to the gates run).
Arm N seeds: `default_rng([20261028, patient, istar, comparator_index, level, rep])`, comparator index white 0, lp40 1,
wander 2, disp 3; displacement replicates use mesh2's lead fields (seed 20261028 + 1000 patient + 100 sd_index + rep).

## Definitions chosen here (A3 silent)

1. Baseline wander: per lead, quadratic c0 + c1 s + c2 s², c ~ N(0, 1), s = t / t_w over the QRS window
   t_w = max target activation + 10 ms (the SNR window); rescaled so max |drift| in the window = level x the lead's
   peak-to-peak in the window; held at its end value after the window.
2. Electrode displacement: target = ECG of θ* activation through `W_disp[sd, rep]`; calibration with the stored
   undisplaced W (mesh2: W_zero differs from W by at most 7.3e-4 relative). One displaced lead field per replicate,
   shared by the three θ*.
3. Final loss reported = emulator loss at θ̂ after round 2 (`fval_emu`), as in the gates run.

## QC (all from `qc.json`; quantiles are 2.5th, 50th, 97.5th percentiles)

- Completion: 737 of 737 constructions, 50 hearts, 4,422 of 4,422 estimates (round 2), 0 NaN, 0 unactivated nodes.
  Arm N: 96,000 of 96,000 rows, 0 NaN. Refinement runs 4,302 (at most 8 per construction); emulator RMS error at
  refinement points 0.004, 0.026, 0.113 ms, maximum absolute 3.95 ms.
- Reference recovery: maximum |θ̂/θ* − 1| 1.6e-5 (v_endo), 1.2e-4 (v_t) for correlation loss, n = 150.
- Fit loss (correlation, `fval_emu`) by set: primary 0.000, 0.046, 0.224 (n 1,482; 52 censored); ukbb_cardiac 0.036,
  0.180, 0.699 (276; 91); RV wall 1.5 mm 0.016, 0.096, 0.290 (153; 13); RV wall extra 1.0 mm 0.023, 0.097, 0.321
  (300; 34).
- Censored, correlation loss, by source (n; low v_t, high v_t, high v_endo): expert 7 of 393 (4, 3, 0); Task027 12 of
  390 (8, 4, 1); Task114 16 of 390 (7, 7, 4); UKB SAX 34 of 378 (31, 2, 2); CineMA 30 of 384 (9, 19, 4); ukbb_cardiac
  91 of 276 (52, 37, 3; also 6 at low v_endo). Amplitude loss: 2, 12, 8, 14, 19, 45. v_t > v_fib flags in
  `estimates.csv`.
- Root flag (> 10 mm), included rows: 8 of 294 non-reference at 1.5 mm, 9 of 393 at 1.0 mm (all sets pooled; per-set
  denominators follow from `estimates.csv`).
- Arm N, correlation loss, median over heart and θ* of the replicate SD of θ̂/θ* (v_t): white 0.0135 (10 dB) to
  0.0013 (30 dB); lp40 0.0585 to 0.0056; wander 0.017, 0.034, 0.064 (5, 10, 20%); displacement 0.030, 0.054, 0.102
  (5, 10, 20 mm). Censored replicates: displacement 3, 20, 76 of 3,000; wander 20% 1; lp40 10 dB 1.

## One-voxel versus A2 fast layer, trio (QC, not an estimand)

One-voxel arm = gates estimates (`results/2026-10-28_G3G4_ministudy/estimates.csv`); same meshes, roots, W and θ*;
only the fast layer and hence the targets differ. Mesh share (median over 6 sources): 101 0.478/0.438 (A2/one-voxel)
at 1.5 mm, 0.424/0.287 at 1.0 mm; 102 0.444/0.409, 0.383/0.262; 104 0.358/0.333, 0.309/0.216.
Between-construction CV of θ̂/θ* (correlation loss, median over θ*), A2 against one-voxel:

| heart | set | v_endo | v_t |
|---|---|---|---|
| 101 | 12 (gates set) | 5.3% vs 14.5% | 25.1% vs 30.9% |
| 102 | 12 | 1.4% vs 2.6% | 23.5% vs 22.9% |
| 104 | 12 | 7.1% vs 15.5% | 20.8% vs 23.8% |
| 101 | 10 (primary sources) | 2.7% vs 14.0% | 10.1% vs 18.9% |
| 102 | 10 | 1.4% vs 2.6% | 13.3% vs 13.3% |
| 104 | 10 | 4.7% vs 12.1% | 11.9% vs 14.8% |

Paired θ̂ ratio A2/one-voxel (108 pairs, correlation loss): median 0.990 (v_endo, 2.5th to 97.5th 0.758 to 1.019),
1.000 (v_t, 0.754 to 1.422). The one-voxel values reproduce the gates preview (14.5%, 30.9% for 101). Three hearts:
descriptive only.

## Outputs

`hearts/heart_patientNNN.h5` (50, 27 GB): targets, grid_T, refinement, calib[round, θ*, loss, (v_endo, v_t, fval)],
noise/<comparator>/theta_hat[θ*, level, rep, loss, 3] with rng_seed. CSVs: `estimates.csv` (round 2),
`estimates_rounds.csv.gz`, `noise_estimates.csv`, `refinement.csv`, `constructions.csv`; `qc.json`; `shards.json`;
`test/` (job 5394180, heart 102, nrep 3, not for analysis). Index: `MANIFEST`.

## Not done or not verified

- Pipeline not rerun end to end a second time. Spot check (login node): heart 101, θ* 0, wander 10%, rep 3,
  correlation loss recomputed from the seed: θ̂ (0.851482, 0.389303) against CSV (0.851475, 0.389335), loss equal to
  7 digits (difference within the Nelder-Mead tolerance). A displacement replicate was not spot-checked (login node
  timed out).
- The A2 layer's own unit-test failure (mesh2) is inherited; not addressed here.
- RV wall 2 mm at 1.5 mm has 1 construction (103 expert); the 1.0 mm extra set is not pre-registered.
