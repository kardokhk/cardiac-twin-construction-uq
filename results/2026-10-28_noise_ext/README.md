# Extended noise grids, Amendment 2 B4 (owner noise-ext, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Governing text: `final/AMENDMENT_2_2026-10-28.md` B4 (declared
extension of E3, added after the full-arm data were seen), over Amendment 1 A3. No pre-registered line, primary output
or existing file was changed. QC and an owner check only; the reported E3 on the extended grid belongs to the
analysis owner.

## Jobs (account p201509, partition cpu, 1 node each)

| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5395828 | test | heart 102: displacement prep, solves, assemble in /dev/shm; all new levels, nrep 3; export | 00:19:56 | 0.33 |
| 5395865 | default | displacement draws, 50 hearts | 00:07:33 | 0.13 |
| 5395866_0..2 | default | torso lead-field solves, 360 workers | 00:55:20, 00:53:53, 00:53:16 | 2.71 |
| 5395867 | default | assemble 50 displaced_ext files, displacement calibrations | 00:42:37 | 0.71 |
| 5395868_0..1 | default | white, lp40, wander calibrations (25 hearts per task) | 01:04:56, 01:04:10 | 2.15 |
| 5395869 | default | CSV export, owner check | 00:02:27 | 0.04 |

Total 6.07 CPU node-hours (cap 15), from elapsed time x 1 node. Logs: `logs_noise_ext_2026-10-28.tar.gz`, per-worker
solve and assemble logs `logs_disp_ext_workers.tar.gz`, test outputs `test/test_job5395828.tar.gz`. Code SHA-256:
`code_sha256.txt`. Rerun: `K=code/full/calib; p=$(sbatch --parsable $K/noise_ext_prep.sh); s=$(sbatch --parsable
--dependency=afterok:$p $K/noise_ext_solve.sh); d=$(sbatch --parsable --dependency=afterok:$s $K/noise_ext_disp.sh);
m=$(sbatch --parsable $K/noise_ext_main.sh); sbatch --dependency=afterok:$d:$m $K/noise_ext_export.sh` (the assemble
step refuses to overwrite existing `displaced_ext` files; move them first).

## Code, software, inputs

New files only: `code/full/calib/run_noise_ext.py` (driver), `export_noise_ext.py` (CSV and owner check),
`noise_ext_{test,prep,solve,disp,main,export}.sh`, `code/full/mesh/displace_ext.py` (imports `displace.py` unchanged;
copies its `prep` and `assemble` with the new SD levels and output path; `solve`, walk, surface and lead-field rules
unchanged). `calib_lib.py` (noise, wander, calibration, bound rule) and `run_armC._load_emulator` are reused unchanged.
Env `cvri-sim` (Python 3.11.17, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pandas 3.0.6, pyamg 5.3.0). Nothing installed.
One calibration per core, 126 workers, `OMP_NUM_THREADS=1`, own `/dev/shm/noiseext_<job>` scratch per job.
No openCARP run was needed (the reference emulator and targets come from arm C).

Inputs, read only: `results/2026-10-28_armC/hearts/heart_patientNNN.h5` (reference construction's final corrected
emulator, round 2 points; stored targets and θ*), `data/processed/acdc_mesh_v2/cons/` reference constructions, torso
09 (as mesh2). Modification times of `armC/noise_estimates.csv`, `armC/hearts/` and `acdc_mesh_v2/displaced/` predate
these jobs.

## Seeds (every stochastic step)

- θ*: unchanged from arm C (Latin hypercube, seed 20261028 + patient id), read from the arm C HDF5.
- White, lp40, wander: `numpy.random.default_rng([20261028, patient, istar, comparator_index, level_key, rep])`,
  comparator index white 0, lp40 1, wander 2; same draw for both losses. `level_key` = level for level ≥ 0. NumPy's
  SeedSequence rejects negative entropy (`ValueError: expected non-negative integer`), so for negative SNR
  `level_key` = 1000 + |level| (−5 → 1005, −10 → 1010, −15 → 1015). The CSV `seed` column holds the list used.
- Displacement: draw seed 20261028 + 1000 patient + 100 sd_index + rep, sd_index 3 (30 mm) and 4 (40 mm), rep 0 to
  19 (no overlap with mesh2's indices 0 to 2); `default_rng(seed).standard_normal((6, 2))` for V1 to V6. Torso solver:
  pyamg with numpy global seed 20261028 before setup. CSV `seed` = draw seed.
- Owner-check bootstrap: `default_rng(20261028)`, 2,000 cluster resamples over hearts.

## Outputs

- `noise_estimates_ext.csv`: 84,000 rows (50 hearts x 3 θ* x 14 levels x 20 replicates x 2 losses), header identical
  to `results/2026-10-28_armC/noise_estimates.csv`; 0 NaN.
- `noise_ext_main_0.h5`, `noise_ext_main_1.h5`, `noise_ext_disp.h5`: `patientNNN/<comparator>/theta_hat[θ*, level,
  rep, loss, (v_endo, v_t, fval)]` and `rng_seed`.
- `data/processed/acdc_mesh_v2/displaced_ext/patientNNN_W_displaced_ext.h5` (50 files, 3.5 GB): `W_disp[k, rep, lead,
  node]`, k = 0, 1 for sd_index 3, 4.
- `disp_ext_prep.json`, `disp_ext_solves.json`, `disp_ext_draws.npz`; `owner_check.json`.

## QC

- Solves: 8,796 distinct electrode nodes from 12,000 draws, CG 33 to 35 iterations, relative residual at most 1.0e-10.
  Snap of walk end to node: median 0.61 mm, maximum 1.29 mm. Zero-offset check, W from new solves against stored W:
  maximum relative difference at most 7.3e-4 over 50 hearts (mesh2: at most 7.3e-4).
- Displaced node distance from the original electrode, median over hearts of per-heart medians: 34.3 mm (SD 30) and
  46.7 mm (SD 40). Relative change of W, median 0.51 and 0.62 (mesh2 at 20 mm: 0.38).
- Spot check on the login node, heart 102, θ* 0, rep 0: white −10 dB and displacement 40 mm recomputed from the seed,
  both losses, equal to the CSV to 6 decimals.
- Censored replicates (estimate at a prior bound), of 3,000 per level and loss, corr / amp: white 0 dB 0 / 2, −5 dB
  1 / 25, −10 dB 24 / 190, −15 dB 166 / 595; lp40 5 dB 11 / 96, 0 dB 91 / 465, −5 dB 412 / 1,034, −10 dB 972 / 1,397,
  −15 dB 1,379 / 1,709; wander 40% 36 / 421, 80% 154 / 1,486; displacement 30 mm 143 / 445, 40 mm 186 / 704. Kept at
  the bound (A1 rule). At heavy censoring the prior box caps the replicate SD, so noise SD at lp40 ≤ −5 dB and at the
  most severe amplitude-loss levels is a lower bound.

## Owner check (not the analysis E3)

Median over 50 hearts of SD_noise (√ of the mean over θ* of the replicate variance of θ̂/θ*) at the new levels, against
the median SD_cons (primary constructions, same rule as `run_analysis.py`; reproduced 0.0284 and 0.1816 for corr, 0.0324
and 0.1597 for amp). Break-even: `run_analysis.crossing` rule on the combined A3 + B4 grid, percentile 95% CI.

| loss, comparator | parameter | SD_cons | SD_noise at new levels | break-even [95% CI] |
|---|---|---|---|---|
| corr, white (5, 0, −5, −10, −15 dB) | v_endo | 0.0284 | 0.0053, 0.0095, 0.0176, 0.0315, 0.0676 | −9.1 dB [−11.1, −8.2] |
| | v_t | 0.1816 | 0.0247, 0.0432, 0.0798, 0.1591, 0.3018 | −11.0 dB [−12.2, −9.5] |
| corr, lp40 | v_endo | 0.0284 | 0.0233, 0.0466, 0.1394, 0.4052, 0.6478 | 3.6 dB [1.7, 4.3] |
| | v_t | 0.1816 | 0.1129, 0.2326, 0.5975, 0.9218, 1.1326 | 1.7 dB [0.8, 3.1] |
| corr, wander (40, 80%) | v_endo | 0.0284 | 0.0181, 0.0354 | 63.7% [50.2, beyond grid > 80] (246 of 2,000 resamples beyond) |
| | v_t | 0.1816 | 0.1472, 0.2747 | 50.5% [39.6, 66.1] |
| corr, displacement (30, 40 mm) | v_endo | 0.0284 | 0.0400, 0.0481 | 19.8 mm [16.2, 31.2] |
| | v_t | 0.1816 | 0.1688, 0.2177 | 32.6 mm [23.8, beyond grid > 40] (68 of 2,000 beyond) |
| amp, white | v_endo / v_t | 0.0324 / 0.1597 | see `owner_check.json` | −3.3 dB [−4.7, −2.0] / −2.6 dB [−4.1, −0.7] |
| amp, lp40 | v_endo / v_t | | | 9.3 dB [7.8, 10.3] / 10.7 dB [8.6, 12.3] |
| amp, wander | v_endo / v_t | | | 24.0% [19.4, 31.2] / 14.9% [11.8, 20.5] |
| amp, displacement | v_endo / v_t | | | 11.6 mm [9.1, 14.1] / 10.3 mm [8.5, 13.7] |

Under the correlation loss, every point break-even now lies inside the extended grid. The CI upper ends for wander
(v_endo) and displacement (v_t) still reach beyond it. Differences from the E3x extrapolation (corr, measured minus
E3x): white −0.2 dB and +1.6 dB, lp40 +0.8 dB and +1.0 dB, wander +3.4 and +4.1 points, displacement −0.2 mm (v_endo, inside the A3 grid already)
and +0.2 mm (v_t); pairs in v_endo, v_t order. Owner arithmetic; the analysis owner reports the official figures.

## Assumptions

1. Negative-SNR seed key 1000 + |level| (numpy cannot take negative entropy); any bijective non-negative key would do.
2. Construction SD for the check = primary constructions (include_primary), matching loss, as in E3.
3. Output split into three HDF5 files (two noise shards, one displacement) rather than per heart, to keep the file
   count low (quota about 90%).
4. Owner check reuses the `run_analysis.crossing` interpolation and percentile rule, reimplemented in
   `export_noise_ext.py`; it was not run through `run_analysis.py`.

## Not done or not verified

- Pipeline not rerun end to end a second time; two replicates spot-checked.
- I did not check which amplitude-loss break-evens already lay inside the A3 grid, so no measured-minus-E3x
  differences are given for the amplitude loss.
- The analysis has not yet concatenated `noise_estimates_ext.csv`; `COMP` levels in `run_analysis.py` would need extending.
