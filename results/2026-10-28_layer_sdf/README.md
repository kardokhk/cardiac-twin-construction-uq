# Amendment 2 B1: smooth-surface fast layer, arm C rerun (owner layer-sdf, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Governing text: `final/AMENDMENT_2_2026-10-28.md` B1 (declared
sensitivity, added after the full-arm data were seen) over `final/AMENDMENT_2026-10-28.md`. No pre-registered line, primary
estimand or primary output was changed; the primary estimand stays on the A2 layer. Owner check only; the integration
owner produces the final table.

## Jobs (account p201509, partition cpu, 1 node each)

| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5395849 | test | B1 unit test (22 cases) and B1 tags for heart 102 | 00:06:39 | 0.11 |
| 5395850 | test | arm C test, heart 102, 10 primary constructions | 00:12:08 | 0.20 |
| 5395851 | default | B1 tags for 494 primary constructions | 00:21:16 | 0.35 |
| 5395852_0..4 | default | arm C rerun, 5 shards (`results/2026-10-28_armC/shards.json`) | 01:09:48, 01:29:09, 01:11:23, 01:09:32, 01:09:32 | 6.16 |
| 5395853 | default | export and owner check | 00:15:42 | 0.26 |

Total 7.09 CPU node-hours (cap 20; projection after the test was about 7.5). Logs: `logs_layer_sdf_2026-10-28.tar.gz`.
Test outputs: `test_jobs5395849-5395850.tar.gz` (not for analysis). Code SHA-256: `code_sha256.txt`.
Rerun: `MODE=layer sbatch --time=01:00:00 code/full/calib/armC_sdf_job.sh`, then `MODE=full sbatch --array=0-4
--time=03:00:00 ...`, then `MODE=export sbatch --time=00:30:00 ...` (unit test: `MODE=unit`, qos test).

## Code

- `code/full/mesh/endo_layer.py`: new functions `sdf_layer`, `smooth_endo_surface`, `dist_to_triangles`,
  `point_triangle_dist` appended. The A2 functions are byte-identical to the previous file (SHA-256 before the addition
  c064601b...59c3a9; copy at `notes/scratch/2026-10-28-endo_layer.pre-B1.py`).
- `code/full/mesh/test_endo_layer_sdf.py`: the A2 test cases, imported from `test_endo_layer.py` (unchanged).
- `code/full/calib/layer_sdf_tags.py`, `run_armC_sdf.py`, `armC_sdf_job.sh`, `export_sdf.py`. `run_armC_sdf.py` imports
  `run_armC.py` and `calib_lib.py` (both unchanged, SHA-256 equal to the arm C record) and replaces only the mesh writer,
  so that element tags come from `layer_sdf.h5:/cons/<id>/etag_sdf`. The reference construction that generates the
  targets carries the B1 layer too. Arm N is skipped; only `include_primary` rows run.

## Software, inputs, seeds

openCARP v20.0 AppImage, `-renumbering 0`, one eikonal per core, 126 workers, `OMP_NUM_THREADS=1`, own
`/dev/shm/b1sdf_<job>_<task>` per task. Env `cvri-sim` (Python 3.11.17, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pandas
3.0.6). Installed: scikit-image 0.25.2 and lazy_loader 0.4 with `pip --no-deps --target
/project/home/p201509/envs/cvri-sim-b1-extra` (556 files, bundled tests removed). `endo_layer` appends this path for
`marching_cubes` only. The `cvri-sim` env itself was not changed.
Inputs: `data/processed/acdc_mesh_v2/MANIFEST.tsv` rows with `include_primary` (494: 50 hearts, 5 sources, 1.5 and 1.0
mm, RV wall 3 mm), construction files read only. Seeds: θ* Latin hypercube seed 20261028 + patient id (identical
to arm C for all 50 hearts, checked); unit-test poses `default_rng(20261028)` (as the A2 test); bootstrap
`default_rng(20261028)` per cell, 2,000 cluster resamples of hearts. The B1 layer draws no random numbers.

## Definitions chosen here (B1 silent)

1. "Native voxel" = the construction's label-grid voxel (1.5 or 1.0 mm), the grid that the mesh and the A2 test use;
   σ = 1.5 or 1.0 mm. The alternative reading (ACDC acquisition voxel, anisotropic, 5 to 10 mm through-plane) was not used.
2. LV (label 3) and RV (label 4) blood are smoothed separately: nearest-neighbour upsampling by 6 (1.5 mm) or 4 (1.0 mm),
   Gaussian filter (truncate 4, zero outside), marching cubes (Lewiner) at level 0.5. The signed distance field is the
   distance to this surface, negative inside. Its zero level set is the surface itself.
3. Cavity side only: a surface triangle is kept when the nearest non-blood native voxel to its centroid is kept
   myocardium. This drops the smooth analogue of A2's blood-background faces (basal cut): median 10.6% of triangles.
4. Distance = exact point-to-triangle distance from the tet centroid; layer if ≤ 1.5 mm. Checked against brute force
   (300 centroids, maximum difference 0) and dense sampling (1,000 random triangles).

## Unit test (`unit_test_sdf.json`, `unit_test_table.csv`), against the ±15% lines

| case | B1 error, 1.5 mm | B1 error, 1.0 mm | A2 error, 1.5 / 1.0 mm (same meshes) |
|---|---|---|---|
| ellipsoid, 8 poses (median, min to max) | −5.2% (−5.4 to −4.9) | −3.0% (−3.1 to −2.9) | +43.4% / +26.4% |
| sphere r 25 mm (LV and RV label) | −5.2% | −3.8% | +44.9% / +25.6% |
| plane, whole face | −9.8% | −15.9% | 0 / 0 |
| plane, flat core of the face | 0 (1.500 of 1.500 mm) | 0 (1.667 of 1.667 mm) | |

Ellipsoid lines: all 16 poses within ±15%. Median volume ratio 1.0/1.5 mm 1.023 (within ±15%). B1 PASS. The A2 values
reproduce `results/2026-10-28_mesh_v2/full/endo_layer_test.json` exactly. The plane whole-face deficit comes from the
rounding of the finite cavity's edges, and the plane is not a B1 acceptance line. The plane core thickness was
recomputed on the login node (`unit_test_sdf_plane.json`) after a fix to the core footprint selection in the test script
(the job 5395849 value selected by centroid and undercounted the border voxels). The layer itself was not changed.

## Layer share of myocardial volume (`layer_share.csv`, `layer_share_summary.csv`; median, 2.5th to 97.5th percentile)

| resolution | B1 | A2 | B1/A2 |
|---|---|---|---|
| 1.5 mm (247) | 29.0% (18.7 to 44.2) | 43.6% | 0.669 |
| 1.0 mm (247) | 29.5% (19.0 to 44.0) | 38.2% | 0.769 |

The B1 layer lies inside the A2 layer in every construction (B1-only volume median 0). Unlike A2, the share does not
depend on resolution. Tags: `layer_sdf.h5` (`etag_sdf`, `endo_dist_sdf_mm`); construction files and `etag` untouched.

## Arm C with B1 (`estimates_sdf.csv`: arm C columns plus `layer` = sdf_B1, round 2)

QC: 494 constructions, 2,964 of 2,964 estimates, 0 NaN; refinement runs 2,827 (at most 9 per construction); emulator
RMS error at refinement points 0.007, 0.037, 0.126 ms (2.5th, 50th, 97.5th), maximum absolute 5.05 ms. Reference
recovery max |θ̂/θ* − 1| 1.3e-4 (v_endo), 2.2e-4 (v_t). Correlation-loss fit 0.000, 0.044, 0.215. Censored: correlation
26 of 1,482 (A2 primary 52), amplitude 7.

Owner check, E1 (median over 50 hearts of between-construction CV of θ̂/θ*, 95% CI), `owner_check_E1.csv`:

| loss | parameter | B1 | A2 (primary) | B1 − A2, hearts paired |
|---|---|---|---|---|
| corr | v_endo | 3.1% [2.8, 3.7] | 2.6% [2.2, 2.9] | +0.5 [+0.2, +1.0] points |
| corr | v_t | 13.0% [12.0, 16.2] | 16.1% [12.5, 18.7] | −3.1 [−4.9, −0.3] points |
| amp | v_endo | 3.6% [2.9, 4.3] | 3.1% [2.5, 3.6] | +0.5 [+0.0, +0.9] points |
| amp | v_t | 11.6% [10.2, 14.1] | 12.8% [11.3, 15.6] | −1.2 [−2.2, +0.2] points |

The A2 column reproduces analysis E1 (2.6% [2.2, 2.9], 16.1% [12.5, 18.7]) exactly, which checks this re-implementation.
Reading: with the thinner smooth layer, construction spread is a little larger for v_endo and smaller for v_t. The
construction spread for v_t stays several times that for v_endo. Known-truth simulation; associations only.

## Not done or not verified

- No independent recomputation of the B1 CV beyond the A2 reproduction; the integration owner's table is final.
- Layer share is reported against A2 only; not compared with the one-voxel layer.
- B1 rests on assumption 1 (σ from the mesh grid voxel). σ from the ACDC in-plane voxel would differ slightly; it was not run.
- Censored estimates were kept at the bound (A1 rule); no drop-censored sensitivity was computed here.
