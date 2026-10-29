# Full arm C constructions, A2 fast layer, RV-wall sensitivity, displaced lead fields (owner mesh2, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Governing text: `final/AMENDMENT_2026-10-28.md` (A1 to A4) and
`final/STUDY_PLAN.md` sections 3 and 4. No pre-registered line or Amendment 1 setting was moved. Scope dirs:
`code/full/mesh/`, `data/processed/acdc_mesh_v2/`, this directory.

## Jobs (account p201509, partition cpu, 1 node each)

| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5393524 | test | test run: unit test, trio constructions, RV wall 2/4 mm (101), displaced W for 101 | 00:16:05 | 0.27 |
| 5393731 | default | stage 1: unit test, 50 references, 750 constructions, smoke tests, displacement prep | 00:13:55 | 0.23 |
| 5393732_0..2 | default | stage 2: 5,813 torso lead-field solves, 3-node array | 00:36:29, 00:35:28, 00:35:35 | 1.79 |
| 5393733 | default | stage 3: 50 displaced-W files, manifest. State FAILED at the final cleanup line only (unbound `$D`); all outputs written; scratch removed by hand; script fixed afterwards | 00:22:21 | 0.37 |

Total 2.66 CPU node-hours (cap 20). Scripts: `code/full/mesh/full_build.sh`, `full_solve.sh`, `full_assemble.sh`
(full run), `mesh2_job.sh` (test). SHA-256 of all code: `full/code_sha256.txt` (the `full_assemble.sh` hash there
predates the one-line cleanup fix).

## Software, inputs, seeds

openCARP v20.0 AppImage, `-renumbering 0`; env `cvri-sim` (Python 3.11.17, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0,
pandas 3.0.6, pyamg 5.3.0). Nothing installed. Code copied from `code/gates/mesh/` (acdc_lib, build, sax_route, carpio;
sax_route and carpio unchanged), imports `code/pilot/I13/eplib.py` and `code/gates/ka/bayer_i09.py` unchanged
(`python -I -B`, no bytecode written). Torso subset reused read-only from `data/processed/acdc_mesh/torso09_near.npz`.
Seeds: Bayer fibres numpy global seed 20261028 (OMP_NUM_THREADS=1); pyamg torso solver seed 20261028; displacement
seed 20261028 + 1000 x patient + 100 x sd_index + rep; unit-test poses default_rng(20261028).

## 1. A2 fast endocardial layer (`endo_layer.py`, `test_endo_layer.py`, `full/endo_layer_test.json`)

Rule: tets whose centroid lies within 1.5 mm (ties included) of a voxel face shared by kept myocardium and LV or RV
blood; exact point-to-square distance. Epicardial and basal faces are excluded by construction. Stored as `etag`;
the gates one-voxel layer as `etag_onevoxel`; the distance as `endo_dist_mm`.

| case | 1.5 mm voxels | 1.0 mm voxels |
|---|---|---|
| plane, known answer from centroid rule | exact (1.5 mm thick) | exact (1.667 mm thick) |
| sphere r 25 mm, A2 layer vs analytic shell | +44.9% | +25.6% |
| ellipsoid (20, 25, 40) mm, 8 poses, A2, median (range) | +43.4% (+41.4 to +47.8) | +26.4% (+25.7 to +26.6) |
| same, gates one-voxel layer | +31.2% (+27.1 to +40.0) | -14.1% (-16.9 to -8.7) |
| median layer volume ratio 1.0/1.5 mm | A2 0.88 | one-voxel 0.67 (from the medians above) |

Acceptance declared in the test file before the first run: plane exact (PASS), ellipsoid within 15% (FAIL), 1.0/1.5
ratio within 15% (PASS). Overall FAIL. Cause: the distance is to the voxel staircase, whose offset volume exceeds the
smooth surface's; the bias is larger at 1.5 mm. A2 halves the resolution dependence of the layer but does not give a
1.5 mm-thick layer. Reported against the line; A2 not changed.

Real hearts (RV wall 3 mm, all meshed constructions), fast-layer share of myocardial volume, median (2.5th to 97.5th
percentile): 1.5 mm A2 0.449 (0.281 to 0.680), one-voxel 0.413 (0.258 to 0.636); 1.0 mm A2 0.394 (0.247 to 0.598),
one-voxel 0.267 (0.167 to 0.411). Trio expert (A2 / one-voxel): 101 0.484/0.443 at 1.5, 0.429/0.290 at 1.0; 102
0.467/0.431, 0.400/0.273; 104 0.347/0.322, 0.298/0.209. Every construction carries `etag_onevoxel`, so the A2
one-voxel sensitivity on the trio needs no rebuild.

## 2. Constructions (all 50 ACDC test hearts)

Regression: all 313 constructions that also exist in the gates run have the identical mesh, identical W (maximum
difference 0.0) and `etag_onevoxel` equal to the gates `etag`; all 50 reference transforms and root points equal the
gates ones (difference 0.0).

| set | attempted | G2 pass (F1 to F9) | included |
|---|---|---|---|
| primary (expert + 4 networks, 1.5 and 1.0 mm) | 500 | 498 | 494 (247 per resolution) |
| ukbb_cardiac sensitivity | 100 | 92 | 92 |
| RV wall 2 and 4 mm, 1.5 mm (plan) | 100 | 51 | 51 |
| RV wall 2 and 4 mm, 1.0 mm (extra, not pre-registered) | 100 | 100 | 100 |

Primary exclusions (both resolutions): cinema_acdc_sax 129 (G1), nnunet_ukb_sax 133 (G1) and 145 (G1, F6 -27.2% /
-26.3%). ukbb_cardiac failures: 114 (F2), 121 and 124 (F1, no RV endocardial nodes), 125 (F3). No included
construction failed the smoke test. Smoke maximum activation median 73.6 ms (1.5 mm) and 73.1 ms (1.0 mm).
Node counts, RV wall 3 mm: median 64,150 (1.5 mm) and 199,786 (1.0 mm).

**Root shift > 10 mm (A4)**, denominator = non-reference constructions of that resolution with a mapped root:
1.5 mm 8 of 245 (primary sources 3 of 199; included primary 3 of 197); 1.0 mm 9 of 295 (primary sources 4 of 249;
included 4 of 247). Flagged primary: 114 cinema (11.2, 10.6 mm), 114 Task114 (12.2, 12.8), 114 UKB SAX (26.6, 26.2),
133 Task114 at 1.0 mm (10.2). Other flags are ukbb_cardiac (109, 113, 118, 131, 136). At 1.0 mm the expert
construction is non-reference and counted.

## 3. RV free-wall thickness (hearts 101, 102, 103, 104, 105, 106, 107, 109, 112, 119)

4 mm: 50 of 50 pass. 2 mm at 1.5 mm: 1 of 50 pass (103 expert); 49 fail F3 (largest component 0.81 to 0.947): a
2 mm wall is thinner than two 1.5 mm voxels and splits from the septum. The 1.0 mm extra set passes 100 of 100 and
gives a 2/3/4 mm comparison at one resolution; it is labelled `rv_wall_extra_1.0mm` and is not pre-registered.

## 4. Displaced lead fields (A3.3; `displace.py`)

18,000 draws (50 hearts x 3 SD x 20 replicates x 6 electrodes) mapped to 5,813 distinct torso surface nodes, each
solved once (33 to 35 CG iterations, relative residual at most 1.0e-10). Surface edge length median 1.70 mm; end
point to node snap median 0.62 mm (maximum 1.27 mm); walked/target length median 0.999 (sphere check of the walk:
geodesic error under 0.01 mm at 20 and 60 mm). Distance of displaced node from the original electrode, median
(2.5th to 97.5th): 5.9 (1.2 to 13.4), 11.7 (2.6 to 27.3), 23.4 (4.5 to 54.2) mm for SD 5, 10, 20 mm. Relative change
of W (all 8 leads, Frobenius) median 0.101, 0.202, 0.382. Zero-offset check: W from new solves against stored W, maximum
relative difference at most 7.3e-4 (per-lead relative L2 at most 3.2e-5 on 101); stored-Z path 4.1e-8.

## Assumptions

1. Inclusion requires all G2 criteria F1 to F9 (the table in `G2_criteria.md`), not only F1 to F8.
2. ukbb_cardiac enters its sensitivity set wherever it meshes, regardless of G1 (A1 wording); 20 of its 92 included
   rows (10 hearts) are G1-unusable. Filter on `g1_usable` to apply G1 too.
3. `root_shift_mm` is the maximum over the 4 roots; each root is in `root_shift_each_mm`.
4. Displacement: surface walk (geodesic approximation), electrode at the nearest boundary node (rule v2 convention).
5. The unit-test acceptance threshold (15%) was declared before the first run; the plan gave none.

## Outputs

- `data/processed/acdc_mesh_v2/MANIFEST.tsv` (800 rows), `cons/` (741 HDF5), `ref/` (50), `displaced/` (50 HDF5,
  `W_disp[sd, rep, lead, node]`), 35 GB, 846 files.
- Here: `full/` and `test/` (`constructions.tsv`, `summary.json`, `endo_layer_test.json`, `disp_prep.json`,
  `disp_solves.json`, `disp_draws.npz`, `code_sha256.txt`), `logs_qc_mesh_v2_2026-10-28.tar.gz` (all logs and QC JSONs).
  Index: `MANIFEST`.

Not done: pipeline not rerun end to end a second time; the full unit test ran twice with identical output.
