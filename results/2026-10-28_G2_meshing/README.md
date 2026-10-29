# Gate G2: ACDC constructions, meshing and timing (owner mesh, I11 gates, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Tasks 1 to 6 of the meshing specification. No pre-registered line was
moved. Criteria for G2 were written to `G2_criteria.md` (SHA-256 `b12892473ffb7b1b1ca4bacc039f88415033c4b87237bad1e026b075ff03c0aa`)
at 03:33 on 2026-10-28, before job 5392468, the first job that meshed an ACDC heart.

## Verdict

**G2 PASS: 50 of 50 hearts meshed from expert labels at 1.5 mm without manual fixes** (line: at least 35 of 50).
All 50 also passed the forward smoke test (eikonal plus 8-lead ECG). Pass in every group (10 of 10 each in DCM,
HCM, MINF, NOR, RV). Reaction-eikonal run time is measured: arm F on 10 hearts x 3 theta* is far below the
100 node-hour line (worst case 1.6 node-hours at 1.5 mm), so arm F stays at 10 hearts.

## Jobs (account p201509, partition cpu, 1 node each)

| job | qos | purpose | elapsed | node-hours |
|---|---|---|---|---|
| 5392468 | test | torso prep, trio references, trio networks 1.5 mm, one 1.0 mm; RE check failed (MPI refused) | 00:05:32 | 0.09 |
| 5392475 | default | all constructions, ellipsoid check, timing; fibres unseeded (superseded, see "Reproducibility") | 00:20:48 | 0.35 |
| 5392520 | default | rebuild of all constructions with seeded fibres, packed timing, RE timing | 00:15:43 | 0.26 |

Total 0.70 CPU node-hours (cap 15). Script: `code/gates/mesh/mesh_job.sh` (`MODE=test|full`, `SKIP_ELLIPSOID`,
`SKIP_PACK`). Logs and all per-construction QC JSONs: `logs_qc_mesh_2026-10-28.tar.gz` (1,361 members).

## Software, inputs, seeds

- openCARP v20.0 AppImage (`/project/home/p201509/opt/opencarp-v20.0/...`), `-renumbering 0` passed explicitly.
- Env `cvri-sim`: Python 3.11.17, numpy 2.4.6, scipy 1.17.1, nibabel 5.4.2, h5py 3.16.0, pandas 3.0.6.
  **Installed by me: pyamg 5.3.0** (`pip install --no-deps`, 242 files; it was missing from cvri-sim; the pilot
  used it from `pilot-I13`).
- Labels: expert `data/raw/acdc/extracted/testing/patientNNN/patientNNN_frameED_gt.nii.gz` (ED from Info.cfg);
  networks from `data/processed/acdc_seg/MANIFEST.tsv` (seg owner, G1).
- Torso: Malik 2026 heart 09 (`data/processed/pilot_I13/malik09.npz`, `heart09.h5`, electrode rule v2, lead fields
  Z, homogeneous 0.2 S/m). Subset (tets within 140 mm of the LV centroid, 1,598,949 nodes, 9,032,632 of 15,208,101
  tets) in `data/processed/acdc_mesh/torso09_near.npz` (SHA-256 22e523e5...3d84).
- Seeds: numpy global seed 20261028 before every Bayer call (pyamg spectral-radius estimate); design seed 13013
  (pilot design, timing only). Nothing else is stochastic.
- Code (SHA-256 of the versions that produced the data): `acdc_lib.py` 0794dcf7...faedfe, `build.py` 9641cf1e...adbf,
  `sax_route.py` e3eecec4...f2a1, `carpio.py` 21423d76...8139, `design_runs.py` b50d685c...91a5,
  `re_timing.py` 89061535...91ac5, `manifest.py` b551dfa1...51a2, `mesh_job.sh` 6048e1df...fe. Reused unchanged:
  `code/pilot/I13/eplib.py` (035c9368...0df), `code/gates/ka/bayer_i09.py` (c61bd95d...c6d2), which runs the I09
  pilot fibre functions verbatim.

## Task 1: construction pipeline (`code/gates/mesh/`)

`sax_route.py` keeps the pilot SAX algebra (per-slice signed distance, linear interpolation between slice centres,
apical cap within half a slice, 3 mm RV wall, 6 tets per voxel, largest 6-connected myocardium). It adds an ACDC input
path and skips the RV-wall step when no RV exists (the pilot crashed there). Checks:
- `test_sax_route.py`: on the pilot path, labels, mesh and lengths are bit-identical to `code/pilot/I11/06_constructions.py`
  at 1.5 and 1.0 mm, with and without perturbations: PASS.
- ka's `ellipsoid_sax.py`, run unchanged with `--route code/gates/mesh/sax_route.py` (96 cases): every error column
  identical to ka's run of the pilot route (maximum absolute difference 0.0). At 1.5 mm and 10 mm spacing the total
  myocardial error is a median of -11.0% (range -15.8% to -4.8%), as ka reports. Files: `full/ellipsoid/`.

Per construction (`build.py`): per-class largest 26-connected component, slices with LV myocardium, reconstruction,
mesh, Bayer fibres (+40/-50 helix, -65/+25 sheet), fast endocardial layer = tets touching a blood-interface node
(tag 2; median 41% of tets on expert 1.5 mm, 2.5th to 97.5th percentile 27% to 62%), root nodes, lead field, one
eikonal run.
- Root nodes: the pilot rule (LV septal mid h 0.50, LV anterior paraseptal h 0.65, LV posterior paraseptal h 0.33,
  RV septal h 0.40; 2.5 mm stimulus patches) applied to the expert 1.5 mm construction; the 4 points are mapped to
  every other construction by nearest LV or RV endocardial node. Shift median 1.5 mm, 97.5th percentile 5.5 mm;
  8 of 263 non-reference constructions have a root more than 10 mm away (maximum 34.7 mm, mostly the RV root,
  6 of 8 are ukbb_cardiac).
- Volumes, expert 1.5 mm, reconstruction against label volume: LV myocardium median -5.2% (2.5th to 97.5th
  percentile -7.6% to -2.1%), LV cavity -7.0% (-12.3% to -2.6%). The negative bias matches the ellipsoid test
  (half a slice lost at the base).
- Mesh size, expert 1.5 mm: median 64,375 nodes (37,240 to 133,954). Trio at 1.0 mm: 112,448 to 226,825 nodes.

## Task 2: ECG and placement in torso 09

ACDC affines carry no spacing or orientation (stored affine diag(-1,-1,1)). Image-physical coordinates are
i*sx, j*sy, k*sz from the header zooms. Heart axis: slice normal, base to apex; the apex end is the stack end with
the smaller mean LV epicardial area of its two outermost LV slices (the rule put the apex at the high slice index in all 50 expert hearts; 8 of 265 attempted network constructions would have chosen the other end on their own labels, and the reference direction was used). Septal
direction: LV cavity to RV cavity centroid, in-plane. Anterior = septal x axis: this is the chirality of 16 of 17
Malik heart frames (heart 07 is the exception, a sign artefact of the pilot frame) including heart 09. The rotation
maps (axis, septal, anterior) onto heart 09's frame (det +1 asserted); the LV myocardial centroid goes to heart 09's.
The expert 1.5 mm construction fixes this transform and the slice direction; every construction of the heart reuses
both. Electrodes: rule v2 of heart 09; 8 leads (I, II, V1 to V6); W = K_i Z, ECG = -W Vm (pilot kernel).
- Diagnostic ICP refinement onto heart 09's surface: rotation median 16.6 degrees (1.4 to 30.7), RMS 7.1 mm before
  and 5.0 mm after. Its weights are stored as `W_icp`; `W` (frame alignment) is primary. On patient 102 the smoke ECGs
  under W and W_icp correlate at r = 0.987 to 1.000 per lead, amplitude ratio 1.18.
- Heart nodes not inside any of the 12 candidate torso tets: median 2,004 per expert construction (1,018 to 4,566),
  extrapolated from the nearest tet (minimum barycentric coordinate -2.5). With 48 candidates 0.7% remain on
  patient 101 (in gaps of the torso mesh, at most 2.8 mm from a torso node); W changed by 0.8% (relative norm). The
  pilot had about 5%; I kept the pilot setting.

## Task 3: S x M on the G1 trio (101 DCM, 102 NOR, 104 HCM)

All 36 constructions (expert + 5 networks, 1.5 and 1.0 mm) built and passed the smoke test (theta = 1.55, 0.29 m/s;
fibre 0.6 m/s). Maximum activation 62.2 to 92.8 ms. Node counts and volume errors: `constructions.tsv`, `summary.txt`.

## Task 4: G2 on 50 expert hearts (1.5 mm)

Meshing-only (F1 to F8): 50/50. Meshing plus smoke (F1 to F9): 50/50. Smoke maximum activation median 77.0 ms (55.2
to 121.8). Non-manifold boundary edges (reported, not a criterion): median 12 (0 to 237) of about 10^5. 9 of 50
hearts have 5 mm slices, 41 have 10 mm.

## Task 5: timing

**Eikonal, packed** (128 single-thread runs on 128 cores, pilot 90-run design on the 3 trio expert constructions,
270 runs per resolution, job 5392520): 1.0 mm median 39.8 s (2.5th to 97.5th percentile 19.4 to 107.5 s; mean 45.9 s;
maximum 157.1 s); 1.5 mm median 11.3 s (5.3 to 23.4 s; mean 11.8 s). Paired 1.0/1.5 ratio median 3.83 (2.66 to 4.89);
A1 assumed 3.4 and 95 s per 1.0 mm run. ACDC meshes are smaller than the pilot's Malik meshes, so runs are faster
than ka's heart 09 timing (283 s at 1.0 mm). Arm C (27,000 runs per resolution, 50 x 6 x 90): 2.7 node-hours at
1.0 mm (plan 5.6) and 0.7 at 1.5 mm (plan 1.6); x3 contingency 8.1 and 2.1. Caveat: the study widens the v_t prior
below 0.15 m/s, and the slowest runs are at low velocity, so these are lower bounds for the widened design.

**Reaction-eikonal** (openCARP RE+, `dream.solve 1`, monodomain, tenTusscherPanfilov, dt 25 us, 200 ms, patient 101
expert 1.5 mm, 67,621 nodes, near the median size). openCARP v20.0 refuses MPI for eikonal/RE physics ("Use openMP
instead"), and it also needs `phys_region ptype 2` (without it a plain monodomain run happens silently).
- Wall time: 190.5 s with 8 threads, 190.5 s with 32 threads, 60.0 s for 50 ms with 1 thread (about 240 s for 200 ms
  by linear extrapolation, not measured). Eight concurrent 8-thread copies: 185.1 s each (unseeded build).
  RE- (no diffusion), 8 threads: 116.1 s.
- Vm ECG against the activation-template ECG (same W): r = 0.960 to 0.991 per lead (V5 lowest), amplitude ratio 1.045.
  Upstroke 1.21 ms after the eikonal activation time; RE activation equals the pure eikonal activation.
- **Arm F extrapolation** (10 hearts x 3 theta* = 30 target runs at 1.5 mm): packed one per core 0.016 node-hours;
  8 threads per run, 16 per node 0.10; worst case, each run alone on a node, 30 x 190.5 s = 1.6 node-hours. At
  1.0 mm (3.1 times the nodes, cost assumed linear in nodes) worst case 5.0. If arm F also needed its own 90-run
  emulator grid per heart: 900 runs, 0.47 node-hours packed. **None exceeds 100 node-hours.**

## Task 6 (bonus): network constructions of all 50 hearts at 1.5 mm

| network | meshed (F1 to F8) | plus smoke |
|---|---|---|
| nnunet_task027_acdc | 50/50 | 50/50 |
| nnunet_task114_mnms | 50/50 | 50/50 |
| cinema_acdc_sax | 50/50 | 50/50 |
| nnunet_ukb_sax | 49/50 (145: F6, LV myocardium -26.3%) | 49/50 |
| ukbb_cardiac_fcn_sa | 46/50 (114: F2 no RV; 121, 124: F1 no RV endocardial nodes; 125: F3 component 0.944) | 46/50 |

## Reproducibility

Job 5392475 built every construction with unseeded fibres. pyamg's smoothed-aggregation setup draws a random start
vector, so the Laplace fields differed at about 1e-10 between calls, and the bislerp step flipped fibres by up to 90
degrees in about 0.6% of elements (patient 102: 983 of 159,162 elements over 1 degree; W changed by 3.8% in relative
norm; smoke ECG r >= 0.99999, relative L2 0.22%). All constructions were rebuilt with the seed in job 5392520. Seeded
builds are identical within a process and across login and compute nodes with one BLAS thread (maximum fibre
difference 3e-8), but differ with 8 threads. Keep `OMP_NUM_THREADS=1` (the job script does).

## Outputs

- `data/processed/acdc_mesh/MANIFEST.tsv` (313 rows; see handoff), `cons/` (313 HDF5, 5.9 GB), `ref/` (50
  per-heart transforms and root points), `runs/` (6 HDF5 of design runs on the trio expert constructions, pilot design,
  seeded fibres), `torso09_near.npz`. 371 files in total, 19 in this directory.
- `constructions.tsv`: every QC field for all 318 attempted constructions; `summary.txt`.
- `manifest.py` reads the QC JSONs from `full/qc` and `test/qc`, which are now in the tarball; extract before rerunning.

## Correction 2026-10-28 (follow-up round, owner mesh)

Superseded values above are kept as written. Job 5392656 (qos test, 1 node, 00:28:31, 0.48 CPU node-hours; script
`code/gates/mesh/re_job.sh`, SHA-256 4cb8f500...7d). Records: `full/re_followup_5392656.jsonl` (one line per run).
Code changed: `re_timing.py` (54f46126...d4f6), `carpio.py` (032f2524...ec7d). Seeds: none new (RE and eikonal are
deterministic; design seed 13013; fibres not rebuilt; `OMP_NUM_THREADS=1` except where a thread count is stated).

### Counts in Task 1 and Task 4

| item | old | new | reason |
|---|---|---|---|
| large root shifts (> 10 mm) from ukbb_cardiac | 6 of 8 | 5 of 8 (others: cinema_acdc_sax, nnunet_task114_mnms, nnunet_ukb_sax, all patient 114) | recount from `constructions.tsv` |
| rows carrying `root_shift_mm` | not stated | 313 of 318 (50 reference rows, all 0; 263 non-reference; the 5 failed constructions have none). The earlier count of 260 was not reproduced; "8 of 263 non-reference" stands | recount |
| minimum node count, expert 1.5 mm | 37,240 | 37,245 | typo; `constructions.tsv` |

The largest shift is at the RV root in 7 of the 8.

### Task 5, eikonal: one basis for A1 (reconciliation with ka)

Basis: median wall time of single-thread openCARP eikonal runs, 90 runs concurrent on one 128-core node, the same
90 theta (pilot design, seed 13013; identical theta arrays checked). ka's heart 09 timing used exactly this
packing. Job 5392520 did not: it pooled 270 runs on 128 workers. The trio was rerun at 90 workers in job 5392656.

| mesh | nodes 1.0 / 1.5 mm | median 1.0 mm | median 1.5 mm | paired 1.0/1.5 ratio, median |
|---|---|---|---|---|
| Malik heart 09 (ka) | 346,716 / 109,705 | 283.0 s | 62.5 s | 4.51 |
| ACDC 101 expert | 210,961 / 67,621 | 41.8 s | 11.6 s | 3.65 |
| ACDC 102 expert | 120,573 / 39,082 | 21.6 s | 6.4 s | 3.74 |
| ACDC 104 expert | 222,803 / 71,168 | 45.0 s | 11.6 s | 4.00 |
| ACDC trio pooled | | 37.4 s | 10.5 s | 3.82 (2.5th to 97.5th 2.75 to 5.02) |

Old (128 workers): pooled medians 39.8 and 11.3 s, ratio 3.83. What explains the difference:
- **theta:** none. Both used the same 90 theta.
- **Packing density:** small. At 90 instead of 128 workers, the per-heart medians fell 3% to 10% at 1.0 mm. At
  1.5 mm they changed by -15% to +5%. The pooled ratio stayed at 3.82 (3.83 before).
- **Node count:** the node ratio is the same, 3.16 for heart 09 and 3.09 to 3.13 for the trio, so it does not
  explain the ratio. The absolute times are another matter. Heart 09 takes 570 to 816 us of wall per node. ACDC
  takes 154 to 219 us (128 workers). On ACDC, wall time scales as nodes^1.10. Extrapolated that way, heart 09
  would take 21 and 76 s, not 62.5 and 283 s.
- **Heart size and geometry:** the likely reason. This is inference, not tested. Heart 09 is large (298 mL
  myocardium, wall length 15.6 mm, against 155 mL and 9.1 mm for 101). Its median maximum activation is about
  100 ms (ACDC 70 to 93 ms). The ratio also rises with heart size within the trio (102: 3.74, 104: 4.00). A
  fixed start-up cost per run weighs more on small 1.5 mm meshes, which lowers their ratio. Packing contention
  raises the ratio: on heart 09 solo the ratio was 150/37.2 = 4.0, and packed it was 4.5.
- A1 (3.4) underestimates the ratio for every heart measured.

**Slowdown from the widened prior** (v_endo 0.5 to 4.0, v_t 0.1 to 0.8 m/s). These are the actual 9 x 9 grid wall
times from `results/2026-10-28_G3G4_ministudy` (`grid_wall_s`, 126 workers per node, tend 1000 ms) against the pilot
design (128 workers, tend 400 ms) on the same 6 expert constructions:

| | median wall | mean wall |
|---|---|---|
| 1.0 mm | x0.98 (0.98, 0.99, 1.07) | x1.36 (1.25, 1.36, 1.56) |
| 1.5 mm | x0.87 (0.78, 0.87, 1.17) | x1.05 (0.95, 1.05, 1.50) |

The tail grows: on 101 expert 1.0 mm the v_endo = 0.5 column takes 117 to 204 s. The tend and worker-count
differences are confounded with the prior.

**Arm C cost (supersedes 2.7 / 0.7 node-hours and x3 contingency 8.1 / 2.1).** The run set is 295 constructions
that passed at 1.5 mm in `constructions.tsv` (50 expert and 245 network) x 90 runs = 26,550 runs per resolution.
The 1.0 mm set assumes the same 295 constructions; only the trio exists at 1.0 mm now. Per-run wall times come
from each construction's nearest-size ministudy construction: its 81 grid wall times, plus 9 draws from them (seed
20261028), scaled by (nodes ratio)^1.13 at 1.5 mm and ^1.19 at 1.0 mm. Those exponents are fitted to the 18
ministudy means per resolution against 1.5 mm node count. Billing is whole-node, 126 workers per node,
longest-first list scheduling:

| resolution | mean run | core-hours | node-hours, 1 node | 4 nodes | 8 nodes |
|---|---|---|---|---|---|
| 1.0 mm | 71.0 s | 523 | 4.2 | 4.2 to 4.4 | 4.2 to 4.7 |
| 1.5 mm | 15.4 s | 114 | 0.90 | 0.90 to 0.95 | 0.91 to 0.99 |

The plan has 5.6 and 1.6. Sensitivity: if ACDC hearts cost per node what heart 09 costs (3 to 4x), the totals
become about 15 and 3 node-hours. ka's heart 09 rate figures (20.4 and 4.4) are upper bounds for ACDC.

### Task 5, reaction-eikonal: honest timing (supersedes the RE wall times and arm F numbers above)

The bug: `re_timing.py` used `tag = f'm{mode}_np{npm}'` for its `/dev/shm` mesh, eikonal and simulation
directories. The 8 concurrent 8-thread copies in job 5392520 therefore shared one set of directories, so the
"185.1 s each" figure is invalid. The 4 concurrent runs (np1, np8, np32, RE- np8) had distinct tags but ran at
the same time. Now each copy has its own directories (tag includes the copy id and PID) and removes them when done.

Setup: patient101 expert 1.5 mm, RE+, 200 ms, dt 25 us. `re_wall_s` is the openCARP call. Billing uses the
copy-start to last-end span.

| condition | old | new: openCARP wall per run | span per node |
|---|---|---|---|
| solo, 1 thread, 200 ms | 60.0 s for 50 ms, "about 240 s" extrapolated | 192.3 s | 205 s |
| solo, 8 threads | 190.5 s (with 3 other runs) | 192.0 s | 208 s |
| solo, 32 threads | 190.5 s (with 3 other runs) | not measured (skipped for time) | |
| 128 copies x 1 thread | not measured | median 404.9 s (381.7 to 417.6) | 432 s |
| 16 copies x 8 threads | 185.1 s (8 copies, shared directories) | median 183.3 s (182.5 to 188.7) | 201 s |
| 4 copies x 32 threads | not measured | median 183.9 s | 197 s |

openCARP v20.0 RE+ does not scale with OpenMP threads here: 1 and 8 threads take the same time. Throughput per
node is 128 runs per 432 s at 1 thread, 16 per 201 s at 8 threads and 4 per 197 s at 32 threads. One run per core
is about 3.7x more efficient than 16 x 8 (3.4 against 12.6 node-seconds per run), despite a 2.1x per-run slowdown from contention. The Vm output is
byte-identical (SHA-256 99da4404...) across all 150 runs, whatever the thread count, packing or region layout.

**ECG agreement: unchanged.** Vm ECG against the activation-template ECG: r = 0.960 to 0.991 per lead (V5 0.960).
Amplitude ratio 1.045. Upstroke 1.21 ms after eikonal activation. All 0 unactivated. Maximum |delta r| against
`full/re_plus_np8.json` is 0.0.

**gregion warning fixed** (`carpio.run_reaction_eikonal(layout=...)`, default now `'default0'`). With regions
written as gregion[0..1], openCARP warned "check_unassigned_tags: 153510 element(s) ... carry tags {1} not
assigned to any gregion_i region; these elements default to region 0". The gregion[0] IDs were apparently not
counted, and tag 1 fell back to region 0, which held tag 1's own parameters, so results were correct. `'default0'`
adds an ID-less default gregion[0] (tag 1 parameters) and writes the regions as gregion[1..2]. The warning is gone.
At 200 ms Vm is byte-identical to `'legacy'` (4 legacy copies, 146 default0 copies). It was also identical at
20 ms on the login node, where `'default0dup'` matched too. `'legacy'` remains selectable.
`run_eikonal` (pure eikonal, pilot verbatim copy) was not changed.

**Arm F cost, whole-node billing** (1.5 mm measured on a 67,621-node heart; ACDC expert median 64,375,
maximum 133,954):

| design | 10 hearts x 3 theta* (30 runs) | 5 hearts x 3 theta* (15 runs) |
|---|---|---|
| packed one per core on one node | 0.06 to 0.12 node-hours (span 205 to 432 s); up to 0.24 for the largest heart | 0.06 to 0.12 |
| each run alone on a node (worst case) | 30 x 205 s = 1.7 | 0.85 |
| 1.0 mm (3.1x nodes, cost assumed linear in nodes, not measured), packed | about 0.37 (up to 0.75) | about 0.37 |
| 1.0 mm, each run alone | 5.3 | 2.6 |
| plus a 90-run RE emulator grid per heart, 1.5 mm, packed 128 per node | 900 runs, 8 node-waves x 432 s = 0.96 | 450 runs, 4 waves = 0.48 |
| same at 1.0 mm (linear scaling assumed) | about 3.0 | about 1.5 |

Old: 0.016 (packed, fractional-node accounting), 0.10, 1.6, 5.0 and 0.47. **Arm F stays far below the 100
node-hour line under every design.**
