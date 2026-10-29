# Known-answer tier additions (owner ka, I11 gates, 2026-10-28)

Project `/project/home/p201509/work/cvri/simulation`. Tasks: STUDY_PLAN section 5 "To add" (a) to (c), OPEN_QUESTIONS
S6 and S7, and the replacement of assumption A1 (d). No pre-registered line was moved.

## Jobs

| job | purpose | walltime | node-hours |
|---|---|---|---|
| 5392244 | qos test, reduced run (TIMEOUT during packed timing, as expected) | 00:30:09 | 0.50 |
| 5392283 | full run, `MODE=full` | 00:45:46 | 0.76 |
| 5392645 | follow-up (qos test): thick shell with fixed analytic reference, boundary conflict audit (`ka_fix.sbatch`) | 00:02:18 | 0.04 |

Total 1.31 CPU node-hours (1.27 before the follow-up). Script: `code/gates/ka/ka.sbatch`. Logs: `logs_ka_2026-10-28.tar.gz`.
Results of the full run are in `full/`.

## Software and inputs

- openCARP v20.0 AppImage (`/project/home/p201509/opt/opencarp-v20.0/...`). Eikonal settings: those of
  `code/pilot/I11/carpio.run_eikonal` (renumbering default 0, max_iter 500), plus I09 `carpio.write_par` for the shell.
- `cvri-sim`: Python 3.11.17, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, h5py 3.16.0.
- New env `/project/home/p201509/envs/ka-ldrb` (micromamba, conda-forge): Python 3.12, fenics-dolfinx 0.10.0,
  basix 0.10.0, ufl 2025.2.1, petsc4py 3.25.5, mpi4py 4.1.2, numpy 2.5.4, numba 0.68.0, plus pip fenicsx-ldrb 0.2.0
  (PyPI, MIT; github.com/finsberg/fenicsx-ldrb, successor of finsberg/ldrb), io4dolfinx 1.5.0, pyamg 5.3.0,
  pandas 3.1.0. **46,641 files.** The first attempt with dolfinx 0.9 failed (io4dolfinx needs at least 0.10).
  The shared micromamba package cache `/project/home/p201509/cache/mamba` holds 68,681 files. I did not clean it,
  because other owners may be using it.
- SHA-256 of the source files:
  - `code/pilot/I11/06_constructions.py`: 9a76f506…a4eac
  - `code/pilot/I09/02_case.py`: ec6f4823…06131
  - `code/pilot/I09/00_known_answer.py`: e4e595ae…7c5e
  - `code/pilot/I13/03_design_runs.py`: b707b4b8…1b9a
  - `code/pilot/I11/carpio.py`: 39edda03…8225
  - `data/processed/pilot_I13/cons/cons09.h5`: 0002919b…633f

  The full hashes, including those of each extracted function, are in `full/sax/ellipsoid_sax.json` and
  `full/fibres/*.json`.
- Seeds:
  - SAX test: `default_rng(20261028)`. Its draws are overwritten, so the test is deterministic.
  - Heart 09 builds: seed 11009 (K = 0, so no draws).
  - Timing designs: seed 13013 (pilot design).

## (a) SAX reconstruction of an analytic shell: `code/gates/ka/ellipsoid_sax.py`

**Callable.** Run it as `python -I ellipsoid_sax.py --route <module.py> --out <dir>`. The script extracts `sdf`,
`smooth_field`, `sax_construction`, `mesh_from_labels` and `lengths` verbatim with `ast` and records their hashes.
Two things change how the route runs, without editing its source:
- Observer perturbations are zeroed by intercepting the parameter `dict(...)` call.
- `smooth_field` returns zeros when its SD is 0.

The extraction is reused by the mesh owner through `load_route` / `make_namespace`.

**Geometry.**
- LV: truncated prolate ellipsoids, endocardium 25/65 mm and epicardium 33/73 mm, with the base 10 mm above the
  equator.
- RV crescent: an ellipsoid minus the LV body, with an assumed 3 mm wall.
- Truth:
  - analytic for the LV: myocardium 95.93 mL, cavity 104.56 mL;
  - midpoint quadrature on a 0.2 mm grid for the RV: wall 27.68 mL, cavity 94.34 mL;
  - total myocardium 123.61 mL.

**Sweep.** Slice offsets were swept over 8 equally spaced positions per spacing.

**Finding.** The pilot route crashes on an LV-only shell, because `zr.min()` is taken over an empty RV cavity, so
only the biventricular shape was run.

Signed relative error, median [min, max] over the 8 offsets, in-plane 1.25 mm (1.6 mm: within 1.1 points for myocardium and LV quantities, up to 1.9 points for RV wall):

| mesh | spacing | total myocardium | LV myocardium | LV cavity | RV cavity | RV wall |
|---|---|---|---|---|---|---|
| 1.5 mm | 10 mm | −11.0% [−15.8, −4.8] | −7.5% [−12.5, −1.7] | −9.5% [−17.8, −2.2] | −13.2% [−21.0, −3.0] | −22.3% |
| 1.0 mm | 10 mm | −9.2% [−13.6, −1.6] | −7.8% [−11.3, −0.1] | −9.0% [−16.7, −0.7] | −12.8% [−19.6, −1.9] | −13.4% |
| 1.5 mm | 5 mm | −6.0% [−9.2, −1.2] | −3.0% [−6.5, +1.1] | −3.1% [−7.6, +1.4] | −4.0% [−8.6, +1.2] | −16.8% |
| 1.0 mm | 5 mm | −4.5% [−7.7, −1.2] | −3.6% [−7.2, 0.0] | −3.9% [−7.8, +0.2] | −5.0% [−9.8, −0.7] | −7.5% |

- The 8 mm spacing lies between these rows (`full/sax/ellipsoid_sax_summary.csv`).
- Direct voxelisation without the SAX step gives a myocardium error of −2.3% at 1.5 mm and 0.0% at 1.0 mm. The
  LV cavity error is +1.4% and +0.8%.
- Myocardial Dice against the voxel truth was a median of 0.89 at 10 mm spacing and 0.93 at 5 mm.
- The median bias is negative in all 12 configurations (2 meshes x 3 spacings x 2 in-plane), but single offsets
  can be positive: total myocardium +1.03% at 1.0 mm, 8 mm spacing, 1.6 mm in-plane, offset 4.0 mm (1 of 96
  cases); LV myocardium up to +1.34% (9 of 96), LV cavity up to +1.39% (6 of 96), RV cavity up to +1.16% (2 of 96);
  RV wall 0 of 96 (see Correction 2026-10-28). The negative median bias comes from tissue lost beyond the outermost slices (no basal extrapolation) and
  from the half-slice apical cap.
- The range across slice offsets (about 11 percentage points at 10 mm) is construction variance that the route
  adds by itself, even with a perfect segmentation.
- At 1.5 mm, the 3 mm RV wall is under-represented (−17% to −22%).

**Line.** No acceptance line is pre-registered (plan: "myocardial volume error reported").

## (b) Bayer (I09) against ldrb: `code/gates/ka/fibres_compare.py`, `bayer_i09.py`

**Inputs.**
- Both codes received identical meshes, boundary sets and apex node(s). The apex came from ldrb's own search.
- ldrb was evaluated in DG_0, the same per-element quantities that I09 uses.
- Angles: helix +40° (endocardium) and −50° (epicardium); sheet −65° and +25°.
- Eikonal settings: v_f 0.6, v_t 0.29, and v_endo 1.55 m/s.

| mesh | helix difference median / p95 | angle between fibres p95 | LV free-wall helix, endo / epi (I09 ; ldrb) | median abs. AT difference | p95 | max | ΔTAT |
|---|---|---|---|---|---|---|---|
| ellipsoid, 1.5 mm voxel (214,650 tets) | 0.0° / 0.0002° | 0.004° | 38.2/−50.1 ; 38.2/−50.1 | 0.0 ms | 0.0001 ms | 2.7 ms | −0.03 ms of 61.3 |
| heart 09 c0ref, 1.5 mm (530,454 tets) | 0.44° / 45.8° | 63.9° | 39.2/−50.0 ; 37.4/−50.0 | 0.11 ms | 1.22 ms | 5.3 ms | +0.38 ms of 96.4 (0.4%) |

- **Transmural profiles.** The LV free-wall profile medians agree within 0.4° in bins φ_epi ≥ 0.1 on heart 09 (1.8° in the innermost bin), and
  exactly on the ellipsoid (+38.2° to −50.1°, monotone). See `full/fibres/*.json`.
- **Heart 09 tail.** The tail is not a rule difference. The boundary data conflict on that mesh:
  - 3,931 of 8,049 LV-endocardial nodes, and 4,770 of 10,852 RV-endocardial nodes, are also epicardial-face nodes.
  - I09 gives the endocardium priority (setdiff), while ldrb applies both Dirichlet conditions.
  - In elements that touch such nodes, the helix difference has a median of 25°. Away from them (372,103 elements),
    it has a median of 0.32° and a p95 of 4.7°. (Not reproduced; see Correction 2026-10-28.)
- ldrb 0.2.0 has a minor bug in its apex locator: it tests x[2] twice and never x[0].

**Line.** No numeric line is pre-registered.

## (c) Thick-shell eikonal: `code/gates/ka/thick_shell.py`

Line: maximum error / maximum analytic time < 2%. The I09 solver settings and the study settings gave identical
numbers. **The table below used a faulty analytic reference for the unsnapped Delaunay rows at 0.75 to 0.375 mm
and the voxel rows at 0.5 and 0.375 mm; corrected values are in Correction 2026-10-28.**

| h (mm) | I09 Delaunay mesh | same mesh, inner nodes snapped to r = a | voxel (Kuhn) mesh* | TAT error, Delaunay |
|---|---|---|---|---|
| 1.5 | 3.73% | 3.73% | 8.5% | +2.1% |
| **1.0** | **2.71% (FAIL)** | 2.71% (FAIL) | 6.8% | +1.4% |
| 0.75 | 3.19% (FAIL) | 2.23% (FAIL) | 5.5% | +1.4% |
| 0.5 | 2.29% (FAIL) | 1.73% (pass) | 6.2% | +1.2% |
| 0.375 | 5.54% (FAIL; one node near the source) | 1.43% (pass) | 5.8% | +0.9% |

\* The voxel error includes the difference between the staircase geometry and the sphere, so it is not a clean
solver error.

- The I09 values at 1.5, 1.0 and 0.75 mm are reproduced exactly.
- At 1.0 mm, the solver is slower than the analytic solution everywhere: mean +1.23 ms, and 57% of nodes exceed 2%.
- The 99th percentile of the error falls with h: 3.6%, 2.6%, 2.2%, 1.7% and 1.4% at 1.5 to 0.375 mm.

## (d) Run time at 1.0 mm (A1): heart 09 c0ref, pilot route, `03_design_runs.py`, 1 thread per run

- **Meshes.** 1.0 mm: 346,716 nodes and 1,791,330 tets. 1.5 mm: 109,705 nodes and 530,454 tets.

| condition | 1.0 mm | 1.5 mm |
|---|---|---|
| solo, 3 θ: median (range) | 150 s (141 to 588) | 37 s (35 to 138) |
| 90 runs packed on one node: median [2.5th, 97.5th percentile] | 283 s [116, 722] | 62.5 s [28, 146] |

- **Packed 1.0/1.5 ratio.** The paired ratio is a median of 4.51 (range 3.60 to 5.16). Assumption A1 used 3.4.
- **Velocity dependence.** Run time depends strongly on θ: the slowest corner, θ = (0.8, 0.15), is the maximum.
- **Arm C cost at heart 09 rates.** 27,000 runs at the mean packed wall time give 20.4 node-hours at 1.0 mm (the
  plan has 5.6) and 4.4 at 1.5 mm (the plan has 1.6).
- **Size caveat.** Heart 09 lies at the top of the pilot's node range (57k to 125k), so these figures are upper-end
  estimates for ACDC hearts, whose sizes are not yet known.

## Correction 2026-10-28 (owner ka, job 5392645)

### Thick-shell analytic reference (section c)

**Reason.** `geod` (copied from `code/pilot/I09/00_known_answer.py`) marks a node "blocked" when the closest point
of the straight segment from the source lies at distance < a. Fibonacci nodes on the inner sphere lie at
r = a - 1.8e-15 mm for about half of them (35, 94, 158, 369 and 681 nodes at 1.5 to 0.375 mm). For such a node on the
lit side the closest point is the node itself, so the test fires and the wrap formula returns a length shorter than
the straight chord (by up to 3.33 ms of time at 0.375 mm). The former diagnostic counted nodes with r < a - 1e-9 and
reported 0. "Snap" moves nodes by about 1e-15 mm. It changed the results only by flipping this test, so it is not a
test of the polyhedral inner boundary, as described above. Fix (`code/gates/ka/thick_shell.py`, `geod_fixed`):
blocked = closest < a(1 - 1e-9), and a blocked geodesic is never shorter than the chord. `code/pilot` was not
changed. Known-answer checks: antipode 36.955 mm, lit pole 5.000 mm, equator 18.478 mm, all exact. The I09 `geod` is
still evaluated, and with it the rerun reproduces every old value exactly, so the solver output did not change.
Output: `full/thick_shell_fixed.csv` (columns `*_i09geod` hold the old reference) and `full/thick_shell_fixed_nodes.npz`.

Max |error| / max analytic T. Old value is followed by the new value; the two solver settings remain identical.

| h (mm) | Delaunay | Delaunay, snapped | voxel (Kuhn) |
|---|---|---|---|
| 1.5 | 3.73% → 3.73% (unchanged) | 3.73% → 3.73% | 8.5% → 8.5% |
| **1.0** | **2.71% → 2.71% (FAIL, unchanged)** | 2.71% → 2.71% (FAIL) | 6.8% → 6.8% |
| 0.75 | 3.19% → 2.23% (FAIL) | 2.23% → 2.23% (FAIL) | 5.5% → 5.5% |
| 0.5 | 2.29% (FAIL) → 1.73% (pass) | 1.73% → 1.73% (pass) | 6.2% → 4.2% |
| 0.375 | 5.54% (FAIL) → 1.43% (pass) | 1.43% → 1.43% (pass) | 5.8% → 3.8% |

- With the fix, the snapped and unsnapped Delaunay meshes give identical results. The maximum now lies on the outer
  surface on the shadow side (116° to 132° from the source), in every Delaunay row.
- **S7 at 1.0 mm:** 2.71%, FAIL, unchanged. The bug altered the reference at 5 inner nodes there, none of them the
  maximum. The 2% line was not moved.
- TAT error, mean signed error, p99 (3.6%, 2.6%, 2.2%, 1.7%, 1.4%) and the 57% of nodes above 2% at 1.0 mm are
  unchanged.
- **The I09 pilot value at 0.75 mm carried the bug.** `results/2026-10-22_pilot_I09/known_answer_solver.csv`,
  `shell_iso_h0.75`, has 0.03194 (3.19%). The corrected value on the same mesh is 2.23%. The pilot values at 1.5 and
  1.0 mm were not affected at the maximum.
- Voxel rows at 0.5 and 0.375 mm also changed. Their maximum had been at lit-side staircase nodes inside the ball.
  The voxel reference for nodes with r < a is still only approximate (wrap with a clamped radius, bounded below by
  the chord).

### Fibre comparison (section b): boundary conflicts and tails

Output: `full/fibres/conflict_nodes.json` (`code/gates/ka/conflict_nodes.py`, which regenerates the meshes and
classification of `fibres_compare.py` and reads the stored `fibres_*.h5`).

- **Heart 09, counts reproduced.** 3,931 of 8,049 LV-endocardial nodes and 4,770 of 10,852 RV-endocardial nodes are
  also epicardial nodes, 8,701 in all.
- **Heart 09, element statistic not reproduced.** The "372,103 elements away, median 0.32°, p95 4.7°" statistic does
  not reproduce, and no code or log for it survives. Using the definition stated above (elements touching no
  conflict node), 473,496 elements are away, with |helix difference| median 0.34° and p95 6.9°. The 56,958 touching
  elements have median 25.2° and p95 77.6°. Counting only LV conflict nodes, 504,719 elements are away, with median
  0.39° and p95 19.8°. The 4.7° value is withdrawn and 6.9° replaces it.
- **Ellipsoid, not identical to round-off.** The two fields agree to round-off in the bulk: helix difference median
  5e-8° and p95 2e-4°, fibre angle p95 0.004°, dAT median 0 ms and p95 8e-5 ms. They are not identical everywhere:
  - The fibre-line angle differs by more than 1° in 6,415 of 214,650 elements (3.0%) and by more than 45° in 6,157,
    with a maximum of 89.97°.
  - The helix difference is above 1° in 193 elements, with a maximum of 66.5°.
  - |dAT| is above 1 ms at 373 of 50,124 nodes, with a maximum of 2.72 ms.
  - 24 RV-endocardial nodes are also epicardial (127 touching elements, helix difference median 14.5°). The large
    angle differences mostly lie elsewhere: 99th percentile 80° in elements away from conflict nodes. Their location
    was not analysed.

### SAX bias sign (section a)

The old wording, "The bias is always negative", was false. Total myocardium is +1.03% at 1.0 mm, 8 mm spacing,
1.6 mm in-plane, offset 4.0 mm. The section now gives the per-quantity count of positive cases (from
`full/sax/ellipsoid_sax.csv`; the `err_*_mL` columns hold relative errors despite their names). The median remains
negative in every configuration. The table values are unchanged.

## Note 2026-10-28
The arm C cost of 20.4 (1.0 mm) and 4.4 (1.5 mm) CPU node-hours above uses Malik heart 09 timings and is superseded
by 4.2 and 0.90 node-hours from ACDC trio timings (`results/2026-10-28_G2_meshing/README.md`, Correction 2026-10-28).
The "p95 4.7°" away from conflict nodes is withdrawn; 6.9° replaces it (definition of the stored dhelix field to be stated).
