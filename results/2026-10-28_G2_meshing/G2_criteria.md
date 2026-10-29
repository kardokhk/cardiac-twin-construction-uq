# Gate G2 failure criteria (owner mesh), fixed before any ACDC heart was meshed

Written 2026-10-28 by the mesh owner, before the first ACDC construction was built. The SHA-256 of this file is
recorded in `README.md` of this directory together with the job ID of the first meshing job, so the order can be
checked. The pre-registered gate line (STUDY_PLAN section 7, NEXT_30_DAYS week 3) is unchanged: **G2 passes if at
least 35 of the 50 ACDC test hearts mesh without manual fixes** from their expert labels at 1.5 mm.

"Without manual fixes" means: one fixed automatic route, one parameter set for all hearts (rules below), no per-heart
editing of labels, parameters or code, and no reruns with changed settings that are counted toward the gate.

## Route (fixed)

Expert ED labels -> per-class largest 26-connected component in the slice stack -> slices that contain LV myocardium
-> apex end chosen by rule (the stack end whose two outermost LV slices have the smaller mean LV epicardial area)
-> per-slice in-plane signed distance -> linear interpolation between slice centres -> apical cap within half a
slice spacing beyond the most apical slice -> assumed 3 mm RV free wall -> 1.5 mm voxels -> 6 tets per voxel,
largest face-connected myocardial component -> Bayer fibres -> one-voxel fast endocardial layer -> root nodes by
rule -> lead field in Malik heart 09 torso -> one forward eikonal run.

## A heart FAILS G2 if any of the following holds

| id | criterion | threshold |
|---|---|---|
| F1 | Any exception or non-zero exit in reconstruction, meshing, fibres, root placement or lead-field interpolation | any |
| F2 | Fewer than 2 slices contain LV myocardium, or the expert labels contain no RV cavity in any LV slice | any |
| F3 | Disconnected myocardium: the largest face-connected (6-connected) myocardial voxel component holds less than 95% of the myocardial voxels of the reconstruction | < 95% |
| F4 | Mesh too small to be a heart | < 2,000 nodes |
| F5 | Inverted or zero-volume tetrahedra after orientation fix | any |
| F6 | LV myocardial volume of the reconstruction differs from the label volume (class 2 voxel count x voxel volume) by more than 25% | abs > 25% |
| F7 | LV cavity volume of the reconstruction differs from the label volume (class 3) by more than 25% | abs > 25% |
| F8 | Bayer fibres non-finite in more than 0.1% of elements | > 0.1% |
| F9 | Smoke test: openCARP eikonal (theta = 1.55, 0.29 m/s; fibre 0.6 m/s; renumbering 0) fails, or any node is unactivated, or the maximum activation time is outside 30 to 400 ms, or the 8-lead ECG contains non-finite values | any |

G2 is reported two ways: (i) meshing-only (F1 to F8), which is the pre-registered "mesh without manual fixes" line,
and (ii) meshing plus smoke test (F1 to F9). The gate verdict uses (i); (ii) is reported beside it.

## Reported but not a failure criterion

- **Non-manifold boundary edges and vertices** (voxel meshes have them where two voxels share only an edge or a
  corner). They are counted and reported. They are not a failure because the openCARP fast iterative eikonal
  solver and the FEM lead-field kernel need only valid, positively oriented tetrahedra and a connected mesh; no
  surface remeshing step follows. Decided before seeing data.
- Rotation of the ICP refinement relative to the frame-based placement (diagnostic for the torso placement).
- Nodes of the placed heart that fall outside every torso tetrahedron (lead field then extrapolated).

## Threshold rationale

- F6 and F7 at 25%: the ka owner's analytic ellipsoid test of the pilot SAX route (`results/2026-10-28_known_answer/README.md`)
  gave LV myocardium errors down to -12.5% and LV cavity errors down to -17.8% at 10 mm slice spacing from slice
  offset alone, with a perfect segmentation. A 20% line would come close to failing a perfect segmentation, so 25%
  is used. It still catches gross failures (missing slices, swapped classes, wrong apex).
- F3 at 95%: small islands are removed by the largest-component rule; losing more than 5% of the tissue means the
  reconstruction split the heart.
- F9 activation range 30 to 400 ms: the eikonal `tend` is 400 ms; below 30 ms means the stimulus covers most of the
  mesh.
