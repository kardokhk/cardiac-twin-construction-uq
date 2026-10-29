# Amendment 1 to the study plan (pre-registered before any full-arm data)

Project root `/project/home/p201509/work/cvri/simulation`. Written 2026-10-28 after gates G1 to G4
(`critique/I11-gates.md`) and before any full arm C, N, F or T run. Decisions U5 to U7 were made by the author on
2026-10-28. Where this file and `final/STUDY_PLAN.md` differ, this file governs. Lines below are not moved after data.

## A1. Construction levels (U5)

- Primary segmentation sources S: expert labels (reference), nnU-Net Task027_ACDC, nnU-Net Task114 M&Ms, nnU-Net UKB SAX,
  CineMA acdc_sax (image-derived crop). Four networks.
- ukbb_cardiac FCN_sa is a declared weak-network sensitivity level, run on every heart where it meshes, reported
  separately, never pooled into the primary estimand.
- A construction enters the primary analysis if its mask passes the G1 usable rule (structure present, largest
  26-connected component, Dice > 0.70) and it passes the G2 mesh criteria. Excluded constructions are counted and listed
  per heart and network. No other exclusion. Fit quality and bound hits are reported beside every estimate; bound
  estimates are censored (primary analysis keeps them at the bound; sensitivity drops them).

## A2. Fast endocardial layer (U6)

- The fast endocardial layer is defined as all myocardial elements whose centroid lies within 1.5 mm of the LV or RV
  endocardial surface, at both mesh resolutions (1.5 mm and 1.0 mm), instead of one voxel. Elements of the trabecular
  cavity side only; the layer is not added to the RV free wall epicardium.
- Fast-layer velocity rule unchanged from the gates run (calibrated v_endo).
- The gates-run one-voxel definition is rerun on the 3 trio hearts as a sensitivity, so the change's effect on the mesh
  share is reported.

## A3. Noise arm (U7)

Primary comparator (unchanged): white Gaussian noise at 10, 15, 20, 25, 30 dB SNR (per lead, signal power over the QRS
window), 20 replicates per level, calibrated on the reference construction's emulator.

Pre-declared secondary comparators, each at the same SNR grid where the definition has an SNR, 20 replicates:

1. Band-limited noise: white noise low-pass filtered at 40 Hz (4th-order zero-phase Butterworth), scaled to the SNR.
2. Baseline wander: per lead, a random quadratic drift across the QRS window, amplitude (peak deviation) 5%, 10% and 20% of
   that lead's QRS peak-to-peak amplitude; coefficients drawn from a standard normal and rescaled.
3. Electrode displacement: precordial electrodes V1 to V6 displaced on the torso surface by an isotropic tangential
   Gaussian offset with SD 5, 10 and 20 mm (limb electrodes fixed); lead field recomputed for each replicate; targets
   recomputed on the reference construction with the displaced lead field, calibrated with the undisplaced emulator.

For each comparator: break-even level (the noise level at which noise-induced SD of θ̂/θ* equals construction SD), with
cluster-bootstrap 95% CI over hearts, reported as "beyond grid" when outside the tested range.

## A4. Other settings fixed for the full run

- Prior: v_endo 0.5 to 4.0 m/s, v_t 0.1 to 0.8 m/s (as in the gates run). θ*: 3 per heart, Latin hypercube in 0.8 to 2.6
  and 0.15 to 0.45 m/s, seed 20261028 + patient id.
- Fibres: Bayer rule, seeded pyamg, OMP_NUM_THREADS=1.
- Primary loss: correlation (1 − mean Pearson r). Secondary: raw-amplitude relative L2.
- Primary estimand, model and bootstrap as in `final/STUDY_PLAN.md` section 4 (median over hearts of
  between-construction CV of θ̂/θ*, cluster bootstrap over hearts, 2,000 resamples, seed 20261028).
- Root-shift flag: a construction whose mapped root node lies more than 10 mm from the reference rule position is
  flagged; denominator = non-reference constructions of the same resolution.
- Arm F on 10 ACDC hearts (2 per group, lowest patient ids passing G2), 3 θ* each, both losses.
- S7 stays a reported FAIL at 1.0 mm (2.71%); resolution-dependent eikonal lag is reported as part of the mesh factor.
