# Analysis plan (extract)

Dated 2026-10-22, written before any simulation of the 50 ACDC test hearts. Sections 1, 3, 4, 5, 7 and 12 are reproduced verbatim. Sections 2, 6 and 8 to 11 (novelty search, compute budget, publication route, timeline, risks and trade-offs) are internal planning and are omitted. "Arm" letters map to the paper as follows: arm C, construction experiment; arm N, ECG noise; arm F, simplified physics; arm T, generic torso. Cross-references to internal files (critique/, NUMBERS.md, final/EVIDENCE.md) are not part of this release.

SHA-256 of the complete dated plan file: `7eec711e22535665f2b15b4084d3dea5399b03dd1a48e7253b2782921aadb24d`.

## 1. Question

**Working title.** How much of a calibrated conduction velocity is the pipeline? Construction-stage variance in
ECG-calibrated eikonal ventricular twins built from public cine CMR.

**One sentence.** Estimate the share of variance in ECG-calibrated conduction velocities (endocardial, v_endo, and
transmural, v_t) of eikonal biventricular twins that is due to construction choices (segmentation network and mesh
resolution; torso and electrode geometry as a secondary factor), relative to calibration error and to ECG target
noise, in the 50 held-out ACDC cine CMR test subjects and 22 public heart-in-torso models (17 Malik 2026 HCM, 5 KCL
healthy), so that twin-pipeline developers and ASME V&V40 credibility assessors know whether construction
uncertainty must enter a twin's uncertainty budget.

**Why it matters.** Solís-Lemus et al. 2026 write that "explicitly quantifying construction-stage uncertainty in
whole-heart models has not been addressed here and represents an important limitation, particularly in the context
of regulatory credibility frameworks" (E01; "here" refers to their own study). They note that earlier work began
to quantify it for left atrial models (their refs 1 and 25; see section 2).
Population twin studies compare calibrated conduction velocity across groups (Qian et al. 2025, 3,461 UK Biobank
twins; E02). If construction variance is of the order of those group differences, the comparisons need a
construction term.

**Who acts, and the chain.**

| link | content | status |
|---|---|---|
| 1 | Between-construction variance of calibrated velocities with known truth, ACDC anatomies | measured by this study |
| 2 | That variance carries over to real twins calibrated to real ECGs | asserted; bounded by the model-form comparator arm (section 3, arm F) |
| 3 | V&V40-style assessment treats geometry input uncertainty as a required term | documented practice (Galappaththige 2022, E03, abstract only) |
| 4 | Developers change practice (segmentation ensembles, construction intervals) | asserted |

The study measures link 1 only. It reports a lower bound on the construction term, not accuracy against patients.

## 3. Design

### Primary arm C: segmentation and mesh construction (ACDC)

- **Hearts.** The 50 ACDC test subjects, 10 per group (NOR, DCM, HCM, MINF, ARV), end-diastolic frame. The test set
  is used because nnU-Net Task027 and CineMA acdc_sax were trained on the ACDC training split. Licence CC BY-NC-SA
  4.0 (E09, read at the primary record 2026-10-22).
- **Segmentation sources S (up to 6 levels).** Expert manual labels (reference) plus up to five networks with public
  weights: nnU-Net Task027_ACDC, nnU-Net Task114 M&Ms, nnU-Net UKB SAX, CineMA acdc_sax (with its ground-truth-centred
  crop removed), ukbb_cardiac FCN_sa (E10). At least three networks must run (gate G1).
- **Mesh settings M (2 levels).** Voxel-to-tetrahedral meshes at 1.5 mm and 1.0 mm. Short-axis reconstruction as in
  the pilot: signed-distance interpolation between slices, apical cap, right-ventricular free wall of assumed 3 mm
  thickness (sensitivity at 2 and 4 mm on 10 hearts). biv-me is not used: it stops without valve and apex landmarks,
  which ACDC lacks (`critique/I11-redteam.md` rung 4).
- **Model.** openCARP v20.0 eikonal, `renumbering=0` (bug #553 is present in v20.0: activation errors up to 42.5 ms
  with `renumbering=1`, measured in the pilot). Fibres by the Bayer 2012 rule (+40° endocardium, −50° epicardium;
  implementation from the I09 pilot, to be cross-checked against the `ldrb` package). One-voxel fast endocardial
  layer. Fibre velocity fixed at 0.6 m/s. Root nodes fixed by rule and mapped from the reference construction.
- **ECG.** Each heart is registered once (reference construction) into one fixed reference torso, Malik heart 09
  (median torso volume, as in the I13 pilot), with rule-placed electrodes; the same transform is applied to every
  construction of that heart, so torso is constant within a heart. Eight independent leads, QRS window, homogeneous
  lead field (kernel verified against openCARP phie_recovery, r ≥ 0.9999998; I13 pilot section 3).
- **Calibrated parameters.** θ = (v_endo, v_t). The fibre velocity and anisotropy ratio are dropped: Camps 2021
  recovered fibre and sheet-normal speeds with absolute errors of 28 to 34 cm/s against true values of 29 to 88 cm/s
  (RT-I11-09).
- **Truth and targets.** Three θ* draws per heart (Latin hypercube inside the prior), targets simulated on the
  reference construction. Every construction is calibrated to the same targets.
- **Calibration.** Cubic-spline emulator on a 9 × 9 log grid plus up to 9 refinement runs per construction (the
  pilot design). Two losses: correlation (1 − mean Pearson r; primary, declared now) and raw-amplitude relative L2.
  The prior is widened beyond the pilot's (v_t 0.15 to 0.5 m/s), because 12 of 102 pilot constructions had an
  estimate at a bound (`critique/I11-pilot.md` section 2); estimates at a bound are reported and treated as censored.

### Arm N: target noise (no new forward runs)

Gaussian noise added to each target at signal-to-noise ratios of 10, 15, 20, 25 and 30 dB, 20 replicates each,
calibrated on the reference construction's emulator. This arm gives the calibration spread due to noise as a
function of SNR. The comparison with arm C is expressed as the **break-even SNR**: the SNR at which noise-induced
variance equals construction-induced variance. This avoids having to defend one assumed clinical noise level.

### Arm T: torso and electrode geometry (secondary, from I13)

The 22 heart-in-torso models: 17 Malik 2026 HCM (E07, CC BY 4.0) and 5 KCL healthy (E11, CC BY 4.0). Own torso and
rule-placed electrodes against two reference torsos (Malik 09 and 17). Two registrations: full rigid ICP and
orientation-preserving (translation only). Fit quality (final loss) and bound hits are reported beside every bias.
The pilot found that full registration rotated hearts by a median of 21° and that own-versus-reference lead
correlation was 0.60 (N015 context; `critique/I13-pilot.md`), so the orientation-preserving arm is the primary
estimate for this factor.

### Arm F: model-form comparator (secondary, 10 ACDC hearts)

Targets generated with full transmembrane-voltage ECGs (openCARP reaction-eikonal) instead of the activation-template
ECG, then calibrated with the eikonal emulator. This gives a same-anatomy estimate of model-form bias to set beside
the construction variance. In the pilot, activation-template and full-Vm ECGs agreed at r = 0.990 to 0.997 (I11
pilot section 3). Run time for reaction-eikonal on these meshes is not measured (gate G2).

## 4. Estimands and analysis

1. **Primary.** For each heart and parameter, the between-construction coefficient of variation of θ̂/θ* across the
   S × M constructions. Reported as the median over hearts with a 95% CI from a cluster bootstrap over hearts
   (2,000 resamples, seed recorded).
2. **Variance shares.** Linear mixed model on log(θ̂/θ*): random heart, fixed segmentation source and mesh setting
   and their interaction, θ* draw, residual. Shares with cluster-bootstrap 95% CIs.
3. **Break-even SNR** (arm N versus arm C), with bootstrap 95% CI.
4. **Free-rule benchmark (pre-registered).** Three-length linear rule (scale, wall thickness, base height, all
   mask-readable) for θ̂/θ̂_ref, in-sample and leave-one-heart-out R². In the pilot this was 0.15 to 0.57 (N013). If
   it reaches 0.9 for both parameters on real ACDC constructions, the conclusion is reported as "a morphometric
   correction suffices", which is a smaller paper.
5. **Torso bias** (arm T): mean signed and absolute relative bias, 95% CI over hearts, with fit quality and censoring.
6. **Exploratory.** Dependence on ACDC group (5 × 10 hearts). Any test is reported with an effect size and 95% CI,
   Holm-corrected across the family, and labelled exploratory. Association, not causation, in all cross-heart
   comparisons.

Reporting follows the project rule: effect size with 95% CI for every claim; median with interpercentile range for
non-normal quantities; no SEM.

## 5. Known-answer tier

Already passed in the pilot (jobs listed in `critique/I11-pilot.md`, `critique/I13-pilot.md`):

| test | acceptance | pilot outcome |
|---|---|---|
| Analytic eikonal slab | ≤ 1.0 ms at 0.2 mm, falling with refinement | 0.97, 0.60, 0.50, 0.30 ms at 0.5, 0.25, 0.2, 0.1 mm |
| Scale invariance (s = 0.5, 1.7, 3) | exact | 0.0 ms |
| Niederer 2011 benchmark (E12), monodomain 0.1 mm | within 1 ms of 42.75 ms | 43.56 ms |
| Renumbered and rotated duplicate, `renumbering=0` | ≤ 10⁻⁶ ms | 7.6 × 10⁻⁶ ms |
| Sphere dipole lead field | RMS ≤ 2% | 0.50% to 0.83% |
| Lead-field reciprocity | ≤ 1% | 1.5 × 10⁻¹⁰ |
| Parameter recovery, own construction | ≤ 2% | ≤ 0.05% |

To add in month 1: (a) short-axis reconstruction of an analytic ellipsoidal shell sampled at ACDC slice spacing,
with myocardial volume error reported; (b) Bayer fibre angles against the `ldrb` reference implementation on two
meshes; (c) thick-shell eikonal, which missed the 2% line in the I09 pilot (2.7% to 3.7%), repeated at 1.0 mm.

**Data the model did not see.** None is available: no public dataset pairs ACDC images with ECGs. The paper states
this, frames the result as a known-truth lower bound, and uses arm F as the bridge to model-form error. This is the
main reviewer exposure (section 8).

## 7. Pre-registered gates and stop rules

| gate | when | test | if it fails |
|---|---|---|---|
| G1 | week 2 | At least 3 networks produce usable LV, myocardium and RV masks on 3 ACDC test hearts without retraining | Use M&Ms-2 when it returns, or reduce S to the networks that run and state it |
| G2 | week 3 | Short-axis meshing succeeds for at least 70% of the 50 hearts; reaction-eikonal run time measured | Below 70%: report on the subset and say so. Arm F over 100 node-hours: run on 5 hearts |
| G3 | week 4 | On 3 real ACDC hearts, between-construction SD of θ̂ ≥ 2 × SD of the target-noise arm at 20 dB in at least 2 hearts for at least one parameter | The construction term is smaller than plausible noise: report as a bound (short paper) |
| G4 | week 4 | Three-length rule R² on the 3-heart pilot | ≥ 0.9 for both parameters: reframe as the morphometric-correction paper |

Lines are not moved after seeing data. If a line proves wrong, results are reported against both lines.

## 12. Reproducibility record

Software: openCARP v20.0 AppImage (released 2026-09-01) at `/project/home/p201509/opt/opencarp-v20.0`; Python env
`cvri-sim` (Python 3.11.17; versions in `environment.md`); pilot env `pilot-I13`. Seeds: 20261022 (general), 13013
(I13 θ* draws), 110NN (I11 constructions). Pilot scripts: `code/pilot/I11/`, `code/pilot/I13/`, `code/pilot/I09/`
(fibres). All pilot jobs and charges: `NUMBERS.md` N005 to N007.
