# G1 segmentation gate (seg owner, I11 gates run), 2026-10-28

Project root /project/home/p201509/work/cvri/simulation. Gate G1 of final/STUDY_PLAN.md section 7 (line not moved).

## Verdict

**G1 PASS.** 5 of 5 networks gave usable LV cavity, LV myocardium and RV cavity masks on all 3 gate hearts without
retraining (pre-registered line: at least 3). Gate trio from Info.cfg: patient101 (DCM), patient102 (NOR),
patient104 (HCM). patient103 is MINF, so 104 is the nearest HCM.

Pre-declared "usable" rule (written in notes/scratch/2026-10-28-seg.md before any output existed): structure present;
largest 3D 26-connected component kept; Dice of that component against the expert label > 0.70. All 9
heart-structure cells must pass for a network to be usable.

## Gate trio, ED frame (Dice of largest component / HD95 mm), from metrics_gate_trio.tsv (job 5392371 for nnU-Net, 5392273 for CineMA and ukbb_cardiac)

| network | 101 DCM LV / MYO / RV | 102 NOR LV / MYO / RV | 104 HCM LV / MYO / RV |
|---|---|---|---|
| nnunet_task027_acdc | 0.981 / 0.919 / 0.971 | 0.976 / 0.921 / 0.964 | 0.968 / 0.916 / 0.951 |
| nnunet_task114_mnms | 0.967 / 0.879 / 0.939 | 0.950 / 0.879 / 0.931 | 0.944 / 0.871 / 0.821 |
| nnunet_ukb_sax | 0.964 / 0.842 / 0.915 | 0.952 / 0.857 / 0.894 | 0.955 / 0.870 / 0.919 |
| cinema_acdc_sax | 0.975 / 0.901 / 0.964 | 0.970 / 0.920 / 0.960 | 0.963 / 0.910 / 0.903 |
| ukbb_cardiac_fcn_sa | 0.963 / 0.797 / 0.918 | 0.946 / 0.837 / 0.894 | 0.874 / 0.727 / 0.760 |

HD95 on the trio ranged from 1.4 to 11.7 mm (full values in the TSV). HD95 is quantised by the grid: 1.4 to 1.9 mm is
one in-plane voxel and 10 mm is one slice, so it mostly counts whether any surface is off by a whole slice.
The 50-subject rerun (job 5392390) reproduced the trio metrics exactly.

## All 50 test subjects, ED (reported, not gated), from metrics_test50.tsv and summary_test50.md

Median with 2.5th to 97.5th percentile over 50 hearts (largest-component Dice); "usable" uses the same rule as the gate.

| network | usable hearts | LV Dice | MYO Dice | RV Dice |
|---|---|---|---|---|
| nnunet_task027_acdc | 50/50 | 0.972 (0.909 to 0.985) | 0.910 (0.856 to 0.930) | 0.964 (0.868 to 0.975) |
| nnunet_task114_mnms | 50/50 | 0.955 (0.888 to 0.972) | 0.866 (0.813 to 0.898) | 0.941 (0.819 to 0.959) |
| nnunet_ukb_sax | 48/50 | 0.959 (0.830 to 0.972) | 0.842 (0.706 to 0.902) | 0.922 (0.791 to 0.959) |
| cinema_acdc_sax | 49/50 | 0.965 (0.895 to 0.981) | 0.896 (0.727 to 0.926) | 0.952 (0.858 to 0.973) |
| ukbb_cardiac_fcn_sa | 36/50 | 0.941 (0.425 to 0.968) | 0.768 (0.398 to 0.869) | 0.875 (0.006 to 0.953) |

Failures: nnunet_ukb_sax 133 (MYO 0.682) and 145 (LV 0.355, MYO 0.388, RV 0.100); cinema 129 (MYO 0.671);
ukbb_cardiac 14 hearts (107, 109, 113, 114, 118, 120, 121, 124, 125, 130, 131, 133, 136, 147), mostly RV near zero.
Downstream arms should treat ukbb_cardiac as a weak construction (or drop it) and must decide how to handle per-heart
failures; this owner does not make that call.

## Methods

- Inputs: ACDC ED frame (Info.cfg ED) of patient101 to 150, data/raw/acdc (see its README; CC BY-NC-SA 4.0).
- nnunet_task027_acdc: nnU-Net 1.7.1, ensemble of 2d + 3d_fullres (5 folds each), ensemble postprocessing.json
  shipped with the weights; default mirroring TTA.
- nnunet_task114_mnms: nnU-Net 1.7.1, nnUNetTrainerV2_MMS, ensemble 2d + 3d_fullres (5 folds each); no ensemble
  postprocessing file is shipped, none applied.
- nnunet_ukb_sax: nnU-Net v2 2.6.2, Dataset100_UKBB_Petersen_SAX, 2d, folds 0 to 4, checkpoint_final.
- cinema_acdc_sax: the ground-truth-centred crop is REMOVED. Pass 1: sliding-window inference (MONAI, roi 192 x 192 x
  16, overlap 0.5, gaussian) over the whole resampled field (1 x 1 x 10 mm); pass 2: 192 x 192 crop centred on the
  bounding box of the largest component of the pass-1 LV prediction (CineMA's own crop rule, predicted LV in place of
  the expert LV). Fallback to the image centre was needed for 0 of 50 hearts (crop_log.json). Seeds 0, 1, 2 ensembled by
  mean softmax; probabilities resampled linearly to the original grid, then argmax.
- ukbb_cardiac_fcn_sa: TF1 checkpoint run via tf.compat.v1 in TensorFlow-CPU 2.17.1 (the TF1 stack does not support
  A100 CUDA); preprocessing copied from deploy_network.py (1st to 99th percentile rescale, pad to multiples of 16),
  native in-plane resolution, no reorientation.
- All outputs mapped to the ACDC convention 0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity and saved as
  uint8 NIfTI with the original image affine and header (250 of 250 checked: shape and affine match the expert label).
- Metrics: code/gates/seg/seglib.py (Dice; symmetric HD95 of surface voxels with anisotropic spacing; largest
  component with 26-connectivity). Known-answer tests: code/gates/seg/test_seglib.py (5 pass, including expert label
  against itself: Dice 1, HD95 0). One Dice value recomputed independently with nibabel (patient145, Task027 and UKB):
  consistent.
- Stochasticity: inference only; PYTHONHASHSEED=0, torch.manual_seed(0) for CineMA. Two runs of the trio gave identical
  metrics.

## Jobs (account p201509, partition gpu, 1 node, all charged whole-node)

| job | qos | purpose | elapsed | GPU node-hours |
|---|---|---|---|---|
| 5392273 | test | 5 networks, trio; nnU-Net v1 failed (read-only weight dirs), CineMA and ukbb_cardiac ran | 00:25:31 | 0.43 |
| 5392337 | test | nnU-Net serial, trio; TIMEOUT (30 min) | 00:30:21 | 0.51 |
| 5392371 | test | nnU-Net parallel on 4 GPUs, trio; G1 evaluation | 00:13:17 | 0.22 |
| 5392390 | default | all 5 networks, 50 subjects, metrics, MANIFEST.tsv | 00:47:39 | 0.79 |
| total | | | | 1.95 GPU, 0 CPU (cap 6 GPU, 1 CPU) |

Logs: logs/cvri-seg-<job>.out; nnU-Net work dirs (minus .npz) tarred as work_<job>.tar.gz.

## Software

- Env /project/home/p201509/envs/cvri-seg (micromamba, conda-forge python 3.11.18): torch 2.5.1+cu121, torchvision
  0.20.1+cu121, nnunet 1.7.1, nnunetv2 2.6.2, cinema 1.0.0 (github.com/mathpluscode/CineMA commit c10daa1d, installed
  --no-deps), monai 1.5.2, numpy 1.26.4, scipy 1.17.1, pandas 2.2.3, SimpleITK 2.5.0, nibabel 5.4.2, timm 1.0.15,
  safetensors 0.5.3. Full freeze: code/gates/seg/requirements-cvri-seg.txt. 48,906 files.
- Env /project/home/p201509/envs/cvri-seg-tf (python 3.11): tensorflow-cpu 2.17.1, numpy 1.26.4, nibabel 5.3.2.
  Freeze: code/gates/seg/requirements-cvri-seg-tf.txt. 28,007 files.
- ukbb_cardiac source github.com/baiwenjia/ukbb_cardiac commit f465832c. GPU driver 610.57.04, A100-SXM4-40GB.
- CineMA declares exact pins (numpy 1.26.4, scipy 1.15.2, scikit-learn 1.6.1 and others) that differ from the env;
  inference ran without error, but the pins were not honoured.

## Scripts

code/gates/seg/: seg_job.sh (Slurm), run_nnunet.py, run_cinema.py, run_ukbb_cardiac.py, evaluate.py, seglib.py,
summarise.py, test_seglib.py. Command for the 50-subject run:
`sbatch --qos=default --time=02:00:00 code/gates/seg/seg_job.sh "101-150" test50 manifest`.
