# Pilot I13 (reference-torso bias), results directory

Written 2026-10-22. Report: `critique/I13-pilot.md`. Shared stage A (known-answer
tests, timing) lives in `results/2026-10-22_pilot_I11/stageA/`.

## Software and data

- openCARP v20.0 AppImage, `openCARP -buildinfo` GIT hash ffce19513ca90235192fff261160f53167921877 (tag "latest"),
  binaries under `/project/home/p201509/opt/opencarp-v20.0/.../usr/bin/`; pure eikonal (`dream.solve=0`),
  single-thread runs, `renumbering=0` (default; see bug #553 in the stage A results).
- Python 3.11.17 venv `/project/home/p201509/envs/pilot-I13` (created 2026-10-22 for this pilot): numpy 2.4.6,
  scipy 1.17.1, pandas 3.0.6, h5py 3.16.0, pyamg 5.3.0. Scripts run with `python -I`, OMP/BLAS threads = 1.
- Malik et al. 2026 deposit, Zenodo 18890229 (CC BY 4.0), 17 heart-in-torso archives downloaded 2026-10-22 to
  `data/raw/pilot/I13/malik/`. The deposit contains meshes only (CARP txt + VTK); no simulation set-ups, no tag
  table, no electrode positions. Tags inferred: 22 RV and 23 LV myocardium (the only fibred tags in all 17),
  20 RV and 21 LV blood, 4 vertebral column.

## Pipeline (code in `code/pilot/I13/`, shared helpers in `code/pilot/I11/carpio.py`)

| step | script | output |
|---|---|---|
| convert archive | `01_convert.py` | `data/processed/pilot_I13/malikNN.npz` |
| heart frame, roots, own-torso lead fields | `02_prep_heart.py`, then `02b_leadfields_v2.py` (electrode rule v2) | `heartNN.h5` |
| 1.5 mm voxel reference construction c0ref (+ I11 alternatives) | `code/pilot/I11/06_constructions.py`, `06b_reweight.py` | `cons/consNN.h5` |
| eikonal design (9 x 9 log grid + 3 theta* + 6 held-out), seed 13013 | `03_design_runs.py`, `03b_design_all.py` | `runs/runs_hNN_c0ref.h5` |
| reference torsos (median and 75th-percentile torso volume: hearts 09 and 17) | `04b_choose_refs.py` | `refs.json` |
| rigid registration onto the reference torso's ventricles + reference lead fields | `04_ref_weights.py` | `ref/w_NN_RR.npz` (rule v2), `ref_rule1/`, `ref_trans/` (translation only) |
| calibration (cubic-spline emulator, grid + Nelder-Mead, both losses) | `05_calibrate.py` | `calib/`, `calib_rule1/`, `calib_trans/` |
| analysis | `08_analyse.py` (bootstrap seed 13013, 10,000 resamples) | `analysis_I13.json`, `analysis_I13_rule1.json`, `analysis_I13_translation_only.json` |
| forward ECG similarity own vs reference | `09_forward_similarity.py` | `forward_similarity.json`, `forward_similarity_rule1.json` |
| KA-3 rigid motion | `07_ka3_rigid.py` | `ka3_rigid.json`, `ka3_rigid_rule1.json` |

Logs: `logs_all.tar.gz`. Batch scripts: `code/pilot/I13/main.sbatch`, `rerun_v2.sbatch`, `trans_only.sbatch`.

## Jobs and node-hours (account p201509, partition cpu, 1 node each)

| job | purpose | elapsed | node-h |
|---|---|---|---|
| 5372255 | stage A test (eikonal KA; ECG KA-E1/E2) | 00:02:23 | 0.04 |
| 5372330 | stage A test (heart 01 prep; timing run failed on a formatting bug) | 00:30:06 | 0.50 |
| 5372406 | eikonal timing on native CT mesh | 00:30:25 | 0.51 |
| 5372685 | main, cancelled after electrode-placement failure in 4 torsos (stage A Niederer and ECG completed) | 00:12:37 | 0.21 |
| 5372756 | main: prep, 119 constructions, 10,710 eikonal runs, rule-v1 calibrations | 01:20:01 | 1.33 |
| 5373547 | electrode rule v2: lead fields, weights, calibrations | 00:19:46 | 0.33 |
| 5373685 | translation-only decomposition arm | 00:02:45 | 0.05 |
| **total (I13 and I11 together)** | | | **2.97** |
