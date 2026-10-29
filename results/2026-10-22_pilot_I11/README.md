# Pilot I11 (construction variance, substitute constructions) and shared stage A, results directory

Written 2026-10-22. Report: `critique/I11-pilot.md`. Software, data, jobs and
node-hours are listed in `results/2026-10-22_pilot_I13/README.md` (the two pilots share one pipeline and the same
jobs; 2.97 node-hours in total).

## Stage A (`stageA/`)

| file | content | script | job |
|---|---|---|---|
| `stageA_eikonal.json` | KA-1 slab convergence, rotated fibres, scale invariance, renumbering (#553) | `code/pilot/I11/stageA_eikonal.py` (seed 20261022) | 5372255 |
| `stageA_niederer.json` | Niederer 2011 monodomain benchmark, 4 (h, dt) cases | `stageA_niederer.py` | 5372756 (identical in 5372685: `stageA_niederer_job5372685.json`) |
| `stageA_ecg.json` | KA-E1 sphere dipole, KA-E2 reciprocity, KA-E3 openCARP phie_recovery, KA-E4 template | `stageA_ecg.py` (seed 20261022) | 5372756 |
| `stageA_ecg_loginrerun_E3m.json` | KA-E3 repeated with openCARP's conductivity convention (login node, 2 MPI ranks, 26 s) | same | login node |
| `timing01.log`, `t1_8.h5`, `t1_32.h5`, `t16.h5` | eikonal wall time on the native Malik 01 myocardial mesh (3.26 M tets) | `code/pilot/I13/03_design_runs.py` | 5372406 |

## I11 substitute-construction test

- Constructions: `code/pilot/I11/06_constructions.py` (seed 110NN for heart NN), 1.5 mm voxels, c0ref plus six
  SAX-like alternatives per heart; stored in `data/processed/pilot_I13/cons/consNN.h5` (meshes, fibres, lengths,
  ECG weights).
- Eikonal runs: shared with I13 (`data/processed/pilot_I13/runs/`), 119 constructions x 90 runs.
- Calibration: `07_calibrate_cons.py` -> `calib/` (electrode rule v2) and `calib_rule1/`.
- Analysis: `09_analyse.py` (cluster bootstrap over hearts, 2,000 resamples, seed 11011) -> `analysis_I11.json`,
  `analysis_I11_rule1.json`.
- Logs: `logs_all.tar.gz`.
