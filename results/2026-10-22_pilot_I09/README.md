# I09 pilot (rung 6), 2026-10-22

Decisive test: is eikonal total activation time (TAT) on KIT biventricular SSM meshes reached by a linear rule in
(LVEDV, LV mass, RVEDV)? Report: `critique/I09-pilot.md`.

## Files

| file | content | made by |
|---|---|---|
| `weights.csv`, `cases.txt` | 300 x 100 mode weights; rows 0-199 N(0, I) (numpy `default_rng(20261022)`), rows 200-299 deposited KIT `instances_weights.csv.gz` | `code/pilot/I09/03_make_cases.py` |
| `I09_pilot_table.h5` (`/cases`), `I09_pilot_table.csv` | one row per mesh: phenotypes, TAT, AT10-90, mean AT (primary and secondary pipelines), inverted tets, Laplace iterations, timings, fibre QC, weights | `02_case.py` per mesh, gathered by `04_decide.py` |
| `decision.txt`, `decision.json`, `decide_stdout.txt` | pre-registered criteria, bootstrap CIs (2,000 resamples, seed 20261022) | `04_decide.py`, rerun on the login node 2026-10-22 14:12 after job 5372272 (identical inputs) |
| `known_answer_solver.csv` | slab and thick-shell eikonal tests | `00_known_answer.py`, job 5372177 |
| `test_gate_outputs.tar.gz` | gate-job JSON rows and logs, mean-shape activation maps (x1.0, x1.2 scale, 0.8 CV) | job 5372177 (`test.sbatch`) |
| `rows_json.tar.gz`, `main_case_logs_5372272.tar.gz` | raw per-case JSON and per-case logs | job 5372272 (`main.sbatch`) |

## Software and data

- openCARP v20.0 AppImage (git hash ffce19513ca9), pure eikonal (`dream.solve = 0`, FIM tol 1e-4 ms), meshtool from the same bundle.
- Python venv `/project/home/p201509/envs/pilot-I09` (system-site-packages over `cvri-sim`, Python 3.11.17,
  numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pyvista 0.49.1) plus pyamg 5.3.0 (installed 2026-10-22).
- KIT SSM, Zenodo 4506463 (CC BY 4.0): `heart_vol.vtu`, `pc_vol.mat`, `var.csv.gz`, `instances_weights.csv.gz`
  from `data/raw/redteam/I10/` (md5 matched the record), `heart_sur.vtp` downloaded 2026-10-22 to
  `data/raw/pilot/I09/` (md5 11069c64928db9faf60d34ffe916b73d, matches record). Derived template files in
  `data/processed/I09/` (`01_template.py`).

## Jobs (account p201509, partition cpu, 1 node each)

| job | purpose | elapsed | node-h |
|---|---|---|---|
| 5372177 | gate, qos test | 00:10:42 | 0.18 |
| 5372272 | main, qos short, 80 cases concurrently | 02:07:45 | 2.13 |
| 5372698 | dependent resume job, cancelled as not needed | 00:00:14 | 0.004 |
| total | | | 2.31 |
