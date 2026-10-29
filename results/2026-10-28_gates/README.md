# Gates G1 to G4, 2026-10-28: index

Project root `/project/home/p201509/work/cvri/simulation`. Verdicts and every number: `critique/I11-gates.md`.
Job IDs and charges: `notes/PROGRESS.md`, section "Jobs (gates run)". Versions: `environment.md` (2026-10-28 entry).

| gate or test | results directory | regenerating script |
|---|---|---|
| G1 segmentation | `results/2026-10-28_G1_segmentation/` (labels in `data/processed/acdc_seg/`, MANIFEST.tsv) | `code/gates/seg/` |
| G2 meshing, timings | `results/2026-10-28_G2_meshing/` (meshes in `data/processed/acdc_mesh/`, MANIFEST.tsv) | `code/gates/mesh/` |
| G3, G4 mini-study | `results/2026-10-28_G3G4_ministudy/` (gates.json, gates.md) | `code/gates/calib/analyse_gates.py` (78 s, login node) |
| Known-answer S6, S7, SAX ellipsoid, A1 | `results/2026-10-28_known_answer/` (see its "Correction 2026-10-28") | `code/gates/ka/` |
| Evidence (retraction, occupancy, leakage, licences) | `results/2026-10-28_evidence/` | `code/gates/lit/` |

Seeds: θ* 20261028 + patient id; noise seeds keyed by (patient, θ* index, SNR, type); fibres seeded pyamg with
OMP_NUM_THREADS=1. openCARP v20.0, renumbering=0.

Not rerun end to end. Independently rerun: `analyse_gates.py` (identical output), Dice for all 250
masks, G2 counts, thick-shell errors with an independent reference. Also rerun: G4 R² and the CV preview.
