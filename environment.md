# Environment

Captured 2026-10-22.

- Machine: MeluXina login node (EuroHPC, LuxProvide), Linux 4.18.0-553 el8_10. Account p201509.
- Login-node system Python 3.6.8 (used only for `code/00_merge_numbers.py`).
- Project environment `cvri-sim`, micromamba (conda-forge), at `/project/home/p201509/envs/cvri-sim`
  (35,702 files). Explicit spec: `/project/home/p201509/env-specs/cvri-sim.explicit.txt`; pip freeze in
  `requirements.txt`.
  - Python 3.11.17, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, statsmodels 0.15.0, SALib 1.6.0, nibabel 5.4.2,
    matplotlib 3.11.2.

## Seeds and runs
Each red-team computation records its own seed and script path in its critique file.

## Added 2026-10-22 for pilots
- openCARP v20.0 (release 2026-09-01), x86_64 AppImage from git.opencarp.org, extracted to
  `/project/home/p201509/opt/opencarp-v20.0/openCARP-v20.0-x86_64_AppImage/squashfs-root` (bundled MPICH, meshtool).
- cvri-sim additions: meshio 5.3.5, vtk 9.7.1, pyvista 0.49.1, h5py 3.16.0 (pip freeze refreshed in requirements.txt).
- carputils: pip build from git.opencarp.org failed on 2026-10-22; not installed.

## Added 2026-10-28 for gates G1 to G4
- cvri-seg (micromamba, `/project/home/p201509/envs/cvri-seg`, 48,906 files): Python 3.11.18, torch 2.5.1+cu121,
  torchvision 0.20.1+cu121, nnunet 1.7.1, nnunetv2 2.6.2, cinema 1.0.0 (git c10daa1d, installed --no-deps; its exact
  numpy/scipy/scikit-learn pins not honoured), monai 1.5.2, numpy 1.26.4, scipy 1.17.1, pandas 2.2.3, SimpleITK 2.5.0,
  nibabel 5.4.2, timm 1.0.15, safetensors 0.5.3. Freeze: `code/gates/seg/requirements-cvri-seg.txt`.
- cvri-seg-tf (`/project/home/p201509/envs/cvri-seg-tf`, 28,007 files): Python 3.11, tensorflow-cpu 2.17.1 running the
  ukbb_cardiac TF1 FCN_sa graph through tf.compat.v1 (native TF1 not attempted; no A100 CUDA support). Deviation from the
  plan's "TF1 if it installs cleanly, otherwise drop". Freeze: `code/gates/seg/requirements-cvri-seg-tf.txt`.
- Cloned sources: ukbb_cardiac f465832c, CineMA c10daa1d (`code/gates/seg/src`). GPU driver 610.57.04.
- ka-ldrb (`/project/home/p201509/envs/ka-ldrb`, 41,197 files): Python 3.12, fenics-dolfinx 0.10.0, basix 0.10.0,
  ufl 2025.2.1, petsc4py 3.25.5, mpi4py 4.1.2, numpy 2.5.4, numba 0.68.0; pip: fenicsx-ldrb 0.2.0, io4dolfinx 1.5.0,
  fenicsx-compat 0.4.0, pyamg 5.3.0, pandas 3.1.0.
- cvri-sim: pyamg 5.3.0 added (pip --no-deps). Bayer fibre construction must run with OMP_NUM_THREADS=1 and a seeded
  pyamg solve; unseeded runs flipped about 0.6% of elements (job 5392475, superseded by 5392520).
- Data: ACDC (training and testing) from the CREATIS Girder primary record, CC BY-NC-SA 4.0, accessed 2026-10-28,
  SHA-256 in `data/raw/acdc/README.md`; five weight sets in `data/raw/weights/README.md`. Both read-only.
- Retraction Watch snapshot: Crossref-distributed CSV, gitlab.com/crossref/retraction-watch-data commit 2c24a2fc,
  data of 2026-10-27.
- Seeds: theta* LHS 20261028 + patient id; noise seeds keyed by (patient, theta* index, SNR, noise type); bootstrap
  seed in `code/gates/calib/calib_lib.py`.
- Three envs were created against the plan's one; p201509 file count rose from 743,349 (74%) to about 912,000 (91%).

## Added 2026-10-29 for Amendment 2 B1

- `/project/home/p201509/envs/cvri-sim-b1-extra`: scikit-image 0.25.2 and lazy_loader 0.4, installed with `pip
  --no-deps` (556 files) added to sys.path by code/full/mesh/endo_layer.py line 132 for the B1 smooth-surface layer (marching cubes). cvri-sim
  itself unchanged. B1 seeds: as arm C (20261028 + patient id for θ*). B4 seeds: default_rng([20261028, patient,
  istar, comparator_index, level_key, rep]) with level_key 1000 + |level| for negative SNR; displacement lead fields
  20261028 + 1000 patient + 100 sd_index + rep, sd_index 3 (30 mm) and 4 (40 mm).

## v03 manuscript analyses (2026-10-29)

- results/2026-10-29_v03_analyses/: code/manuscript/v03_analyses.py (SHA-256 and input hashes in results.json), cvri-sim env, bootstrap 2,000 resamples, seed scheme as the primary analysis (20261028). Tests tests/test_v03_analyses.py (20 pass, plain assert runner; pytest not installed in cvri-sim).
- Figures: code/figures/fig1_pipeline.py, graphical_abstract.py, fig2_v03.py to fig5_v03.py; hashes in figures/manuscript/README.md. Fonts Nimbus Sans (figures), Liberation Sans (graphical abstract).
- Manuscript build: code/manuscript/build_draft.py (suffix -v03), build_table_networks.py, make_supplement_methods_v03.py, build_supplement.py (v03 mode); login-node python3 3.6.8.

## Public release (2026-10-29)

- GitHub kardokhk/cardiac-twin-construction-uq v1.0.0 (commit 46d915d); Zenodo doi:10.5281/zenodo.23770610 (concept 10.5281/zenodo.23770609). Built with code/release/make_release.py; requirements.txt in the release from `pip list --format=freeze` of cvri-sim (Python 3.11.17).
