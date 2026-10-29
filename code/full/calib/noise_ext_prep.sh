#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-noiseext-prep
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%j.out
# Owner noise-ext, stage D1: displacement draws (SD 30, 40 mm; sd_index 3, 4) for 50 hearts into the shared scratch S.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh
S=data/processed/acdc_mesh_v2/_scratch_disp_ext; rm -rf $S; mkdir -p $S/logs
date -Is; hostname; sha256sum $M/displace.py $M/displace_ext.py code/full/calib/*noise_ext* code/full/calib/calib_lib.py code/full/calib/run_armC.py code/full/calib/csv_export.py > results/2026-10-28_noise_ext/code_sha256.txt
$PY $M/displace_ext.py prep $S $(seq -s, 101 150)
date -Is; echo "prep finished"
