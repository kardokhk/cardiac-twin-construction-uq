#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-noiseext-export
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%j.out
# Owner noise-ext: noise_estimates_ext.csv and owner_check.json from the three result HDF5 files.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; R=results/2026-10-28_noise_ext
date -Is
$PY code/full/calib/export_noise_ext.py $R $R/noise_ext_main_0.h5 $R/noise_ext_main_1.h5 $R/noise_ext_disp.h5
rc=$?; date -Is; echo "done rc=$rc"; exit $rc
