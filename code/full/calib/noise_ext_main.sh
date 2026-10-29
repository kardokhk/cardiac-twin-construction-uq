#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --array=0-1
#SBATCH --job-name=cvri-noiseext-main
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%A_%a.out
# Owner noise-ext: white, lp40 (5 to -15 dB) and wander (40, 80%), 20 replicates, both losses.
# Array task 0: hearts 101-125; task 1: hearts 126-150.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; K=code/full/calib
a=$SLURM_ARRAY_TASK_ID; W=/dev/shm/noiseext_${SLURM_ARRAY_JOB_ID}_$a; mkdir -p $W; export TMPDIR=$W
date -Is; hostname
$PY $K/run_noise_ext.py --out results/2026-10-28_noise_ext/noise_ext_main_$a.h5 --hearts $(seq $((101+25*a)) $((125+25*a))) \
  --workers 126 --comparators white lp40 wander
rc=$?; rm -rf "$W"; date -Is; echo "done rc=$rc"; exit $rc
