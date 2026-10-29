#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --array=0-2
#SBATCH --job-name=cvri-noiseext-solve
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%A_%a.out
# Owner noise-ext, stage D2: torso lead-field solves (displace.solve unchanged), workers a*120 .. a*120+119 of 360.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh
S=data/processed/acdc_mesh_v2/_scratch_disp_ext; NW=360; a=$SLURM_ARRAY_TASK_ID
date -Is; hostname
seq $((a*120)) $((a*120+119)) | xargs -P 120 -I{} sh -c "$PY $M/displace_ext.py solve $S {} $NW > $S/logs/solve_{}.log 2>&1"
date -Is; echo "array task $a finished"
