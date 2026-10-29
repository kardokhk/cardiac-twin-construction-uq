#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --array=0-2
#SBATCH --job-name=cvri-mesh2-solve
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%A_%a.out
# mesh2 full run, stage 2: torso lead-field solves for the distinct displaced electrode nodes; array task a runs
# workers a*120 .. a*120+119 of 360 (one solve per core, OMP_NUM_THREADS=1). Outputs Z_<worker>.npy in S.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh
R=results/2026-10-28_mesh_v2/full; S=data/processed/acdc_mesh_v2/_scratch_disp; NW=360; a=$SLURM_ARRAY_TASK_ID
date -Is; hostname
seq $((a*120)) $((a*120+119)) | xargs -P 120 -I{} sh -c "$PY $M/displace.py solve $S {} $NW > $R/logs/disp_solve_{}.log 2>&1"
date -Is; echo "array task $a finished"
