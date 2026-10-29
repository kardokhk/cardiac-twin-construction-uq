#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-ka-fix
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# I11 gates, owner ka, follow-up 2026-10-28: (1) thick shell rerun with the fixed geod (all h, 3 families, 2 solvers);
# (2) boundary conflict audit for the fibre comparison. Deterministic: no random draws; OMP_NUM_THREADS=1.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation
cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
PYL="/project/home/p201509/envs/ka-ldrb/bin/python -I"
K=code/gates/ka
R=results/2026-10-28_known_answer/full
W=/dev/shm/kafix_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
date -Is; hostname
$PY $K/thick_shell.py $W/shell $R/thick_shell_fixed.csv --h 1.5 1.0 0.75 0.5 0.375 --workers 30 > $R/shell_fixed.log 2>&1 &
$PYL $K/conflict_nodes.py $R/fibres $R/fibres/conflict_nodes.json > $R/conflict_nodes.log 2>&1 &
wait
rm -rf $W
date -Is; echo done
