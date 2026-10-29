#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-analysis
#SBATCH --output=results/2026-10-28_analysis/logs/%x-%j.out
# Usage (from project root): sbatch --qos=test --time=00:30:00 code/full/analysis/analysis_job.sh <nboot> <outdir> <figdir>
set -euo pipefail
NBOOT=${1:-2000}; OUT=${2:-results/2026-10-28_analysis}; FIG=${3:-figures/drafts}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=16
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache MPLCONFIGDIR=/project/home/p201509/cache/mpl
PY=/project/home/p201509/envs/cvri-sim/bin/python
date; hostname
T1=$(date +%s); $PY code/full/analysis/run_analysis.py --nboot "$NBOOT" --out "$OUT" --figs "$FIG"
echo "run_analysis wall $(( $(date +%s) - T1 )) s"; sha256sum code/full/analysis/*.py code/full/analysis/*.sh > "$OUT/code_sha256.txt"
date
