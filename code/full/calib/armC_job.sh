#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-armC
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_armC/logs/%x-%A_%a.out
# Full arms C and N, owner calibC. One array task = one node = one shard of hearts (shards.json).
# MODE=test: heart 102 only, nrep 3, into results/2026-10-28_armC/test (submit with --qos=test --time=00:30:00 --array=0).
# MODE=full: shard $SLURM_ARRAY_TASK_ID of results/2026-10-28_armC/shards.json, nrep 20.
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"
K=code/full/calib
TAG=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
W=/dev/shm/calibC_$TAG; mkdir -p $W; export TMPDIR=$W     # own scratch per array task
date -Is; hostname; echo "MODE=$MODE TAG=$TAG"
$PY $K/test_calib.py || exit 1
if [ "$MODE" = test ]; then
  OUT=results/2026-10-28_armC/test
  $PY $K/run_armC.py --out $OUT --hearts 102 --workers 126 --nrep 3
  $PY $K/csv_export.py $OUT
else
  OUT=results/2026-10-28_armC
  HEARTS=$($PY -c "import json;print(' '.join(map(str,json.load(open('$OUT/shards.json'))['shards'][$SLURM_ARRAY_TASK_ID])))")
  echo "hearts: $HEARTS"
  $PY $K/run_armC.py --out $OUT --hearts $HEARTS --workers 126 --nrep 20
fi
rc=$?
rm -rf "$W"
date -Is; echo "done rc=$rc"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
exit $rc
