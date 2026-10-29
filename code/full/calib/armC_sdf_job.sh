#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-b1sdf
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_layer_sdf/logs/%x-%A_%a.out
# Amendment 2 B1 (owner layer-sdf). One node per task.
# MODE=unit    : B1 unit test (test_endo_layer_sdf.py) + layer tags for heart 102 into test/  (qos test)
# MODE=ctest   : arm C rerun, heart 102 only, into test/                                         (qos test)
# MODE=layer   : layer tags for all primary constructions -> layer_sdf.h5, layer_share.csv
# MODE=full    : arm C rerun, shard $SLURM_ARRAY_TASK_ID of results/2026-10-28_armC/shards.json
# MODE=export  : estimates_sdf.csv and owner check
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"
K=code/full/calib
O=results/2026-10-28_layer_sdf
TAG=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
W=/dev/shm/b1sdf_$TAG; mkdir -p $W; export TMPDIR=$W     # own scratch per job/task
date -Is; hostname; echo "MODE=$MODE TAG=$TAG"
case $MODE in
  unit)
    mkdir -p $O/test
    $PY code/full/mesh/test_endo_layer_sdf.py $O/unit_test_sdf.json 22
    $PY $K/layer_sdf_tags.py $O/test/layer_sdf_102.h5 $O/test/layer_share_102.csv --hearts 102 --workers 10 ;;
  ctest)
    $PY $K/test_calib.py || exit 1
    $PY $K/run_armC_sdf.py --layer-h5 $O/test/layer_sdf_102.h5 --out $O/test --hearts 102 --workers 126 ;;
  layer)
    $PY $K/layer_sdf_tags.py $O/layer_sdf.h5 $O/layer_share.csv --workers 64 ;;
  full)
    $PY $K/test_calib.py || exit 1
    HEARTS=$($PY -c "import json;print(' '.join(map(str,json.load(open('results/2026-10-28_armC/shards.json'))['shards'][$SLURM_ARRAY_TASK_ID])))")
    echo "hearts: $HEARTS"
    $PY $K/run_armC_sdf.py --layer-h5 $O/layer_sdf.h5 --out $O --hearts $HEARTS --workers 126 ;;
  export)
    $PY $K/export_sdf.py $O ;;
esac
rc=$?
rm -rf "$W"
date -Is; echo "done rc=$rc"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
exit $rc
