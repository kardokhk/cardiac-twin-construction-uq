#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-armF
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_armF/logs/%x-%j.out
# I11 full run, arm F (model-form comparator), owner armF, 2026-10-28.
# MODE=test: heart 102 only (+ RE dt 12.5 us on it), into test/. MODE=full: the 10 A4 hearts, convergence heart 101
# (RE dt 12.5 us and the 1.0 mm unit). Usage: sbatch --qos=test --time=00:30:00 --export=ALL,MODE=test armF_job.sh
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"
K=code/full/armF
W=/dev/shm/armF_${SLURM_JOB_ID}; mkdir -p $W; export TMPDIR=$W
date -Is; hostname; $PY -c "import os; print('affinity cpus', len(os.sched_getaffinity(0)))"; echo "MODE=$MODE"; sha256sum $K/*.py $K/*.sh
$PY $K/test_armf.py || exit 1
if [ "$MODE" = test ]; then
  OUT=results/2026-10-28_armF/test
  $PY $K/run_armF.py --out $OUT --hearts 102 --workers 124 --conv-heart 102
else
  OUT=results/2026-10-28_armF
  $PY $K/run_armF.py --out $OUT --hearts 101 106 104 105 103 112 102 107 109 119 --workers 124 \
      --conv-heart 101 --conv-mesh 1.0
fi
rc=$?
date -Is; echo "run rc=$rc"
if [ -f $K/analyse_armF.py ]; then $PY $K/analyse_armF.py $OUT > $OUT/analyse_armF.log 2>&1; tail -60 $OUT/analyse_armF.log; fi
rm -rf "$W"
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
