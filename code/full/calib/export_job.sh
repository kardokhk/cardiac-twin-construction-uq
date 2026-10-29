#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-armC-export
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_armC/logs/%x-%j.out
# CSV export and QC for arms C and N (owner calibC). MODE=test: mini driver run (heart 102, 2 constructions plus the
# reference, nrep 2) into test2/, then export and QC of test/ and test2/. MODE=full: export and QC of the full run.
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 EXPORT_WORKERS=50
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"
K=code/full/calib
W=/dev/shm/calibCx_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
date -Is; hostname; echo "MODE=$MODE"
if [ "$MODE" = test ]; then
  $PY $K/run_armC.py --out results/2026-10-28_armC/test2 --hearts 102 --workers 126 --nrep 2 \
      --cons-filter 'task027_acdc__1.5__rv3.0|cinema_acdc_sax__1.0__rv3.0'
  for D in test test2; do $PY $K/csv_export.py results/2026-10-28_armC/$D; done
  $PY $K/qc_armC.py results/2026-10-28_armC/test | head -60
else
  $PY $K/csv_export.py results/2026-10-28_armC
  $PY $K/qc_armC.py results/2026-10-28_armC > results/2026-10-28_armC/qc_print.txt; head -80 results/2026-10-28_armC/qc_print.txt
fi
rc=$?
rm -rf "$W"
date -Is; echo "done rc=$rc"
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
exit $rc
