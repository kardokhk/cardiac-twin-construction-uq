#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-calib
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_G3G4_ministudy/logs/%x-%j.out
# I11 gates G3/G4 mini-study, owner calib. MODE=test: heart 102 only, 3 noise replicates, into test/.
# MODE=full: hearts 101 102 104, 20 replicates. Usage: sbatch --qos=test --time=00:30:00 --export=ALL,MODE=test calib_job.sh
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
K=code/gates/calib
W=/dev/shm/calib_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
date -Is; hostname; echo "MODE=$MODE"
$PY $K/test_calib.py || exit 1
if [ "$MODE" = test ]; then
  OUT=results/2026-10-28_G3G4_ministudy/test
  $PY $K/run_ministudy.py --out $OUT --hearts 102 --workers 126 --nrep 3
else
  OUT=results/2026-10-28_G3G4_ministudy
  $PY $K/run_ministudy.py --out $OUT --hearts 101 102 104 --workers 126 --nrep 20
fi
date -Is; echo "analysis"
$PY $K/analyse_gates.py $OUT > $OUT/analyse_gates.log 2>&1; tail -40 $OUT/analyse_gates.log
rm -rf $W
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
