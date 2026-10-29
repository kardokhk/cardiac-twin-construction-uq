#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-armT-v2
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_armT/logs/%x-%j.out
# Arm T follow-up 2026-10-28 (owner armT): per-arm refinement rerun (t06) and identity/ICP/axis checks (t07).
# MODE=test: hearts 09 17 K3.  MODE=full: 22 hearts.  Phase-1 runs from job 5393781 are reused, not rerun.
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
K=code/full/armT; OUT=results/2026-10-28_armT
W=/dev/shm/armT_v2_$SLURM_JOB_ID; mkdir -p $W/logs $W/chk; export TMPDIR=$W
date -Is; hostname; echo "MODE=$MODE"
$PY $K/test_armt.py || exit 1
if [ "$MODE" = test ]; then HS="09 17 K3"; KS="K3"; else
  HS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 K1 K2 K3 K4 K5"; KS="K1 K2 K3 K4 K5"; fi
echo "== checks (identity, icp equivalence, KCL axes)"; date -Is
for h in 09 17; do $PY $K/t07_checks.py identity $h $W/chk/identity_$h.json > $W/logs/chk_identity_$h.log 2>&1 & done
for h in $HS; do $PY $K/t07_checks.py icpeq $h $W/chk/icpeq_$h.json > $W/logs/chk_icpeq_$h.log 2>&1 & done
for h in $KS; do $PY $K/t07_checks.py axes $h $W/chk/axes_$h.json > $W/logs/chk_axes_$h.log 2>&1 & done
wait
tail -n 3 $W/logs/chk_*.log
$PY -c "import json,glob,os; json.dump({os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob('$W/chk/*.json'))}, open('$OUT/checks_v2.json','w'), indent=1)"
echo "== per-arm refinement rerun"; date -Is
$PY $K/t06_rerun_perarm.py --workers 126 --hearts $HS 2>&1 | tee $W/logs/runs_v2.log
date -Is
tar czf $OUT/logs/logs_v2_${MODE}_$SLURM_JOB_ID.tar.gz -C $W logs
rm -rf $W
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
