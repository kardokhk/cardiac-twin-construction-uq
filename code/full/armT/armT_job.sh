#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-armT
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_armT/logs/%x-%j.out
# I11 arm T (owner armT). MODE=test: K3 + references 09, 17 only (sbatch --qos=test --time=00:30:00 --export=ALL,MODE=test).
# MODE=full: 5 KCL conversions, 22 hearts.
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
K=code/full/armT; OUT=results/2026-10-28_armT; D=$OUT/data
W=/dev/shm/armT_$SLURM_JOB_ID; mkdir -p $W/logs; export TMPDIR=$W
date -Is; hostname; echo "MODE=$MODE"
$PY $K/test_armt.py || exit 1
if [ "$MODE" = test ]; then KS="3"; HS="09 17 K3"; else KS="1 2 3 4 5"; HS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 K1 K2 K3 K4 K5"; fi
if [ "${SKIP_PREP:-0}" != 1 ]; then
echo "== convert KCL"; date -Is
for k in $KS; do
  $PY $K/t01_convert_kcl.py data/raw/kcl_torso/KCL_torso$k.zip $D/kcl$k.npz $D/kcl${k}_convert.json > $W/logs/convert_K$k.log 2>&1 &
done; wait
tail -n 2 $W/logs/convert_*.log
echo "== prep"; date -Is
for h in $HS; do $PY $K/t02_prep.py $h > $W/logs/prep_$h.log 2>&1 & done; wait
grep -h -E "done|Error|error|fast layer|pilot c0ref" $W/logs/prep_*.log
fi
echo "== register"; date -Is
for h in $HS; do $PY $K/t03_register.py $h > $W/logs/reg_$h.log 2>&1 & done; wait
grep -h -E "heart|Error|error" $W/logs/reg_*.log
echo "== runs and calibration"; date -Is
$PY $K/t04_runs.py --workers 126 --hearts $HS 2>&1 | tee $W/logs/runs.log | grep -v "phase 1 [0-9]"
date -Is
tar czf $OUT/logs/logs_${MODE}_$SLURM_JOB_ID.tar.gz -C $W logs
rm -rf $W
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
