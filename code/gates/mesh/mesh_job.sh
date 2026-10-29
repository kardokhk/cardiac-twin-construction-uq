#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-mesh
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# I11 gates, owner mesh: ACDC constructions, gate G2, eikonal and reaction-eikonal timing.
# MODE=test : torso prep, trio references, trio networks at 1.5 mm, one expert at 1.0 mm, short RE check
# MODE=full : everything (idempotent: existing passed constructions are not rebuilt)
# SKIP_ELLIPSOID=1, SKIP_PACK=1: skip the route ellipsoid check and the 8-copy RE packing check
# Usage: sbatch --qos=test --time=00:30:00 --export=ALL,MODE=test mesh_job.sh
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
M=code/gates/mesh
R=results/2026-10-28_G2_meshing/$MODE; mkdir -p $R/qc $R/logs
D=data/processed/acdc_mesh
W=/dev/shm/mesh_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
NETS="nnunet_task027_acdc nnunet_task114_mnms nnunet_ukb_sax cinema_acdc_sax ukbb_cardiac_fcn_sa"
TRIO="101 102 104"
ALL=$(seq 101 150)
date -Is; hostname; echo "MODE=$MODE"
[ -f $D/torso09_near.npz ] || $PY $M/build.py prep_torso > $R/logs/prep_torso.log 2>&1
date -Is; echo "phase 1: reference constructions (expert, 1.5 mm)"
if [ "$MODE" = test ]; then REFS="$TRIO"; else REFS="$ALL"; fi
for p in $REFS; do echo $p; done | xargs -P 64 -I{} sh -c "$PY $M/build.py ref {} $R/qc > $R/logs/ref_{}.log 2>&1"
grep -h -E "pass|fail" $R/logs/ref_*.log | sort | head -60
date -Is; echo "phase 2: other constructions"
rm -f $W/cons.txt
if [ "$MODE" = test ]; then
  for p in $TRIO; do for n in $NETS; do echo "$p $n 1.5" >> $W/cons.txt; done; done
  echo "101 expert 1.0" >> $W/cons.txt
else
  for p in $TRIO; do for n in $NETS; do echo "$p $n 1.0" >> $W/cons.txt; done; echo "$p expert 1.0" >> $W/cons.txt; done
  for p in $ALL; do for n in $NETS; do echo "$p $n 1.5" >> $W/cons.txt; done; done
fi
# skip constructions already built and passed in an earlier run
while read p n v; do f=$D/cons/patient${p}__${n}__${v}.h5; \
  if [ -f $f ] && $PY -c "import h5py,sys; sys.exit(0 if h5py.File('$f','r').attrs.get('smoke_ok', False) else 1)"; then :; \
  else echo "$p $n $v"; fi; done < $W/cons.txt > $W/todo.txt
wc -l < $W/todo.txt
# 1.0 mm builds first (longest), then 1.5 mm
sort -k3,3r $W/todo.txt | xargs -P 100 -L 1 sh -c "$PY $M/build.py cons \$0 \$1 \$2 $R/qc > $R/logs/cons_\$0_\$1_\$2.log 2>&1"
date -Is; echo "phase 2 done"
if [ "$MODE" = test ]; then
  $PY $M/re_timing.py $D/cons/patient101__expert__1.5.h5 $R/re_test.json 1 8 20 > $R/logs/re_test.log 2>&1
  tail -3 $R/logs/re_test.log
else
  date -Is; echo "phase 3: SAX ellipsoid check of this route (ka script, unchanged)"
  [ "${SKIP_ELLIPSOID:-0}" = 0 ] && $PY code/gates/ka/ellipsoid_sax.py --route $M/sax_route.py --out $R/ellipsoid --vox 1.5 1.0 --spacing 10 5 8 \
      --inplane 1.25 1.6 --noff 8 --truth-h 0.2 --workers 64 > $R/logs/ellipsoid.log 2>&1
  date -Is; echo "phase 4: packed eikonal timing, pilot 90-run design on the trio expert constructions, 128 workers"
  for v in 1.0 1.5; do
    $PY $M/design_runs.py $D/runs 128 $D/cons/patient101__expert__$v.h5 $D/cons/patient102__expert__$v.h5 \
        $D/cons/patient104__expert__$v.h5 > $R/logs/design_$v.log 2>&1
    tail -4 $R/logs/design_$v.log; date -Is
  done
  date -Is; echo "phase 5: reaction-eikonal timing on patient101 expert 1.5 mm (concurrent: RE+ 1, 8, 32 OpenMP threads; RE- 8 threads)"
  F=$D/cons/patient101__expert__1.5.h5
  $PY $M/re_timing.py $F $R/re_plus_np1.json 1 1 50 > $R/logs/re_plus_np1.log 2>&1 &
  $PY $M/re_timing.py $F $R/re_plus_np8.json 1 8 200 > $R/logs/re_plus_np8.log 2>&1 &
  $PY $M/re_timing.py $F $R/re_plus_np32.json 1 32 200 > $R/logs/re_plus_np32.log 2>&1 &
  $PY $M/re_timing.py $F $R/re_minus_np8.json 2 8 200 > $R/logs/re_minus_np8.log 2>&1 &
  wait
  if [ "${SKIP_PACK:-0}" = 0 ]; then
  date -Is; echo "phase 5b: RE+ 8 threads, eight concurrent copies (packing check)"
  for i in 1 2 3 4 5 6 7 8; do $PY $M/re_timing.py $F $R/re_plus_pack8_$i.json 1 8 200 > $R/logs/re_plus_pack8_$i.log 2>&1 & done
  wait
  fi
fi
date -Is; echo "phase 6: manifest"
$PY $M/manifest.py $R > $R/logs/manifest.log 2>&1; tail -30 $R/logs/manifest.log
rm -rf $W
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
