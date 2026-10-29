#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-ka
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# I11 gates, owner ka: known-answer tests (a) SAX ellipsoid, (b) fibres vs ldrb, (c) thick shell, (d) 1.0 mm timing.
# MODE=test (qos test, reduced) or MODE=full. Usage: sbatch [--qos=test --time=00:30:00] --export=MODE=test ka.sbatch
set -uo pipefail
MODE=${MODE:-full}
ROOT=/project/home/p201509/work/cvri/simulation
cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
PYL="/project/home/p201509/envs/ka-ldrb/bin/python -I"
K=code/gates/ka
R=results/2026-10-28_known_answer/$MODE; mkdir -p $R
W=/dev/shm/ka_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
date -Is; hostname
if [ "$MODE" = test ]; then
  A_ARGS="--vox 1.5 --spacing 10 --inplane 1.25 --noff 2 --truth-h 0.5 --workers 2"
  C_ARGS="--h 1.5 --workers 6"
  FIBMESH="ellipsoid"
  NSOLO=1; PACK=8
else
  A_ARGS="--vox 1.5 1.0 --spacing 10 5 8 --inplane 1.25 1.6 --noff 8 --truth-h 0.2 --workers 24"
  C_ARGS="--h 1.5 1.0 0.75 0.5 0.375 --workers 30"
  FIBMESH="ellipsoid heart09"
  NSOLO=3; PACK=90
fi
# (a), (c), (b) concurrently (about 24 + 24 + 2 cores)
$PY $K/ellipsoid_sax.py --out $R/sax $A_ARGS > $R/sax.log 2>&1 &
$PY $K/thick_shell.py $W/shell $R/thick_shell.csv $C_ARGS > $R/shell.log 2>&1 &
for m in $FIBMESH; do $PYL $K/fibres_compare.py $m $R/fibres $W/fib > $R/fibres_$m.log 2>&1 & done
# (d) build heart 09 reference construction at 1.0 and 1.5 mm with the verbatim pilot route (K = 0)
for v in 1.0 1.5; do
  mkdir -p $W/t$v/cons $W/t$v/mesh
  $PY code/pilot/I11/06_constructions.py data/processed/pilot_I13/heart09.h5 data/processed/pilot_I13/malik09.npz \
      $W/t$v/cons $W/t$v/mesh 0 11009 $v > $R/build_$v.log 2>&1 &
done
wait
date -Is; echo "a,b,c and builds done"
# (d) timing on an otherwise idle node: solo runs (1 worker, 1 thread), then packed runs
for v in 1.0 1.5; do
  M=$W/t$v/mesh/h09_c0ref
  $PY code/pilot/I13/03_design_runs.py $M ${M}_stim.vtx $R/timing_solo_$v.h5 13013 --workers 1 --threads 1 --limit $NSOLO --tmp $W > $R/timing_solo_$v.log 2>&1
  date -Is
done
for v in 1.0 1.5; do
  M=$W/t$v/mesh/h09_c0ref
  $PY code/pilot/I13/03_design_runs.py $M ${M}_stim.vtx $R/timing_pack_$v.h5 13013 --workers $PACK --threads 1 --limit $PACK --tmp $W > $R/timing_pack_$v.log 2>&1
  date -Is
done
for v in 1.0 1.5; do head -c 300 $W/t$v/mesh/h09_c0ref.pts | head -1 > $R/mesh_nodes_$v.txt; head -1 $W/t$v/mesh/h09_c0ref.elem >> $R/mesh_nodes_$v.txt; cp $W/t$v/cons/cons09.json $R/cons09_$v.json; done
rm -rf $W
date -Is; echo done
