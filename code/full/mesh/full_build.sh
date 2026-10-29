#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-mesh2-build
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# mesh2 full run, stage 1: A2 layer unit test, 50 references, 750 other constructions with smoke tests,
# displacement prep for 50 hearts (into the shared scratch S, read by stage 2 on other nodes).
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh
R=results/2026-10-28_mesh_v2/full; mkdir -p $R/qc $R/logs
D=data/processed/acdc_mesh_v2; mkdir -p $D/cons $D/ref $D/displaced
S=$D/_scratch_disp; rm -rf $S; mkdir -p $S                         # shared across stage 2 nodes, removed in stage 3
W=/dev/shm/mesh2_${SLURM_JOB_ID}; mkdir -p $W/build; export TMPDIR=$W/build
rm -f $D/cons/*.h5 $D/ref/*.npz $D/displaced/*.h5                   # no reuse of test-job outputs
date -Is; hostname; sha256sum $M/*.py $M/*.sh > $R/code_sha256.txt
$PY $M/test_endo_layer.py $R/endo_layer_test.json > $R/logs/endo_layer_test.log 2>&1 &
date -Is; echo "phase 1: references"
seq 101 150 | xargs -P 50 -I{} sh -c "$PY $M/build.py ref {} $R/qc > $R/logs/ref_{}.log 2>&1"
cat $R/logs/ref_*.log | awk '{print $2, $3}' | sort | uniq -c
date -Is; echo "phase 2: displacement prep (background) and constructions"
$PY $M/displace.py prep $S $(seq -s, 101 150) > $R/logs/disp_prep.log 2>&1 &
$PY $M/lists.py todo full > $W/todo.txt; wc -l < $W/todo.txt
xargs -P 100 -L 1 sh -c "$PY $M/build.py cons \$0 \$1 \$2 \$3 $R/qc > $R/logs/cons_\$0_\$1_\$2_\$3.log 2>&1" < $W/todo.txt
date -Is; echo "constructions finished"; cat $R/logs/cons_*.log | awk '{print $2, $3}' | sort | uniq -c
wait
tail -6 $R/logs/disp_prep.log; cp $S/prep.json $R/disp_prep.json 2>/dev/null
$PY $M/manifest.py $R > $R/logs/manifest.log 2>&1; tail -40 $R/logs/manifest.log
tail -8 $R/logs/endo_layer_test.log
rm -rf $W; date -Is; echo "stage 1 finished"
