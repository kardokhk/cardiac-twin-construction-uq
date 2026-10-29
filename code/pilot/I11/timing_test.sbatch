#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=pilotI11I13-timing-test
#SBATCH --output=logs/%x-%j.out
# Eikonal timing on Malik heart 01 (3.26 M tets), then ECG KA-E3/E4 and coarse Niederer (pilots I11/I13)
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation
PY="/project/home/p201509/envs/pilot-I13/bin/python -I"
OUT=$ROOT/results/2026-10-22_pilot_I11/stageA; mkdir -p $OUT
W=/dev/shm/pilotT_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
cd $ROOT
M=data/processed/pilot_I13/mesh/heart01
for t in 8 32; do
  date -Is; timeout 600 $PY code/pilot/I13/03_design_runs.py $M ${M}_stim.vtx $W/t1_$t.h5 13013 --workers 1 --threads $t --limit 1 >> $OUT/timing01.log 2>&1
done
date -Is
timeout 900 $PY code/pilot/I13/03_design_runs.py $M ${M}_stim.vtx $W/t16.h5 13013 --workers 16 --threads 8 --limit 16 >> $OUT/timing01.log 2>&1
date -Is
cp $W/*.h5 $OUT/ 2>/dev/null
timeout 300 $PY code/pilot/I11/stageA_ecg.py $W/ecg $OUT/stageA_ecg.json > $OUT/stageA_ecg.log 2>&1 &
timeout 300 $PY code/pilot/I11/stageA_niederer.py $W/nied $OUT/stageA_niederer_coarse.json "0.5:50:8,0.2:10:32" > $OUT/stageA_niederer_coarse.log 2>&1 &
wait
date -Is
rm -rf $W
