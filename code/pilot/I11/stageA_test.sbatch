#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=pilotI11I13-stageA-test
#SBATCH --output=logs/%x-%j.out
# Stage A known-answer tests (eikonal, ECG) + heart 01 preparation + eikonal timing (pilots I11/I13, 2026-10-22)
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation
PY="/project/home/p201509/envs/pilot-I13/bin/python -I"
OUT=$ROOT/results/2026-10-22_pilot_I11/stageA; mkdir -p $OUT
W=/dev/shm/pilotA_$SLURM_JOB_ID; mkdir -p $W
export TMPDIR=$W
cd $ROOT
date -Is
$PY code/pilot/I11/stageA_niederer.py $W/nied $OUT/stageA_niederer_coarse.json "0.5:50:8,0.2:10:32" > $OUT/stageA_niederer_coarse.log 2>&1 &
P1=$!
OMP_NUM_THREADS=8 $PY code/pilot/I11/stageA_ecg.py $W/ecg $OUT/stageA_ecg.json > $OUT/stageA_ecg.log 2>&1 &
P2=$!
mkdir -p data/processed/pilot_I13/mesh
OMP_NUM_THREADS=16 $PY code/pilot/I13/02_prep_heart.py data/processed/pilot_I13/malik01.npz data/processed/pilot_I13 data/processed/pilot_I13/mesh > $OUT/prep01.log 2>&1
echo "prep exit $?"; date -Is
wait $P1 $P2
date -Is
$PY code/pilot/I13/03_design_runs.py data/processed/pilot_I13/mesh/heart01 data/processed/pilot_I13/mesh/heart01_stim.vtx $W/timing01_w16t8.h5 13013 --workers 16 --threads 8 --limit 16 > $OUT/timing01.log 2>&1
$PY code/pilot/I13/03_design_runs.py data/processed/pilot_I13/mesh/heart01 data/processed/pilot_I13/mesh/heart01_stim.vtx $W/timing01_w32t4.h5 13013 --workers 32 --threads 4 --limit 32 >> $OUT/timing01.log 2>&1
cp $W/timing01_*.h5 $OUT/ 2>/dev/null
date -Is
rm -rf $W
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State
