#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=pilotI13-transonly
#SBATCH --output=logs/%x-%j.out
# I13 decomposition arm: heart translated into the reference torso keeping its own orientation (electrode rule v2).
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
PY="/project/home/p201509/envs/pilot-I13/bin/python -I"
C13=code/pilot/I13; D=data/processed/pilot_I13; R13=results/2026-10-22_pilot_I13
mkdir -p $D/ref_trans $R13/calib_trans $R13/logs_trans
W=/dev/shm/pilotT_$SLURM_JOB_ID; mkdir -p $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TRANSLATION_ONLY=1
HEARTS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17"; R1=09; R2=17
for h in $HEARTS; do for r in $R1 $R2; do [ "$h" != "$r" ] && \
  echo "$PY $C13/04_ref_weights.py $D $D/cons $D $h $r $D/ref_trans/w_${h}_$r.npz > $R13/logs_trans/ref_${h}_$r.log 2>&1"; done; done > $W/c.cmds
xargs -P 32 -I CMD bash -c CMD < $W/c.cmds
for h in $HEARTS; do
  refs=""
  [ "$h" != "$R1" ] && refs="$refs C_R1=$D/ref_trans/w_${h}_$R1.npz"
  [ "$h" != "$R2" ] && refs="$refs C_R2=$D/ref_trans/w_${h}_$R2.npz"
  echo "$PY $C13/05_calibrate.py $D/cons/cons$h.h5 $D/runs/runs_h${h}_c0ref.h5 $R13/calib_trans/calib$h.json --ref $refs > $R13/logs_trans/calib$h.log 2>&1"
done > $W/d.cmds
xargs -P 17 -I CMD bash -c CMD < $W/d.cmds
rm -rf $W; date -Is
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State
