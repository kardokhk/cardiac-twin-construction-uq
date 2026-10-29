#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=pilotI13I11-rule2
#SBATCH --output=logs/%x-%j.out
# Electrode rule v2 (heart- and spine-anchored): recompute own-torso lead fields, construction ECG weights,
# reference-torso weights, KA-3 and all calibrations.  Eikonal design runs are reused unchanged.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
PY="/project/home/p201509/envs/pilot-I13/bin/python -I"
C13=code/pilot/I13; C11=code/pilot/I11
D=data/processed/pilot_I13; R13=results/2026-10-22_pilot_I13; R11=results/2026-10-22_pilot_I11
mkdir -p $R13/calib $R11/calib $R13/logs_v2 $R11/logs_v2
W=/dev/shm/pilotC_$SLURM_JOB_ID; mkdir -p $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HEARTS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17"
date -Is; echo "lead fields v2"
for h in $HEARTS; do echo "$PY $C13/02b_leadfields_v2.py $D/heart$h.h5 $D/malik$h.npz > $R13/logs_v2/lf$h.log 2>&1"; done > $W/a.cmds
xargs -P 17 -I CMD bash -c CMD < $W/a.cmds
date -Is; echo "reweight constructions"
for h in $HEARTS; do echo "$PY $C11/06b_reweight.py $D/heart$h.h5 $D/cons/cons$h.h5 $D/malik$h.npz > $R11/logs_v2/rew$h.log 2>&1"; done > $W/b.cmds
xargs -P 17 -I CMD bash -c CMD < $W/b.cmds
date -Is; echo "reference weights"
read R1 R2 < <($PY $C13/04b_choose_refs.py $D)
echo "refs $R1 $R2"
for h in $HEARTS; do for r in $R1 $R2; do
  echo "$PY $C13/04_ref_weights.py $D $D/cons $D $h $r $D/ref/w_${h}_$r.npz > $R13/logs_v2/ref_${h}_$r.log 2>&1"; done; done > $W/c.cmds
xargs -P 24 -I CMD bash -c CMD < $W/c.cmds
$PY $C13/07_ka3_rigid.py $D/heart01.h5 $D/cons/cons01.h5 $D/malik01.npz $R13/ka3_rigid.json > $R13/logs_v2/ka3.log 2>&1 &
date -Is; echo "calibrations"
for h in $HEARTS; do
  refs=""
  [ "$h" != "$R1" ] && refs="$refs B_R1=$D/ref/w_${h}_$R1.npz"
  [ "$h" != "$R2" ] && refs="$refs B_R2=$D/ref/w_${h}_$R2.npz"
  echo "$PY $C13/05_calibrate.py $D/cons/cons$h.h5 $D/runs/runs_h${h}_c0ref.h5 $R13/calib/calib$h.json --ref $refs > $R13/logs_v2/calib$h.log 2>&1"
  echo "$PY $C11/07_calibrate_cons.py $D/cons/cons$h.h5 $D/runs $h $R11/calib/calib$h.json > $R11/logs_v2/calib$h.log 2>&1"
done > $W/d.cmds
xargs -P 34 -I CMD bash -c CMD < $W/d.cmds
wait
$PY $C13/09_forward_similarity.py $D $R13/forward_similarity.json > $R13/logs_v2/forward.log 2>&1
date -Is; echo done
rm -rf $W
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
