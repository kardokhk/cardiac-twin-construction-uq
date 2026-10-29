#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=pilotI13I11-main
#SBATCH --output=logs/%x-%j.out
# Pilots I13 (reference-torso bias) and I11 (construction variance, substitute constructions), 2026-10-22.
# Phases are idempotent (skip when outputs exist). Seeds: design 13013, constructions 11000+heart.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
PY="/project/home/p201509/envs/pilot-I13/bin/python -I"
C13=code/pilot/I13; C11=code/pilot/I11
D=data/processed/pilot_I13; Z=data/raw/pilot/I13/malik
R13=results/2026-10-22_pilot_I13; R11=results/2026-10-22_pilot_I11
mkdir -p $D/cons $D/consmesh $D/runs $D/ref $R13/logs $R11/logs $R13/calib $R11/calib $R11/stageA
W=/dev/shm/pilotB_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
HEARTS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17"
date -Is; echo "phase 0: stage A extras (background)"
$PY $C11/stageA_ecg.py $W/ecg $R11/stageA/stageA_ecg.json > $R11/stageA/stageA_ecg.log 2>&1 &
$PY $C11/stageA_niederer.py $W/nied $R11/stageA/stageA_niederer.json "0.5:50:4,0.2:10:8,0.1:10:16,0.1:5:16" > $R11/stageA/stageA_niederer.log 2>&1 &
date -Is; echo "phase 1: convert"
for h in $HEARTS; do [ -f $D/malik$h.npz ] || echo $h; done | xargs -P 16 -I{} sh -c "$PY $C13/01_convert.py $Z/{}.zip $D > $R13/logs/convert{}.log 2>&1"
date -Is; echo "phase 2: prep (lead fields)"
for h in $HEARTS; do [ -f $D/heart$h.h5 ] || echo $h; done | xargs -P 16 -I{} sh -c "$PY $C13/02_prep_heart.py $D/malik{}.npz $D - > $R13/logs/prep{}.log 2>&1"
date -Is; echo "phase 3: constructions (K=6, 1.5 mm voxels)"
for h in $HEARTS; do [ -f $D/cons/cons$h.json ] || echo $h; done | xargs -P 17 -I{} sh -c "$PY $C11/06_constructions.py $D/heart{}.h5 $D/malik{}.npz $D/cons $D/consmesh 6 110{} 1.5 > $R11/logs/cons{}.log 2>&1"
date -Is; echo "phase 4: eikonal design runs"
rm -f $W/list.txt
for h in $HEARTS; do for c in c0ref c1 c2 c3 c4 c5 c6; do
  echo "$D/consmesh/h${h}_$c $D/consmesh/h${h}_${c}_stim.vtx $D/runs/runs_h${h}_$c.h5 13013" >> $W/list.txt; done; done
$PY $C13/03b_design_all.py $W/list.txt --workers 124 > $R13/logs/design_all.log 2>&1
date -Is; echo "phase 5: reference torsos, KA-3"
read R1 R2 < <($PY $C13/04b_choose_refs.py $D)
echo "refs $R1 $R2"
rm -f $W/ref.cmds
for h in $HEARTS; do for r in $R1 $R2; do [ -f $D/ref/w_${h}_$r.npz ] || \
  echo "$PY $C13/04_ref_weights.py $D $D/cons $D $h $r $D/ref/w_${h}_$r.npz > $R13/logs/ref_${h}_$r.log 2>&1" >> $W/ref.cmds; done; done
[ -f $W/ref.cmds ] && xargs -P 24 -I CMD bash -c CMD < $W/ref.cmds
$PY $C13/07_ka3_rigid.py $D/heart01.h5 $D/cons/cons01.h5 $D/malik01.npz $R13/ka3_rigid.json > $R13/logs/ka3.log 2>&1 &
date -Is; echo "phase 6: calibrations"
rm -f $W/cal.cmds
for h in $HEARTS; do
  refs=""
  [ "$h" != "$R1" ] && refs="$refs B_R1=$D/ref/w_${h}_$R1.npz"
  [ "$h" != "$R2" ] && refs="$refs B_R2=$D/ref/w_${h}_$R2.npz"
  echo "$PY $C13/05_calibrate.py $D/cons/cons$h.h5 $D/runs/runs_h${h}_c0ref.h5 $R13/calib/calib$h.json --ref $refs > $R13/logs/calib$h.log 2>&1" >> $W/cal.cmds
  echo "$PY $C11/07_calibrate_cons.py $D/cons/cons$h.h5 $D/runs $h $R11/calib/calib$h.json > $R11/logs/calib$h.log 2>&1" >> $W/cal.cmds
done
xargs -P 34 -I CMD bash -c CMD < $W/cal.cmds
wait
date -Is; echo "done"
cp -r $W/nied/*.log $R11/stageA/ 2>/dev/null
rm -rf $W
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
