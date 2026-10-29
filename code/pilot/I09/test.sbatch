#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-I09-test
#SBATCH --output=logs/%x-%j.out
# I09 pilot gate: known-answer tests + scale/CV invariance on the KIT mean shape + timing of 2 instances.
set -uo pipefail
cd /project/home/p201509/work/cvri/simulation
PY="/project/home/p201509/envs/pilot-I09/bin/python -I"
C=code/pilot/I09
R=results/2026-10-22_pilot_I09
T=$R/test; mkdir -p $T
W=/dev/shm/I09_$SLURM_JOB_ID; mkdir -p $W
WT=$R/weights.csv
date
$PY $C/00_known_answer.py $W/ka $R/known_answer_solver.csv > $T/ka.log 2>&1 &
$PY $C/02_case.py --case mean_s1 --weights zero --out $T/mean_s1.json --work $W --save-at $W/mean_s1.npz --omp 4 > $T/mean_s1.log 2>&1 &
$PY $C/02_case.py --case mean_s12 --weights zero --scale 1.2 --out $T/mean_s12.json --work $W --save-at $W/mean_s12.npz --omp 4 > $T/mean_s12.log 2>&1 &
$PY $C/02_case.py --case mean_cv08 --weights zero --cvfac 0.8 --out $T/mean_cv08.json --work $W --save-at $W/mean_cv08.npz --omp 4 > $T/mean_cv08.log 2>&1 &
$PY $C/02_case.py --case G000 --weights $WT --row 0 --out $T/G000.json --work $W --omp 1 > $T/G000.log 2>&1 &
$PY $C/02_case.py --case U001 --weights $WT --row 200 --out $T/U001.json --work $W --omp 1 > $T/U001.log 2>&1 &
wait
date
$PY - <<'PY'
import numpy as np
W="/dev/shm/I09_"+__import__('os').environ['SLURM_JOB_ID']
a=np.load(W+"/mean_s1.npz"); b=np.load(W+"/mean_s12.npz"); c=np.load(W+"/mean_cv08.npz")
for k in a.files:
    x,y,z=a[k].astype(float),b[k].astype(float),c[k].astype(float); m=x>0
    print(k,"TAT",x.max(),"scale1.2 ratio TAT",y.max()/x.max(),"nodewise ratio min/max",(y[m]/x[m]).min(),(y[m]/x[m]).max(),
          "| cv0.8 ratio TAT",z.max()/x.max(),"nodewise",(z[m]/x[m]).min(),(z[m]/x[m]).max())
np.savez_compressed("results/2026-10-22_pilot_I09/test/mean_shape_AT.npz", s1=a["primary"], s12=b["primary"], cv08=c["primary"])
PY
rm -rf $W
