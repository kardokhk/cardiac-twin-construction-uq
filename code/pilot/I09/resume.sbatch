#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-I09-resume
#SBATCH --output=logs/%x-%j.out
# I09 pilot main arm: 200 Gaussian + 100 deposited-uniform KIT instances, packed with xargs -P.
set -uo pipefail
cd /project/home/p201509/work/cvri/simulation
PY="/project/home/p201509/envs/pilot-I09/bin/python -I"
C=code/pilot/I09; R=results/2026-10-22_pilot_I09
ROWS=$R/rows; LOGS=/dev/shm/I09logs_$SLURM_JOB_ID; mkdir -p $ROWS $LOGS
W=/dev/shm/I09_$SLURM_JOB_ID; mkdir -p $W
NP=${NP:-80}
date
cat $R/cases.txt | xargs -P $NP -L 1 bash -c \
  "[ -s $ROWS/\$0.json ] && exit 0; $PY $C/02_case.py --case \$0 --row \$2 --weights $R/weights.csv --out $ROWS/\$0.json --work $W --omp 1 > $LOGS/\$0.log 2>&1 || echo FAIL \$0"
date
tar czf $R/main_case_logs_$SLURM_JOB_ID.tar.gz -C $LOGS .
rm -rf $W $LOGS
$PY $C/04_decide.py $ROWS $R/weights.csv $R > $R/decide_stdout.txt 2>&1
date
