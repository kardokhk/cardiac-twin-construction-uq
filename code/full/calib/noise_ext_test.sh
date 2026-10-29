#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-noiseext-test
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%j.out
# Owner noise-ext, Amendment 2 B4, test on heart 102: full displacement pipeline (prep, 120 solves workers, assemble)
# into /dev/shm, then all new levels with nrep 3, then export + owner check. Outputs tarred into results/.../test/.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh; K=code/full/calib
W=/dev/shm/noiseext_$SLURM_JOB_ID; S=$W/scr; mkdir -p $S $W/disp $W/out; export TMPDIR=$W
export NOISE_EXT_DISP_DIR=$W/disp
date -Is; hostname
$PY $M/displace_ext.py prep $S 102 > $W/prep.log 2>&1; tail -3 $W/prep.log
seq 0 119 | xargs -P 120 -I{} sh -c "$PY $M/displace_ext.py solve $S {} 120 > $S/solve_{}.log 2>&1"
date -Is; cat $S/solve_*.log | head -3
$PY $M/displace_ext.py assemble $S 102 2>&1 | tail -2
date -Is
$PY $K/run_noise_ext.py --out $W/out/noise_ext_test.h5 --hearts 102 --workers 126 --nrep 3
date -Is
$PY $K/export_noise_ext.py $W/out $W/out/noise_ext_test.h5 2>&1 | tail -60
mkdir -p results/2026-10-28_noise_ext/test
cp $S/prep.json $W/out/disp_prep_test.json
tar czf results/2026-10-28_noise_ext/test/test_job$SLURM_JOB_ID.tar.gz -C $W out prep.log $(cd $W; ls scr/*.log)
rm -rf "$W"
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
