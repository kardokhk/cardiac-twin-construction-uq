#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-noiseext-disp
#SBATCH --output=/project/home/p201509/work/cvri/simulation/results/2026-10-28_noise_ext/logs/%x-%j.out
# Owner noise-ext, stage D3: assemble 50 displaced_ext files, solve summary, then displacement calibrations (30, 40 mm).
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh; K=code/full/calib
R=results/2026-10-28_noise_ext; S=data/processed/acdc_mesh_v2/_scratch_disp_ext
W=/dev/shm/noiseext_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
date -Is; n=$(ls $S/Zmeta_*.npz | wc -l); echo "worker outputs: $n"
seq 101 150 | xargs -P 16 -I{} sh -c "$PY $M/displace_ext.py assemble $S {} > $S/logs/asm_{}.log 2>&1"
grep -h "W_zero" $S/logs/asm_*.log | wc -l
$PY -c "
import numpy as np, glob, json
its=[];rr=[]
for f in glob.glob('$S/Zmeta_*.npz'):
    z=np.load(f); its+=list(z['its']); rr+=list(z['relres'])
m=np.load('$S/meta.npz')
np.savez_compressed('$R/disp_ext_draws.npz', **{k: m[k] for k in ('R','Xe','Lw','enode','snap_mm','euclid_node','euclid_walk','uniq','pids')})
json.dump(dict(n_solves=len(its), its_min=int(min(its)), its_max=int(max(its)), relres_max=float(max(rr))), open('$R/disp_ext_solves.json','w'))
print(open('$R/disp_ext_solves.json').read())"
cp $S/prep.json $R/disp_ext_prep.json
n=$(ls data/processed/acdc_mesh_v2/displaced_ext/*.h5 | wc -l); echo "displaced_ext files: $n"
if [ "$n" = 50 ]; then
  $PY $K/run_noise_ext.py --out $R/noise_ext_disp.h5 --hearts $(seq 101 150) --workers 126 --comparators disp
  rc=$?
  tar czf $R/logs_disp_ext_workers.tar.gz -C $S logs && rm -rf $S
else
  rc=1
fi
rm -rf "$W"; date -Is; echo "done rc=$rc"
exit $rc
