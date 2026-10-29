#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-mesh2-asm
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# mesh2 full run, stage 3: per-heart displaced ECG weights (one HDF5 per heart), solve summary, manifest, cleanup.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"; M=code/full/mesh
R=results/2026-10-28_mesh_v2/full; S=data/processed/acdc_mesh_v2/_scratch_disp
date -Is; n=$(ls $S/Zmeta_*.npz | wc -l); echo "worker outputs: $n"
seq 101 150 | xargs -P 16 -I{} sh -c "$PY $M/displace.py assemble $S {} > $R/logs/disp_asm_{}.log 2>&1"
cat $R/logs/disp_asm_*.log | grep -c "W_zero"
$PY -c "
import numpy as np, glob, json
its=[];rr=[]
for f in glob.glob('$S/Zmeta_*.npz'):
    z=np.load(f); its+=list(z['its']); rr+=list(z['relres'])
m=np.load('$S/meta.npz')
np.savez_compressed('$R/disp_draws.npz', **{k: m[k] for k in ('R','Xe','Lw','enode','snap_mm','euclid_node','euclid_walk','uniq','pids')})
json.dump(dict(n_solves=len(its), its_min=int(min(its)), its_max=int(max(its)), relres_max=float(max(rr))), open('$R/disp_solves.json','w'))
print(open('$R/disp_solves.json').read())"
$PY $M/manifest.py $R > $R/logs/manifest.log 2>&1; tail -5 $R/logs/manifest.log
n=$(ls data/processed/acdc_mesh_v2/displaced/*.h5 | wc -l); echo "displaced files: $n"
[ "$n" = 50 ] && rm -rf $S
date -Is; echo "stage 3 finished"
