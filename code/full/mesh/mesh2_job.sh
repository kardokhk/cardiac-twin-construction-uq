#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-mesh2
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# I11 full arm, owner mesh2: A2 layer unit test, all constructions (both resolutions, RV-wall sensitivity), smoke
# tests, electrode-displacement lead fields (A3.3), manifest.
# MODE=test: trio only, displacement for patient 101 only.  MODE=full: everything.
# Optional: PB (parallel builds, default 64), NW (solve workers, default 120), SKIP_BUILD=1, SKIP_DISP=1.
# Usage: sbatch --qos=test --time=00:30:00 --export=ALL,MODE=test mesh2_job.sh
set -uo pipefail
MODE=${MODE:-full}; PB=${PB:-64}; NW=${NW:-120}
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I -B"
M=code/full/mesh
R=results/2026-10-28_mesh_v2/$MODE; mkdir -p $R/qc $R/logs
D=data/processed/acdc_mesh_v2; mkdir -p $D/cons $D/ref $D/displaced
W=/dev/shm/mesh2_${SLURM_JOB_ID}; mkdir -p $W/build; export TMPDIR=$W/build      # job-private scratch
S=$W/disp; mkdir -p $S                                                            # displacement scratch
date -Is; hostname; echo "MODE=$MODE PB=$PB NW=$NW"
sha256sum $M/*.py $M/*.sh > $R/code_sha256.txt
if [ "$MODE" = test ]; then HEARTS="101 102 104"; DH="101"; LT=test; else HEARTS=$(seq 101 150); DH=$(seq -s, 101 150); LT=full; fi
$PY $M/test_endo_layer.py $R/endo_layer_test.json > $R/logs/endo_layer_test.log 2>&1 &
if [ "${SKIP_BUILD:-0}" = 0 ]; then
  date -Is; echo "phase 1: reference constructions"
  for p in $HEARTS; do echo $p; done | xargs -P 50 -I{} sh -c "$PY $M/build.py ref {} $R/qc > $R/logs/ref_{}.log 2>&1"
  grep -h -E "pass|fail" $R/logs/ref_*.log | awk '{print $2}' | sort | uniq -c
  date -Is; echo "phase 2: other constructions"
  $PY $M/lists.py todo $LT > $W/todo.txt; wc -l < $W/todo.txt
  xargs -P $PB -L 1 sh -c "$PY $M/build.py cons \$0 \$1 \$2 \$3 $R/qc > $R/logs/cons_\$0_\$1_\$2_\$3.log 2>&1" < $W/todo.txt
  date -Is; echo "phase 2 done"; cat $R/logs/cons_*.log | awk '{print $2, $3}' | sort | uniq -c
fi
if [ "${SKIP_DISP:-0}" = 0 ]; then
  date -Is; echo "phase 3: displacement prep"
  [ "$MODE" = test ] && DH=101
  $PY $M/displace.py prep $S $DH > $R/logs/disp_prep.log 2>&1; tail -5 $R/logs/disp_prep.log
  date -Is; echo "phase 4: lead-field solves ($NW workers)"
  seq 0 $((NW-1)) | xargs -P $NW -I{} sh -c "$PY $M/displace.py solve $S {} $NW > $R/logs/disp_solve_{}.log 2>&1"
  cat $R/logs/disp_solve_*.log | tail -3
  date -Is; echo "phase 5: assemble"
  echo $DH | tr ',' '\n' | xargs -P 16 -I{} sh -c "$PY $M/displace.py assemble $S {} > $R/logs/disp_asm_{}.log 2>&1"
  cat $R/logs/disp_asm_*.log | head -60
  cp $S/prep.json $R/disp_prep.json
  $PY -c "
import numpy as np, glob, json
its=[];rr=[]
for f in glob.glob('$S/Zmeta_*.npz'):
    z=np.load(f); its+=list(z['its']); rr+=list(z['relres'])
m=np.load('$S/meta.npz')
np.savez_compressed('$R/disp_draws.npz', **{k: m[k] for k in ('R','Xe','Lw','enode','snap_mm','euclid_node','euclid_walk','uniq','pids')})
json.dump(dict(n_solves=len(its), its_min=int(min(its)), its_max=int(max(its)), relres_max=float(max(rr))), open('$R/disp_solves.json','w'))
print(open('$R/disp_solves.json').read())"
fi
wait
date -Is; echo "phase 6: manifest"
$PY $M/manifest.py $R > $R/logs/manifest.log 2>&1; tail -60 $R/logs/manifest.log
tail -25 $R/logs/endo_layer_test.log
du -sh $W; rm -rf $W
date -Is; echo done
