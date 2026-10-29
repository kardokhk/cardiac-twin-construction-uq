#!/bin/bash -l
#SBATCH --account=p201509
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --job-name=cvri-mesh-re
#SBATCH --output=/project/home/p201509/work/cvri/simulation/logs/%x-%j.out
# I11 gates follow-up (owner mesh, 2026-10-28): honest reaction-eikonal (RE+, full Vm) timing on patient101 expert
# 1.5 mm with a separate mesh/eikonal/sim directory per concurrent copy (re_timing.py tag includes copy id + PID).
# Phases, in priority order, each skipped if it would not finish inside the 30 min test QOS:
#   P2 solo, 1 thread, 200 ms (layout default0)
#   P3 128 copies x 1 thread (124 default0 + 4 legacy; legacy vs default0 Vm hash check at 200 ms)
#   P4 16 copies x 8 threads, P5 4 copies x 32 threads
#   P8 eikonal packing-density check: pilot 90-run design on the trio expert constructions with 90 workers
#      (job 5392520 used 128), 1.5 then 1.0 mm
#   P6 solo 8 threads, P7 solo 32 threads
# Per-copy JSONs stay in /dev/shm; each phase appends its records to one JSONL file in results.
set -uo pipefail
ROOT=/project/home/p201509/work/cvri/simulation; cd $ROOT
export PIP_CACHE_DIR=/project/home/p201509/cache/pip XDG_CACHE_HOME=/project/home/p201509/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/project/home/p201509/envs/cvri-sim/bin/python -I"
M=code/gates/mesh
D=data/processed/acdc_mesh
F=$D/cons/patient101__expert__1.5.h5
OUT=results/2026-10-28_G2_meshing/full/re_followup_${SLURM_JOB_ID}.jsonl
W=/dev/shm/mesh_re_$SLURM_JOB_ID; mkdir -p $W; export TMPDIR=$W
LIMIT=1740
date -Is; hostname; nproc; sha256sum $M/re_timing.py $M/carpio.py $M/re_job.sh
collect() {  # $1 phase name, $2 phase wall s; appends every JSON in $W/$1/ as one record per line
  $PY - "$1" "$2" "$W/$1" "$OUT" <<'EOF'
import json, sys, glob, os
ph, wall, d, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(out, 'a') as fo:
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        r = json.load(open(f)); r.pop('log_tail', None) if r.get('copy_id', '') not in ('0', 'solo') else None
        r['phase'] = ph; r['phase_wall_s'] = wall; fo.write(json.dumps(r) + '\n')
    for f in sorted(glob.glob(os.path.join(d, '*.err'))):
        if os.path.getsize(f) and not os.path.exists(f[:-4] + '.json'):
            fo.write(json.dumps(dict(phase=ph, failed=os.path.basename(f), tail=open(f).read()[-1500:])) + '\n')
EOF
}
run_phase() {  # $1 name, $2 estimate s, $3 copies, $4 threads, $5 legacy copies (first N)
  local name=$1 est=$2 n=$3 th=$4 nleg=${5:-0}
  if [ $((SECONDS + est)) -gt $LIMIT ]; then echo "SKIP $name (elapsed $SECONDS s, estimate $est s)"; \
    echo "{\"phase\": \"$name\", \"skipped\": true, \"elapsed_s\": $SECONDS}" >> $OUT; return; fi
  mkdir -p $W/$name; date -Is; echo "phase $name: $n copies x $th threads"
  local t0=$SECONDS
  for i in $(seq 1 $n); do
    L=default0; [ $i -le $nleg ] && L=legacy
    cid=$i; [ $n = 1 ] && cid=solo
    $PY $M/re_timing.py $F $W/$name/c$i.json 1 $th 200 $cid $L > $W/$name/c$i.err 2>&1 &
  done
  wait
  local dt=$((SECONDS - t0)); echo "phase $name wall $dt s"; collect $name $dt
}
run_phase P2_solo_t1 330 1 1
run_phase P3_pack128_t1 600 128 1 4
run_phase P4_pack16_t8 330 16 8
run_phase P5_pack4_t32 300 4 32
if [ $((SECONDS + 300)) -le $LIMIT ]; then
  date -Is; echo "phase P8: eikonal design, 90 workers"
  for v in 1.5 1.0; do
    t0=$SECONDS
    $PY $M/design_runs.py $W/runs90_$v 90 $D/cons/patient101__expert__$v.h5 $D/cons/patient102__expert__$v.h5 \
        $D/cons/patient104__expert__$v.h5 > $W/design90_$v.log 2>&1
    tail -4 $W/design90_$v.log
    $PY - $W/runs90_$v $v $((SECONDS - t0)) $OUT <<'EOF'
import h5py, glob, json, sys, os
d, v, wall, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
with open(out, 'a') as fo:
    for f in sorted(glob.glob(os.path.join(d, '*.h5'))):
        h = h5py.File(f, 'r')
        fo.write(json.dumps(dict(phase='P8_eikonal_90workers', vox=v, construction=os.path.basename(f), workers=90,
                                 n_nodes=int(h.attrs['n_nodes']), pool_wall_s=float(h.attrs['pool_wall_s']),
                                 wall_s=h['wall_s'][:].tolist(), theta=h['theta'][:].tolist(), phase_wall_s=wall)) + '\n')
EOF
  done
else
  echo "SKIP P8 (elapsed $SECONDS s)"; echo "{\"phase\": \"P8_eikonal_90workers\", \"skipped\": true, \"elapsed_s\": $SECONDS}" >> $OUT
fi
run_phase P6_solo_t8 230 1 8
run_phase P7_solo_t32 230 1 32
rm -rf $W
date -Is; echo done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,NNodes,State,MaxRSS
