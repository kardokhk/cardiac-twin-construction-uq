"""Reaction-eikonal (full transmembrane voltage) timing and ECG on one ACDC construction (owner mesh, I11 gates,
2026-10-28). openCARP v20.0, dream.solve = 1 (RE+, with diffusion) or 2 (RE-), monodomain, tenTusscherPanfilov,
default conductivities, renumbering 0, dt 25 us, Vm written every 0.5 ms. ECG from Vm: ECG_l(t) = -sum_n W_ln Vm_n(t)
(same weights W as the activation-template ECG). The template ECG from the same run's eikonal activation (act.igb)
and from a pure-eikonal run are compared with the Vm ECG (Pearson r per lead over the QRS window).
Usage: python -I re_timing.py <cons.h5> <out.json> <mode> <omp_threads> [tend_ms] [copy_id] [layout] [vm_out.npy]
(openCARP v20.0 refuses MPI for eikonal/RE physics, so parallelism is OpenMP threads in one rank.)
Follow-up 2026-10-28: every copy gets its own mesh, eikonal and simulation directory under $TMPDIR (tag includes
copy_id and the PID); the original tag f'm{mode}_np{npm}' was shared by concurrent copies with the same mode and
thread count (the 8-copy packing check of job 5392520 wrote into one directory). `layout` is passed to
carpio.run_reaction_eikonal (default 'default0', the warning-free layout). The scratch directories are removed at the end.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import acdc_lib as A, carpio, eplib
import h5py

f, out, mode, npm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tend = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0
copy_id = sys.argv[6] if len(sys.argv) > 6 else '0'
layout = sys.argv[7] if len(sys.argv) > 7 else 'default0'
vm_out = sys.argv[8] if len(sys.argv) > 8 else None
tmp = os.environ.get('TMPDIR', '/tmp')
tag = f'm{mode}_np{npm}_c{copy_id}_{layout}_p{os.getpid()}'
t_start = time.time()
mb, vtx = A.materialize(f, os.path.join(tmp, 're_' + tag))
th = A.THETA_SMOKE
regions = [(1, A.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])]
W = h5py.File(f, 'r')['W'][:]
res = dict(construction=f, mode={1: 'RE+', 2: 'RE-'}[mode], omp_threads=npm, tend_ms=tend, dt_us=25.0, theta=th,
           n_nodes=int(W.shape[1]), copy_id=copy_id, layout=layout, tag=tag, host=os.uname()[1])
act_e, wall_e = A.eikonal(mb, vtx, th, os.path.join(tmp, 'eik_' + tag))
res['eikonal_wall_s'] = wall_e; res['eikonal_act_max_ms'] = float(act_e.max())
vm, act, wall, log = carpio.run_reaction_eikonal(mb, vtx, regions, os.path.join(tmp, 'sim_' + tag), mode=mode,
                                                 np_mpi=npm, tend=tend, dt_us=25.0, spacedt=0.5, keep=False,
                                                 layout=layout)
res['re_wall_s'] = wall; res['vm_frames'] = int(vm.shape[0])
res['re_t0_unix'] = t_start
res['warn_unassigned'] = [l for l in log.splitlines() if l.startswith('WARNING-LINE')]
import hashlib
res['vm_sha256'] = hashlib.sha256(np.ascontiguousarray(vm).tobytes()).hexdigest()
if vm_out:
    np.save(vm_out, vm)
dt = 0.5
ecg_vm = -(vm.astype(np.float64) @ W.T).T                      # (8, T)
up = vm > -40.0; ok = up.any(0); first = up.argmax(0)
res['re_unactivated'] = int((~ok).sum())
res['re_upstroke_minus_act_ms_median'] = float(np.median(first[ok] * dt - act_e[ok]))
nt = vm.shape[0]
ecg_t = eplib.ECGModel(W, tmax=(nt - 1) * dt, dt=dt)(act_e)
ecg_t_re = eplib.ECGModel(W, tmax=(nt - 1) * dt, dt=dt)(act) if act is not None else None
qrs = slice(0, min(nt, int((act_e.max() + 20) / dt)))
def corr(a, b):
    return [float(np.corrcoef(a[l, qrs], b[l, qrs])[0, 1]) for l in range(8)]
# remove the resting baseline (rest Vm contributes a constant because sum_n W_ln = 0 only up to round-off)
base = ecg_vm[:, :1]
res['r_vm_vs_template_eik'] = corr(ecg_vm - base, ecg_t)
if ecg_t_re is not None:
    res['r_vm_vs_template_re_act'] = corr(ecg_vm - base, ecg_t_re)
res['amp_ratio_vm_over_template'] = float(np.linalg.norm((ecg_vm - base)[:, qrs]) / np.linalg.norm(ecg_t[:, qrs]))
res['log_tail'] = log[-1500:]
res['t_end_unix'] = time.time()
import shutil
for d in ('re_', 'eik_'):
    shutil.rmtree(os.path.join(tmp, d + tag), ignore_errors=True)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'log_tail'}), flush=True)
