"""Stage A ECG known-answer tests (shared by I11 and I13).
KA-E1 current dipole at the centre of a homogeneous insulated sphere (R = 100 mm): surface potential
      3 p.cos(theta) / (4 pi sigma R^2) (classical result, e.g. Plonsey and Barr, Bioelectricity). Our lead-field
      FEM gives phi_e - phi_ref = p . grad Z_e(x0). Acceptance: RMS relative error <= 2%.
KA-E2 reciprocity: lead-field ECG vs direct forward solve K phi = -K_i Vm for a random smooth Vm field.
      Acceptance: max relative difference <= 1%.
KA-E3 independent-code check: openCARP v20.0 monodomain on the Niederer slab writes vm.igb and phie_recovery
      (infinite-medium extracellular potential) at 6 points; we recompute phie from the same vm.igb with our
      weights W = K_i Z_inf, Z_inf = 1/(4 pi sigma r). Acceptance: per-point correlation >= 0.999 and constant
      amplitude ratio (sigma convention).
KA-E4 template error: our activation-time ECG model (sigmoid upstroke template) vs the full-Vm computation of
      KA-E3, correlation per point (information only; it measures the template approximation).
Usage: python -I stageA_ecg.py <workdir> <out.json>
"""
import sys, os, json, time, subprocess
import numpy as np
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I13'))
import carpio, eplib

work, out = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
rng = np.random.default_rng(20261022)
res = {}

# ---------------- KA-E1 sphere dipole
R = 100.0
def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
res['KA_E1'] = []
for nsurf, nin in ([(4000, 40000)] if os.environ.get('SMALL') else [(4000, 40000), (12000, 200000)]):
    surf = fib_sphere(nsurf) * R
    r = R * rng.random(nin) ** (1 / 3) * 0.985
    inn = fib_sphere(nin)[rng.permutation(nin)] * r[:, None]
    pts = np.vstack([np.zeros((1, 3)), surf, inn])
    tets = Delaunay(pts).simplices.astype(np.int64)
    vol = np.abs(carpio.tet_volumes(pts, tets)); tets = tets[vol > 1e-9]
    K = eplib.stiffness(pts, tets, sigma=eplib.SIG_BULK)
    el = 1 + rng.choice(nsurf, 12, replace=False)
    ref = el[-1]
    Z = eplib.solve_leadfields(K, el[:-1], ref, verbose=False)
    near = np.nonzero(np.linalg.norm(pts, axis=1) < 12.0)[0]
    A = np.column_stack([np.ones(len(near)), pts[near]])
    grad = np.linalg.lstsq(A, Z[near], rcond=None)[0][1:]          # (3, ne)
    p = np.array([0.3, -0.5, 1.0]) * 1000.0                          # arbitrary dipole
    num = p @ grad
    xh = pts[el] / np.linalg.norm(pts[el], axis=1, keepdims=True)
    phi = 3 * (xh @ p) / (4 * np.pi * eplib.SIG_BULK * R ** 2)
    ana = phi[:-1] - phi[-1]
    err = float(np.sqrt(((num - ana) ** 2).mean()) / np.sqrt((ana ** 2).mean()))
    r_ = dict(nodes=len(pts), tets=len(tets), rms_rel_err=err, max_rel_err=float(np.abs(num - ana).max() / np.abs(ana).max()))
    res['KA_E1'].append(r_); print('KA_E1', r_, flush=True)

# ---------------- KA-E2 reciprocity (on the finer ball)
heart = np.linalg.norm(pts - np.array([10, -20, 5]), axis=1) < 35
hel = heart[tets].all(1)
Ki = eplib.stiffness(pts, tets[hel], sigma=eplib.SIG_IL)
Vm = np.where(heart, np.sin(pts[:, 0] / 9.0) * np.cos(pts[:, 1] / 13.0) * 50 + pts[:, 2], 0.0)
src = -(Ki @ Vm)
keep = np.ones(len(pts)); keep[ref] = 0
import scipy.sparse as sp, pyamg
Dk = sp.diags(keep); Ad = (Dk @ K @ Dk + sp.diags(1 - keep)).tocsr()
b = src.copy(); b[ref] = 0
phi_d = pyamg.smoothed_aggregation_solver(Ad, symmetry='symmetric').solve(b, tol=1e-12, accel='cg', maxiter=1000)
direct = phi_d[el[:-1]] - phi_d[ref]
lf = -(Z.T @ (Ki @ Vm))
res['KA_E2'] = dict(max_rel_diff=float(np.abs(direct - lf).max() / np.abs(direct).max()))
print('KA_E2', res['KA_E2'], flush=True)

# ---------------- KA-E3 / E4 openCARP phie_recovery on Niederer slab
h = 0.25
spts, stets, stags = carpio.box_mesh((20, 7, 3), h)
base = os.path.join(work, 'nslab')
carpio.write_mesh(base, spts * 1000, stets, stags)
rp = np.array([[10, 3.5, 8], [10, 3.5, 23], [-5, 3.5, 1.5], [25, -5, 1.5], [10, 15, -4], [3, 3, 40.0]])
with open(os.path.join(work, 'rec.pts'), 'w') as f:
    f.write(f'{len(rp)}\n'); np.savetxt(f, rp * 1000, fmt='%.1f')
sb = os.path.join(work, 'stimblock.par')
open(sb, 'w').write('stim[0].elec.p0 = -1 -1 -1\nstim[0].elec.p1 = 1500.5 1500.5 1500.5\n')
par = ['+F', sb, '-meshname', base, '-simID', os.path.join(work, 'nslab_run'), '-tend', '80', '-dt', '20', '-spacedt', '0.1',
       '-timedt', '1', '-bidomain', '0', '-num_imp_regions', '1', '-imp_region[0].im', 'tenTusscherPanfilov',
       '-num_gregions', '1', '-gregion[0].num_IDs', '1', '-gregion[0].ID[0]', '1',
       '-gregion[0].g_il', '0.17', '-gregion[0].g_el', '0.62', '-gregion[0].g_it', '0.019', '-gregion[0].g_et', '0.24',
       '-gregion[0].g_in', '0.019', '-gregion[0].g_en', '0.24',
       '-num_stim', '1', '-stim[0].crct.type', '0', '-stim[0].elec.geom_type', '2',
       '-stim[0].ptcl.start', '0', '-stim[0].ptcl.duration', '2', '-stim[0].pulse.strength', '35.71',
       '-phie_rec_ptf', os.path.join(work, 'rec')]
mpirun = os.path.join(os.path.dirname(carpio.OPENCARP), 'mpirun')
t0 = time.time()
with open(os.path.join(work, 'nslab_run.log'), 'w') as lf:
    r = subprocess.run([mpirun, '-launcher', 'fork', '-np', os.environ.get('NP', '16'), carpio.OPENCARP] + par, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1'))
print('openCARP slab run', r.returncode, f'{time.time()-t0:.0f}s', flush=True)
if r.returncode != 0:
    print(open(os.path.join(work, 'nslab_run.log')).read()[-3000:])
vm = carpio.read_igb(os.path.join(work, 'nslab_run', 'vm.igb'))            # (T, N)
ph = carpio.read_igb(os.path.join(work, 'nslab_run', 'phie_recovery.igb'))  # (T, 6)
fibs = np.tile([1.0, 0, 0], (len(stets), 1))
Kis = eplib.stiffness(spts, stets, fib=fibs, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
rr = np.linalg.norm(spts[:, None, :] - rp[None], axis=2)
Zinf = 1.0 / (4 * np.pi * eplib.SIG_BULK * rr)
W = (Kis @ Zinf).T                                         # (6, N)
ours = -(vm @ W.T)                                         # (T, 6)
T = min(len(ours), len(ph))
corr = [float(np.corrcoef(ours[:T, k], ph[:T, k])[0, 1]) for k in range(6)]
ratio = [float(np.sum(ph[:T, k] * ours[:T, k]) / np.sum(ours[:T, k] ** 2)) for k in range(6)]
res['KA_E3'] = dict(corr=corr, amp_ratio_opencarp_over_ours=ratio, frames=T)
# variant: monodomain (harmonic-mean) conductivity tensor and sigma = g_bath default 1.0 S/m (hypothesis for openCARP's convention)
sml, smt = 0.17 * 0.62 / (0.17 + 0.62), 0.019 * 0.24 / (0.019 + 0.24)
Kim = eplib.stiffness(spts, stets, fib=fibs, sig_l=sml, sig_t=smt)
Wm = (Kim @ (1.0 / (4 * np.pi * 1.0 * rr))).T
oursm = -(vm @ Wm.T)
res['KA_E3_monodomain_tensor_gbath1'] = dict(
    corr=[float(np.corrcoef(oursm[:T, k], ph[:T, k])[0, 1]) for k in range(6)],
    amp_ratio=[float(np.sum(ph[:T, k] * oursm[:T, k]) / np.sum(oursm[:T, k] ** 2)) for k in range(6)])
print('KA_E3m', res['KA_E3_monodomain_tensor_gbath1'], flush=True)
print('KA_E3', res['KA_E3'], flush=True)
# template model from activation times
ta = np.full(vm.shape[1], np.nan)
dtv = 0.1
above = vm > -40.0
first = above.argmax(0); ok = above.any(0)
v0 = vm[np.maximum(first - 1, 0), np.arange(vm.shape[1])]; v1 = vm[first, np.arange(vm.shape[1])]
ta = (first - 1 + (-40.0 - v0) / np.maximum(v1 - v0, 1e-9)) * dtv
ta[~ok] = 1e9
res['KA_E4'] = {}
for tau in [0.5, 1.0, 2.0]:
    em = eplib.ECGModel(W, tmax=(T - 1) * dtv, dt=dtv, tau=tau)
    sim = em(ta)
    c4 = [float(np.corrcoef(sim[k, :T], ours[:T, k])[0, 1]) for k in range(6)]
    res['KA_E4'][f'tau{tau}'] = c4
print('KA_E4', res['KA_E4'], flush=True)
json.dump(res, open(out, 'w'), indent=1)
