"""Stage A (shared by pilots I11 and I13): openCARP v20.0 eikonal known-answer tests.
KA-1  homogeneous anisotropic slab (Niederer 2011 geometry 20 x 7 x 3 mm), point stimulus at a corner,
      fibres along x: t = sqrt((x/vl)^2 + (y/vt)^2 + (z/vn)^2). Error vs resolution and wall time.
KA-1b same with fibres rotated 45 deg in the xy-plane (tests fibre handling), analytic in the rotated frame.
SCALE lengths x s and velocities x s: activation times unchanged (exact invariance of the eikonal equation).
KA-4 / #553  node renumbering: (a) openCARP renumbering=0 vs 1 on the same mesh; (b) a randomly permuted and
      rigidly rotated copy of the mesh, mapped back. Expected identical to solver tolerance.
Usage: python -I stageA_eikonal.py <workdir> <out.json> [threads]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carpio

work, out = sys.argv[1], sys.argv[2]
thr = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(work, exist_ok=True)
VL, VT = 0.6, 0.3
res = {'threads': thr, 'vl': VL, 'vt': VT}
rng = np.random.default_rng(20261022)


def slab(h, name, fib=None, scale=1.0, perm=None, R=None):
    pts, tets, tags = carpio.box_mesh((20, 7, 3), h)
    pts = pts * scale
    f = np.tile([1.0, 0, 0], (len(tets), 1)) if fib is None else np.tile(fib, (len(tets), 1))
    if R is not None:
        pts = pts @ R.T; f = f @ R.T
    stim = 0
    if perm is not None:                       # perm[new] = old
        invp = np.argsort(perm)
        pts = pts[perm]; tets = invp[tets]; stim = invp[0]
    base = os.path.join(work, name)
    carpio.write_mesh(base, pts * 1000, carpio.fix_orientation(pts, tets), tags, f)
    carpio.write_vtx(base + '.vtx', [stim])
    return base, pts


def analytic(p, vl, vt, rot=None):
    q = p if rot is None else p @ rot      # coordinates in fibre frame
    return np.sqrt((q[:, 0] / vl) ** 2 + (q[:, 1] / vt) ** 2 + (q[:, 2] / vt) ** 2)


# KA-1 convergence
res['KA1'] = []
for h in [0.5, 0.25, 0.2, 0.1]:
    base, pts = slab(h, f'slab_h{h}')
    act, wall = carpio.run_eikonal(base, base + '.vtx', [(1, VL, VT, VT)], os.path.join(work, f'run_h{h}'),
                                   threads=thr, tend=100)
    ta = analytic(pts, VL, VT)
    e = act - ta
    r = dict(h=h, nodes=len(pts), max_abs_err_ms=float(np.abs(e).max()), mean_err_ms=float(e.mean()),
             rms_err_ms=float(np.sqrt((e ** 2).mean())), last_act_ms=float(act.max()), last_act_exact=float(ta.max()),
             wall_s=wall)
    res['KA1'].append(r); print('KA1', r, flush=True)

# KA-1b rotated fibres
c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
Rf = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])       # columns: fibre, sheet, normal directions
base, pts = slab(0.2, 'slab_rot', fib=Rf[:, 0])
act, wall = carpio.run_eikonal(base, base + '.vtx', [(1, VL, VT, VT)], os.path.join(work, 'run_rot'), threads=thr, tend=100)
ta = analytic(pts, VL, VT, rot=Rf)
res['KA1b_rot45'] = dict(h=0.2, max_abs_err_ms=float(np.abs(act - ta).max()), rms_err_ms=float(np.sqrt(((act - ta) ** 2).mean())), wall_s=wall)
print('KA1b', res['KA1b_rot45'], flush=True)

# scale invariance
base0, _ = slab(0.25, 'slab_s1')
a0, _ = carpio.run_eikonal(base0, base0 + '.vtx', [(1, VL, VT, VT)], os.path.join(work, 'run_s1'), threads=thr, tend=100)
res['SCALE'] = []
for sc in [0.5, 1.7, 3.0]:
    base, _ = slab(0.25, f'slab_s{sc}', scale=sc)
    a1, _ = carpio.run_eikonal(base, base + '.vtx', [(1, VL * sc, VT * sc, VT * sc)], os.path.join(work, f'run_s{sc}'),
                               threads=thr, tend=100)
    r = dict(scale=sc, max_abs_diff_ms=float(np.abs(a1 - a0).max()), rel_max=float(np.abs(a1 - a0).max() / a0.max()))
    res['SCALE'].append(r); print('SCALE', r, flush=True)

# #553 renumbering
a_r1, _ = carpio.run_eikonal(base0, base0 + '.vtx', [(1, VL, VT, VT)], os.path.join(work, 'run_renum1'), threads=thr,
                             tend=100, extra=['-renumbering', '1'])
res['RENUM_param'] = dict(max_abs_diff_ms=float(np.abs(a_r1 - a0).max()))
# permuted + rotated copy
th = 0.7
Rr = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
pts0, _, _ = carpio.box_mesh((20, 7, 3), 0.25)
perm = rng.permutation(len(pts0))
base, _ = slab(0.25, 'slab_perm', perm=perm, R=Rr)
ap, _ = carpio.run_eikonal(base, base + '.vtx', [(1, VL, VT, VT)], os.path.join(work, 'run_perm'), threads=thr, tend=100)
a_back = np.empty_like(ap); a_back[perm] = ap
ap1, _ = carpio.run_eikonal(base, base + '.vtx', [(1, VL, VT, VT)], os.path.join(work, 'run_perm_r1'), threads=thr,
                            tend=100, extra=['-renumbering', '1'])
a_back1 = np.empty_like(ap1); a_back1[perm] = ap1
res['KA4_perm_rot'] = dict(max_abs_diff_ms=float(np.abs(a_back - a0).max()),
                           max_abs_diff_renum1_ms=float(np.abs(a_back1 - a0).max()))
print('RENUM', res['RENUM_param'], res['KA4_perm_rot'], flush=True)
json.dump(res, open(out, 'w'), indent=1)
