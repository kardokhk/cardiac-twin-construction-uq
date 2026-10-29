"""I09 pilot step 2: one KIT SSM instance -> phenotypes, Laplace-Dirichlet rule-based fibres, eikonal TAT.

Fixed across all meshes (pre-registered pipeline, written 2026-10-22 before any main-arm run):
  geometry   x = mean + sum_i sqrt(var_i) w_i pc_i (pc_vol, node-interleaved), times an optional scale factor;
             topology of heart_vol.vtu (478,820 nodes, 2,555,157 tets, mean edge 0.82 mm on the mean shape).
  phenotypes LVEDV, RVEDV: volume enclosed by the LV (RV) endocardial faces closed by a fan from the centroid of
             the basal orifice loop (divergence theorem). Myocardial volume: sum of tet volumes. RV free wall:
             elements whose centroid is nearer the RV than the LV endocardium and whose RV-endo plus epi distance
             is below 4.5 mm (the KIT RV wall is a 3 mm normal offset). LV mass = 1.05 g/mL x (myocardium - RV
             free wall), so the septum counts as LV, as in CMR convention.
  fibres     Bayer et al. 2012 (doi:10.1007/s10439-012-0593-5) LDRB: Laplace fields psi (apex node 0, base 1),
             phi_epi, phi_lv, phi_rv (Dirichlet on LV endo, RV endo, epi), d = phi_rv/(phi_lv+phi_rv);
             alpha_endo +40, alpha_epi -50, beta_endo -65, beta_epi +25 degrees; bislerp with right-multiplied
             quaternion symmetries. One fibre direction per element (transversely isotropic conduction).
  eikonal    openCARP v20.0, dream.solve = 0 (pure eikonal, fast iterative method, tol 1e-4 ms).
             Myocardium (tag 1): v_f 600, v_t = v_n 300 mm/s. Fast endocardial layer (tag 2, elements touching
             LV or RV endocardium): isotropic 2000 mm/s in the PRIMARY run; same as myocardium in the
             SECONDARY (no-fast-layer) run. Stimuli: 4 template patches (3 LV endocardial, 1 RV free wall,
             3 mm radius), all at t = 0.
  TAT        max activation time over all nodes minus 0; AT10-90 = 90th minus 10th percentile.
Usage: python -I 02_case.py --case ID --weights FILE --row K --scale S --cvfac C --out JSON --work DIR
       [--save-at NPY] [--primary-only]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pyamg
import scipy.sparse as sp
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carpio  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument("--case", required=True)
P.add_argument("--weights", required=True, help="CSV of weight rows (no header), or 'zero'")
P.add_argument("--row", type=int, default=0)
P.add_argument("--scale", type=float, default=1.0)
P.add_argument("--cvfac", type=float, default=1.0)
P.add_argument("--out", required=True)
P.add_argument("--work", required=True)
P.add_argument("--proc", default="data/processed/I09")
P.add_argument("--var", default="data/raw/redteam/I10/var.csv.gz")
P.add_argument("--save-at", default="")
P.add_argument("--primary-only", action="store_true")
P.add_argument("--omp", type=int, default=1)
A = P.parse_args()

ALPHA_ENDO, ALPHA_EPI, BETA_ENDO, BETA_EPI = 40.0, -50.0, -65.0, 25.0
VF, VT, VFAST = 600.0, 300.0, 2000.0
RHO = 1.05
RVFW_THICK = 4.5

t_start = time.time()
res = {"case": A.case, "row": A.row, "scale": A.scale, "cvfac": A.cvfac, "ok": False}
tm = np.load(os.path.join(A.proc, "template.npz"))
X0, T = tm["X0"], tm["T"].astype(np.int64)
pc = np.load(os.path.join(A.proc, "pc_vol_f32.npy"), mmap_mode="r")
var = np.loadtxt(A.var)
if A.weights == "zero":
    w = np.zeros(100)
else:
    w = np.loadtxt(A.weights, delimiter=",")[A.row]
res["w_mahal2"] = float(w @ w)
X = (X0 + (pc @ (np.sqrt(var) * w).astype(np.float32)).astype(np.float64).reshape(-1, 3)) * A.scale

# element geometry
x0 = X[T[:, 0]]
J = np.stack([X[T[:, 1]] - x0, X[T[:, 2]] - x0, X[T[:, 3]] - x0], 1)  # rows = edges
det = np.linalg.det(J)
vol = det / 6.0
res["n_inverted"] = int((vol <= 0).sum())

Ji = np.linalg.inv(J)
G = np.empty((T.shape[0], 4, 3))
G[:, 1:, :] = np.transpose(Ji, (0, 2, 1))
G[:, 0, :] = -G[:, 1:, :].sum(1)
avol = np.abs(vol)


def cavity_volume(faces, be, beloop, nloop):
    v = np.einsum("ij,ij->i", X[faces[:, 0]], np.cross(X[faces[:, 1]], X[faces[:, 2]])).sum()
    for l in range(nloop):
        e = be[beloop == l]
        c = X[np.unique(e)].mean(0)
        # cap triangles (v, u, c) close the surface consistently with face orientation
        v += np.einsum("ij,ij->i", X[e[:, 1]], np.cross(X[e[:, 0]], np.broadcast_to(c, (e.shape[0], 3)))).sum()
    return abs(v) / 6.0 / 1000.0  # mL


res["LVEDV_mL"] = cavity_volume(tm["faces_lv"], tm["be_lv"], tm["beloop_lv"], int(tm["nloop_lv"]))
res["RVEDV_mL"] = cavity_volume(tm["faces_rv"], tm["be_rv"], tm["beloop_rv"], int(tm["nloop_rv"]))
cent = X[T].mean(1)


def fdist(faces):
    return cKDTree(X[faces].mean(1)).query(cent, workers=A.omp)[0]


bF, fcls = tm["bF"], tm["fcls"]
d_lv, d_rv, d_epi = fdist(bF[fcls == 3]), fdist(bF[fcls == 4]), fdist(bF[fcls == 2])
rvfw = (d_rv < d_lv) & (d_rv + d_epi < RVFW_THICK)
res["myo_mL"] = avol.sum() / 1000.0
res["RVFW_mL"] = avol[rvfw].sum() / 1000.0
res["LVmass_g"] = RHO * (res["myo_mL"] - res["RVFW_mL"])

# Laplace fields
n = X.shape[0]
Ke = avol[:, None, None] * np.einsum("eik,ejk->eij", G, G)
rows = np.repeat(T, 4, axis=1).ravel()
cols = np.tile(T, (1, 4)).ravel()
K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
del Ke, rows, cols


def solve(dn, gv):
    mask = np.zeros(n, bool); mask[dn] = True
    I = np.where(~mask)[0]
    u = np.zeros(n); u[dn] = gv
    Kii = K[I][:, I].tocsr()
    rhs = -(K[I][:, dn] @ gv)
    ml = pyamg.smoothed_aggregation_solver(Kii, symmetry="symmetric")
    resid = []
    u[I] = ml.solve(rhs, tol=1e-10, accel="cg", maxiter=500, residuals=resid)
    return u, len(resid)


t0 = time.time()
lv, rv, epi = tm["lv_nodes"], tm["rv_nodes"], tm["epi_nodes"]
epi = np.setdiff1d(epi, np.union1d(lv, rv))
dn = np.concatenate([lv, rv, epi])
# one hierarchy for the three transmural problems: phi_lv + phi_rv + phi_epi = 1 by linearity
mask = np.zeros(n, bool); mask[dn] = True
I = np.where(~mask)[0]
Kii = K[I][:, I].tocsr(); Kid = K[I][:, dn]
ml = pyamg.smoothed_aggregation_solver(Kii, symmetry="symmetric")
fields = {}
iters = []
for name, g in (("lv", np.r_[np.ones(lv.size), np.zeros(rv.size + epi.size)]),
                ("rv", np.r_[np.zeros(lv.size), np.ones(rv.size), np.zeros(epi.size)])):
    u = np.zeros(n); u[dn] = g
    r = []
    u[I] = ml.solve(-(Kid @ g), tol=1e-10, accel="cg", maxiter=500, residuals=r)
    fields[name] = u; iters.append(len(r))
fields["epi"] = 1.0 - fields["lv"] - fields["rv"]
apex, base = tm["apex_nodes"], tm["base_nodes"]
psi, it_psi = solve(np.concatenate([apex, base]), np.r_[np.zeros(apex.size), np.ones(base.size)])
iters.append(it_psi)
res["laplace_iters"] = iters
res["t_laplace_s"] = time.time() - t0
del K, Kii, Kid, ml


def egrad(u):
    return np.einsum("ei,eij->ej", u[T], G)


def emean(u):
    return u[T].mean(1)


g_psi, g_lv, g_rv, g_epi = egrad(psi), egrad(fields["lv"]), egrad(fields["rv"]), egrad(fields["epi"])
e_lv, e_rv, e_epi = emean(fields["lv"]), emean(fields["rv"]), emean(fields["epi"])
depth = e_rv / np.maximum(e_lv + e_rv, 1e-12)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def axis(gpsi, gphi):
    e1 = unit(gpsi)
    e2 = gphi - np.einsum("ij,ij->i", e1, gphi)[:, None] * e1
    e2 = unit(e2)
    e0 = np.cross(e1, e2)
    return np.stack([e0, e1, e2], axis=2)  # columns


def orient(Q, a, b):
    a, b = np.radians(a), np.radians(b)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    z, o = np.zeros_like(a), np.ones_like(a)
    Ra = np.stack([np.stack([ca, -sa, z], -1), np.stack([sa, ca, z], -1), np.stack([z, z, o], -1)], 1)
    Rb = np.stack([np.stack([o, z, z], -1), np.stack([z, cb, sb], -1), np.stack([z, -sb, cb], -1)], 1)
    return Q @ Ra @ Rb


def rot2quat(R):
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((R.shape[0], 4))
    c = np.argmax(np.stack([tr, m00, m11, m22], 1), 1)
    s = np.sqrt(np.maximum(1 + tr, 1e-30)) * 2
    k = c == 0
    q[k] = np.stack([0.25 * s[k], (R[k, 2, 1] - R[k, 1, 2]) / s[k], (R[k, 0, 2] - R[k, 2, 0]) / s[k],
                     (R[k, 1, 0] - R[k, 0, 1]) / s[k]], 1)
    s = np.sqrt(np.maximum(1 + m00 - m11 - m22, 1e-30)) * 2
    k = c == 1
    q[k] = np.stack([(R[k, 2, 1] - R[k, 1, 2]) / s[k], 0.25 * s[k], (R[k, 0, 1] + R[k, 1, 0]) / s[k],
                     (R[k, 0, 2] + R[k, 2, 0]) / s[k]], 1)
    s = np.sqrt(np.maximum(1 + m11 - m00 - m22, 1e-30)) * 2
    k = c == 2
    q[k] = np.stack([(R[k, 0, 2] - R[k, 2, 0]) / s[k], (R[k, 0, 1] + R[k, 1, 0]) / s[k], 0.25 * s[k],
                     (R[k, 1, 2] + R[k, 2, 1]) / s[k]], 1)
    s = np.sqrt(np.maximum(1 + m22 - m00 - m11, 1e-30)) * 2
    k = c == 3
    q[k] = np.stack([(R[k, 1, 0] - R[k, 0, 1]) / s[k], (R[k, 0, 2] + R[k, 2, 0]) / s[k],
                     (R[k, 1, 2] + R[k, 2, 1]) / s[k], 0.25 * s[k]], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat2rot(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def bislerp(Qa, Qb, t):
    qa, qb = rot2quat(Qa), rot2quat(Qb)
    w, x, y, z = qa.T
    cands = np.stack([qa, np.stack([-x, w, z, -y], 1), np.stack([-y, -z, w, x], 1),
                      np.stack([-z, y, -x, w], 1)], 0)  # qa, qa*i, qa*j, qa*k
    dots = np.einsum("cei,ei->ce", cands, qb)
    k = np.argmax(np.abs(dots), 0)
    qm = cands[k, np.arange(qa.shape[0])]
    dm = dots[k, np.arange(qa.shape[0])]
    qm[dm < 0] *= -1
    dm = np.abs(dm).clip(0, 1)
    th = np.arccos(dm)
    st = np.sin(th)
    small = st < 1e-6
    wa = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, st))
    wb = np.where(small, t, np.sin(t * th) / np.where(small, 1, st))
    q = wa[:, None] * qm + wb[:, None] * qb
    return quat2rot(q / np.linalg.norm(q, axis=1, keepdims=True))


t0 = time.time()
a_s = ALPHA_ENDO * (1 - depth) - ALPHA_ENDO * depth
b_s = BETA_ENDO * (1 - depth) - BETA_ENDO * depth
a_w = ALPHA_ENDO * (1 - e_epi) + ALPHA_EPI * e_epi
b_w = BETA_ENDO * (1 - e_epi) + BETA_EPI * e_epi
Q_lv = orient(axis(g_psi, -g_lv), a_s, b_s)
Q_rv = orient(axis(g_psi, g_rv), a_s, b_s)
Q_endo = bislerp(Q_lv, Q_rv, depth)
Q_epi = orient(axis(g_psi, g_epi), a_w, b_w)
Q = bislerp(Q_endo, Q_epi, e_epi)
fib = Q[:, :, 0]
res["t_fibre_s"] = time.time() - t0
res["fib_nonfinite"] = int((~np.isfinite(fib)).any(1).sum())
fib = np.nan_to_num(fib, nan=1.0)
fib = unit(fib)

# fibre QC: helix angle in the LV free wall (depth < 0.1) near endo (phi_epi < 0.1) and epi (phi_epi > 0.9),
# relative to the local (circumferential, apicobasal) frame of the epi axis system
Qe = axis(g_psi, g_epi)
lvfw = depth < 0.1
ha = np.degrees(np.arctan2(np.einsum("ij,ij->i", fib, Qe[:, :, 1]), np.einsum("ij,ij->i", fib, Qe[:, :, 0])))
ha = (ha + 90) % 180 - 90
res["helix_endo_deg"] = float(np.median(ha[lvfw & (e_epi < 0.1)]))
res["helix_epi_deg"] = float(np.median(ha[lvfw & (e_epi > 0.9)]))
res["helix_mid_deg"] = float(np.median(ha[lvfw & (np.abs(e_epi - 0.5) < 0.05)]))
del Q, Q_lv, Q_rv, Q_endo, Q_epi, Qe, G, J, Ji

# eikonal runs
os.makedirs(A.work, exist_ok=True)
wd = os.path.join(A.work, A.case)
os.makedirs(wd, exist_ok=True)


def wbin(path, hdr, arr):
    with open(path, "wb") as f:
        h = hdr.encode().ljust(1024, b"\0")
        f.write(h)
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


wbin(os.path.join(wd, "m.bpts"), f"{n} 0 666", X * 1000.0)
wbin(os.path.join(wd, "m.blon"), f"1 {T.shape[0]} 0 666", fib)
os.symlink(os.path.abspath(os.path.join(A.proc, "tmpl.belem")), os.path.join(wd, "m.belem"))
names = [str(s) for s in tm["stim_names"]]
for i, s in enumerate(names):
    carpio.write_vtx(os.path.join(wd, f"stim{i}.vtx"), tm["stim_" + s])


def par(fast):
    c = A.cvfac
    L = ["meshname = m", "experiment = 0", "tend = 1", "dt = 1.0", "spacedt = 1.0", "timedt = 1.0",
         "dream.solve = 0", "dream.fim.tol = 1e-4", "dream.fim.max_iter = 5000",
         "num_phys_regions = 1", "phys_region[0].ptype = 2", "phys_region[0].num_IDs = 2",
         "phys_region[0].ID[0] = 1", "phys_region[0].ID[1] = 2",
         "num_imp_regions = 1", "imp_region[0].im = MitchellSchaeffer", "imp_region[0].num_IDs = 2",
         "imp_region[0].ID[0] = 1", "imp_region[0].ID[1] = 2",
         "num_gregions = 2",
         "gregion[0].num_IDs = 1", "gregion[0].ID[0] = 1",
         f"gregion[0].dream.vel_l = {VF * c}", f"gregion[0].dream.vel_t = {VT * c}",
         f"gregion[0].dream.vel_n = {VT * c}",
         "gregion[1].num_IDs = 1", "gregion[1].ID[0] = 2"]
    if fast:
        L += [f"gregion[1].dream.vel_l = {VFAST * c}", f"gregion[1].dream.vel_t = {VFAST * c}",
              f"gregion[1].dream.vel_n = {VFAST * c}"]
    else:
        L += [f"gregion[1].dream.vel_l = {VF * c}", f"gregion[1].dream.vel_t = {VT * c}",
              f"gregion[1].dream.vel_n = {VT * c}"]
    L += [f"num_stim = {len(names)}"]
    for i in range(len(names)):
        L += [f"stim[{i}].crct.type = 0", f"stim[{i}].elec.vtx_file = stim{i}", f"stim[{i}].ptcl.start = 0.0",
              f"stim[{i}].ptcl.npls = 1", f"stim[{i}].ptcl.duration = 1.0", f"stim[{i}].pulse.strength = 100.0"]
    return "\n".join(L) + "\n"


ats = {}
for label, fast in (("primary", True), ("secondary", False)):
    if label == "secondary" and A.primary_only:
        continue
    with open(os.path.join(wd, f"{label}.par"), "w") as f:
        f.write(par(fast))
    t0 = time.time()
    with open(os.path.join(wd, f"{label}.log"), "w") as fh:
        rc = subprocess.run([carpio.OPENCARP, "+F", f"{label}.par", "-simID", label], cwd=wd, stdout=fh,
                            stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS=str(A.omp)),
                            timeout=3600).returncode
    res[f"t_eik_{label}_s"] = time.time() - t0
    res[f"rc_{label}"] = rc
    try:
        at = carpio.read_igb(os.path.join(wd, label, "act.igb"))[-1].astype(np.float64)
    except Exception as e:  # noqa: BLE001
        res[f"err_{label}"] = repr(e)
        continue
    bad = ~np.isfinite(at) | (at < 0) | (at > 1e5)
    res[f"n_unactivated_{label}"] = int(bad.sum())
    good = at[~bad]
    res[f"TAT_{label}_ms"] = float(good.max())
    res[f"AT10_90_{label}_ms"] = float(np.quantile(good, 0.9) - np.quantile(good, 0.1))
    res[f"ATmean_{label}_ms"] = float(good.mean())
    ats[label] = at.astype(np.float32)

if A.save_at:
    np.savez_compressed(A.save_at, **ats)
res["ok"] = all(res.get(f"rc_{l}", 1) == 0 and f"TAT_{l}_ms" in res
                for l in (("primary",) if A.primary_only else ("primary", "secondary")))
res["t_total_s"] = time.time() - t_start
shutil.rmtree(wd, ignore_errors=True)
with open(A.out, "w") as f:
    json.dump(res, f)
print(json.dumps(res))
