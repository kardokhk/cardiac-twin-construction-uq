"""Arm T per-heart preparation (owner armT, 2026-10-28). One heart in, one HDF5 out (results/2026-10-28_armT/data/heart_<id>.h5).

Native heart frame, root nodes, rule-v2 electrodes and own-torso lead fields:
  Malik hearts: taken from the I13 pilot (data/processed/pilot_I13/heartNN.h5, built by code/pilot/I13/02_prep_heart.py
  and 02b_leadfields_v2.py), read only.
  KCL hearts: computed here with the same code (frame and root rule copied from 02_prep_heart.py, electrodes
  eplib.place_electrodes_v2 with the bone-based spine proxy from t01, lead fields eplib.solve_leadfields).
Construction (all 22 hearts identical): voxelise the deposited ventricles (tags 20-23) at 1.5 mm in the heart frame
(copied from code/pilot/I11/06_constructions.py, reference construction c0ref only), voxel -> 6 tets, Bayer fibres
seeded, A2 fast layer, roots mapped, own-torso ECG weights W_own.
Usage: python -I t02_prep.py <hid>
"""
import json
import os
import sys
import time

import numpy as np
import h5py
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402
eplib, carpio = L.eplib, L.carpio

hid = sys.argv[1]
VOX = L.VOX
t0 = time.time()


def log(*a):
    print(f'[{hid} {time.time() - t0:6.0f}s]', *a, flush=True)


d = np.load(L.npz_of(hid))
tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64); tg = d['tags']
meta = dict(heart=hid, patient_id_seed=L.pid_of(hid), group=L.group_of(hid))

if hid.startswith('K'):
    # ---------------- native frame and roots (verbatim logic of code/pilot/I13/02_prep_heart.py)
    myo_m = np.isin(tg, [22, 23])
    mt = tt[myo_m]
    gnodes, inv = np.unique(mt, return_inverse=True)
    P = tp[gnodes]
    lvb = np.unique(tt[tg == 21]); rvb = np.unique(tt[tg == 20])
    lv_endo_g = np.intersect1d(np.unique(tt[tg == 23]), lvb)
    rv_endo_g = np.intersect1d(np.unique(tt[np.isin(tg, [22, 23])]), rvb)
    assert len(lv_endo_g) > 1000 and len(rv_endo_g) > 1000, (len(lv_endo_g), len(rv_endo_g))
    lv_endo = np.searchsorted(gnodes, lv_endo_g); rv_endo = np.searchsorted(gnodes, rv_endo_g)
    lvn = np.searchsorted(gnodes, np.unique(tt[tg == 23]))
    c_lv = P[lvn].mean(0)
    w, V = np.linalg.eigh(np.cov((P[lvn] - c_lv).T))
    ax = V[:, -1]; ax = ax if ax @ np.array([1, -1, -1]) > 0 else -ax
    s = (P - c_lv) @ ax
    s_ap, s_bs = s[lvn].max(), s[lvn].min()
    h = (s_ap - s) / (s_ap - s_bs)
    c_lvb = tp[lvb].mean(0); c_rvb = tp[rvb].mean(0)
    sep = (c_rvb - c_lvb); sep -= (sep @ ax) * ax; sep /= np.linalg.norm(sep)
    ant = np.cross(ax, sep); ant = ant if ant[1] < 0 else -ant

    def pick(cands, hval, direction):
        m = cands[np.abs(h[cands] - hval) < 0.05]
        v = P[m] - c_lv; v -= np.outer(v @ ax, ax)
        return m[np.argmax(v @ direction)]
    roots = [pick(lv_endo, 0.50, sep), pick(lv_endo, 0.65, (sep + ant) / np.sqrt(2)),
             pick(lv_endo, 0.33, (sep - ant) / np.sqrt(2)), pick(rv_endo, 0.40, -sep)]
    frame = np.vstack([c_lv, ax, sep, ant]); roots_xyz = P[np.array(roots)]
    heart_centroid = P.mean(0)
    myo_native_mL = float(np.abs(carpio.tet_volumes(tp, mt)).sum() / 1000)
    del mt, inv
    # ---------------- electrodes (rule v2) and own-torso lead fields
    bf = eplib.boundary_faces(tt)
    xyz, en = eplib.place_electrodes_v2(tp, bf, heart_centroid, d['spine_pts'].astype(np.float64))
    log('electrodes v2 relative to heart (mm):', dict(zip(eplib.ELECS, np.round(xyz - heart_centroid).astype(int).tolist())))
    K = eplib.stiffness(tp, tt, sigma=eplib.SIG_BULK)
    np.random.seed(L.FIB_SEED)          # pyamg spectral-radius start vector (affects iterations only, tol 1e-10)
    Z = np.column_stack([eplib.solve_leadfields(K, en[:-1], en[-1], verbose=False), np.zeros(len(tp))])
    del K
    torso_mL = float(np.abs(carpio.tet_volumes(tp, tt)).sum() / 1000)
    elec_xyz = xyz
    src = 'computed (KCL)'
    log('lead fields done')
else:
    with h5py.File(os.path.join(L.PILOT, f'heart{hid}.h5'), 'r') as H:
        assert H.attrs['electrode_rule'] == 'v2'
        frame = H['frame'][:]; roots_xyz = H['pts'][:][H['roots'][:]]; elec_xyz = H['elec_xyz'][:]
        Z = H['Z'][:].astype(np.float64); torso_mL = float(H.attrs['torso_volume_mL'])
        heart_centroid = H.attrs['heart_centroid']; myo_native_mL = float(H.attrs['myo_volume_mL'])
    src = 'I13 pilot heartNN.h5 (rule v2)'

# ---------------- voxelise deposited ventricles at VOX in heart frame (code/pilot/I11/06_constructions.py)
c = frame[0]; Rf = np.vstack(frame[1:4])
sel = np.isin(tg, [20, 21, 22, 23])
ht, htag = tt[sel], tg[sel]
LAB = {23: 1, 22: 2, 21: 3, 20: 4}
U = (tp[np.unique(ht)] - c) @ Rf.T
lo = np.floor(U.min(0) / VOX) - 4; hi = np.ceil(U.max(0) / VOX) + 4
shape = tuple((hi - lo).astype(int) + 1)
g = (np.stack(np.meshgrid(*[np.arange(s_) for s_ in shape], indexing='ij'), -1).reshape(-1, 3) + lo) * VOX
X = g @ Rf + c
cent = tp[ht].mean(1); tree = cKDTree(cent)
lab = np.zeros(len(X), np.int8)
bb = tree.query(X, k=1)[0] < 3.0
idx = np.nonzero(bb)[0]
_, cand = tree.query(X[idx], k=16)
best = np.full(len(idx), -np.inf); bl = np.zeros(len(idx), np.int8)
lut = np.zeros(256, np.int8)
for k_, v_ in LAB.items():
    lut[k_] = v_
for k in range(16):
    Tk = ht[cand[:, k]]; Xk = tp[Tk]
    D = np.stack([Xk[:, 1] - Xk[:, 0], Xk[:, 2] - Xk[:, 0], Xk[:, 3] - Xk[:, 0]], axis=2)
    lam = np.linalg.solve(D, (X[idx] - Xk[:, 0])[:, :, None])[:, :, 0]
    s_ = np.minimum(1 - lam.sum(1), lam.min(1))
    up = s_ > best; best[up] = s_[up]
    bl[up] = lut[htag[cand[up, k]]]
lab[idx] = np.where(best > -1e-6, bl, 0)
lab = lab.reshape(shape)
del X, g, cent, tree, cand
log('voxel grid', shape, {k: int((lab == v).sum()) for k, v in LAB.items()})
pilot_match = None
if not hid.startswith('K'):
    with h5py.File(os.path.join(L.PILOT, 'cons', f'cons{hid}.h5'), 'r') as C:
        pl = C['c0ref']['lab'][:]
    pilot_match = float((pl == lab).mean()) if pl.shape == lab.shape else -1.0
    log('label agreement with pilot c0ref', pilot_match)

# ---------------- mesh (06_constructions.mesh_from_labels, verbatim algebra)
myo = (lab == 1) | (lab == 2)
cc_, ncc = ndi.label(myo)
n_myo = int(myo.sum())
if ncc > 1:
    sizes = ndi.sum(myo, cc_, range(1, ncc + 1)); myo = cc_ == (1 + int(np.argmax(sizes)))
vi = np.argwhere(myo)
shp = np.array(lab.shape) + 1
corner = lambda a, b, cc: ((vi[:, 0] + a) * shp[1] + (vi[:, 1] + b)) * shp[2] + (vi[:, 2] + cc)   # noqa: E731
cn = [corner(a, b, cc) for a in (0, 1) for b in (0, 1) for cc in (0, 1)]
paths = [(4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3)]
tets = np.concatenate([np.column_stack([cn[0], cn[p], cn[q], cn[7]]) for p, q in paths])
used, inv = np.unique(tets, return_inverse=True)
tets = inv.reshape(tets.shape)
ijk = np.column_stack(np.unravel_index(used, tuple(shp))).astype(np.float64) - 0.5
pts = ((ijk + lo) * VOX) @ Rf + c
tets = carpio.fix_orientation(pts, tets)


def corners_of(mask):
    v = np.argwhere(mask)
    return np.unique(np.concatenate([((v[:, 0] + a) * shp[1] + (v[:, 1] + b)) * shp[2] + (v[:, 2] + cc)
                                     for a in (0, 1) for b in (0, 1) for cc in (0, 1)]))


iface = np.isin(used, corners_of((lab == 3) | (lab == 4)))
lv_if = np.isin(used, corners_of(lab == 3))
tet_lab = np.tile(lab[vi[:, 0], vi[:, 1], vi[:, 2]].astype(np.int8), 6)
log('mesh', len(pts), 'nodes', len(tets), 'tets', 'kept', round(len(vi) / n_myo, 4))

# ---------------- Bayer fibres, seeded (as code/gates/mesh/acdc_lib.bayer, own heart frame)
import bayer_i09  # noqa: E402
ax = frame[1]
s = (pts - c) @ ax
bfh = eplib.boundary_faces(tets)
bn = np.unique(bfh)
lvn_ = np.nonzero(lv_if)[0]; rvn_ = np.nonzero(iface & ~lv_if)[0]
apex = np.nonzero(s >= s.max() - 0.75 * VOX)[0]
base = bn[s[bn] <= s.min() + 0.75 * VOX]
np.random.seed(L.FIB_SEED)
bo, bmeta = bayer_i09.bayer_fibres(pts, tets, lvn_, rvn_, bn, apex, base)
fib = bo['fib']; nonfin = int(bo['nonfinite'])
log('fibres', 'nonfinite', nonfin, 'apex', len(apex), 'base', len(base))

# ---------------- A2 fast endocardial layer: element centroid within 1.5 mm of the LV or RV cavity surface
endo_tris = bfh[iface[bfh].all(1)]
dcen = L.dist_to_surface(pts[tets].mean(1), pts, endo_tris)
etag = np.where(dcen <= L.D_FAST, 2, 1).astype(np.int8)
etag_1vox = np.where(iface[tets].any(1), 2, 1)
log('fast layer A2 share', round(float((etag == 2).mean()), 4), 'one-voxel share', round(float((etag_1vox == 2).mean()), 4))

# ---------------- roots mapped (06_constructions)
lvn = np.nonzero(lv_if)[0]; rvn = np.nonzero(iface & ~lv_if)[0]
rnodes = np.array([lvn[cKDTree(pts[lvn]).query(r)[1]] for r in roots_xyz[:3]] + [rvn[cKDTree(pts[rvn]).query(roots_xyz[3])[1]]])
stim = np.unique(np.concatenate([cKDTree(pts).query_ball_point(pts[r], L.R_ROOT) for r in rnodes]))
root_shift = np.linalg.norm(pts[rnodes] - roots_xyz, axis=1)

# ---------------- own-torso ECG weights
_cen = tp[tt].mean(1); _lo, _hi = pts.min(0) - 25, pts.max(0) + 25
tt_near = tt[np.all((_cen > _lo) & (_cen < _hi), axis=1)]; del _cen
Zs, sc = eplib.locate_and_interp(tp, tt_near, Z, pts)
Ki = eplib.stiffness(pts, tets, fib=fib.astype(np.float64), sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
W_own = eplib.ecg_weights(Ki, Zs)
outside = int((sc < -1e-6).sum())
lvmyo = lab == 1
edt = ndi.distance_transform_edt(lvmyo, sampling=VOX)
zz = np.nonzero(lvmyo.any((1, 2)))[0]
myo_mL = float(myo.sum()) * VOX ** 3 / 1000

with h5py.File(L.heart_file(hid), 'w') as f:
    g_ = f.create_group('cons')
    g_['pts'] = pts; g_['tets'] = tets.astype(np.int32); g_['etag'] = etag; g_['fib'] = fib.astype(np.float32)
    g_['stim'] = stim.astype(np.int32); g_['roots'] = rnodes.astype(np.int32); g_['W_own'] = W_own
    g_['lv_endo'] = lvn.astype(np.int32); g_['rv_endo'] = rvn.astype(np.int32); g_['tet_lab'] = tet_lab
    g_['surf_nodes'] = bn.astype(np.int32)
    g_.create_dataset('lab', data=lab, compression='gzip', compression_opts=4)
    f['frame'] = frame; f['roots_xyz_native'] = roots_xyz; f['elec_xyz'] = elec_xyz
    f.attrs.update(meta)
    f.attrs.update(source=src, torso_volume_mL=torso_mL, myo_native_mL=myo_native_mL, myo_cons_mL=myo_mL,
                   heart_centroid=heart_centroid, vox=VOX, n_nodes=len(pts), n_tets=len(tets),
                   fast_share_A2=float((etag == 2).mean()), fast_share_1vox=float((etag_1vox == 2).mean()),
                   fib_nonfinite=nonfin, bayer_meta=json.dumps(bmeta), fib_seed=L.FIB_SEED,
                   root_shift_mm=root_shift, outside_nodes_own=outside, kept_frac=len(vi) / n_myo,
                   L_wall=float(4 * edt[lvmyo].mean()), L_base=float(zz.max() - zz.min() + 1) * VOX,
                   pilot_label_agreement=-2.0 if pilot_match is None else pilot_match,
                   electrode_rule='v2', v_fib=L.V_FIB, d_fast_mm=L.D_FAST)
log('done', L.heart_file(hid))
