"""Arm T registration of one heart into the reference torsos (owner armT, 2026-10-28). Logic of the I13 pilot
code/pilot/I13/04_ref_weights.py, applied to the arm T constructions:
  icp   : full rigid registration. Initial rotation maps the heart frame (axis, septal, anterior) onto the reference
          heart's frame (det fixed to +1 as in the pilot), then ICP (eplib.icp, 50 iterations, 90th-percentile trimming)
          of the construction surface onto the reference construction's surface.
  trans : orientation-preserving (PRIMARY, plan section 3): the heart keeps its own orientation; its centroid is
          translated onto the ICP-registered centroid.
ECG weights W = K_i Z_ref(moved nodes), with Z_ref the reference torso's rule-v2 lead fields (I13 pilot heartRR.h5).
A heart is never its own reference; for hid in REFS the identity check (heart into its own torso through the same
code path) is stored instead, and must reproduce W_own.
Writes group reg/<RR>_<icp|trans> into results/2026-10-28_armT/data/heart_<hid>.h5.
Usage: python -I t03_register.py <hid>
"""
import os
import sys
import time

import numpy as np
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402
eplib = L.eplib

hid = sys.argv[1]
t0 = time.time()
H = h5py.File(L.heart_file(hid), 'r')
C = H['cons']
P = C['pts'][:]; tets = C['tets'][:].astype(np.int64); fib = C['fib'][:].astype(np.float64)
sh = C['surf_nodes'][:]; Fh = H['frame'][:]; W_own = C['W_own'][:]
H.close()
Ki = eplib.stiffness(P, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
res = {}
for rid in L.REFS:
    with h5py.File(L.heart_file(rid), 'r') as Rh:
        Pr = Rh['cons/pts'][:]; sr = Rh['cons/surf_nodes'][:]; Fr = Rh['frame'][:]
    with h5py.File(os.path.join(L.PILOT, f'heart{rid}.h5'), 'r') as Rp:
        Zr = Rp['Z'][:].astype(np.float64); elec_ref = Rp['elec_xyz'][:]
    d = np.load(L.npz_of(rid)); tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
    A = np.vstack(Fh[1:4]); B = np.vstack(Fr[1:4])
    R0 = B.T @ A
    if np.linalg.det(R0) < 0:
        R0 = B.T @ np.diag([1, 1, -1]) @ A
    t0v = Pr.mean(0) - R0 @ P.mean(0)
    if rid == hid:      # identity check: the heart's own construction in its own torso, no ICP (ICP on identical
        Ricp, ticp, rms = np.eye(3), np.zeros(3), 0.0      # point sets trims every pair and returns NaN)
    else:
        Ricp, ticp, rms = eplib.icp(P[sh], Pr[sr], R0, t0v)
    ang = L.rot_angle_deg(Ricp)
    cm = P.mean(0) @ Ricp.T + ticp
    variants = {'icp': (Ricp, ticp, rms)}
    if rid != hid:
        tt_ = cm - P.mean(0)
        variants['trans'] = (np.eye(3), tt_, float(np.sqrt((cKDTree(Pr[sr]).query(P[sh] + tt_)[0] ** 2).mean())))
    for reg, (R, t, rms_) in variants.items():
        moved = P @ R.T + t
        lo, hi = moved.min(0) - 15, moved.max(0) + 15
        cen = tp[tt].mean(1)
        near = np.all((cen > lo) & (cen < hi), axis=1)
        Zs, score = eplib.locate_and_interp(tp, tt[near], Zr, moved)
        W = eplib.ecg_weights(Ki, Zs)
        rel = float(np.linalg.norm(W - W_own) / np.linalg.norm(W_own))
        key = f'{rid}_{reg}' if rid != hid else f'{rid}_self'
        res[key] = dict(W=W, R=R, t=t, icp_rms=rms_, rot_deg_icp=ang, rot_deg_applied=L.rot_angle_deg(R),
                        outside=int((score < -1e-6).sum()), rel_diff_vs_own=rel,
                        translation_mm=float(np.linalg.norm(moved.mean(0) - P.mean(0))), elec_xyz_ref=elec_ref)
        print(f'heart {hid} -> torso {rid} [{reg}]: surface RMS {rms_:.2f} mm, ICP rotation {ang:.1f} deg, '
              f'outside {res[key]["outside"]}, |W-W_own|/|W_own| {rel:.3f}, {time.time()-t0:.0f}s', flush=True)
with h5py.File(L.heart_file(hid), 'a') as f:
    if 'reg' in f:
        del f['reg']
    for k, v in res.items():
        g = f.create_group(f'reg/{k}')
        g['W'] = v.pop('W'); g['R'] = v.pop('R'); g['t'] = v.pop('t'); g['elec_xyz_ref'] = v.pop('elec_xyz_ref')
        g.attrs.update(v)
print('done', hid, f'{time.time()-t0:.0f}s')
