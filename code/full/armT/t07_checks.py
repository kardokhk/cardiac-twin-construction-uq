"""Arm T follow-up checks 2026-10-28 (owner armT). Three checks, one heart per call, results as JSON on stdout-file.

  identity <hid>   (hid in REFS) heart into its own torso through the ICP and translation-only code paths that
                   t03_register.py lines 46-48 skipped. The registration and weight code below is copied verbatim from
                   t03_register.py (frame initialisation, eplib.icp, centroid translation, locate_and_interp on candidate
                   tets within 15 mm, ecg_weights). Cases:
                     a  eplib.icp (unchanged) from the frame initialisation  -> expected NaN (strict '<' 90th-percentile
                        trim keeps no pair when all distances are 0)
                     b  icp_le (same algorithm, trim 'd <= p90') from the frame initialisation
                     c  eplib.icp from a perturbed start (10 deg about (1,1,1)/sqrt3, translation (3,-2,4) mm)
                     d  icp_le from the same perturbed start
                   For every case with finite R, t: ICP path W (R, t) and translation path W (I, centroid shift), each
                   |W - W_own| / |W_own|, rotation angle and translation recovered.
  icpeq <hid>      eplib.icp versus icp_le on the real registrations (heart into each reference != heart): max |R - R'|,
                   |t - t'|, and reproduction of the stored reg/<RR>_icp R and t.
  axes <hid>       (KCL hearts) translation-only lead r in each reference torso when the KCL heart is read under axis
                   conventions LPS (as used), RAS (x, y flipped: proper 180 deg rotation about z), LAS and RPS (one axis
                   flipped: mirror). The heart is reoriented about its own centroid and its centroid placed where the
                   stored translation-only registration put it. Lead r as in t04 w_post: 1 - loss_corr(ref ECG, own ECG)
                   at the 3 theta* activation maps, median over theta*. Stiffness is invariant under orthogonal maps of
                   geometry and fibres together, so Ki is computed once.
Usage: python -I t07_checks.py <identity|icpeq|axes> <hid> <out.json>
"""
import json
import os
import sys
import time

import numpy as np
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402
eplib = L.eplib

mode, hid, outp = sys.argv[1:4]
t0 = time.time()


def icp_le(src, dst, R0, t0_, iters=50):
    """eplib.icp with the trim written 'd <= p90' (keeps all pairs when every distance is equal)."""
    tree = cKDTree(dst); R, t = R0, t0_
    for _ in range(iters):
        moved = src @ R.T + t
        d, j = tree.query(moved)
        keep = d <= np.percentile(d, 90)
        R, t = eplib.kabsch(src[keep], dst[j[keep]])
    moved = src @ R.T + t
    d, _ = tree.query(moved)
    return R, t, float(np.sqrt((d ** 2).mean()))


def rotmat(axis, deg):
    k = np.asarray(axis, float); k /= np.linalg.norm(k); a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


H = h5py.File(L.heart_file(hid), 'r')
C = H['cons']
P = C['pts'][:]; tets = C['tets'][:].astype(np.int64); fib = C['fib'][:].astype(np.float64)
sh = C['surf_nodes'][:]; Fh = H['frame'][:]; W_own = C['W_own'][:]
stored = {k: dict(R=H[f'reg/{k}/R'][:], t=H[f'reg/{k}/t'][:], W=H[f'reg/{k}/W'][:]) for k in H['reg']}
tmax = float(H['runs'].attrs['tmax_ms']); Tt = H['runs/target_T'][:]
H.close()


def ref_data(rid):
    with h5py.File(L.heart_file(rid), 'r') as Rh:
        Pr = Rh['cons/pts'][:]; sr = Rh['cons/surf_nodes'][:]; Fr = Rh['frame'][:]
    A = np.vstack(Fh[1:4]); B = np.vstack(Fr[1:4])
    R0 = B.T @ A
    if np.linalg.det(R0) < 0:
        R0 = B.T @ np.diag([1, 1, -1]) @ A
    t0v = Pr.mean(0) - R0 @ P.mean(0)
    return Pr, sr, R0, t0v


def torso(rid):
    with h5py.File(os.path.join(L.PILOT, f'heart{rid}.h5'), 'r') as Rp:
        Zr = Rp['Z'][:].astype(np.float64)
    d = np.load(L.npz_of(rid)); tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
    return Zr, tp, tt


def weights(moved, Ki, Zr, tp, tt):
    lo, hi = moved.min(0) - 15, moved.max(0) + 15
    cen = tp[tt].mean(1)
    near = np.all((cen > lo) & (cen < hi), axis=1)
    Zs, score = eplib.locate_and_interp(tp, tt[near], Zr, moved)
    return eplib.ecg_weights(Ki, Zs), int((score < -1e-6).sum())


def rel(W):
    return float(np.linalg.norm(W - W_own) / np.linalg.norm(W_own))


res = dict(mode=mode, heart=hid)
if mode == 'identity':
    assert hid in L.REFS
    Ki = eplib.stiffness(P, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
    Pr, sr, R0, t0v = ref_data(hid)
    Zr, tp, tt = torso(hid)
    Rp_ = rotmat([1, 1, 1], 10.0); tp_ = np.array([3.0, -2.0, 4.0])
    Rpert = Rp_ @ R0; tpert = Rp_ @ t0v + tp_ + (P.mean(0) - Rp_ @ P.mean(0))   # rotate about the centroid, then shift
    res['R0_minus_I_max'] = float(np.abs(R0 - np.eye(3)).max()); res['t0v_norm_mm'] = float(np.linalg.norm(t0v))
    res['perturbation'] = dict(rot_deg=10.0, axis=[1, 1, 1], shift_mm=tp_.tolist(),
                               start_rot_deg=L.rot_angle_deg(Rpert),
                               start_rms_mm=float(np.sqrt((cKDTree(Pr[sr]).query(P[sh] @ Rpert.T + tpert)[0] ** 2).mean())))
    cases = {'a_eplib_icp_frame_init': (eplib.icp, R0, t0v), 'b_icp_le_frame_init': (icp_le, R0, t0v),
             'c_eplib_icp_perturbed': (eplib.icp, Rpert, tpert), 'd_icp_le_perturbed': (icp_le, Rpert, tpert)}
    for name, (fn, Ra, ta) in cases.items():
        with np.errstate(all='ignore'):
            try:
                Ricp, ticp, rms = fn(P[sh], Pr[sr], Ra, ta)
            except Exception as e:      # noqa: BLE001
                res[name] = dict(error=str(e)[:200]); continue
        r = dict(finite=bool(np.isfinite(Ricp).all() and np.isfinite(ticp).all()))
        if r['finite']:
            cm = P.mean(0) @ Ricp.T + ticp
            Wi, oi = weights(P @ Ricp.T + ticp, Ki, Zr, tp, tt)
            tt_ = cm - P.mean(0)
            Wt, ot = weights(P + tt_, Ki, Zr, tp, tt)
            r.update(icp_rms_mm=rms, rot_deg=L.rot_angle_deg(Ricp), t_norm_mm=float(np.linalg.norm(ticp)),
                     trans_shift_mm=float(np.linalg.norm(tt_)), icp_rel_W=rel(Wi), trans_rel_W=rel(Wt),
                     icp_outside=oi, trans_outside=ot)
        res[name] = r
        print(name, r, f'{time.time()-t0:.0f}s', flush=True)
elif mode == 'icpeq':
    for rid in L.REFS:
        if rid == hid:
            continue
        Pr, sr, R0, t0v = ref_data(rid)
        Ra, ta, ra = eplib.icp(P[sh], Pr[sr], R0, t0v)
        Rb, tb, rb = icp_le(P[sh], Pr[sr], R0, t0v)
        s = stored[f'{rid}_icp']
        res[rid] = dict(eplib_vs_stored_R=float(np.abs(Ra - s['R']).max()), eplib_vs_stored_t=float(np.abs(ta - s['t']).max()),
                        le_vs_eplib_R=float(np.abs(Rb - Ra).max()), le_vs_eplib_t_mm=float(np.abs(tb - ta).max()),
                        le_vs_eplib_rot_deg=L.rot_angle_deg(Rb @ Ra.T), rms_eplib=ra, rms_le=rb)
        print(rid, res[rid], flush=True)
elif mode == 'axes':
    assert hid.startswith('K')
    Ki = eplib.stiffness(P, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
    eo = eplib.ECGModel(W_own, tmax=tmax)
    own_ecg = [eo(Tt[i]) for i in range(len(Tt))]
    conv = {'LPS_as_used': np.eye(3), 'RAS': np.diag([-1.0, -1.0, 1.0]), 'LAS_mirror': np.diag([1.0, -1.0, 1.0]),
            'RPS_mirror': np.diag([-1.0, 1.0, 1.0])}
    dj = json.load(open(os.path.join(L.DATA, f'kcl{hid[1:]}_convert.json')))
    cw = {k: np.array(v) for k, v in dj['centroids_world_mm'].items()}
    Pm = P.mean(0)
    for rid in L.REFS:
        Zr, tp, tt = torso(rid)
        s = stored[f'{rid}_trans']
        cm = Pm + s['t']
        out = {}
        for name, S in conv.items():
            moved = (P - Pm) @ S.T + cm
            W, o = weights(moved, Ki, Zr, tp, tt)
            ea = eplib.ECGModel(W, tmax=tmax)
            rs = [1 - float(eplib.loss_corr(ea(Tt[i]), own_ecg[i])) for i in range(len(Tt))]
            # anatomy under this reading (world = LPS-as-used since M_world_to_lps = I for all 5 KCL hearts)
            c = {k: S @ v for k, v in cw.items()}
            out[name] = dict(lead_r_per_theta=rs, lead_r_median=float(np.median(rs)), outside=o, rel_W_vs_own=rel(W),
                             heart_anterior_of_body_mm=float(c['body'][1] - c['heart'][1]),
                             lvblood_left_of_rvblood_mm=float(c['lv_blood'][0] - c['rv_blood'][0]),
                             reproduces_stored_W=(float(np.linalg.norm(W - s['W']) / np.linalg.norm(s['W']))
                                                  if name == 'LPS_as_used' else None))
            print(hid, rid, name, {k: v for k, v in out[name].items() if k != 'lead_r_per_theta'}, flush=True)
        res[rid] = out
    res['M_world_to_lps'] = dj['M_world_to_lps']; res['axis_purity'] = dj['axis_purity']
res['seconds'] = round(time.time() - t0, 1)
json.dump(res, open(outp, 'w'), indent=1)
print('done', mode, hid, res['seconds'])
