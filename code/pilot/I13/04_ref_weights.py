"""Place heart <hid> in reference torso <rid> (rigid registration onto the reference torso's own ventricles,
as in reference-torso pipelines such as Qian 2025) and compute its ECG weights with the reference torso's
lead fields and rule-placed electrodes.  Also supports the identity check (hid == rid), which exercises the
registration + interpolation path and must reproduce W_own.
Usage: python -I 04_ref_weights.py <h5dir> <consdir> <npzdir> <hid> <rid> <out.npz>
(operates on the 1.5 mm voxel reference construction c0ref of each heart)
"""
import sys, os, time
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplib
from scipy.spatial import cKDTree

h5dir, consdir, npzdir, hid, rid, out = sys.argv[1:7]
t0 = time.time()
H = h5py.File(os.path.join(h5dir, f'heart{hid}.h5'), 'r'); Rf = h5py.File(os.path.join(h5dir, f'heart{rid}.h5'), 'r')
CH = h5py.File(os.path.join(consdir, f'cons{hid}.h5'), 'r')['c0ref']; CR = h5py.File(os.path.join(consdir, f'cons{rid}.h5'), 'r')['c0ref']
P = CH['pts'][:]; tets = CH['tets'][:].astype(np.int64); fib = CH['fib'][:].astype(np.float64)
Fh = H['frame'][:]; Fr = Rf['frame'][:]
Pr = CR['pts'][:]
W_own = CH['W'][:]
# surfaces
sh = np.unique(eplib.boundary_faces(tets)); sr = np.unique(eplib.boundary_faces(CR['tets'][:].astype(np.int64)))
A = np.vstack(Fh[1:4]); B = np.vstack(Fr[1:4])          # rows: axis, septal, anterior
R0 = B.T @ A
if np.linalg.det(R0) < 0:
    R0 = B.T @ np.diag([1, 1, -1]) @ A
t0v = Pr.mean(0) - R0 @ P.mean(0)
R, t, rms = eplib.icp(P[sh], Pr[sr], R0, t0v)
if os.environ.get('TRANSLATION_ONLY'):
    # decomposition arm: keep the heart's own orientation, translate its centroid onto the registered centroid
    cm = P.mean(0) @ R.T + t
    R = np.eye(3); t = cm - P.mean(0)
    rms = float(np.sqrt((cKDTree(Pr[sr]).query(P[sh] + t)[0] ** 2).mean()))
ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
moved = P @ R.T + t
print(f'heart {hid} -> torso {rid}: ICP surface RMS {rms:.2f} mm, rotation {ang:.1f} deg, '
      f'translation {np.linalg.norm(t + R @ P.mean(0) - P.mean(0)):.1f} mm', flush=True)
# reference torso lead field at moved heart nodes
d = np.load(os.path.join(npzdir, f'malik{rid}.npz'))
tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
lo, hi = moved.min(0) - 15, moved.max(0) + 15
cen = tp[tt].mean(1)
near = np.all((cen > lo) & (cen < hi), axis=1)
Zr = Rf['Z'][:].astype(np.float64)
Zs, score = eplib.locate_and_interp(tp, tt[near], Zr, moved)
outside = int((score < -1e-6).sum())
Ki = eplib.stiffness(P, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
W = eplib.ecg_weights(Ki, Zs)
rel = float(np.linalg.norm(W - W_own) / np.linalg.norm(W_own))
np.savez(out, W=W, R=R, t=t, icp_rms=rms, rot_deg=ang, outside=outside, rel_diff_vs_own=rel,
         elec_xyz_ref=Rf['elec_xyz'][:], centroid_moved=moved.mean(0))
print(f'  outside-tet nodes {outside}, |W_ref - W_own|/|W_own| = {rel:.3f}, {time.time()-t0:.0f}s', flush=True)
