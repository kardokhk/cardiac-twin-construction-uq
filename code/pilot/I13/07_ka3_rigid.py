"""KA-3 (I13): rigid motion of heart, torso and electrodes together leaves the ECG unchanged.
Rotates and translates the whole heart-in-torso mesh of one heart, re-solves the lead fields with the same
electrode nodes, recomputes ECG weights, and compares 8-lead ECGs from one activation map (first design run
or eikonal-free synthetic activation = distance from root nodes / 0.5 m/s).  Acceptance: RMS difference <= 0.1%.
Usage: python -I 07_ka3_rigid.py <heart.h5> <cons.h5> <malik.npz> <out.json>
"""
import sys, os, json
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplib
from scipy.spatial import cKDTree
hf, cf, npz, out = sys.argv[1:5]
H = h5py.File(hf, 'r'); C = h5py.File(cf, 'r')['c0ref']
d = np.load(npz); tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
a, b, c = 0.4, -0.7, 1.1
Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
R = Rz @ Ry @ Rx; t = np.array([123.0, -45.0, 300.0])
tp2 = tp @ R.T + t
en = H['elec_nodes'][:]
K = eplib.stiffness(tp2, tt, sigma=eplib.SIG_BULK)
Z = np.column_stack([eplib.solve_leadfields(K, en[:-1], en[-1], verbose=False), np.zeros(len(tp))])
P = C['pts'][:] @ R.T + t; fib = C['fib'][:].astype(np.float64) @ R.T
ct = C['tets'][:].astype(np.int64)
Ki = eplib.stiffness(P, ct, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
Zs, _ = eplib.locate_and_interp(tp2, tt, Z, P)
W2 = eplib.ecg_weights(Ki, Zs)
W1 = C['W'][:]
P0 = C['pts'][:]
ta = cKDTree(P0[C['stim'][:]]).query(P0)[0] / 0.5
e1 = eplib.ECGModel(W1, tmax=ta.max() + 20)(ta); e2 = eplib.ECGModel(W2, tmax=ta.max() + 20)(ta)
r = dict(rms_rel_diff=float(np.sqrt(((e1 - e2) ** 2).mean()) / np.sqrt((e1 ** 2).mean())),
         W_rel_diff=float(np.linalg.norm(W1 - W2) / np.linalg.norm(W1)))
print(r); json.dump(r, open(out, 'w'), indent=1)
