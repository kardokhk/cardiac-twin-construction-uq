"""Recompute ECG weights W of every construction of one heart with the heart's current own-torso lead field Z
(rule v2); rule-v1 weights kept as W_rule1.  Meshes are unchanged.  Usage: python -I 06b_reweight.py <heart.h5> <cons.h5> <malik.npz>"""
import sys, os, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I13'))
import eplib
hf, cf, npz = sys.argv[1:4]
H = h5py.File(hf, 'r'); Z = H['Z'][:].astype(np.float64); P0 = H['pts'][:]
d = np.load(npz); tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
cen = tp[tt].mean(1); lo, hi = P0.min(0) - 25, P0.max(0) + 25
tn = tt[np.all((cen > lo) & (cen < hi), axis=1)]
with h5py.File(cf, 'a') as C:
    for n in C:
        g = C[n]
        pts = g['pts'][:]; tets = g['tets'][:].astype(np.int64); fib = g['fib'][:].astype(np.float64)
        Zs, sc = eplib.locate_and_interp(tp, tn, Z, pts)
        Ki = eplib.stiffness(pts, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
        W = eplib.ecg_weights(Ki, Zs)
        if 'W_rule1' not in g:
            g.move('W', 'W_rule1')
        else:
            del g['W']
        g['W'] = W
        print(os.path.basename(cf), n, 'rel change vs rule1 %.3f' % (np.linalg.norm(W - g['W_rule1'][:]) / np.linalg.norm(g['W_rule1'][:])), flush=True)
