"""I13: forward ECG similarity between own-torso and reference-torso ECGs at theta* (same activation), per heart.
Usage: python -I 09_forward_similarity.py <D> <out.json>"""
import sys, os, json, numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplib
D, out = sys.argv[1:3]
refs = json.load(open(os.path.join(D, 'refs.json')))
res = {}
for h in [f'{i:02d}' for i in range(1, 18)]:
    C = h5py.File(os.path.join(D, 'cons', f'cons{h}.h5'), 'r')['c0ref']; R = h5py.File(os.path.join(D, 'runs', f'runs_h{h}_c0ref.h5'), 'r')
    T = R['T'][:]; k = R['kind'][:]
    e_own = eplib.ECGModel(C['W'][:], tmax=float(T.max()) + 20)
    for rn in ['R1', 'R2']:
        r = refs[rn]
        if r == h:
            continue
        z = np.load(os.path.join(D, 'ref', f'w_{h}_{r}.npz'))
        e_ref = eplib.ECGModel(z['W'], tmax=float(T.max()) + 20)
        cs, ar = [], []
        for i in np.nonzero(k == 1)[0]:
            a, b = e_own(T[i]), e_ref(T[i])
            cs.append([np.corrcoef(a[l], b[l])[0, 1] for l in range(8)])
            ar.append(np.sqrt((b ** 2).sum() / (a ** 2).sum()))
        res[f'{h}_{rn}'] = dict(mean_lead_corr=float(np.mean(cs)), min_lead_corr=float(np.min(cs)), rms_amp_ratio=float(np.mean(ar)),
                                rot_deg=float(z['rot_deg']), icp_rms=float(z['icp_rms']))
json.dump(res, open(out, 'w'), indent=1)
v = np.array([[d['mean_lead_corr'], d['rms_amp_ratio'], d['rot_deg']] for d in res.values()])
print('n', len(v), 'mean lead corr median %.3f IQR %.3f-%.3f; amp ratio median %.2f range %.2f-%.2f; rotation median %.1f range %.1f-%.1f deg' % (
    np.median(v[:, 0]), *np.percentile(v[:, 0], [25, 75]), np.median(v[:, 1]), v[:, 1].min(), v[:, 1].max(), np.median(v[:, 2]), v[:, 2].min(), v[:, 2].max()))
print('corr(mean lead corr, rotation) = %.2f' % np.corrcoef(v[:, 0], v[:, 2])[0, 1])
