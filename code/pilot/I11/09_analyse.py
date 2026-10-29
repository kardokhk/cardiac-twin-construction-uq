"""Pilot I11 analysis (closes D3).  For each heart and each alternative construction c (c1..c6):
  r_c = mean over the 3 theta* draws of theta_hat_c / theta_hat_ref   (per parameter, per loss)
  l_c = L_c / L_ref for L in (L_scale, L_wall, L_base)                 (mask-readable lengths)
Rules fitted on the pooled alternative constructions:
  R1 single length, slope 1 (idea rung 2):  r = l_scale             (R^2 against the mean of r)
  R2 single length, fitted:                 r = a + b l_scale
  R3 three-length linear rule (red team):   r = a + b1 l_scale + b2 l_wall + b3 l_base
R^2 in-sample (pre-registered statistic), adjusted R^2, leave-one-heart-out predictive R^2; 95% CIs by cluster
bootstrap over hearts (2,000 resamples, seed 11011).  Pre-registered: I11's 12-lead arm dies if R3 gives R^2 >= 0.9
for both parameters.  Primary loss declared before data: correlation-based; amplitude loss reported alongside.
Spread: per-heart between-construction SD of log theta_hat (alternatives) vs noise = SD of reference-construction
recovery error (theta_hat_ref - theta*)/theta*.
Usage: python -I 09_analyse.py <calib_dir> <out.json>
"""
import sys, os, json, glob
import numpy as np
cdir, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(11011)
B_LO, B_HI = np.array([0.8, 0.15]), np.array([3.0, 0.5])
LK = ['L_scale', 'L_wall', 'L_base']
files = sorted(glob.glob(os.path.join(cdir, 'calib*.json')))
res = dict(n_hearts=len(files))


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b, A @ b


for loss in ['corr', 'amp']:
    rows = []   # heart, construction, r(2), l(3), bound flag
    noise = []
    spread = {}
    for f in files:
        J = json.load(open(f)); hid = J['hid']; C = J['constructions']
        ref = {tuple(np.round(c['theta_star'], 6)): np.array(c['theta_hat']) for c in C['c0ref']['calib'] if c['loss'] == loss}
        for ts, th in ref.items():
            noise.append((th - np.array(ts)) / np.array(ts))
        Lref = np.array([C['c0ref']['attrs'][k] for k in LK])
        logs = []
        for n, c in C.items():
            if n == 'c0ref':
                continue
            rr, bnd = [], False
            for cc in c['calib']:
                if cc['loss'] != loss:
                    continue
                th = np.array(cc['theta_hat']); ts = tuple(np.round(cc['theta_star'], 6))
                bnd |= bool(np.any(np.isclose(th, B_LO, rtol=1e-3) | np.isclose(th, B_HI, rtol=1e-3)))
                rr.append(th / ref[ts])
            rr = np.array(rr)
            logs.append(np.log(rr).mean(0))
            rows.append(dict(hid=hid, c=n, r=rr.mean(0), l=np.array([c['attrs'][k] for k in LK]) / Lref, bound=bnd))
        spread[hid] = np.std(np.array(logs), axis=0, ddof=1)
    hids = sorted({r['hid'] for r in rows})
    H = np.array([r['hid'] for r in rows]); Y = np.array([r['r'] for r in rows]); X = np.array([r['l'] for r in rows])
    noise = np.array(noise); sdn = noise.std(0, ddof=1)
    out_l = dict(n_constructions=len(rows), n_hearts=len(hids), n_bound_hits=int(sum(r['bound'] for r in rows)),
                 noise_sd=sdn.tolist(), noise_mean=noise.mean(0).tolist(),
                 length_ratio_cv=(X.std(0, ddof=1)).tolist())
    for j, p in enumerate(['v_endo', 'v_t']):
        y = Y[:, j]
        sp = np.array([spread[h][j] for h in hids])
        d = dict(ratio_mean=float(y.mean()), ratio_sd=float(y.std(ddof=1)),
                 per_heart_between_construction_cv=sp.round(4).tolist(),
                 median_between_construction_cv=float(np.median(sp)),
                 hearts_spread_ge_2x_noise=int((sp >= 2 * sdn[j]).sum()))

        def stats_(idx):
            yy, XX, HH = y[idx], X[idx], H[idx]
            s = {}
            s['R1_slope1'] = r2(yy, XX[:, 0])
            s['R2_fitted1'] = r2(yy, fit(XX[:, :1], yy)[1])
            b3, yh3 = fit(XX, yy)
            s['R3_three_length'] = r2(yy, yh3)
            n_, k_ = len(yy), 3
            s['R3_adj'] = 1 - (1 - s['R3_three_length']) * (n_ - 1) / (n_ - k_ - 1)
            # leave-one-heart-out predictive R^2
            pred = np.empty_like(yy)
            for h in np.unique(HH):
                tr = HH != h
                b, _ = fit(XX[tr], yy[tr])
                pred[~tr] = np.column_stack([np.ones((~tr).sum()), XX[~tr]]) @ b
            s['R3_LOHO'] = r2(yy, pred)
            s['R3_coef'] = b3.tolist()
            return s
        full = stats_(np.arange(len(y)))
        bs = []
        for _ in range(2000):
            pick = rng.choice(hids, len(hids), replace=True)
            idx = np.concatenate([np.nonzero(H == h)[0] for h in pick])
            try:
                bs.append(stats_(idx))
            except Exception:
                pass
        for k in ['R1_slope1', 'R2_fitted1', 'R3_three_length', 'R3_adj', 'R3_LOHO']:
            v = np.array([b[k] for b in bs])
            d[k] = float(full[k]); d[k + '_ci'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        d['R3_coef'] = full['R3_coef']
        nb = np.array([not r['bound'] for r in rows])
        if nb.sum() > 8:
            sub = stats_(np.nonzero(nb)[0])
            d['R3_three_length_no_bound_hits'] = float(sub['R3_three_length']); d['n_no_bound_hits'] = int(nb.sum())
        out_l[p] = d
    out_l['kill_three_length_both_ge_0.9'] = bool(out_l['v_endo']['R3_three_length'] >= 0.9 and out_l['v_t']['R3_three_length'] >= 0.9)
    res[loss] = out_l
json.dump(res, open(out, 'w'), indent=1)
for loss in ['corr', 'amp']:
    o = res[loss]
    print(f"[{loss}] constructions {o['n_constructions']} over {o['n_hearts']} hearts; bound hits {o['n_bound_hits']}; "
          f"noise SD {np.round(np.array(o['noise_sd'])*100,2)}%; length-ratio SD {np.round(np.array(o['length_ratio_cv'])*100,2)}%")
    for p in ['v_endo', 'v_t']:
        d = o[p]
        print(f"  {p}: ratio {d['ratio_mean']:.3f} (SD {d['ratio_sd']*100:.2f}%), median between-cons CV {d['median_between_construction_cv']*100:.2f}%, "
              f"hearts spread>=2xnoise {d['hearts_spread_ge_2x_noise']}/{o['n_hearts']}")
        for k in ['R1_slope1', 'R2_fitted1', 'R3_three_length', 'R3_adj', 'R3_LOHO']:
            print(f"     {k}: {d[k]:.3f} [{d[k+'_ci'][0]:.3f}, {d[k+'_ci'][1]:.3f}]")
        if 'n_no_bound_hits' in d:
            print(f"     R3 excluding bound-hit constructions (n={d['n_no_bound_hits']}): {d['R3_three_length_no_bound_hits']:.3f}")
    print('  KILL (R3 >= 0.9 both):', o['kill_three_length_both_ge_0.9'])
