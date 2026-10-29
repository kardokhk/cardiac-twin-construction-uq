"""Pilot I13 analysis: reference-torso bias in calibrated (v_endo, v_t).
Per heart h, reference arm B, loss L, parameter p:
  e_A = (theta_hat_A - theta*)/theta*  (own torso; emulator + calibration error, the noise floor)
  e_B = (theta_hat_B - theta*)/theta*  (reference torso)
  per-heart bias b_h = mean over the 3 theta* draws of e_B;  paired bias d_h = mean of (e_B - e_A).
Population: mean b_h with 95% CI (t interval and percentile bootstrap over hearts, 10,000 resamples, seed 13013).
Noise: SD of e_A over all hearts and draws.  Red-team criterion (pilot spec): |b_h| >= 2 SD(noise) for at least
one parameter in at least one heart under at least one loss.  Population criterion (pilot specification): the 95% CI of
mean |b_h| lies above 2 SD(noise) for at least one parameter.
Usage: python -I 08_analyse.py <calib_dir> <out.json>
"""
import sys, os, json, glob
import numpy as np
from scipy import stats
cdir, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(13013)
files = sorted(glob.glob(os.path.join(cdir, 'calib*.json')))
rows = []
for f in files:
    r = json.load(open(f)); hid = os.path.basename(f)[5:7]
    for c in r['calib']:
        th = np.array(c['theta_hat'])
        bnd = bool(np.any(np.isclose(th, [0.8, 0.15], rtol=1e-3) | np.isclose(th, [3.0, 0.5], rtol=1e-3)))
        rows.append(dict(hid=hid, arm=c['arm'], loss=c['loss'], ts=tuple(np.round(c['theta_star'], 6)), e=np.array(c['rel_err']),
                         bound=bnd, fval=c['fval']))
arms = sorted({r['arm'] for r in rows if r['arm'] != 'A_own'})
res = {'n_files': len(files), 'arms': arms, 'params': ['v_endo', 'v_t']}


def boot_ci(x, fn=np.mean, n=10000):
    x = np.asarray(x); bs = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


for loss in ['corr', 'amp']:
    A = [r for r in rows if r['arm'] == 'A_own' and r['loss'] == loss]
    eA = np.array([r['e'] for r in A])
    sdA = eA.std(0, ddof=1)
    res[f'{loss}_noise'] = dict(n=len(eA), mean=eA.mean(0).tolist(), sd=sdA.tolist(), max_abs=np.abs(eA).max(0).tolist())
    for arm in arms:
        B = [r for r in rows if r['arm'] == arm and r['loss'] == loss]
        hearts = sorted({r['hid'] for r in B})
        bh, dh, absb = [], [], []
        for h in hearts:
            eb = np.array([r['e'] for r in B if r['hid'] == h])
            ea = {r['ts']: r['e'] for r in A if r['hid'] == h}
            ed = np.array([r['e'] - ea[r['ts']] for r in B if r['hid'] == h])
            bh.append(eb.mean(0)); dh.append(ed.mean(0)); absb.append(np.abs(eb).mean(0))
        bh, dh, absb = np.array(bh), np.array(dh), np.array(absb)
        n = len(hearts); tq = stats.t.ppf(0.975, n - 1)
        out_arm = dict(n_hearts=n, hearts=hearts, n_calibrations=len(B), n_bound_hits=int(sum(r['bound'] for r in B)),
                       median_final_loss=float(np.median([r['fval'] for r in B])),
                       median_final_loss_ownarm=float(np.median([r['fval'] for r in A])))
        for j, p in enumerate(['v_endo', 'v_t']):
            m, s = bh[:, j].mean(), bh[:, j].std(ddof=1)
            ma = absb[:, j].mean()
            out_arm[p] = dict(
                mean_bias=float(m), ci_t=[float(m - tq * s / np.sqrt(n)), float(m + tq * s / np.sqrt(n))],
                ci_boot=boot_ci(bh[:, j]), sd_between_hearts=float(s),
                mean_abs_bias=float(ma), ci_boot_abs=boot_ci(absb[:, j]),
                paired_mean=float(dh[:, j].mean()), paired_ci_boot=boot_ci(dh[:, j]),
                per_heart=bh[:, j].round(4).tolist(),
                n_hearts_abs_bias_ge_2sd_noise=int((np.abs(bh[:, j]) >= 2 * sdA[j]).sum()),
                two_sd_noise=float(2 * sdA[j]),
                pop_criterion=bool(boot_ci(absb[:, j])[0] > 2 * sdA[j]))
        res[f'{loss}_{arm}'] = out_arm
crit_rt = any(res[f'{l}_{a}'][p]['n_hearts_abs_bias_ge_2sd_noise'] > 0 for l in ['corr', 'amp'] for a in arms for p in ['v_endo', 'v_t'])
crit_pop = any(res[f'{l}_{a}'][p]['pop_criterion'] for l in ['corr', 'amp'] for a in arms for p in ['v_endo', 'v_t'])
res['criterion_redteam_any_heart'] = bool(crit_rt); res['criterion_population'] = bool(crit_pop)
json.dump(res, open(out, 'w'), indent=1)
for k, v in res.items():
    if isinstance(v, dict) and 'v_endo' in v:
        print(k, 'n', v['n_hearts'], 'bound hits', v['n_bound_hits'], '/', v['n_calibrations'], 'median final loss %.3f (own arm %.1e)' % (v['median_final_loss'], v['median_final_loss_ownarm']))
        for p in ['v_endo', 'v_t']:
            q = v[p]
            print(f"   {p}: mean bias {q['mean_bias']*100:+.2f}% CI [{q['ci_boot'][0]*100:+.2f},{q['ci_boot'][1]*100:+.2f}]  "
                  f"mean|b| {q['mean_abs_bias']*100:.2f}% CI [{q['ci_boot_abs'][0]*100:.2f},{q['ci_boot_abs'][1]*100:.2f}]  "
                  f"2SDnoise {q['two_sd_noise']*100:.2f}%  hearts>=2SD {q['n_hearts_abs_bias_ge_2sd_noise']}/{v['n_hearts']}")
    elif k.endswith('_noise'):
        print(k, {kk: (np.round(np.array(vv) * 100, 2).tolist() if kk != 'n' else vv) for kk, vv in v.items()})
print('criterion (red team, any heart):', crit_rt, ' population:', crit_pop)
