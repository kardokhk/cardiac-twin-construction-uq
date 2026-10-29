"""Arm F analysis (owner armF, 2026-10-28). Usage: python -I -B analyse_armF.py <out_dir>
Reads <out_dir>/heart_patient*.h5, writes estimates_armF.csv, summary_armF.json and prints a digest.

Primary arm F quantity (declared in this script before the full run): per heart and parameter, the mean over its 3
theta* of log(theta_hat/theta*) for RE targets, correlation loss, QRS window, final (round 2) emulator, 1.5 mm unit;
population model-form bias = exp(mean over hearts), 95% CI from the t distribution over hearts (df = n - 1), with a
cluster-bootstrap percentile CI over hearts (2,000 resamples, seed 20261028) beside it. Estimates at a prior bound are
kept at the bound (primary) and dropped (sensitivity). The control set (pure-eikonal targets) is summarised the same
way, and the paired contrast RE/control (same heart, theta*, loss, window) isolates model form from emulator error.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import h5py
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armf_lib as F  # noqa: E402

out = sys.argv[1]
files = sorted(glob.glob(os.path.join(out, 'heart_patient*.h5')))
LN = [l for l, _ in F.LOSSES]
rows = []
cmp_rows = []
for fn in files:
    with h5py.File(fn, 'r') as H:
        p = int(H.attrs['patient']); grp = H.attrs['group']; ts = H['theta_star'][:]
        for vk in H['units']:
            g = H['units'][vk]
            tm = g['targets/tmpl']
            for sname in g['targets']:
                s = g['targets'][sname]
                if 'calib' not in s:
                    continue
                C = s['calib'][:]                       # round x istar x loss x window x 4
                for i in range(3):
                    for j, ln in enumerate(LN):
                        for w, wn in enumerate(F.WINDOWS):
                            th = C[2, i, j, w, :2]
                            lo, hi = F.bound_hit(th)
                            rows.append(dict(patient=p, group=grp, vox=float(vk), set=sname, istar=i, loss=ln,
                                             window=wn, v_endo_star=ts[i, 0], v_t_star=ts[i, 1],
                                             v_endo_hat=th[0], v_t_hat=th[1], ratio_v_endo=th[0] / ts[i, 0],
                                             ratio_v_t=th[1] / ts[i, 1], fval=C[2, i, j, w, 2],
                                             loss_at_theta_star=C[2, i, j, w, 3],
                                             v_endo_hat_round0=C[0, i, j, w, 0], v_t_hat_round0=C[0, i, j, w, 1],
                                             bound_v_endo=('lo' if lo[0] else 'hi' if hi[0] else ''),
                                             bound_v_t=('lo' if lo[1] else 'hi' if hi[1] else ''),
                                             censored=bool(lo.any() or hi.any()),
                                             act_max_ms=float(s['act_max_ms'][i]), tmax_ms=float(g.attrs['tmax_ms'])))
                if sname.startswith('re'):
                    for i in range(3):
                        e_re = s['ecg'][i]; e_t = tm['ecg'][i]; e_ra = s['ecg_tmpl_from_re_act'][i]
                        nw = F.qrs_nw(float(s['act_max_ms'][i]))
                        act = s['act'][i]; up = s['act_up'][i]; ae = tm['act'][i]
                        okm = (act >= 0) & (ae < 1e6)
                        d = act[okm] - ae[okm]
                        cmp_rows.append(dict(
                            patient=p, vox=float(vk), set=sname, istar=i,
                            r_qrs=F.lead_r(e_re[:, :nw], e_t[:, :nw]), r_full=F.lead_r(e_re, e_t),
                            r_qrs_re_act=F.lead_r(e_re[:, :nw], e_ra[:, :nw]),
                            amp_ratio_qrs=float(np.linalg.norm(e_re[:, :nw]) / np.linalg.norm(e_t[:, :nw])),
                            relL2_qrs=float(np.linalg.norm(e_re[:, :nw] - e_t[:, :nw]) / np.linalg.norm(e_t[:, :nw])),
                            act_diff_median=float(np.median(d)), act_diff_p95abs=float(np.percentile(np.abs(d), 95)),
                            act_max_re=float(act[act >= 0].max()), act_max_eik=float(ae.max()),
                            upstroke_minus_eik_median=float(np.nanmedian(up - ae)),
                            unactivated=int(s['unactivated'][i]), re_wall_s=float(s['wall_s'][i]),
                            frames=int(s['frames'][i]), dt_us=float(s.attrs['dt_us']),
                            n_warn=int(s['n_warn_unassigned'][i])))

keys = list(rows[0].keys())
with open(os.path.join(out, 'estimates_armF.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, keys); wr.writeheader(); wr.writerows(rows)

hearts = sorted({r['patient'] for r in rows})
rng_seed = 20261028


def sel(set_, loss='corr', window='qrs', vox=1.5, drop_cens=False):
    return [r for r in rows if r['set'] == set_ and r['loss'] == loss and r['window'] == window and r['vox'] == vox
            and not (drop_cens and r['censored'])]


def per_heart(rs, par):
    d = {}
    for r in rs:
        d.setdefault(r['patient'], []).append(np.log(r['ratio_' + par]))
    return {h: float(np.mean(v)) for h, v in d.items()}


def pop(ph):
    x = np.array(list(ph.values())); n = len(x)
    if n < 2:
        return dict(n_hearts=n, gm=float(np.exp(x.mean())) if n else None)
    m, sd = x.mean(), x.std(ddof=1)
    tq = stats.t.ppf(0.975, n - 1)
    rng = np.random.default_rng(rng_seed)
    bs = np.array([rng.choice(x, n, replace=True).mean() for _ in range(2000)])
    return dict(n_hearts=n, gm_ratio=float(np.exp(m)), ci_t=[float(np.exp(m - tq * sd / np.sqrt(n))),
                float(np.exp(m + tq * sd / np.sqrt(n)))],
                ci_boot=[float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))],
                sd_log_between_hearts=float(sd))


def desc(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), median=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)),
                p97_5=float(np.percentile(v, 97.5)), min=float(v.min()), max=float(v.max()))


summ = dict(hearts=hearts, files=[os.path.basename(f) for f in files], bootstrap_seed=rng_seed, estimates={})
for set_ in sorted({r['set'] for r in rows}):
    for vox in sorted({r['vox'] for r in rows if r['set'] == set_}):
        for loss in LN:
            for window in F.WINDOWS:
                for dc in (False, True):
                    rs = sel(set_, loss, window, vox, dc)
                    if not rs:
                        continue
                    k = f'{set_}|{vox}|{loss}|{window}|{"drop_censored" if dc else "keep_censored"}'
                    e = dict(n_estimates=len(rs), n_censored=int(sum(r['censored'] for r in sel(set_, loss, window, vox))),
                             fval=desc([r['fval'] for r in rs]),
                             fval_minus_loss_at_truth=desc([r['fval'] - r['loss_at_theta_star'] for r in rs]))
                    for par in ('v_endo', 'v_t'):
                        ph = per_heart(rs, par)
                        e[par] = dict(pop(ph), per_heart_gm={str(h): float(np.exp(v)) for h, v in ph.items()},
                                      ratio_all=desc([r['ratio_' + par] for r in rs]),
                                      sd_log_all=float(np.std([np.log(r['ratio_' + par]) for r in rs], ddof=1))
                                      if len(rs) > 1 else None,
                                      bound_lo=int(sum(r['bound_' + par] == 'lo' for r in rs)),
                                      bound_hi=int(sum(r['bound_' + par] == 'hi' for r in rs)))
                    summ['estimates'][k] = e

# paired contrast RE / control (same heart, theta*, loss, window, unit)
idx = {(r['patient'], r['vox'], r['set'], r['istar'], r['loss'], r['window']): r for r in rows}
contrast = {}
for set_ in sorted({r['set'] for r in rows if r['set'].startswith('re')}):
    for loss in LN:
        for window in F.WINDOWS:
            for vox in sorted({r['vox'] for r in rows if r['set'] == set_}):
                k = f'{set_}/tmpl|{vox}|{loss}|{window}'
                e = {}
                for par in ('v_endo', 'v_t'):
                    ph = {}
                    for r in rows:
                        if r['set'] == set_ and r['loss'] == loss and r['window'] == window and r['vox'] == vox:
                            c = idx[(r['patient'], vox, 'tmpl', r['istar'], loss, window)]
                            ph.setdefault(r['patient'], []).append(np.log(r['ratio_' + par] / c['ratio_' + par]))
                    e[par] = pop({h: float(np.mean(v)) for h, v in ph.items()})
                contrast[k] = e
summ['contrast_re_over_control'] = contrast

# dt and mesh convergence on the convergence heart
conv = {}
for r in rows:
    if r['set'] == 're_dt12p5' and r['loss'] == 'corr':
        b = idx[(r['patient'], r['vox'], 're', r['istar'], 'corr', r['window'])]
        conv.setdefault(f'dt12.5_vs_dt25|{r["window"]}', []).append(
            dict(istar=r['istar'], v_endo=r['ratio_v_endo'] / b['ratio_v_endo'], v_t=r['ratio_v_t'] / b['ratio_v_t']))
for fn in files:
    with h5py.File(fn, 'r') as H:
        g = H['units']['1.5']['targets']
        if 're_dt12p5' in g:
            a, b = g['re']['ecg'][:], g['re_dt12p5']['ecg'][:]
            conv['dt_ecg'] = [dict(istar=i, min_r=float(min(F.lead_r(a[i], b[i]))),
                                   relL2=float(np.linalg.norm(a[i] - b[i]) / np.linalg.norm(a[i])),
                                   act_maxabs_ms=float(np.abs(g['re']['act'][i] - g['re_dt12p5']['act'][i]).max()))
                              for i in range(3)]
        if '1.0' in H['units']:
            a = H['units']['1.5']['targets']['re']['ecg'][:]; b = H['units']['1.0']['targets']['re']['ecg'][:]
            n = min(a.shape[2], b.shape[2])
            conv['mesh_ecg_re_1.0_vs_1.5'] = [dict(istar=i, min_r=float(min(F.lead_r(a[i][:, :n], b[i][:, :n]))),
                                                   relL2=float(np.linalg.norm(a[i][:, :n] - b[i][:, :n]) /
                                                               np.linalg.norm(a[i][:, :n])))
                                              for i in range(3)]
summ['convergence'] = conv
# ECG model-form comparison
cm = {}
for set_ in sorted({c['set'] for c in cmp_rows}):
    for vox in sorted({c['vox'] for c in cmp_rows if c['set'] == set_}):
        cs = [c for c in cmp_rows if c['set'] == set_ and c['vox'] == vox]
        cm[f'{set_}|{vox}'] = dict(
            n_targets=len(cs),
            r_qrs_all_leads=desc(np.concatenate([c['r_qrs'] for c in cs])),
            r_qrs_mean_per_target=desc([np.mean(c['r_qrs']) for c in cs]),
            r_qrs_re_act_all_leads=desc(np.concatenate([c['r_qrs_re_act'] for c in cs])),
            r_full_all_leads=desc(np.concatenate([c['r_full'] for c in cs])),
            amp_ratio_qrs=desc([c['amp_ratio_qrs'] for c in cs]), relL2_qrs=desc([c['relL2_qrs'] for c in cs]),
            act_diff_median_ms=desc([c['act_diff_median'] for c in cs]),
            act_diff_p95abs_ms=desc([c['act_diff_p95abs'] for c in cs]),
            act_max_re_minus_eik_ms=desc([c['act_max_re'] - c['act_max_eik'] for c in cs]),
            upstroke_minus_eik_median_ms=desc([c['upstroke_minus_eik_median'] for c in cs]),
            unactivated_total=int(sum(c['unactivated'] for c in cs)), warn_total=int(sum(c['n_warn'] for c in cs)),
            re_wall_s=desc([c['re_wall_s'] for c in cs]))
summ['ecg_model_form'] = cm
summ['ecg_per_target'] = cmp_rows
json.dump(summ, open(os.path.join(out, 'summary_armF.json'), 'w'), indent=1, default=float)


def fmt(e):
    return (f"GM {e['gm_ratio']:.3f} [t {e['ci_t'][0]:.3f}, {e['ci_t'][1]:.3f}; boot {e['ci_boot'][0]:.3f}, "
            f"{e['ci_boot'][1]:.3f}]" if 'gm_ratio' in e else str(e))


print('hearts', hearts)
for k, e in summ['estimates'].items():
    if k.endswith('keep_censored'):
        print(k, '| v_endo', fmt(e['v_endo']), '| v_t', fmt(e['v_t']), f"| cens {e['n_censored']}/{e['n_estimates']}",
              f"| fval med {e['fval']['median']:.4f}")
for k, e in contrast.items():
    print('contrast', k, '| v_endo', fmt(e['v_endo']), '| v_t', fmt(e['v_t']))
for k, e in cm.items():
    print('ecg', k, json.dumps({kk: (vv['median'], vv['min']) if isinstance(vv, dict) else vv for kk, vv in e.items()}))
print('convergence', json.dumps(conv))
