"""Arm T analysis (owner armT, 2026-10-28). Plan section 4 item 5 and the arm T specification.

Per heart h and cell (loss, reference torso RR, registration): relative error e = theta_hat / theta* - 1 for each of the
3 theta*; per-heart signed bias = mean of e over theta*, per-heart absolute bias = mean of |e|. Population estimate =
mean over hearts, 95% CI by cluster (heart) percentile bootstrap, 2,000 resamples, numpy default_rng(20261028) freshly
seeded for every cell. Censoring: primary keeps bound estimates at the bound; sensitivity drops them (per-heart mean
over the remaining theta*, hearts with none left dropped). Reported beside every bias: final fit loss (median and
interquartile range over the cell's estimates) and bound hits (count of estimates with either parameter at a prior bound).
Sets: HCM (Malik, reference heart excluded: 16), healthy (KCL, 5), pooled (21, HCM-heavy).
S5: Spearman rho between the ICP rotation angle and per-heart |bias| (per reference, registration, loss), and the
paired difference ICP minus translation-only in per-heart |bias|, each with a heart bootstrap 95% CI. Exploratory.
Usage: python -I t05_analyse.py      (reads results/2026-10-28_armT/{estimates.csv, post.json, data/heart_*.h5})
       python -I t05_analyse.py --est estimates_v2.csv --round 2 --suffix _v2   (follow-up 2026-10-28: per-arm
       refinement, t06_rerun_perarm.py; writes analysis_v2.json and tables_v2.md only; defaults reproduce the original)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402

NB = 2000
SEED = 20261028
_ap = argparse.ArgumentParser()
_ap.add_argument('--est', default='estimates.csv'); _ap.add_argument('--round', type=int, default=1)
_ap.add_argument('--suffix', default='')
ARGS = _ap.parse_args()
E_all = pd.read_csv(os.path.join(L.OUT, ARGS.est), dtype={'heart': str, 'ref': str})
E = E_all[E_all['round'] == ARGS.round].copy()
post = json.load(open(os.path.join(L.OUT, 'post.json')))
hearts = sorted(E.heart.unique(), key=lambda h: (h.startswith('K'), h))

reg = {}
for h in hearts:
    with h5py.File(L.heart_file(h), 'r') as f:
        for k in f['reg']:
            reg[(h, k)] = dict(f[f'reg/{k}'].attrs)
        reg[(h, 'attrs')] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f.attrs.items() if k != 'bayer_meta'}


def boot_mean(x, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (NB, len(x)))
    b = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_spearman(x, y, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(NB):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        bs.append(spearmanr(x[i], y[i])[0])
    return rho, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def sets_of(ref):
    hcm = [h for h in hearts if not h.startswith('K') and h != ref]
    hl = [h for h in hearts if h.startswith('K')]
    return {'HCM': hcm, 'healthy': hl, 'pooled_HCM_heavy': hcm + hl}


out = dict(n_bootstrap=NB, seed=SEED, cells=[], own=[], forward=[], s5=[], registration=[])
# ---------------------------------------------------------------- own-torso control (known answer: |e| <= 2%)
own = E[E.ref == 'own']
for ln in ('corr', 'amp'):
    o = own[own.loss == ln]
    out['own'].append(dict(loss=ln, n=len(o), max_abs_rel_v_endo=float(o.rel_v_endo.abs().max()),
                           max_abs_rel_v_t=float(o.rel_v_t.abs().max()), median_fval=float(o.fval.median()),
                           bound_hits=int(o.bound_hit.sum())))
out['report_round'] = ARGS.round; out['estimates_file'] = ARGS.est; out['own_by_round'] = []
for r_ in sorted(E_all['round'].unique()):
    for ln in ('corr', 'amp'):
        o = E_all[(E_all.ref == 'own') & (E_all.loss == ln) & (E_all['round'] == r_)]
        out['own_by_round'].append(dict(round=int(r_), loss=ln, n=len(o), max_abs_rel_v_endo=float(o.rel_v_endo.abs().max()),
                                        max_abs_rel_v_t=float(o.rel_v_t.abs().max()),
                                        median_abs_rel_v_endo=float(o.rel_v_endo.abs().median()),
                                        median_abs_rel_v_t=float(o.rel_v_t.abs().median()), bound_hits=int(o.bound_hit.sum())))
# ---------------------------------------------------------------- bias cells
per_heart = {}
for ln in ('corr', 'amp'):
    for ref in L.REFS:
        for rg in ('trans', 'icp'):
            sub = E[(E.loss == ln) & (E.ref == ref) & (E.reg == rg)]
            for sname, hs in sets_of(ref).items():
                s = sub[sub.heart.isin(hs)]
                hs_ = [h for h in hs if h in set(s.heart)]
                cell = dict(loss=ln, ref=ref, reg=rg, set=sname, n_hearts=len(hs_), n_estimates=len(s),
                            bound_hits=int(s.bound_hit.sum()), fval_median=float(s.fval.median()),
                            fval_q25=float(s.fval.quantile(0.25)), fval_q75=float(s.fval.quantile(0.75)))
                for p in ('v_endo', 'v_t'):
                    g = s.groupby('heart')[f'rel_{p}']
                    sg = g.mean().reindex(hs_); ab = g.apply(lambda x: x.abs().mean()).reindex(hs_)
                    if sname == 'pooled_HCM_heavy':
                        per_heart[(ln, ref, rg, p)] = ab
                    m, lo, hi = boot_mean(sg.values); cell[f'{p}_signed'] = [m, lo, hi]
                    m, lo, hi = boot_mean(ab.values); cell[f'{p}_abs'] = [m, lo, hi]
                    sc = s[s.bound_hit == 0]
                    if len(sc):
                        abc = sc.groupby('heart')[f'rel_{p}'].apply(lambda x: x.abs().mean())
                        m, lo, hi = boot_mean(abc.values); cell[f'{p}_abs_uncensored_only'] = [m, lo, hi, int(len(abc))]
                out['cells'].append(cell)
# ---------------------------------------------------------------- registration and forward similarity
rows = []
for h in hearts:
    for ref in L.REFS:
        if ref == h:
            continue
        for rg in ('icp', 'trans'):
            k = f'{ref}_{rg}'
            a = reg[(h, k)]
            sim = np.array(post[h]['sim'][k])
            rows.append(dict(heart=h, group=L.group_of(h), ref=ref, reg=rg, rot_deg_icp=float(a['rot_deg_icp']),
                             icp_rms_mm=float(a['icp_rms']), outside=int(a['outside']), rel_W=float(a['rel_diff_vs_own']),
                             lead_r=float(np.median(sim[:, 0])), amp_ratio=float(np.median(sim[:, 1]))))
R = pd.DataFrame(rows)
if not ARGS.suffix:
    R.to_csv(os.path.join(L.OUT, 'registration_forward.csv'), index=False)
for ref in L.REFS:
    for rg in ('icp', 'trans'):
        for sname, hs in sets_of(ref).items():
            r = R[(R.ref == ref) & (R.reg == rg) & R.heart.isin(hs)]
            q = lambda c: [float(r[c].median()), float(r[c].quantile(0.25)), float(r[c].quantile(0.75)),  # noqa: E731
                           float(r[c].min()), float(r[c].max())]
            out['forward'].append(dict(ref=ref, reg=rg, set=sname, n=len(r), lead_r=q('lead_r'), amp_ratio=q('amp_ratio'),
                                       rot_deg_icp=q('rot_deg_icp'), icp_rms_mm=q('icp_rms_mm'),
                                       outside_nodes_max=int(r.outside.max())))
# identity checks
for h in L.REFS:
    k = (h, f'{h}_self')
    if k in reg:
        out['registration'].append(dict(heart=h, identity_rel_W=float(reg[k]['rel_diff_vs_own']),
                                        identity_rot_deg=float(reg[k]['rot_deg_icp'])))
# ---------------------------------------------------------------- S5
for ref in L.REFS:
    hs = sets_of(ref)['pooled_HCM_heavy']
    rot = R[(R.ref == ref) & (R.reg == 'icp')].set_index('heart').rot_deg_icp.reindex(hs)
    for rg in ('icp', 'trans'):
        lr = R[(R.ref == ref) & (R.reg == rg)].set_index('heart').lead_r.reindex(hs)
        rho = boot_spearman(rot.values, lr.values)
        out['s5'].append(dict(ref=ref, reg=rg, what='spearman(rotation, lead_r)', n=len(hs), rho=rho))
        for ln in ('corr', 'amp'):
            for p in ('v_endo', 'v_t'):
                ab = per_heart[(ln, ref, rg, p)].reindex(hs)
                out['s5'].append(dict(ref=ref, reg=rg, loss=ln, param=p, what='spearman(rotation, abs bias)', n=len(hs),
                                      rho=boot_spearman(rot.values, ab.values)))
    for ln in ('corr', 'amp'):
        for p in ('v_endo', 'v_t'):
            dif = (per_heart[(ln, ref, 'icp', p)] - per_heart[(ln, ref, 'trans', p)]).reindex(hs)
            out['s5'].append(dict(ref=ref, loss=ln, param=p, what='mean paired diff abs bias icp - trans', n=len(hs),
                                  diff=boot_mean(dif.values)))
# ---------------------------------------------------------------- heart table
ht = []
for h in hearts:
    a = reg[(h, 'attrs')]
    ht.append(dict(heart=h, group=L.group_of(h), seed_id=L.pid_of(h), torso_mL=a['torso_volume_mL'], myo_cons_mL=a['myo_cons_mL'],
                   n_nodes=a['n_nodes'], fast_share_A2=a['fast_share_A2'], fast_share_1vox=a['fast_share_1vox'],
                   root_shift_max_mm=float(np.max(a['root_shift_mm'])), outside_own=a['outside_nodes_own'],
                   pilot_label_agreement=a['pilot_label_agreement'],
                   held_rms_ms_median=float(np.median(np.array(post[h]['held'])[:, 0])),
                   held_corr_loss_max=float(np.max(np.array(post[h]['held'])[:, 2]))))
if not ARGS.suffix:
    pd.DataFrame(ht).to_csv(os.path.join(L.OUT, 'hearts.csv'), index=False)
out['hearts'] = ht
json.dump(out, open(os.path.join(L.OUT, f'analysis{ARGS.suffix}.json'), 'w'), indent=1)


# ---------------------------------------------------------------- markdown tables
def f(v, pct=True):
    k = 100 if pct else 1
    return f'{v[0]*k:+.1f}% [{v[1]*k:+.1f}, {v[2]*k:+.1f}]'


def fa(v):
    return f'{v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}]'


lines = ['| loss | ref | registration | set | n | v_endo signed | v_endo abs | v_t signed | v_t abs | bound hits | fit loss median [IQR] |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for c in out['cells']:
    lines.append(f"| {c['loss']} | {c['ref']} | {c['reg']} | {c['set']} | {c['n_hearts']} | {f(c['v_endo_signed'])} | "
                 f"{fa(c['v_endo_abs'])} | {f(c['v_t_signed'])} | {fa(c['v_t_abs'])} | {c['bound_hits']}/{c['n_estimates']} | "
                 f"{c['fval_median']:.3f} [{c['fval_q25']:.3f}, {c['fval_q75']:.3f}] |")
lines += ['', '| ref | registration | set | n | lead r median [IQR] | amplitude ratio median [IQR] | ICP rotation deg median [min, max] |',
          '|---|---|---|---|---|---|---|']
for c in out['forward']:
    lines.append(f"| {c['ref']} | {c['reg']} | {c['set']} | {c['n']} | {c['lead_r'][0]:.2f} [{c['lead_r'][1]:.2f}, {c['lead_r'][2]:.2f}] | "
                 f"{c['amp_ratio'][0]:.2f} [{c['amp_ratio'][1]:.2f}, {c['amp_ratio'][2]:.2f}] | "
                 f"{c['rot_deg_icp'][0]:.1f} [{c['rot_deg_icp'][3]:.1f}, {c['rot_deg_icp'][4]:.1f}] |")
lines += ['', '| S5 item | ref | registration | loss | parameter | n | estimate [95% CI] |', '|---|---|---|---|---|---|---|']
for c in out['s5']:
    v = c.get('rho', c.get('diff'))
    lines.append(f"| {c['what']} | {c['ref']} | {c.get('reg', 'icp - trans')} | {c.get('loss', '-')} | {c.get('param', '-')} | "
                 f"{c['n']} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |")
open(os.path.join(L.OUT, f'tables{ARGS.suffix}.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(json.dumps(out['own'], indent=0)); print(json.dumps(out['registration'], indent=0))
