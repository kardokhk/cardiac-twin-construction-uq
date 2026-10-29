"""CSV export and owner check for the Amendment 2 B4 extended noise grids (owner noise-ext, 2026-10-28).
Usage: python -I -B export_noise_ext.py <out_dir> <shard.h5> [<shard.h5> ...]
Writes <out_dir>/noise_estimates_ext.csv in exactly the column layout of results/2026-10-28_armC/noise_estimates.csv,
and <out_dir>/owner_check.json: per new level the median over hearts of SD_noise (sqrt of the mean over theta* of the
replicate variance of theta_hat/theta*, as run_analysis.noise_sd) next to the median over hearts of SD_cons
(run_analysis.cv_per_heart on include_primary rows of results/2026-10-28_armC/estimates.csv, matching loss), and the
break-even on the combined A3 + B4 grid with the run_analysis.crossing rule (log SD linear in severity, -dB or ln
level; "beyond grid" outside), with a cluster-bootstrap 95% CI over hearts (2,000 resamples, seed 20261028, same
percentile rule as run_analysis.breakeven_block). Owner check only; the analysis owner computes the reported E3.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_lib as C            # noqa: E402
from csv_export import flags      # noqa: E402
import run_noise_ext as X        # noqa: E402

COLS = ['patient', 'group', 'istar', 'v_endo_star', 'v_t_star', 'comparator', 'level', 'level_unit', 'rep', 'seed',
        'loss', 'v_endo_hat', 'v_t_hat', 'fval_emu', 'ratio_v_endo', 'ratio_v_t', 'bound_lo_v_endo', 'bound_hi_v_endo',
        'bound_lo_v_t', 'bound_hi_v_t', 'censored', 'flag_vt_gt_vfib']
LN = [l for l, _ in C.LOSSES]
P = ['v_endo', 'v_t']
SEED = 20261028
NB = 2000
SEV = dict(white=lambda L: -np.asarray(L, float), lp40=lambda L: -np.asarray(L, float),
           wander=lambda L: np.log(np.asarray(L, float)), disp=lambda L: np.log(np.asarray(L, float)))


def export(out, shards):
    rows = []
    for fn in shards:
        with h5py.File(fn, 'r') as O:
            for gp in sorted(O):
                g = O[gp]; p = int(g.attrs['patient']); grp = g.attrs['group']; ts = g['theta_star'][:]
                for cn in X.LEVELS_EXT:
                    if cn not in g:
                        continue
                    nz = g[f'{cn}/theta_hat'][:]; sd = g[f'{cn}/rng_seed'][:]; levs = list(g[f'{cn}/theta_hat'].attrs['levels'])
                    ci = int(g[f'{cn}/theta_hat'].attrs['comparator_index'])
                    for i in range(nz.shape[0]):
                        for li, lev in enumerate(levs):
                            for rp in range(nz.shape[2]):
                                seed = (int(sd[i, li, rp]) if cn == 'disp' else
                                        '-'.join(map(str, X.rep_seed_ext(p, i, ci, lev, rp))))
                                for j, l in enumerate(LN):
                                    th = nz[i, li, rp, j, :2]
                                    rows.append(dict(patient=p, group=grp, istar=i, v_endo_star=ts[i, 0],
                                                     v_t_star=ts[i, 1], comparator=cn, level=int(lev),
                                                     level_unit=X.UNIT[cn], rep=rp, seed=seed, loss=l,
                                                     v_endo_hat=th[0], v_t_hat=th[1], fval_emu=nz[i, li, rp, j, 2],
                                                     ratio_v_endo=th[0] / ts[i, 0], ratio_v_t=th[1] / ts[i, 1],
                                                     **flags(th)))
    df = pd.DataFrame(rows)[COLS]
    df.to_csv(os.path.join(out, 'noise_estimates_ext.csv'), index=False)
    return df


def noise_sd(nz, loss):
    rows = []
    for (h, c, L), g in nz[nz.loss == loss].groupby(['patient', 'comparator', 'level']):
        d = dict(patient=int(h), comparator=c, level=float(L))
        for p in P:
            vs = [np.var(gi[f'ratio_{p}'].values, ddof=1) for _, gi in g.groupby('istar') if len(gi) > 1]
            d[f'sd_{p}'] = float(np.sqrt(np.mean(vs)))
        rows.append(d)
    return pd.DataFrame(rows)


def sd_cons(loss):
    E = pd.read_csv(os.path.join(C.ROOT, 'results', '2026-10-28_armC', 'estimates.csv'))
    E = E[(E.loss == loss) & E.include_primary]
    out = {}
    for h, eh in E.groupby('patient'):
        out[int(h)] = {p: float(np.sqrt(np.mean([np.var(g[f'ratio_{p}'].values, ddof=1)
                                                 for _, g in eh.groupby('istar') if len(g) >= 2]))) for p in P}
    return pd.DataFrame(out).T


def crossing(s_levels, ly_in, t):
    o = np.argsort(s_levels); s = s_levels[o]; ly = np.log(np.asarray(ly_in))[o]; t = np.log(t)
    if not np.all(np.isfinite(ly)):
        return np.nan
    if t < ly[0]:
        return -np.inf
    for k in range(len(s) - 1):
        if (ly[k] - t) * (ly[k + 1] - t) <= 0 and ly[k] != ly[k + 1]:
            return float(s[k] + (t - ly[k]) / (ly[k + 1] - ly[k]) * (s[k + 1] - s[k]))
    return np.inf


def to_level(c, s, levels):
    if np.isposinf(s):
        return f'beyond grid: < {min(levels)} dB' if c in ('white', 'lp40') else f'beyond grid: > {max(levels)}'
    if np.isneginf(s):
        return f'beyond grid: > {max(levels)} dB' if c in ('white', 'lp40') else f'beyond grid: < {min(levels)}'
    if not np.isfinite(s):
        return 'nan'
    return round(float(-s), 2) if c in ('white', 'lp40') else round(float(np.exp(s)), 2)


def owner_check(out, ext):
    old = pd.read_csv(os.path.join(C.ROOT, 'results', '2026-10-28_armC', 'noise_estimates.csv'))
    allz = pd.concat([old, ext], ignore_index=True)
    res = {}
    for loss in LN:
        nsd = noise_sd(allz, loss); sc = sd_cons(loss)
        for c in X.LEVELS_EXT:
            sub = nsd[nsd.comparator == c]
            levels = sorted(sub.level.unique(), key=lambda L: SEV[c]([L])[0])
            for p in P:
                M_ = sub.pivot(index='patient', columns='level', values=f'sd_{p}')[levels]
                hearts = sorted(set(M_.index) & set(sc.index))
                M = M_.loc[hearts].values; sdc = sc.loc[hearts, p].values
                medn = np.median(M, 0); medc = float(np.median(sdc)); sv = SEV[c](levels)
                s0 = crossing(sv, medn, medc)
                rng = np.random.default_rng(SEED); bs = []
                for _ in range(NB):
                    pk = rng.integers(0, len(hearts), len(hearts))
                    bs.append(crossing(sv, np.median(M[pk], 0), float(np.median(sdc[pk]))))
                bs = np.array(bs)
                lo = np.percentile(bs, 2.5, method='lower'); hi = np.percentile(bs, 97.5, method='higher')
                ci = [to_level(c, x, levels) for x in ((hi, lo) if c in ('white', 'lp40') else (lo, hi))]
                res[f'{loss}|{c}|{p}'] = dict(
                    loss=loss, comparator=c, parameter=p, n_hearts=len(hearts), sd_cons_median=medc,
                    levels=[float(x) for x in levels], sd_noise_median=[float(x) for x in medn],
                    new_levels=[float(x) for x in X.LEVELS_EXT[c]], breakeven_combined_grid=to_level(c, s0, levels),
                    ci95=ci, boot_beyond_most_severe=int(np.isposinf(bs).sum()),
                    boot_beyond_least_severe=int(np.isneginf(bs).sum()))
    json.dump(dict(label='owner check (noise-ext), not the analysis E3; combined A3 + B4 grid', seed=SEED, n_boot=NB,
                   results=res), open(os.path.join(out, 'owner_check.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, 'SDcons %.4f' % v['sd_cons_median'],
              ' '.join(f'{l:g}:{s:.4f}' for l, s in zip(v['levels'], v['sd_noise_median'])),
              '| BE', v['breakeven_combined_grid'], v['ci95'], flush=True)


if __name__ == '__main__':
    o = sys.argv[1]
    df = export(o, sys.argv[2:])
    print('rows', len(df), 'NaN', int(df[['v_endo_hat', 'v_t_hat']].isna().sum().sum()),
          'censored by comparator/level/loss:')
    print(df.groupby(['comparator', 'level', 'loss']).censored.sum().to_string(), flush=True)
    owner_check(o, df)
