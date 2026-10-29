"""Gates G3 and G4 and the primary-estimand preview from saved mini-study outputs (owner calib, 2026-10-28).
Reads only <dir>/estimates.csv, noise_estimates.csv, constructions.csv, refinement.csv (written by csv_export.py from
the per-heart HDF5). Writes <dir>/gates.json and <dir>/gates.md. Seed 20261028 for every bootstrap (2,000 resamples).
Usage: python -I analyse_gates.py <dir>

Definitions (fixed before the full run; see README):
  y = theta_hat / theta*  (final round 2 estimates; per construction, theta* draw, parameter, loss)
  SD_cons(heart, p)   = sqrt(mean over the 3 theta* of the between-construction variance of y, ddof 1)
                        over all S x M constructions of the heart, reference included (12 when all built)
  SD_noise(heart, p, SNR) = sqrt(mean over the 3 theta* of the between-replicate variance of y, ddof 1), arm N
  G3 (pre-registered): heart-parameter passes if SD_cons >= 2 x SD_noise(20 dB); G3 passes if, for at least one
                       parameter, at least 2 of 3 hearts pass. Primary: correlation loss, white noise, all sources.
  Break-even SNR: SNR at which SD_noise = SD_cons, from linear interpolation of log10 SD_noise against SNR;
                  outside 10 to 30 dB, extrapolated from the nearest end point with the least-squares slope over all
                  five SNRs (flagged extrap<10 or extrap>30).
  G4: r_c = mean over theta* of theta_hat_c / theta_hat_ref (non-reference constructions), lengths l = L_c / L_ref;
      r = a + b1 l_scale + b2 l_wall + b3 l_base, pooled; in-sample R^2, adjusted R^2, leave-one-heart-out R^2;
      cluster bootstrap over hearts. Reframe trigger: in-sample R^2 >= 0.9 for both parameters.
  Preview: CV_i = SD/mean of y over constructions for each theta*; per heart, median and range over theta*.
           Segmentation vs mesh: per heart, two-way (source x voxel) sums of squares of log y after removing the
           theta* mean, shares of the total (descriptive).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

t_start = time.time()
D = sys.argv[1]
SEED = 20261028
NB = 2000
P = ['v_endo', 'v_t']
SNRS = [10, 15, 20, 25, 30]
WEAK = 'ukbb_cardiac_fcn_sa'
E = pd.read_csv(os.path.join(D, 'estimates.csv'))
N = pd.read_csv(os.path.join(D, 'noise_estimates.csv'))
CON = pd.read_csv(os.path.join(D, 'constructions.csv'))
REF = pd.read_csv(os.path.join(D, 'refinement.csv'))
EF = E[E['round'] == 2].copy()
HEARTS = sorted(EF.patient.unique())
res = dict(hearts=[int(h) for h in HEARTS], seed=SEED, n_boot=NB)


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b, A @ b


def pooled_sd(groups):
    v = [np.var(g, ddof=1) for g in groups if len(g) > 1]
    return float(np.sqrt(np.mean(v))) if v else np.nan


def breakeven(sd_c, sdn):
    """sdn: SD_noise at SNRS (decreasing with SNR). Returns (snr, flag)."""
    x = np.array(SNRS, float); ly = np.log10(sdn); t = np.log10(sd_c)
    if not np.all(np.isfinite(ly)) or not np.isfinite(t):
        return np.nan, 'nan'
    for k in range(len(x) - 1):
        a, b = ly[k], ly[k + 1]
        if (a - t) * (b - t) <= 0 and a != b:
            return float(x[k] + (t - a) / (b - a) * (x[k + 1] - x[k])), 'interp'
    # outside 10 to 30 dB: extrapolate from the nearest end point with the least-squares slope over all 5 SNRs
    # (theory for a locally linear estimator: slope -1/20 per dB); NaN if that slope is not negative
    slope = np.polyfit(x, ly, 1)[0]
    if slope >= 0:
        return np.nan, 'nonmonotone'
    if t > ly[0]:
        return float(x[0] + (t - ly[0]) / slope), 'extrap<10'
    return float(x[-1] + (t - ly[-1]) / slope), 'extrap>30'


def g3_block(loss, ntype, subset, drop_censored=False):
    rng = np.random.default_rng(SEED)
    out = {}
    e = EF[(EF.loss == loss)]
    if subset == 'strong':
        e = e[e.source != WEAK]
    n = N[(N.loss == loss) & (N.noise_type == ntype)]
    if drop_censored:
        e = e[~e.censored]; n = n[~n.censored]
    for h in HEARTS:
        eh = e[e.patient == h]; nh = n[n.patient == h]
        cons = sorted(eh.construction.unique())
        d = dict(n_constructions=len(cons), n_censored_cons_estimates=int(EF[(EF.loss == loss) & (EF.patient == h)].censored.sum()))
        for p in P:
            col = f'ratio_{p}'
            Y = eh.pivot_table(index='construction', columns='istar', values=col)      # cons x theta*
            sd_c = pooled_sd([Y[i].dropna().values for i in Y.columns])
            sdn = np.array([pooled_sd([nh[(nh.snr_db == s) & (nh.istar == i)][col].values for i in range(3)]) for s in SNRS])
            be, flag = breakeven(sd_c, sdn)
            ratio20 = sd_c / sdn[SNRS.index(20)]
            # bootstrap: constructions within heart, replicates within (theta*, SNR)
            bs_be, bs_ratio = [], []
            Yv = Y.values
            NR = {(s, i): nh[(nh.snr_db == s) & (nh.istar == i)][col].values for s in SNRS for i in range(3)}
            for _ in range(NB):
                pick = rng.integers(0, len(Yv), len(Yv))
                Yb = Yv[pick]
                sdcb = pooled_sd([Yb[:, j][np.isfinite(Yb[:, j])] for j in range(Yb.shape[1])])
                sdnb = np.array([pooled_sd([NR[(s, i)][rng.integers(0, len(NR[(s, i)]), len(NR[(s, i)]))] if len(NR[(s, i)]) else np.array([])
                                            for i in range(3)]) for s in SNRS])
                bs_be.append(breakeven(sdcb, sdnb)[0]); bs_ratio.append(sdcb / sdnb[2])
            bs_be = np.array(bs_be); bs_ratio = np.array(bs_ratio)
            d[p] = dict(sd_cons=sd_c, sd_noise={str(s): float(v) for s, v in zip(SNRS, sdn)},
                        ratio_sd_cons_to_noise20=float(ratio20),
                        ratio_ci=[float(np.nanpercentile(bs_ratio, 2.5)), float(np.nanpercentile(bs_ratio, 97.5))],
                        pass_2x=bool(sd_c >= 2 * sdn[SNRS.index(20)]),
                        breakeven_snr_db=be, breakeven_flag=flag,
                        breakeven_ci=[float(np.nanpercentile(bs_be, 2.5)), float(np.nanpercentile(bs_be, 97.5))],
                        breakeven_boot_finite=int(np.isfinite(bs_be).sum()))
        out[str(h)] = d
    per_p = {p: sum(out[str(h)][p]['pass_2x'] for h in HEARTS) for p in P}
    out['hearts_passing'] = per_p
    out['G3_pass'] = bool(any(v >= 2 for v in per_p.values()))
    return out


res['G3'] = {}
for loss in ['corr', 'amp']:
    for ntype in ['white', 'lp40']:
        for subset in ['all', 'strong']:
            res['G3'][f'{loss}|{ntype}|{subset}'] = g3_block(loss, ntype, subset)
    res['G3'][f'{loss}|white|all|no_censored'] = g3_block(loss, 'white', 'all', drop_censored=True)


# ------------------------------------------------------------------------------------------------ G4
def g4_block(loss, subset):
    rng = np.random.default_rng(SEED)
    e = EF[EF.loss == loss]
    if subset == 'strong':
        e = e[e.source != WEAK]
    rows = []
    for h in HEARTS:
        eh = e[e.patient == h]
        refc = eh[eh.is_ref].set_index('istar')
        cr = CON[(CON.patient == h) & CON.is_ref].iloc[0]
        for c, g in eh[~eh.is_ref].groupby('construction'):
            g = g.set_index('istar')
            r = [float((g[f'{p}_hat'] / refc[f'{p}_hat']).mean()) for p in P]
            cc = CON[(CON.construction == c)].iloc[0]
            rows.append(dict(h=h, c=c, r=r, l=[cc.L_scale / cr.L_scale, cc.L_wall / cr.L_wall, cc.L_base / cr.L_base],
                             cens=bool(g.censored.any())))
    H = np.array([r['h'] for r in rows]); Y = np.array([r['r'] for r in rows]); X = np.array([r['l'] for r in rows])
    out = dict(n_constructions=len(rows), n_hearts=len(HEARTS), n_with_censored=int(sum(r['cens'] for r in rows)),
               length_ratio_sd=X.std(0, ddof=1).tolist(), length_ratio_mean=X.mean(0).tolist())
    for j, p in enumerate(P):
        y = Y[:, j]

        def stats_(idx):
            yy, XX, HH = y[idx], X[idx], H[idx]
            b, yh = fit(XX, yy)
            s = dict(R2=r2(yy, yh))
            n_ = len(yy)
            s['R2_adj'] = 1 - (1 - s['R2']) * (n_ - 1) / (n_ - 4)
            if len(np.unique(HH)) >= 2:
                pred = np.empty_like(yy)
                for hh in np.unique(HH):
                    tr = HH != hh
                    bb, _ = fit(XX[tr], yy[tr])
                    pred[~tr] = np.column_stack([np.ones((~tr).sum()), XX[~tr]]) @ bb
                s['R2_LOHO'] = r2(yy, pred)
            else:
                s['R2_LOHO'] = np.nan
            s['coef'] = b.tolist()
            return s
        full = stats_(np.arange(len(y)))
        bs = []
        for _ in range(NB):
            pick = rng.choice(HEARTS, len(HEARTS), replace=True)
            idx = np.concatenate([np.nonzero(H == hh)[0] for hh in pick])
            bs.append(stats_(idx))
        d = dict(ratio_mean=float(y.mean()), ratio_sd=float(y.std(ddof=1)), coef=full['coef'])
        for k in ['R2', 'R2_adj', 'R2_LOHO']:
            v = np.array([b[k] for b in bs], float)
            d[k] = full[k]
            d[k + '_ci'] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            d[k + '_boot_finite'] = int(np.isfinite(v).sum())
        out[p] = d
    out['reframe_trigger_R2_ge_0.9_both'] = bool(out['v_endo']['R2'] >= 0.9 and out['v_t']['R2'] >= 0.9)
    return out


res['G4'] = {f'{l}|{s}': g4_block(l, s) for l in ['corr', 'amp'] for s in ['all', 'strong']}

# ------------------------------------------------------------------------------------------------ preview
prev = {}
for loss in ['corr', 'amp']:
    e = EF[EF.loss == loss]
    pl = {}
    for h in HEARTS:
        eh = e[e.patient == h]
        d = {}
        for p in P:
            col = f'ratio_{p}'
            cvs = [float(g[col].std(ddof=1) / g[col].mean()) for _, g in eh.groupby('istar')]
            y = np.log(eh[col].values)
            yc = y - eh.groupby('istar')[col].transform(lambda s: np.log(s).mean()).values
            dd = eh.assign(yc=yc)
            tot = float((dd.yc ** 2).sum())
            mS = dd.groupby('source').yc.mean(); mM = dd.groupby('vox').yc.mean()
            nS = dd.groupby('source').size(); nM = dd.groupby('vox').size()
            ssS = float((nS * mS ** 2).sum()); ssM = float((nM * mM ** 2).sum())
            mSM = dd.groupby(['source', 'vox']).yc.mean(); nSM = dd.groupby(['source', 'vox']).size()
            ssSM = float((nSM * mSM ** 2).sum()) - ssS - ssM
            d[p] = dict(cv_per_theta_star=cvs, cv_median=float(np.median(cvs)), cv_min=float(min(cvs)), cv_max=float(max(cvs)),
                        ratio_median=float(eh[col].median()),
                        ratio_p2_5=float(np.percentile(eh[col], 2.5)), ratio_p97_5=float(np.percentile(eh[col], 97.5)),
                        ss_share_segmentation=ssS / tot, ss_share_mesh=ssM / tot, ss_share_interaction=ssSM / tot,
                        ss_share_residual=1 - (ssS + ssM + ssSM) / tot, balanced=bool(len(eh) == 12 * 3))
        pl[str(h)] = d
    pl['median_over_hearts'] = {p: float(np.median([pl[str(h)][p]['cv_median'] for h in HEARTS])) for p in P}
    prev[loss] = pl
res['preview'] = prev

# ------------------------------------------------------------------------------------------------ fit quality
fq = {}
for loss in ['corr', 'amp']:
    e = EF[EF.loss == loss]
    fq[loss] = dict(fval_median=float(e.fval_emu.median()), fval_p2_5=float(e.fval_emu.quantile(0.025)),
                    fval_p97_5=float(e.fval_emu.quantile(0.975)), fval_max=float(e.fval_emu.max()),
                    n_estimates=int(len(e)), n_censored=int(e.censored.sum()),
                    n_censored_by_bound={k: int(e[k].sum()) for k in ['bound_lo_v_endo', 'bound_hi_v_endo', 'bound_lo_v_t', 'bound_hi_v_t']},
                    n_vt_gt_vfib=int(e.flag_vt_gt_vfib.sum()),
                    ref_recovery_rel_err={p: e[e.is_ref][f'ratio_{p}'].sub(1).abs().max() for p in P},
                    round0_to_final_max_rel_change={p: float(np.max(np.abs(
                        E[(E['round'] == 2) & (E.loss == loss)][f'{p}_hat'].values /
                        E[(E['round'] == 0) & (E.loss == loss)][f'{p}_hat'].values - 1))) for p in P})
    nz = N[N.loss == loss]
    fq[loss]['noise_n_censored'] = int(nz.censored.sum()); fq[loss]['noise_n'] = int(len(nz))
fq['refinement'] = dict(n_runs=int(len(REF)), emu0_rms_ms_median=float(REF.emu0_rms_ms.median()),
                        emu0_rms_ms_p97_5=float(REF.emu0_rms_ms.quantile(0.975)), emu0_rms_ms_max=float(REF.emu0_rms_ms.max()),
                        emu0_maxabs_ms_max=float(REF.emu0_maxabs_ms.max()),
                        per_construction_max=int(REF.groupby('construction').size().max()))
res['fit_quality'] = json.loads(json.dumps(fq, default=float))
res['runtime_s'] = time.time() - t_start
json.dump(res, open(os.path.join(D, 'gates.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------------------------------------ markdown
L = [f'# Gates G3 and G4: numbers (generated by code/gates/calib/analyse_gates.py; runtime {res["runtime_s"]:.0f} s)', '']
for key in ['corr|white|all', 'amp|white|all', 'corr|white|strong', 'corr|lp40|all', 'amp|lp40|all', 'corr|white|all|no_censored']:
    g = res['G3'][key]
    L += [f'## G3 [{key}]: pass = {g["G3_pass"]}; hearts passing {g["hearts_passing"]}', '',
          '| heart | param | SD_cons | SD_noise 20 dB | ratio [95% CI] | 2x pass | break-even SNR dB [95% CI] (flag) |',
          '|---|---|---|---|---|---|---|']
    for h in HEARTS:
        for p in P:
            d = g[str(h)][p]
            L.append(f'| {h} | {p} | {d["sd_cons"]:.4f} | {d["sd_noise"]["20"]:.4f} | {d["ratio_sd_cons_to_noise20"]:.2f} '
                     f'[{d["ratio_ci"][0]:.2f}, {d["ratio_ci"][1]:.2f}] | {d["pass_2x"]} | {d["breakeven_snr_db"]:.1f} '
                     f'[{d["breakeven_ci"][0]:.1f}, {d["breakeven_ci"][1]:.1f}] ({d["breakeven_flag"]}) |')
    L.append('')
for key, g in res['G4'].items():
    L += [f'## G4 [{key}]: n = {g["n_constructions"]} constructions, {g["n_hearts"]} hearts; reframe trigger = {g["reframe_trigger_R2_ge_0.9_both"]}', '',
          '| param | R2 [95% CI] | adj R2 | LOHO R2 [95% CI] |', '|---|---|---|---|']
    for p in P:
        d = g[p]
        L.append(f'| {p} | {d["R2"]:.3f} [{d["R2_ci"][0]:.3f}, {d["R2_ci"][1]:.3f}] | {d["R2_adj"]:.3f} | '
                 f'{d["R2_LOHO"]:.3f} [{d["R2_LOHO_ci"][0]:.3f}, {d["R2_LOHO_ci"][1]:.3f}] |')
    L.append('')
for loss in ['corr', 'amp']:
    L += [f'## Preview [{loss}]: between-construction CV of theta_hat/theta*', '',
          '| heart | param | CV median (range over theta*) | share seg | share mesh | share SxM | resid |', '|---|---|---|---|---|---|---|']
    for h in HEARTS:
        for p in P:
            d = res['preview'][loss][str(h)][p]
            L.append(f'| {h} | {p} | {100*d["cv_median"]:.2f}% ({100*d["cv_min"]:.2f} to {100*d["cv_max"]:.2f}) | '
                     f'{d["ss_share_segmentation"]:.2f} | {d["ss_share_mesh"]:.2f} | {d["ss_share_interaction"]:.2f} | {d["ss_share_residual"]:.2f} |')
    L.append('')
L += ['## Fit quality', '', '```', json.dumps(res['fit_quality'], indent=1, default=float), '```']
open(os.path.join(D, 'gates.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'runtime {time.time() - t_start:.1f}s')
