"""Export and owner check for the B1 arm C rerun (owner layer-sdf, 2026-10-28). Usage: python -I -B export_sdf.py <dir>

Writes into <dir>:
  estimates_sdf.csv          round 2, columns of results/2026-10-28_armC/estimates.csv plus `layer` = 'sdf_B1'
  estimates_rounds_sdf.csv.gz all rounds, same columns plus layer
  refinement_sdf.csv         refinement runs (as csv_export.py)
  owner_check_E1.json / .csv E1-style CV for the B1 layer and paired difference from primary (A2), both losses
Per-heart rows come from csv_export.export_heart (imported, unchanged). The CV and bootstrap follow
code/full/analysis/run_analysis.py (cv_per_heart, boot_median, boot_paired_diff) re-implemented here line for line:
CV(h) = median over the 3 theta* of SD/mean (ddof 1) of theta_hat/theta* across the heart's primary constructions
(reference included); estimand = median over hearts; 95% percentile CI from 2,000 cluster resamples of hearts,
fresh numpy default_rng(20261028) per cell; paired difference with hearts resampled jointly. Owner check only; the
integration owner produces the final table.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_export as X      # noqa: E402

SEED = 20261028
NB = 2000
P = ['v_endo', 'v_t']
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PRIMARY_CSV = os.path.join(ROOT, 'results', '2026-10-28_armC', 'estimates.csv')


def q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def cv_per_heart(rows):
    out = {}
    for h, eh in rows.groupby('patient'):
        d = dict(n_cons=int(eh.construction.nunique()), n_censored=int(eh.censored.sum()))
        for p in P:
            cvs = [np.std(g[f'ratio_{p}'].values, ddof=1) / np.mean(g[f'ratio_{p}'].values)
                   for _, g in eh.groupby('istar') if len(g) >= 2]
            d[f'cv_{p}'] = float(np.median(cvs)) if cvs else np.nan
        out[int(h)] = d
    return pd.DataFrame(out).T


def boot_median(v):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    bs = np.median(v[rng.integers(0, len(v), (NB, len(v)))], axis=1)
    return float(np.median(v)), q(bs)


def boot_paired_diff(a, b):
    rng = np.random.default_rng(SEED)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].values.astype(float); b = b.loc[idx].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    pick = rng.integers(0, len(a), (NB, len(a)))
    bs = np.median(a[pick], axis=1) - np.median(b[pick], axis=1)
    return float(np.median(a) - np.median(b)), q(bs), int(len(a))


def main(out):
    hs = sorted(int(os.path.basename(f)[13:16]) for f in glob.glob(os.path.join(out, 'hearts', 'heart_patient*.h5')))
    est, ref = [], []
    for p in hs:
        e, _, r, _ = X.export_heart(out, p)
        est += e; ref += r
    E = pd.DataFrame(est); E['layer'] = 'sdf_B1'
    cols = list(pd.read_csv(PRIMARY_CSV, nrows=1).columns)
    assert set(cols) <= set(E.columns), set(cols) - set(E.columns)
    E2 = E[E['round'] == 2][cols + ['layer']]
    E2.to_csv(os.path.join(out, 'estimates_sdf.csv'), index=False)
    E[cols[:cols.index('round')] + ['round'] + cols[cols.index('round') + 1:] + ['layer']].to_csv(
        os.path.join(out, 'estimates_rounds_sdf.csv.gz'), index=False)
    pd.DataFrame(ref).to_csv(os.path.join(out, 'refinement_sdf.csv'), index=False)
    A2 = pd.read_csv(PRIMARY_CSV)
    A2 = A2[A2.include_primary.astype(bool)]
    res = dict(n_hearts=len(hs), seed=SEED, n_boot=NB, primary_csv='results/2026-10-28_armC/estimates.csv',
               n_estimates_sdf=int(len(E2)), cells={})
    rows = []
    for loss in ('corr', 'amp'):
        s = cv_per_heart(E2[(E2.loss == loss) & E2.include_primary.astype(bool)])
        a = cv_per_heart(A2[A2.loss == loss])
        for p in P:
            ms, cs = boot_median(s[f'cv_{p}']); ma, ca = boot_median(a[f'cv_{p}'])
            d, cd, n = boot_paired_diff(s[f'cv_{p}'], a[f'cv_{p}'])
            cell = dict(loss=loss, parameter=p, cv_sdf=ms, ci_sdf=cs, cv_a2=ma, ci_a2=ca, diff_sdf_minus_a2=d,
                        ci_diff=cd, n_hearts_paired=n, n_cons_sdf=int(s.n_cons.sum()), n_cons_a2=int(a.n_cons.sum()),
                        n_censored_sdf=int(s.n_censored.sum()), n_censored_a2=int(a.n_censored.sum()))
            res['cells'][f'{loss}_{p}'] = cell; rows.append(cell)
            print(f'{loss} {p}: B1 {100*ms:.1f}% [{100*cs[0]:.1f}, {100*cs[1]:.1f}]  A2 {100*ma:.1f}% '
                  f'[{100*ca[0]:.1f}, {100*ca[1]:.1f}]  diff {100*d:+.1f} [{100*cd[0]:+.1f}, {100*cd[1]:+.1f}] n {n}')
    json.dump(res, open(os.path.join(out, 'owner_check_E1.json'), 'w'), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'owner_check_E1.csv'), index=False)
    # per-heart table
    s = cv_per_heart(E2[(E2.loss == 'corr')]); a = cv_per_heart(A2[A2.loss == 'corr'])
    s.join(a, lsuffix='_sdf', rsuffix='_a2').to_csv(os.path.join(out, 'per_heart_cv_corr.csv'), index_label='patient')
    # QC
    qc = dict(n_rows=int(len(E2)), nan=int(E2[['v_endo_hat', 'v_t_hat']].isna().sum().sum()),
              censored_by_loss=E2.groupby('loss').censored.sum().astype(int).to_dict(),
              ref_recovery_max_abs={p: float((E2[E2.is_reference & (E2.loss == 'corr')][f'ratio_{p}'] - 1).abs().max())
                                    for p in P})
    print(json.dumps(qc))
    json.dump(qc, open(os.path.join(out, 'qc_sdf.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
