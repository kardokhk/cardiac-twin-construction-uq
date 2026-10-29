"""Derived values quoted in the manuscript that are not in RESULTS.md.
Usage (project root): /project/home/p201509/envs/cvri-sim/bin/python -I code/manuscript/derived_values.py
Writes results/2026-10-28_analysis/derived_for_manuscript.json."""
import json
import numpy as np
import pandas as pd

E = pd.read_csv('results/2026-10-28_armC/estimates.csv')
E = E[(E['round'] == 2) & (E.loss == 'corr') & (E.include_primary.astype(bool))]
cv = E.groupby(['patient', 'istar'])['ratio_v_t'].agg(lambda s: s.std(ddof=1) / s.mean()).groupby('patient').median()
sd = E.groupby(['patient', 'istar'])['ratio_v_t'].var(ddof=1).groupby('patient').mean().pow(0.5)
T = pd.read_csv('results/2026-10-28_armT/estimates_v2.csv', dtype={'heart': str, 'ref': str})
t = T[(T.ref == '09') & (T.reg == 'trans') & (T.loss == 'corr') & (T['round'] == 2)]
ph = t.groupby('heart').rel_v_t.apply(lambda s: s.abs().mean())
out = {
    'v_t_cv_max_pct': round(100 * float(cv.max()), 1), 'v_t_cv_max_heart': int(cv.idxmax()),
    'n_v_t_above_fibre_0p6': int((E.v_t_hat > 0.6).sum()), 'n_primary_estimates': int(len(E)),
    'constructions_per_heart': {str(k): int(v) for k, v in E.groupby('patient').construction.nunique().value_counts().items()},
    'v_t_construction_sd_median': round(float(sd.median()), 4),
    'armT_ref09_trans_corr_v_t_abs_median_over_hearts_pct': round(100 * float(ph.median()), 1),
    'armT_ref09_trans_n_hearts': int(len(ph)),
}
json.dump(out, open('results/2026-10-28_analysis/derived_for_manuscript.json', 'w'), indent=1)
print(out)
