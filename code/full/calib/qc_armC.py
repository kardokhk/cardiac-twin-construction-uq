"""QC of arms C and N plus the A2 versus one-voxel trio comparison (owner calibC, 2026-10-28). No estimands.
Usage: python -I -B qc_armC.py <out_dir>   (reads estimates.csv, noise_estimates.csv, refinement.csv, constructions.csv;
the gates one-voxel trio estimates from results/2026-10-28_G3G4_ministudy/estimates.csv; mesh shares from
data/processed/acdc_mesh_v2/MANIFEST.tsv). Writes <out_dir>/qc.json and prints it.
Definitions: quantiles are the 2.5th, 50th and 97.5th percentiles. Trio CV (same rule as the gates preview,
code/gates/calib/analyse_gates.py): per theta*, SD (ddof 1) / mean of theta_hat/theta* over constructions; per heart the
median over the 3 theta*. Sets: 'gates12' = 6 sources x 2 resolutions at RV wall 3 mm (the gates set);
'primary10' = the 5 primary sources x 2 resolutions.
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
D = sys.argv[1]
E = pd.read_csv(os.path.join(D, 'estimates.csv'))
N = pd.read_csv(os.path.join(D, 'noise_estimates.csv'))
R = pd.read_csv(os.path.join(D, 'refinement.csv'))
CON = pd.read_csv(os.path.join(D, 'constructions.csv'))
M = pd.read_csv(os.path.join(ROOT, 'data/processed/acdc_mesh_v2/MANIFEST.tsv'), sep='\t')
inc = M[M.include_primary.astype(bool) | M.include_sensitivity.astype(bool)]
q = lambda x: [float(v) for v in np.nanpercentile(x, [2.5, 50, 97.5])] if len(x) else None
out = {}
out['completion'] = dict(
    manifest_included=int(len(inc)), calibrated=int(CON.construction.nunique()),
    n_missing=int(len(set(inc.construction_id) - set(CON.construction))),
    missing=sorted(set(inc.construction_id) - set(CON.construction)),
    hearts=int(E.patient.nunique()), estimate_rows=int(len(E)), expected_rows=int(len(inc) * 3 * 2),
    nan_estimates=int(E[['v_endo_hat', 'v_t_hat']].isna().any(axis=1).sum()),
    unactivated_constructions=int((CON.unactivated > 0).sum()),
    by_set={f'{k[0]}|{k[1]}|rv{k[2]}': int(v) for k, v in
            CON.groupby(['analysis_set', 'resolution_mm', 'rv_wall_mm']).construction.nunique().items()},
    noise_rows=int(len(N)), noise_expected=int(E.patient.nunique() * 3 * 16 * 20 * 2),
    noise_nan=int(N[['v_endo_hat', 'v_t_hat']].isna().any(axis=1).sum()))
fl = {}
for (loss, src), g in E.groupby(['loss', 'source']):
    fl[f'{loss}|{src}'] = dict(n=int(len(g)), fval_q=q(g.fval_emu), censored=int(g.censored.sum()),
                               lo_v_endo=int(g.bound_lo_v_endo.sum()), hi_v_endo=int(g.bound_hi_v_endo.sum()),
                               lo_v_t=int(g.bound_lo_v_t.sum()), hi_v_t=int(g.bound_hi_v_t.sum()),
                               vt_gt_vfib=int(g.flag_vt_gt_vfib.sum()))
out['fit_and_bounds_by_loss_source'] = fl
fs = {}
for (loss, aset), g in E.groupby(['loss', 'analysis_set']):
    fs[f'{loss}|{aset}'] = dict(n=int(len(g)), fval_q=q(g.fval_emu), censored=int(g.censored.sum()))
out['fit_and_bounds_by_loss_set'] = fs
rf = E[E.is_reference]
out['reference_recovery_max_abs_rel_err'] = {
    l: dict(v_endo=float((g.ratio_v_endo - 1).abs().max()), v_t=float((g.ratio_v_t - 1).abs().max()),
            n=int(len(g))) for l, g in rf.groupby('loss')}
out['refinement'] = dict(n_runs=int(len(R)), emu0_rms_ms_q=q(R.emu0_rms_ms), emu0_maxabs_ms_max=float(R.emu0_maxabs_ms.max()),
                         per_construction_max=int(CON.n_refinement.max()))
nb = {}
for (loss, c, lev), g in N.groupby(['loss', 'comparator', 'level']):
    nb[f'{loss}|{c}|{lev}'] = dict(n=int(len(g)), censored=int(g.censored.sum()),
                                   sd_ratio_v_endo_q=None, sd_ratio_v_t_q=None)
    s = g.groupby(['patient', 'istar']).agg(a=('ratio_v_endo', 'std'), b=('ratio_v_t', 'std'))
    nb[f'{loss}|{c}|{lev}'].update(sd_ratio_v_endo_q=q(s.a), sd_ratio_v_t_q=q(s.b))
out['noise_by_level'] = nb
# root flags
nonref = E[(~E.is_reference) & (E.loss == 'corr') & (E.istar == 0)]
out['root_flag'] = {f'{k}': int(v) for k, v in nonref.groupby('resolution_mm').root_flag.sum().items()}
out['root_flag_denominator'] = {f'{k}': int(v) for k, v in nonref.groupby('resolution_mm').size().items()}
# trio A2 versus one-voxel
G = pd.read_csv(os.path.join(ROOT, 'results/2026-10-28_G3G4_ministudy/estimates.csv'))
G = G[G['round'] == 2].copy(); G['resolution_mm'] = G.vox
T = E[E.patient.isin([101, 102, 104]) & (E.rv_wall_mm == 3.0)].copy()
tr = {}
for name, src_ok in [('gates12', lambda s: True), ('primary10', lambda s: s != 'ukbb_cardiac_fcn_sa')]:
    for arm, X in [('A2', T), ('onevoxel_gates', G)]:
        X = X[X.source.map(src_ok)]
        for (p, loss), g in X.groupby(['patient', 'loss']):
            for par in ['v_endo', 'v_t']:
                cv = [float(h[f'ratio_{par}'].std(ddof=1) / h[f'ratio_{par}'].mean()) for _, h in g.groupby('istar')]
                tr[f'{name}|{arm}|{loss}|{p}|{par}'] = dict(cv_median_over_theta=float(np.median(cv)),
                                                            cv_each_theta=cv, n_cons=int(g.construction.nunique()))
out['trio_cv'] = tr
key = ['patient', 'source', 'resolution_mm', 'istar', 'loss']
P = T.merge(G, on=key, suffixes=('_a2', '_1v'))
out['trio_paired'] = {f'{l}|{par}': dict(n=int(len(g)), median_ratio_a2_over_1v=float(np.median(g[f'ratio_{par}_a2'] / g[f'ratio_{par}_1v'])),
                                           q=q(g[f'ratio_{par}_a2'] / g[f'ratio_{par}_1v']))
                      for l, g in P.groupby('loss') for par in ['v_endo', 'v_t']}
mt = M[M.heart.isin([101, 102, 104]) & (M.rv_wall_mm == 3.0) & M.g2_pass.astype(bool)]
out['trio_mesh_share'] = {f'{p}|{v}': dict(A2=float(g.endo_vol_frac.median()), onevoxel=float(g.endo_vol_frac_onevoxel.median()),
                                           n=int(len(g)))
                          for (p, v), g in mt.groupby(['heart', 'resolution_mm'])}
json.dump(out, open(os.path.join(D, 'qc.json'), 'w'), indent=1)
print(json.dumps({k: (dict(out[k], missing=out[k]['missing'][:20]) if k == 'completion' else out[k]) for k in ['completion', 'reference_recovery_max_abs_rel_err', 'refinement', 'root_flag',
                                      'root_flag_denominator', 'trio_paired', 'trio_mesh_share']}, indent=1))
