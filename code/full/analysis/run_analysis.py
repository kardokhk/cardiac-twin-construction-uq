"""I11 full-arm analysis (owner analysis, 2026-10-28). Every estimand of STUDY_PLAN section 4 under Amendment 1.

Usage: python -I code/full/analysis/run_analysis.py [--nboot 2000] [--out results/2026-10-28_analysis] [--figs figures/drafts]
Run from the project root. Reads only upstream CSVs (listed in INPUTS). Writes results.json, RESULTS.md, tables/*.csv,
and DRAFT figures with their raw-data CSVs. Every bootstrap: cluster bootstrap over hearts, 2,000 resamples, a fresh
numpy default_rng(20261028) per estimand cell (so each cell is reproducible on its own).

Definitions (fixed here, before reading any full-arm estimand value; README states them again):
  y = theta_hat / theta*, round-2 (arm C) estimates. Primary set: include_primary (A1 entry rule; 4 primary networks +
      expert, 1.5 and 1.0 mm, RV wall 3 mm, reference included); primary loss corr. Bound estimates kept at the bound.
  E1 CV(h, p) = median over the 3 theta* of SD(y)/mean(y) across the heart's constructions (ddof 1);
      estimand = median over hearts; percentile CI from resampling hearts.
  E2 LMM: log y ~ source + resolution + source:resolution + theta*-index (sum-to-zero coding) + random heart intercept,
      REML (closed-form profile likelihood for one random intercept; checked against statsmodels MixedLM).
      Share of term k = V_k / (V_S + V_M + V_SxM + V_theta + s2_heart + s2_resid), V_k = variance over observations
      (ddof 0) of that term's fixed-effect contribution X_k b_k. Also the within-heart construction share
      (V_S + V_M + V_SxM) / (V_S + V_M + V_SxM + s2_resid). CI: refit on resampled hearts (duplicates relabelled).
  E3 SD_cons(h, p) = sqrt(mean over theta* of var(y) across constructions); SD_noise(h, p, c, L) likewise across the
      20 replicates. Population curves = median over hearts. Break-even = level at which the median SD_noise equals the
      median SD_cons, log SD linearly interpolated against severity (s = -SNR dB for white/lp40; s = ln level for
      wander % and displacement mm). Outside the grid -> "beyond grid" (no extrapolated point). Matching loss.
  E4 Three-length rule exactly as code/gates/calib/analyse_gates.py g4_block, on the primary non-reference
      constructions: r = mean over theta* of theta_hat_c / theta_hat_ref; flag if in-sample R2 >= 0.9 for both
      parameters (plan wording), LOHO also reported against 0.9.
  E5 Arm T: per heart mean over theta* of theta_hat/theta* - 1 (signed, abs), mean over hearts, round 2 of estimates_v2.csv (per-arm refinement). Arm F: per
      heart mean over theta* of log(theta_hat/theta*), exp of the mean over hearts (geometric mean), t and bootstrap CI.
  E6 Exploratory (ACDC group, 5 x 10): Kruskal-Wallis per parameter (epsilon^2 effect size) and each group against
      the other 40 hearts (difference of median CV); bootstrap CIs stratified by group; Holm over the 12 tests.
  E7 Sensitivities: amplitude loss; ukbb_cardiac added (all rows, and G1-usable only); censored dropped; root-flagged
      dropped; RV wall 4 mm (S x M, 10 hearts) and 2 mm (1.0 mm only); one-voxel layer (trio, from the gates run);
      1.5 mm only and 1.0 mm only. Paired difference to the primary on the same hearts with bootstrap CI.

Correction 2026-10-28 (follow-up round; additive, no pre-registered definition changed, primary numbers unchanged):
  C1 (E1) The reference construction (expert, 1.5 mm, RV wall 3 mm) is a member of S x M (STUDY_PLAN section 3 lists
      the expert labels as a level of S; section 4 item 1 takes the CV across the S x M constructions), so the primary
      set keeps it; its theta_hat/theta* is about 1 by construction (targets simulated on it). Added sensitivity:
      reference construction removed (set primary_no_ref), same bootstrap seed.
  C2 (E2) Reference rows are fixed near log y = 0, so the random heart intercept absorbs each heart's mean offset of the
      non-reference constructions from the reference: the 'heart' share is relabelled "heart-level mean construction
      offset plus between-heart variation" (JSON key unchanged: 'heart'). Added E2c, descriptive only: per heart,
      sequential (type I, order S, M, S x M) sums of squares of log y after removing the theta* mean, on
      non-reference rows only; median and IQR over hearts. No new inferential model; a heart x source variance-component
      model is listed as a candidate for a later decision.
  C3 (E6) The effect size (H - k + 1)/(n - k) is eta^2_H, not epsilon^2; relabelled, and epsilon^2_R = H/(n - 1) added
      (same bootstrap resamples, so the eta^2_H CI is unchanged).
  C4 Root-shift flag: the A4 denominator read literally (all non-reference constructions of the same resolution with a
      mapped root, from the mesh MANIFEST, every analysis set) reported beside the primary-set denominator, with the
      mesh2 and calibC denominators reconciled.
  C5 The statsmodels ConvergenceWarning from the MixedLM cross-check is captured per parameter and reported.
  C6 (E3) Descriptive only, not pre-registered: break-even level by log-linear extrapolation (OLS of log median noise
      SD on severity over all grid levels), labelled extrapolated; the primary break-even stays "beyond grid".

Amendment 2, 2026-10-28 (final/AMENDMENT_2_2026-10-28.md, declared before these were run, added after full-arm data;
additive, every earlier results.json value unchanged):
  B2 (E2d, secondary model) log y ~ S * M + theta* + (1 | heart) + (1 | heart:source), REML, primary non-reference rows,
      both losses. Own two-component REML (per-heart Woodbury blocks, relative SDs on L-BFGS-B bounds at 0, 4 starts),
      statsmodels MixedLM (vc_formula) cross-check on the point fits. Expert S x M column dropped (no expert 1.5 mm row).
      Cluster bootstrap over hearts, 2,000, rng(20261028) per cell, rng.choice per resample as E2. Within-heart
      construction share = (V_S + V_M + V_SxM + s2_heart:source) / (same + s2_resid). Boundary and convergence flags
      counted over bootstrap fits.
  B3 (E7) RV wall 2 mm: the 1.0 mm set relabelled as the stated substitute for the unmeshable 1.5 mm level (no new runs).
  B1 (integration owner) Smooth-surface fast layer, declared sensitivity: E1 CV on results/2026-10-28_layer_sdf/
      estimates_sdf.csv (same cv_per_heart rule), both losses; paired difference from A2 = difference of medians over hearts,
      hearts resampled jointly (boot_paired_diff, 2,000, rng(20261028)); median of per-heart differences beside it; layer
      share of myocardial volume and the unit test against its +-15% lines, from the layer-sdf owner's tables.
  B4 (integration owner) E3 on the extended grid (A3 + B4 levels; armC noise_estimates.csv concatenated with
      results/2026-10-28_noise_ext/noise_estimates_ext.csv), same A3 crossing rule, "beyond grid" kept, both losses, plus
      censored dropped. COMP (A3 grid) is unchanged; the extended grid is COMP_EXT. Measured minus E3x with a joint
      resample CI (same rng sequence). Self-check: the generalised block on the A3 grid reproduces E3 exactly.
"""
import warnings
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--nboot', type=int, default=2000)
ap.add_argument('--out', default='results/2026-10-28_analysis')
ap.add_argument('--figs', default='figures/drafts')
ap.add_argument('--nofig', action='store_true')
A = ap.parse_args()
NB = A.nboot
SEED = 20261028
P = ['v_endo', 'v_t']
OUT = A.out
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)

INPUTS = dict(
    armC='results/2026-10-28_armC/estimates.csv',
    armC_cons='results/2026-10-28_armC/constructions.csv',
    armN='results/2026-10-28_armC/noise_estimates.csv',
    onevox='results/2026-10-28_G3G4_ministudy/estimates.csv',
    armT='results/2026-10-28_armT/estimates_v2.csv',  # corrected per-arm refinement (2026-10-28)
    armT_json='results/2026-10-28_armT/analysis_v2.json',
    armF='results/2026-10-28_armF/estimates_armF.csv',
    armF_json='results/2026-10-28_armF/summary_armF.json',
    manifest='data/processed/acdc_mesh_v2/MANIFEST.tsv',
)
E = pd.read_csv(INPUTS['armC'])
CON = pd.read_csv(INPUTS['armC_cons'])
NZ = pd.read_csv(INPUTS['armN'])
G = pd.read_csv(INPUTS['onevox'])
TT = pd.read_csv(INPUTS['armT'], dtype={'heart': str, 'ref': str})
FF = pd.read_csv(INPUTS['armF'])
MAN = pd.read_csv(INPUTS['manifest'], sep='\t')
assert (E['round'] == 2).all()
PRIMARY_SOURCES = ['expert', 'nnunet_task027_acdc', 'nnunet_task114_mnms', 'nnunet_ukb_sax', 'cinema_acdc_sax']
WEAK = 'ukbb_cardiac_fcn_sa'
GROUP = E.groupby('patient').group.first().to_dict()
R = dict(inputs=INPUTS, seed=SEED, n_boot=NB)


def q(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [np.nan, np.nan]


def ipr(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), p2_5=float(np.percentile(x, 2.5)), p97_5=float(np.percentile(x, 97.5)), n=int(len(x)))


# ================================================================== E1: construction CV
def cv_per_heart(rows):
    """rows: one loss. Returns DataFrame heart x [cv_v_endo, cv_v_t, sd_v_endo, sd_v_t, n_cons, ...]."""
    out = []
    for h, eh in rows.groupby('patient'):
        d = dict(patient=int(h), group=GROUP[h], n_cons=int(eh.construction.nunique()))
        for p in P:
            col = f'ratio_{p}'
            cvs, vs = [], []
            for i, g in eh.groupby('istar'):
                v = g[col].values
                if len(v) >= 2:
                    cvs.append(np.std(v, ddof=1) / np.mean(v)); vs.append(np.var(v, ddof=1))
            d[f'cv_{p}'] = float(np.median(cvs)) if len(cvs) else np.nan
            d[f'cv_min_{p}'] = float(np.min(cvs)) if cvs else np.nan
            d[f'cv_max_{p}'] = float(np.max(cvs)) if cvs else np.nan
            d[f'sd_{p}'] = float(np.sqrt(np.mean(vs))) if vs else np.nan
        d['n_censored'] = int(eh.censored.sum()); d['n_est'] = int(len(eh))
        out.append(d)
    return pd.DataFrame(out).set_index('patient')


def boot_median(v, nb=None):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    nb = nb or NB
    bs = np.median(v[rng.integers(0, len(v), (nb, len(v)))], axis=1)
    return float(np.median(v)), q(bs)


def boot_paired_diff(a, b):
    """median(a) - median(b) over common hearts, hearts resampled jointly."""
    rng = np.random.default_rng(SEED)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].values; b = b.loc[idx].values
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    pick = rng.integers(0, len(a), (NB, len(a)))
    bs = np.median(a[pick], axis=1) - np.median(b[pick], axis=1)
    return float(np.median(a) - np.median(b)), q(bs), int(len(a))


def summarise_cv(tab, label, src):
    d = dict(label=label, source_csv=src, n_hearts=int(tab[f'cv_{P[0]}'].notna().sum()),
             n_constructions=int(tab.n_cons.sum()), n_estimates=int(tab.n_est.sum()), n_censored=int(tab.n_censored.sum()))
    for p in P:
        m, ci = boot_median(tab[f'cv_{p}'])
        d[p] = dict(cv_median=m, ci=ci, per_heart=ipr(tab[f'cv_{p}'].dropna()),
                    sd_median=float(tab[f'sd_{p}'].median()), sd_ci=boot_median(tab[f'sd_{p}'])[1])
    return d


def sel(loss, primary=True, extra=None, drop=None):
    e = E[E.loss == loss]
    m = e.include_primary.copy() if primary else pd.Series(False, index=e.index)
    if extra is not None:
        m = m | extra(e)
    e = e[m]
    if drop is not None:
        e = e[~drop(e)]
    return e


SETS = {}
SETS['primary'] = sel('corr')
SETS['amp_loss'] = sel('amp')
SETS['ukbb_added_all'] = sel('corr', extra=lambda e: e.analysis_set == 'ukbb_sensitivity')
SETS['ukbb_added_g1usable'] = sel('corr', extra=lambda e: (e.analysis_set == 'ukbb_sensitivity') & e.g1_usable)
SETS['censored_dropped'] = sel('corr', drop=lambda e: e.censored)
SETS['censored_dropped_amp'] = sel('amp', drop=lambda e: e.censored)
SETS['root_flag_dropped'] = sel('corr', drop=lambda e: e.root_flag)
SETS['res_1.5_only'] = sel('corr', drop=lambda e: e.resolution_mm != 1.5)
SETS['res_1.0_only'] = sel('corr', drop=lambda e: e.resolution_mm != 1.0)
H10 = sorted(E[E.analysis_set == 'rv_wall_sensitivity'].patient.unique())
ec = E[E.loss == 'corr']
SETS['rv3_10hearts_SxM'] = ec[ec.include_primary & ec.patient.isin(H10)]
SETS['rv4_10hearts_SxM'] = ec[(ec.rv_wall_mm == 4.0) & ec.patient.isin(H10)]
SETS['rv3_10hearts_1.0mm'] = ec[ec.include_primary & ec.patient.isin(H10) & (ec.resolution_mm == 1.0)]
SETS['rv2_10hearts_1.0mm'] = ec[(ec.rv_wall_mm == 2.0) & (ec.resolution_mm == 1.0) & ec.patient.isin(H10)]
SETS['rv4_10hearts_1.0mm'] = ec[(ec.rv_wall_mm == 4.0) & (ec.resolution_mm == 1.0) & ec.patient.isin(H10)]
SETS['rv3_10hearts_1.5mm'] = ec[ec.include_primary & ec.patient.isin(H10) & (ec.resolution_mm == 1.5)]
SETS['rv4_10hearts_1.5mm'] = ec[(ec.rv_wall_mm == 4.0) & (ec.resolution_mm == 1.5) & ec.patient.isin(H10)]
# one-voxel trio: gates run (round 2), primary sources only, against A2 on the same 10 S x M constructions
TRIO = [101, 102, 104]
g2 = G[(G['round'] == 2) & (G.loss == 'corr')].copy()
SETS['trio_onevoxel_10'] = g2[g2.source.isin(PRIMARY_SOURCES)]
SETS['trio_A2_10'] = SETS['primary'][SETS['primary'].patient.isin(TRIO)]
SETS['trio_onevoxel_12'] = g2
SETS['trio_A2_12'] = ec[ec.patient.isin(TRIO) & (ec.include_primary | (ec.analysis_set == 'ukbb_sensitivity'))]
# C1/C2 (2026-10-28): primary set without the reference construction (expert, 1.5 mm, RV wall 3 mm)
SETS['primary_no_ref'] = SETS['primary'][~SETS['primary'].is_reference.astype(bool)]
SETS['amp_no_ref'] = SETS['amp_loss'][~SETS['amp_loss'].is_reference.astype(bool)]

CVT = {k: cv_per_heart(v) for k, v in SETS.items()}
SRC = {k: (INPUTS['onevox'] if 'onevoxel' in k else INPUTS['armC']) for k in SETS}
R['E1_primary'] = summarise_cv(CVT['primary'], 'primary: corr loss, primary sources x {1.5, 1.0} mm', INPUTS['armC'])
CVT['primary'].to_csv(os.path.join(OUT, 'tables', 'per_heart_cv_primary.csv'))
R['E1_per_theta_mean_check'] = {p: float(np.median([np.mean([np.std(g[f'ratio_{p}'], ddof=1) / np.mean(g[f'ratio_{p}'])
                                                              for _, g in eh.groupby('istar')])
                                                     for _, eh in SETS['primary'].groupby('patient')])) for p in P}

SENS_PAIRS = [  # (label, set, comparator set for the paired difference)
    ('amplitude loss', 'amp_loss', 'primary'),
    ('ukbb_cardiac added (all rows)', 'ukbb_added_all', 'primary'),
    ('ukbb_cardiac added (G1-usable rows)', 'ukbb_added_g1usable', 'primary'),
    ('censored dropped (corr)', 'censored_dropped', 'primary'),
    ('censored dropped (amp)', 'censored_dropped_amp', 'amp_loss'),
    ('root-flagged dropped', 'root_flag_dropped', 'primary'),
    ('1.5 mm only (S at 1.5 mm)', 'res_1.5_only', 'primary'),
    ('1.0 mm only (S at 1.0 mm)', 'res_1.0_only', 'primary'),
    ('1.0 mm only minus 1.5 mm only', 'res_1.0_only', 'res_1.5_only'),
    ('RV wall 3 mm, S x M, 10 hearts', 'rv3_10hearts_SxM', 'primary'),
    ('RV wall 4 mm, S x M, 10 hearts', 'rv4_10hearts_SxM', 'rv3_10hearts_SxM'),
    ('RV wall 4 mm, 1.5 mm only, 10 hearts', 'rv4_10hearts_1.5mm', 'rv3_10hearts_1.5mm'),
    ('RV wall 4 mm, 1.0 mm only, 10 hearts', 'rv4_10hearts_1.0mm', 'rv3_10hearts_1.0mm'),
    ('RV wall 2 mm, 1.0 mm only, 10 hearts (extra set, not pre-registered)', 'rv2_10hearts_1.0mm', 'rv3_10hearts_1.0mm'),
    ('trio A2, 10 primary S x M', 'trio_A2_10', 'trio_onevoxel_10'),
    ('trio A2, 12 (with ukbb_cardiac)', 'trio_A2_12', 'trio_onevoxel_12'),
    ('trio one-voxel, 10 primary S x M', 'trio_onevoxel_10', 'trio_A2_10'),
    ('trio one-voxel, 12', 'trio_onevoxel_12', 'trio_A2_12'),
    ('reference construction removed (expert 1.5 mm RV 3 mm; added 2026-10-28)', 'primary_no_ref', 'primary'),
]
sens = []
for lab, k, kc in SENS_PAIRS:
    d = summarise_cv(CVT[k], lab, SRC[k]); d['set'] = k; d['vs'] = kc
    for p in P:
        diff, ci, n = boot_paired_diff(CVT[k][f'cv_{p}'], CVT[kc][f'cv_{p}'])
        d[p]['diff_vs'] = dict(diff=diff, ci=ci, n_common=n)
    if k.startswith('trio'):
        d['per_heart'] = {int(h): {p: float(CVT[k].loc[h, f'cv_{p}']) for p in P} for h in CVT[k].index}
    sens.append(d)
R['E7_sensitivity_cv'] = sens
_ref = SETS['primary'][SETS['primary'].is_reference.astype(bool)]
R['E1_reference_membership'] = dict(
    note=('The reference construction (expert labels, 1.5 mm, RV wall 3 mm) is a member of S x M per STUDY_PLAN '
          'section 3 (expert labels are a level of S) and section 4 item 1 (CV across the S x M constructions), so the '
          'primary set includes it. Targets are simulated on it, so its theta_hat/theta* is about 1 by construction. '
          'Sensitivity without it: E7 set primary_no_ref.'),
    n_reference_constructions=int(_ref.construction.nunique()),
    reference_ratio={p: ipr(_ref[f'ratio_{p}']) for p in P},
    no_ref={p: next(d[p] for d in sens if d['set'] == 'primary_no_ref') for p in P})
rv2_15 = E[(E.rv_wall_mm == 2.0) & (E.resolution_mm == 1.5)]
R['E7_rv2_at_1.5mm'] = dict(n_constructions=int(rv2_15.construction.nunique()), constructions=sorted(rv2_15.construction.unique()),
                            note='RV wall 2 mm at 1.5 mm: 49 of 50 fail G2 F3 (mesh2); CV not estimable')
# Amendment 2 B3 (added after full-arm data; no new runs): RV wall lower bracket, 1.0 mm set as the stated substitute.
# Values are the existing E7 rows (rv2_10hearts_1.0mm against rv3_10hearts_1.0mm; rv4 1.0 mm as the upper bracket), relabelled.
_m2 = MAN[(MAN.rv_wall_mm == 2.0) & (MAN.resolution_mm == 1.5)]
_nf2 = int(_m2.exclusion_reason.astype(str).str.startswith('G2 F3').sum())
_rv2 = next(d for d in sens if d['set'] == 'rv2_10hearts_1.0mm')
_rv4 = next(d for d in sens if d['set'] == 'rv4_10hearts_1.0mm')
R['E7_B3_rv2_lower_bracket'] = dict(
    label=(f'RV wall 2 mm lower bracket, 1.0 mm set, {_rv2["n_hearts"]} hearts (Amendment 2 B3): not pre-registered; '
           f'1.5 mm level not estimable ({_nf2} of {len(_m2)} fail G2 F3)'),
    **{'n_1.5mm_constructions': int(len(_m2)), 'n_1.5mm_fail_G2_F3': _nf2},
    set=_rv2['set'], vs=_rv2['vs'], n_hearts=_rv2['n_hearts'], n_constructions=_rv2['n_constructions'],
    n_estimates=_rv2['n_estimates'], n_censored=_rv2['n_censored'], **{p: _rv2[p] for p in P},
    upper_bracket_same_resolution=dict(set=_rv4['set'], vs=_rv4['vs'], **{p: _rv4[p] for p in P}))


# ================================================================== E2: variance shares (REML, one random intercept)
def design(rows):
    import patsy
    d = rows.assign(ly=0.0, res=rows.resolution_mm.map(lambda x: f'{x:.1f}'), th=rows.istar.astype(str))
    dm = patsy.dmatrix('C(source, Sum) * C(res, Sum) + C(th, Sum)', d, return_type='dataframe')
    terms = {}
    for t, sl in dm.design_info.term_slices.items():
        n = t.name()
        key = ('intercept' if n == 'Intercept' else 'segmentation_S' if n == 'C(source, Sum)' else 'mesh_M' if n == 'C(res, Sum)'
               else 'SxM' if ':' in n else 'theta_star' if n == 'C(th, Sum)' else n)
        terms[key] = sl
    return dm.values, terms


def reml_fit(y, X, g):
    """Random-intercept LMM by REML. g: integer cluster codes 0..k-1. Returns beta, s2_heart, s2_resid, converged."""
    k = g.max() + 1
    n_i = np.bincount(g, minlength=k).astype(float)
    Sx = np.zeros((k, X.shape[1])); np.add.at(Sx, g, X)
    Sy = np.bincount(g, weights=y, minlength=k)
    XtX = X.T @ X; Xty = X.T @ y; yty = y @ y
    N, p = X.shape

    def parts(lam):
        w = lam / (1 + lam * n_i)
        A = XtX - (Sx * w[:, None]).T @ Sx
        b = Xty - (Sx * (w * Sy)[:, None]).sum(0)
        beta = np.linalg.solve(A, b)
        rr = yty - (w * Sy ** 2).sum() - beta @ b    # r' H^-1 r at the GLS beta
        return A, beta, rr

    def nll(llam):
        lam = np.exp(llam)
        A, beta, rr = parts(lam)
        s2 = rr / (N - p)
        return 0.5 * ((N - p) * np.log(s2) + np.log1p(lam * n_i).sum() + np.linalg.slogdet(A)[1])
    r = optimize.minimize_scalar(nll, bounds=(-20, 8), method='bounded', options=dict(xatol=1e-8))
    llam = r.x if nll(r.x) < nll(-20) else -20
    lam = np.exp(llam)
    A, beta, rr = parts(lam)
    s2 = rr / (N - p)
    return beta, lam * s2, s2, bool(r.success)


def shares_arr(y, X, terms, g):
    beta, s2h, s2e, ok = reml_fit(y, X, g)
    V = {t: float(np.var(X[:, sl] @ beta[sl])) for t, sl in terms.items() if t != 'intercept'}
    V['heart'] = float(s2h); V['residual'] = float(s2e)
    tot = sum(V.values())
    sh = {t: v / tot for t, v in V.items()}
    cons = V['segmentation_S'] + V['mesh_M'] + V['SxM']
    sh['within_heart_construction'] = cons / (cons + V['residual'])
    sh['segmentation_of_construction'] = V['segmentation_S'] / cons
    return sh, V, ok


def shares_boot(rows, p):
    rows = rows.reset_index(drop=True)
    y = np.log(rows[f'ratio_{p}'].values)
    X, terms = design(rows)
    hid = rows.patient.values
    point, V, ok = shares_arr(y, X, terms, pd.factorize(hid)[0])
    rng = np.random.default_rng(SEED)
    hearts = np.array(sorted(np.unique(hid)))
    byh = {h: np.nonzero(hid == h)[0] for h in hearts}
    bs, nfail = [], 0
    for b in range(NB):
        pick = rng.choice(hearts, len(hearts), replace=True)
        idx = np.concatenate([byh[h] for h in pick])
        gb = np.concatenate([np.full(len(byh[h]), j) for j, h in enumerate(pick)])
        try:
            s_, _, okb = shares_arr(y[idx], X[idx], terms, gb); bs.append(s_); nfail += (not okb)
        except Exception:
            nfail += 1
    out = {t: dict(share=point[t], ci=q([s_[t] for s_ in bs])) for t in point}
    return dict(shares=out, variance=V, converged=ok, n_boot_ok=len(bs), n_boot_flag=nfail,
                n_rows=int(len(rows)), n_hearts=int(len(hearts)), n_constructions=int(rows.construction.nunique()))


def ss_shares_per_heart(rows, p):
    """Gates-preview definition (analyse_gates.py): per heart, two-way (source x resolution) sums of squares of log y
    after removing the theta* mean; shares of the total. Median over hearts, bootstrap CI over hearts."""
    col = f'ratio_{p}'
    per = []
    for h, eh in rows.groupby('patient'):
        ly = np.log(eh[col].values)
        yc = ly - eh.groupby('istar')[col].transform(lambda s_: np.log(s_).mean()).values
        dd = eh.assign(yc=yc, res=eh.resolution_mm)
        tot = float((dd.yc ** 2).sum())
        mS = dd.groupby('source').yc.mean(); nS = dd.groupby('source').size()
        mM = dd.groupby('res').yc.mean(); nM = dd.groupby('res').size()
        mSM = dd.groupby(['source', 'res']).yc.mean(); nSM = dd.groupby(['source', 'res']).size()
        ssS = float((nS * mS ** 2).sum()); ssM = float((nM * mM ** 2).sum()); ssSM = float((nSM * mSM ** 2).sum()) - ssS - ssM
        per.append(dict(patient=int(h), segmentation_S=ssS / tot, mesh_M=ssM / tot, SxM=ssSM / tot,
                        residual_theta=1 - (ssS + ssM + ssSM) / tot))
    per = pd.DataFrame(per).set_index('patient')
    return {t: dict(zip(['median', 'ci'], boot_median(per[t].values))) for t in per.columns}, per


# statsmodels cross-check on the primary set (point estimate only)
try:
    import statsmodels.formula.api as smf
    chk = {}
    for p in P:
        d = SETS['primary'].assign(ly=np.log(SETS['primary'][f'ratio_{p}']), res=SETS['primary'].resolution_mm.astype(str))
        with warnings.catch_warnings(record=True) as wl:  # C5: record, per parameter, instead of printing once
            warnings.simplefilter('always')
            m = smf.mixedlm('ly ~ C(source, Sum)*C(res, Sum) + C(istar, Sum)', d, groups=d.patient).fit(reml=True, method='powell')
        y = d.ly.values; X, _ = design(SETS['primary']); g = pd.factorize(SETS['primary'].patient)[0]
        beta, s2h, s2e, _ = reml_fit(y, X, g)
        chk[p] = dict(warnings=[f'{w_.category.__name__}: {w_.message}' for w_ in wl],
                      warning_note=('statsmodels MixedLM.fit warns "MLE may be on the boundary" whenever an estimated '
                                    'random-effect variance is below the fixed absolute threshold 0.01 '
                                    '(mixed_linear_model.py, statsmodels 0.15.0); applies to the cross-check fit only, '
                                    'not to the own REML fit that produces the shares'),
                      statsmodels_converged=bool(m.converged),
                      statsmodels_s2_heart=float(m.cov_re.iloc[0, 0]), statsmodels_s2_resid=float(m.scale),
                      own_s2_heart=float(s2h), own_s2_resid=float(s2e),
                      max_abs_beta_diff=float(np.max(np.abs(np.sort(beta) - np.sort(m.fe_params.values)))))
    R['E2_statsmodels_check'] = chk
except Exception as ex:  # report, do not hide
    R['E2_statsmodels_check'] = dict(error=repr(ex))

import multiprocessing as mp  # noqa: E402
CTX = mp.get_context('fork')   # workers inherit SETS; each task seeds its own rng(SEED)
NPROC = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('NPROC', '4')))


def _e2_task(kp):
    return kp, shares_boot(SETS[kp[0]], kp[1])


E2_SETS = ['primary', 'amp_loss', 'ukbb_added_g1usable', 'censored_dropped', 'root_flag_dropped']
with CTX.Pool(min(NPROC, 10)) as pool:
    e2 = dict(pool.map(_e2_task, [(k, p) for k in E2_SETS for p in P]))
R['E2_shares'] = {k: {p: e2[(k, p)] for p in P} for k in E2_SETS}
R['E2b_ss_per_heart'] = {}
for k in ['primary', 'amp_loss']:
    R['E2b_ss_per_heart'][k] = {}
    for p in P:
        R['E2b_ss_per_heart'][k][p], per = ss_shares_per_heart(SETS[k], p)
        per.to_csv(os.path.join(OUT, 'tables', f'ss_shares_per_heart_{k}_{p}.csv'))


def ss_typeI_per_heart(rows, p):
    """C2 (2026-10-28), descriptive: per heart, log y minus its theta* mean; sequential (type I) sums of squares in the
    order source, resolution, source x resolution, residual (within source x resolution cell, i.e. theta* x
    construction). Sequential because removing the reference leaves the expert at 1.0 mm only (unbalanced cells).
    Shares of the heart's total SS. Returns (summary over hearts: median, 25th, 75th percentile; per-heart table)."""
    col = f'ratio_{p}'
    per = []
    for h, eh in rows.groupby('patient'):
        ly = np.log(eh[col].values)
        yc = ly - eh.groupby('istar')[col].transform(lambda s_: np.log(s_).mean()).values
        src = eh.source.values; res = eh.resolution_mm.map(lambda x: f'{x:.1f}').values
        cell = np.array([f'{a}|{b}' for a, b in zip(src, res)])
        tot = float(((yc - yc.mean()) ** 2).sum())

        def rss_groups(lab):
            return float(sum(((yc[lab == u] - yc[lab == u].mean()) ** 2).sum() for u in np.unique(lab)))
        Xa = np.column_stack([np.ones(len(yc)), pd.get_dummies(src, drop_first=True).values.astype(float),
                              pd.get_dummies(res, drop_first=True).values.astype(float)])
        rss_add = float(((yc - Xa @ np.linalg.lstsq(Xa, yc, rcond=None)[0]) ** 2).sum())
        rS, rC = rss_groups(src), rss_groups(cell)
        per.append(dict(patient=int(h), n_constructions=int(eh.construction.nunique()), segmentation_S=(tot - rS) / tot,
                        mesh_M=(rS - rss_add) / tot, SxM=(rss_add - rC) / tot, residual_theta=rC / tot))
    per = pd.DataFrame(per).set_index('patient')
    summ = {t: dict(median=float(per[t].median()), p25=float(per[t].quantile(.25)), p75=float(per[t].quantile(.75)))
            for t in ['segmentation_S', 'mesh_M', 'SxM', 'residual_theta']}
    return summ, per


R['E2c_within_heart_nonref'] = dict(
    label='DESCRIPTIVE (added 2026-10-28, not pre-registered, no inference): per-heart sequential SS shares, '
          'non-reference rows only; median and IQR over hearts',
    candidate_model_for_user='log y ~ S * M + theta* + (1 | heart) + (1 | heart:source), i.e. a heart x source variance '
                             'component; not fitted here', sets={})
for k, nm in [('primary_no_ref', 'corr loss, primary non-reference (4 networks x 2 resolutions + expert 1.0 mm)'),
              ('amp_no_ref', 'amplitude loss, primary non-reference')]:
    R['E2c_within_heart_nonref']['sets'][k] = dict(description=nm, n_hearts=int(SETS[k].patient.nunique()))
    for p in P:
        summ, per = ss_typeI_per_heart(SETS[k], p)
        R['E2c_within_heart_nonref']['sets'][k][p] = summ
        if k == 'primary_no_ref':
            per.to_csv(os.path.join(OUT, 'tables', f'ss_typeI_per_heart_nonref_{p}.csv'))
_nets = SETS['primary_no_ref'][SETS['primary_no_ref'].source != 'expert']
R['E2c_within_heart_nonref']['sets']['networks_only'] = dict(
    description='corr loss, 4 networks x 2 resolutions only (balanced where all present; order-free check)',
    n_hearts=int(_nets.patient.nunique()), **{p: ss_typeI_per_heart(_nets, p)[0] for p in P})
# ================================================================== E2d: secondary model, Amendment 2 B2, added after full-arm data
# log(theta_hat/theta*) ~ S * M + theta* + (1 | heart) + (1 | heart:source), REML, non-reference primary rows, both losses.
# Declared secondary model (final/AMENDMENT_2_2026-10-28.md B2); E2 stays the pre-registered model and is not changed.
# Own REML (profiled residual variance, relative standard deviations t = sigma_k / sigma_e >= 0 on L-BFGS-B bounds, so
# a zero component is reachable and counted as a boundary fit), cross-checked against statsmodels MixedLM (vc_formula
# for heart:source) on the point fits. Cluster bootstrap over hearts: 2,000 resamples, fresh default_rng(20261028) per
# (set, parameter) cell, duplicated hearts relabelled as distinct hearts (and distinct heart:source groups).
B2_SOURCES = sorted(PRIMARY_SOURCES)          # global heart:source column order (absent source in a heart = zero column)
B2_BOUND = 1e-4                               # t below this (variance ratio < 1e-8) = boundary fit
B2_STARTS = [(1.0, 1.0), (0.2, 0.2), (2.0, 0.2), (0.2, 2.0)]


def design_b2(rows):
    """Fixed-effect design S * M + theta* (sum-to-zero coding, as E2). Non-reference rows have no expert 1.5 mm cell, so
    the expert S x M column is aliased and dropped (fallback if the design is still rank deficient: keep columns left to
    right only when they raise the rank). Returns X, term column indices, dropped names, kept names."""
    import patsy
    d = rows.assign(ly=0.0, res=rows.resolution_mm.map(lambda x: f'{x:.1f}'), th=rows.istar.astype(str))
    dm = patsy.dmatrix('C(source, Sum) * C(res, Sum) + C(th, Sum)', d, return_type='dataframe')
    names = list(dm.columns)
    tkey = {}
    for t, sl in dm.design_info.term_slices.items():
        n = t.name()
        key = ('intercept' if n == 'Intercept' else 'segmentation_S' if n == 'C(source, Sum)' else 'mesh_M' if n == 'C(res, Sum)'
               else 'SxM' if ':' in n else 'theta_star' if n == 'C(th, Sum)' else n)
        for j in range(sl.start, sl.stop):
            tkey[j] = key
    Xf = dm.values
    # the expert has no 1.5 mm row, so its S x M column is the one that does not describe a cell: drop it first
    keep = [j for j in range(Xf.shape[1]) if not (tkey[j] == 'SxM' and '[S.expert]' in names[j])]
    if np.linalg.matrix_rank(Xf[:, keep]) < len(keep) or np.linalg.matrix_rank(Xf[:, keep]) < np.linalg.matrix_rank(Xf):
        keep, rk = [], 0                       # fallback: greedy left-to-right rank selection
        for j in range(Xf.shape[1]):
            r_ = np.linalg.matrix_rank(Xf[:, keep + [j]])
            if r_ > rk:
                keep.append(j); rk = r_
    X = Xf[:, keep]
    terms = {}
    for jj, j in enumerate(keep):
        terms.setdefault(tkey[j], []).append(jj)
    dropped = [names[j] for j in range(Xf.shape[1]) if j not in keep]
    return X, {k: np.array(v) for k, v in terms.items()}, dropped, [names[j] for j in keep]


def b2_blocks(rows, p):
    """Per-heart sufficient statistics for the two-component REML. Z_i = [1, source indicators (B2_SOURCES)]."""
    rows = rows.reset_index(drop=True)
    y = np.log(rows[f'ratio_{p}'].values)
    X, terms, dropped, cols = design_b2(rows)
    Zs = np.column_stack([np.ones(len(rows))] + [(rows.source.values == s).astype(float) for s in B2_SOURCES])
    hearts = np.array(sorted(rows.patient.unique()))
    B = dict(XtX=[], Xty=[], yty=[], n=[], ZtZ=[], ZtX=[], Zty=[], Xsum=[])
    for h in hearts:
        ix = np.nonzero(rows.patient.values == h)[0]
        Xi, yi, Zi = X[ix], y[ix], Zs[ix]
        B['XtX'].append(Xi.T @ Xi); B['Xty'].append(Xi.T @ yi); B['yty'].append(yi @ yi); B['n'].append(len(ix))
        B['ZtZ'].append(Zi.T @ Zi); B['ZtX'].append(Zi.T @ Xi); B['Zty'].append(Zi.T @ yi); B['Xsum'].append(Xi.sum(0))
    B = {k: np.array(v, float) for k, v in B.items()}
    return B, hearts, terms, dropped, cols, (y, X, rows)


def b2_crit(t, B, idx):
    """REML criterion (-2 log restricted likelihood, constants dropped) at relative SDs t = (t_heart, t_heart:source),
    residual variance profiled out. idx: heart block indices (duplicates allowed). Returns crit, beta, s2_resid, ok."""
    th, ths = t
    d = np.array([th] + [ths] * len(B2_SOURCES))
    ZtZ = B['ZtZ'][idx]; ZtX = B['ZtX'][idx] * d[None, :, None]; Zty = B['Zty'][idx] * d[None, :]
    Mi = np.eye(len(d))[None] + d[None, :, None] * ZtZ * d[None, None, :]
    Lc = np.linalg.cholesky(Mi)
    logdetM = 2 * np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum()
    MA = np.linalg.solve(Mi, ZtX); Mb = np.linalg.solve(Mi, Zty[..., None])[..., 0]
    XVX = B['XtX'][idx].sum(0) - np.einsum('hqi,hqj->ij', ZtX, MA)
    XVy = B['Xty'][idx].sum(0) - np.einsum('hqi,hq->i', ZtX, Mb)
    yVy = B['yty'][idx].sum() - np.einsum('hq,hq->', Zty, Mb)
    N = B['n'][idx].sum(); pX = XVX.shape[0]
    sgn, logdetX = np.linalg.slogdet(XVX)
    if sgn <= 0:
        return np.inf, None, np.nan, False
    beta = np.linalg.solve(XVX, XVy)
    s2 = (yVy - beta @ XVy) / (N - pX)
    return float((N - pX) * np.log(s2) + logdetM + logdetX), beta, float(s2), True


def b2_fit(B, idx, terms):
    """Multi-start L-BFGS-B on t >= 0. Returns dict of variance components, term variances, shares and flags."""
    best, nconv = None, 0
    for st0 in B2_STARTS:
        r_ = optimize.minimize(lambda t: b2_crit(t, B, idx)[0], np.array(st0), method='L-BFGS-B',
                               bounds=[(0.0, 1e3), (0.0, 1e3)], options=dict(ftol=1e-13, gtol=1e-9, maxiter=500))
        nconv += bool(r_.success)
        if best is None or r_.fun < best.fun:
            best = r_
    crit, beta, s2e, okX = b2_crit(best.x, B, idx)
    th, ths = [float(x) for x in best.x]
    N = B['n'][idx].sum(); XtX = B['XtX'][idx].sum(0); xs = B['Xsum'][idx].sum(0)
    V = {}
    for k, cols in terms.items():
        if k == 'intercept':
            continue
        b = beta[cols]
        V[k] = float(b @ XtX[np.ix_(cols, cols)] @ b / N - (xs[cols] @ b / N) ** 2)
    V['heart'] = th ** 2 * s2e; V['heart_source'] = ths ** 2 * s2e; V['residual'] = s2e
    tot = sum(V.values())
    sh = {k: v / tot for k, v in V.items()}
    fixc = V['segmentation_S'] + V['mesh_M'] + V['SxM']
    cons = fixc + V['heart_source']
    sh['within_heart_construction'] = cons / (cons + V['residual'])
    sh['within_heart_construction_fixed_only'] = fixc / (fixc + V['heart_source'] + V['residual'])
    sh['source_of_construction'] = (V['segmentation_S'] + V['heart_source']) / cons
    sh['heart_source_of_construction'] = V['heart_source'] / cons
    return dict(V=V, shares=sh, crit=crit, beta=beta, t=[th, ths], best_success=bool(best.success),
                all_starts_success=bool(nconv == len(B2_STARTS)), rank_ok=bool(okX),
                boundary_heart=bool(th < B2_BOUND), boundary_heart_source=bool(ths < B2_BOUND))


def _b2_lite(f_):
    return dict(V=f_['V'], shares=f_['shares'], best_success=f_['best_success'], all_starts_success=f_['all_starts_success'],
                rank_ok=f_['rank_ok'], boundary_heart=f_['boundary_heart'], boundary_heart_source=f_['boundary_heart_source'])


def _e2d_chunk(task):
    """Worker: bootstrap fits for resamples [a, b) of one (set, parameter) cell. Picks were drawn in the parent."""
    kp, a, b = task
    B, hearts, terms = B2PREP[kp][:3]
    out = []
    for pick in B2PICKS[kp][a:b]:
        try:
            out.append(_b2_lite(b2_fit(B, pick, terms)))
        except Exception as ex:
            out.append(dict(error=repr(ex)))
    return kp, a, out


def _e2d_point(kp):
    B, hearts, terms = B2PREP[kp][:3]
    return kp, b2_fit(B, np.arange(len(hearts)), terms)


def e2d_summary(kp, pt, bs_all):
    k, p = kp
    B, hearts, terms, dropped, cols, _ = B2PREP[kp]
    bs = [f_ for f_ in bs_all if 'error' not in f_]
    flags = dict(error=len(bs_all) - len(bs),
                 optimizer_not_converged=sum(not f_['best_success'] for f_ in bs),
                 any_start_not_converged=sum(not f_['all_starts_success'] for f_ in bs),
                 rank_deficient=sum(not f_['rank_ok'] for f_ in bs),
                 boundary_heart=sum(f_['boundary_heart'] for f_ in bs),
                 boundary_heart_source=sum(f_['boundary_heart_source'] for f_ in bs),
                 boundary_either=sum(f_['boundary_heart'] or f_['boundary_heart_source'] for f_ in bs))
    comp = {c: dict(value=pt['V'][c], ci=q([f_['V'][c] for f_ in bs])) for c in pt['V']}
    shr = {c: dict(share=pt['shares'][c], ci=q([f_['shares'][c] for f_ in bs])) for c in pt['shares']}
    rows = SETS[k]
    return dict(variance=comp, shares=shr, t_relative_sd=pt['t'], reml_crit=pt['crit'],
                point_flags=dict(optimizer_converged=pt['best_success'], all_starts_converged=pt['all_starts_success'],
                                 boundary_heart=pt['boundary_heart'], boundary_heart_source=pt['boundary_heart_source']),
                boot_flags=flags, n_boot_ok=len(bs), fixed_columns=cols, dropped_aliased_columns=dropped,
                n_rows=int(len(rows)), n_hearts=int(len(hearts)), n_constructions=int(rows.construction.nunique()),
                n_heart_source_groups=int(rows.groupby(['patient', 'source']).ngroups))


E2D_SETS = {'primary_no_ref': 'correlation loss (primary), primary non-reference rows',
            'amp_no_ref': 'amplitude loss (secondary), primary non-reference rows'}
B2CELLS = [(k, p) for k in E2D_SETS for p in P]
B2PREP = {kp: b2_blocks(SETS[kp[0]], kp[1]) for kp in B2CELLS}
B2PICKS = {}
for kp in B2CELLS:                          # same draw sequence as shares_boot: fresh rng(SEED), rng.choice per resample
    _rng = np.random.default_rng(SEED); _n = len(B2PREP[kp][1])
    B2PICKS[kp] = [_rng.choice(np.arange(_n), _n, replace=True) for _ in range(NB)]
B2CH = 20
with CTX.Pool(max(1, min(NPROC, 126))) as pool:
    _pts = dict(pool.map(_e2d_point, B2CELLS))
    _chunks = pool.map(_e2d_chunk, [(kp, a, min(a + B2CH, NB)) for kp in B2CELLS for a in range(0, NB, B2CH)])
_bs = {kp: [] for kp in B2CELLS}
for kp, a, out in sorted(_chunks, key=lambda t: (B2CELLS.index(t[0]), t[1])):
    _bs[kp].extend(out)
e2d = {kp: e2d_summary(kp, _pts[kp], _bs[kp]) for kp in B2CELLS}
R['E2d_secondary_heart_source'] = dict(
    label=('SECONDARY MODEL, Amendment 2 B2, added after full-arm data (declared before it was fitted; E2 stays the '
           'pre-registered model)'),
    model='log(theta_hat/theta*) ~ S * M + theta* + (1 | heart) + (1 | heart:source), REML',
    rows='include_primary, non-reference (4 networks x {1.5, 1.0} mm + expert 1.0 mm), round 2',
    definitions=dict(
        term_variance='V_k = variance over observations (ddof 0) of X_k b_k, sum-to-zero coding (as E2)',
        share='component / (V_S + V_M + V_SxM + V_theta + s2_heart + s2_heart_source + s2_resid)',
        within_heart_construction='(V_S + V_M + V_SxM + s2_heart_source) / (V_S + V_M + V_SxM + s2_heart_source + s2_resid)',
        within_heart_construction_fixed_only='(V_S + V_M + V_SxM) / (V_S + V_M + V_SxM + s2_heart_source + s2_resid)',
        source_of_construction='(V_S + s2_heart_source) / (V_S + V_M + V_SxM + s2_heart_source)',
        heart_source_of_construction='s2_heart_source / (V_S + V_M + V_SxM + s2_heart_source)',
        boundary=f'relative SD sigma_k / sigma_e < {B2_BOUND} (L-BFGS-B lower bound 0)',
        aliasing='no expert 1.5 mm cell in non-reference rows: one S x M column is aliased and dropped (fixed_columns)'),
    bootstrap=f'{NB} cluster resamples over hearts, seed {SEED}, fresh rng per cell; percentile 95% CI',
    sets={k: dict(description=v, **{p: e2d[(k, p)] for p in P}) for k, v in E2D_SETS.items()})

# cross-check: statsmodels MixedLM with vc_formula for heart:source, same fixed design, point fits only
try:
    import statsmodels.formula.api as smf
    xchk = {}
    for k in E2D_SETS:
        for p in P:
            B, hearts, terms, dropped, cols, (y, X, rows) = B2PREP[(k, p)]
            dd = pd.DataFrame(X, columns=[f'x{j}' for j in range(X.shape[1])])
            dd['ly'] = y; dd['heart'] = rows.patient.values; dd['source'] = rows.source.astype(str).values
            fml = 'ly ~ 0 + ' + ' + '.join(f'x{j}' for j in range(X.shape[1]))
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                m = smf.mixedlm(fml, dd, groups='heart', re_formula='1', vc_formula={'heart_source': '0 + C(source)'}).fit(
                    reml=True, method=['lbfgs', 'powell'])
            own = e2d[(k, p)]['variance']
            s2h, s2hs, s2e = float(m.cov_re.iloc[0, 0]), float(m.vcomp[0]), float(m.scale)
            t_sm = [np.sqrt(max(s2h, 0) / s2e), np.sqrt(max(s2hs, 0) / s2e)]
            ib = np.arange(len(hearts))
            crit_sm, beta_sm_own, _, _ = b2_crit(t_sm, B, ib)
            beta_own = b2_crit(e2d[(k, p)]['t_relative_sd'], B, ib)[1]
            xchk[f'{k}|{p}'] = dict(
                statsmodels=dict(s2_heart=s2h, s2_heart_source=s2hs, s2_resid=s2e, converged=bool(m.converged)),
                own=dict(s2_heart=own['heart']['value'], s2_heart_source=own['heart_source']['value'], s2_resid=own['residual']['value']),
                max_rel_diff_variance=float(max(abs(a - b) / max(abs(b), 1e-300) for a, b in
                                                [(s2h, own['heart']['value']), (s2hs, own['heart_source']['value']), (s2e, own['residual']['value'])])),
                max_abs_beta_diff=float(np.max(np.abs(m.fe_params.values - beta_own))),
                own_crit_at_own_optimum=e2d[(k, p)]['reml_crit'], own_crit_at_statsmodels_estimate=float(crit_sm),
                warnings=[f'{w_.category.__name__}: {w_.message}' for w_ in wl])
    R['E2d_secondary_heart_source']['statsmodels_check'] = xchk
except Exception as ex:  # report, do not hide
    R['E2d_secondary_heart_source']['statsmodels_check'] = dict(error=repr(ex))
print(f'E2d done {time.time() - T0:.0f}s', flush=True)

print(f'E2 done {time.time() - T0:.0f}s', flush=True)


# ================================================================== E3: break-even
COMP = dict(white=('dB SNR', [10, 15, 20, 25, 30], lambda L: -np.asarray(L, float)),
            lp40=('dB SNR', [10, 15, 20, 25, 30], lambda L: -np.asarray(L, float)),
            wander=('% of lead QRS peak-to-peak', [5, 10, 20], lambda L: np.log(np.asarray(L, float))),
            disp=('mm SD', [5, 10, 20], lambda L: np.log(np.asarray(L, float))))


def noise_sd(nz, loss, drop_cens=False):
    n = nz[nz.loss == loss]
    if drop_cens:
        n = n[~n.censored]
    rows = []
    for (h, c, L), g in n.groupby(['patient', 'comparator', 'level']):
        d = dict(patient=int(h), comparator=c, level=float(L))
        for p in P:
            vs = [np.var(gi[f'ratio_{p}'].values, ddof=1) for _, gi in g.groupby('istar') if len(gi) > 1]
            d[f'sd_{p}'] = float(np.sqrt(np.mean(vs)))
        d['n_censored'] = int(g.censored.sum()); d['n'] = int(len(g))
        rows.append(d)
    return pd.DataFrame(rows)


def crossing(levels, sev_fn, sdn, sdc):
    """sdn at levels (ordered least to most severe). Returns (severity, flag)."""
    s = sev_fn(levels); o = np.argsort(s); s = s[o]; ly = np.log(np.asarray(sdn)[o]); t = np.log(sdc)
    if not np.all(np.isfinite(ly)) or not np.isfinite(t):
        return np.nan, 'nan'
    if t < ly[0]:
        return -np.inf, 'beyond grid: noise SD above construction SD at the least severe level'
    for k in range(len(s) - 1):
        if (ly[k] - t) * (ly[k + 1] - t) <= 0 and ly[k] != ly[k + 1]:
            return float(s[k] + (t - ly[k]) / (ly[k + 1] - ly[k]) * (s[k + 1] - s[k])), 'interp'
    return np.inf, 'beyond grid: construction SD above noise SD at the most severe level'


def sev_to_level(c, s):
    """Severity back to native units; beyond-grid values become labels (SNR: more severe = lower dB)."""
    unit, levels = COMP[c][0], COMP[c][1]
    if np.isposinf(s):
        return f'beyond grid: < {min(levels)} dB' if unit == 'dB SNR' else f'beyond grid: > {max(levels)}'
    if np.isneginf(s):
        return f'beyond grid: > {max(levels)} dB' if unit == 'dB SNR' else f'beyond grid: < {min(levels)}'
    if not np.isfinite(s):
        return 'nan'
    return float(-s) if unit == 'dB SNR' else float(np.exp(s))


def breakeven_block(cvtab, nsd, loss_label):
    out = {}
    for c, (unit, levels, sev) in COMP.items():
        for p in P:
            sub = nsd[nsd.comparator == c]
            piv = sub.pivot(index='patient', columns='level', values=f'sd_{p}')[[float(x) for x in levels]]
            hearts = np.array(sorted(set(piv.index) & set(cvtab.index)))
            sdc = cvtab.loc[hearts, f'sd_{p}'].values; M = piv.loc[hearts].values
            med_n = np.median(M, axis=0); med_c = float(np.median(sdc))
            s0, flag = crossing(np.array(levels, float), sev, med_n, med_c)
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(NB):
                pk = rng.integers(0, len(hearts), len(hearts))
                bs.append(crossing(np.array(levels, float), sev, np.median(M[pk], 0), float(np.median(sdc[pk])))[0])
            bs = np.array(bs)
            lo, hi = np.percentile(bs, [2.5, 97.5], method='lower'), np.percentile(bs, [2.5, 97.5], method='higher')
            ci_sev = [float(lo[0]), float(hi[1])]
            # per heart
            ph = [crossing(np.array(levels, float), sev, M[j], sdc[j]) for j in range(len(hearts))]
            n_in = sum(f == 'interp' for _, f in ph)
            n_above = sum(np.isposinf(s) for s, _ in ph); n_below = sum(np.isneginf(s) for s, _ in ph)
            out[f'{c}|{p}'] = dict(comparator=c, parameter=p, unit=unit, levels=levels, n_hearts=int(len(hearts)),
                                   sd_cons_median=med_c, sd_noise_median={str(l): float(v) for l, v in zip(levels, med_n)},
                                   ratio_cons_to_noise={str(l): float(med_c / v) for l, v in zip(levels, med_n)},
                                   breakeven=sev_to_level(c, s0), flag=flag,
                                   ci=[sev_to_level(c, x) for x in (ci_sev[::-1] if unit == 'dB SNR' else ci_sev)], ci_severity=ci_sev,
                                   boot_beyond_most_severe=int(np.isposinf(bs).sum()), boot_beyond_least_severe=int(np.isneginf(bs).sum()),
                                   per_heart_interp=int(n_in), per_heart_beyond_most_severe=int(n_above),
                                   per_heart_beyond_least_severe=int(n_below), loss=loss_label)
    return out


NSD = {('corr', False): noise_sd(NZ, 'corr'), ('amp', False): noise_sd(NZ, 'amp'), ('corr', True): noise_sd(NZ, 'corr', True)}
NSD[('corr', False)].to_csv(os.path.join(OUT, 'tables', 'noise_sd_per_heart_corr.csv'), index=False)
R['E3_breakeven'] = dict(primary=breakeven_block(CVT['primary'], NSD[('corr', False)], 'corr'),
                         amp_loss=breakeven_block(CVT['amp_loss'], NSD[('amp', False)], 'amp'),
                         censored_dropped=breakeven_block(CVT['censored_dropped'], NSD[('corr', True)], 'corr, censored dropped'),
                         ukbb_added_g1usable=breakeven_block(CVT['ukbb_added_g1usable'], NSD[('corr', False)], 'corr'))
R['E3_noise_censored'] = NZ.groupby(['loss', 'comparator', 'level']).censored.sum().reset_index().to_dict('records')


def loglin_cross(levels, sev_fn, sdn, sdc):
    """C6: OLS of log(median noise SD) on severity over all grid levels; severity where the line meets log(SD_cons)."""
    s = sev_fn(levels); ly = np.log(np.asarray(sdn, float))
    if not (np.all(np.isfinite(ly)) and np.isfinite(sdc) and sdc > 0):
        return np.nan, np.nan
    b1, b0 = np.polyfit(s, ly, 1)
    return (float((np.log(sdc) - b0) / b1) if b1 > 0 else np.nan), float(b1)


def extrap_block(cvtab, nsd):
    out = {}
    for c, (unit, levels, sev) in COMP.items():
        for p in P:
            piv = nsd[nsd.comparator == c].pivot(index='patient', columns='level', values=f'sd_{p}')[[float(x) for x in levels]]
            hearts = np.array(sorted(set(piv.index) & set(cvtab.index)))
            sdc = cvtab.loc[hearts, f'sd_{p}'].values; M = piv.loc[hearts].values
            L = np.array(levels, float)
            s0, slope = loglin_cross(L, sev, np.median(M, 0), float(np.median(sdc)))
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(NB):
                pk = rng.integers(0, len(hearts), len(hearts))
                bs.append(loglin_cross(L, sev, np.median(M[pk], 0), float(np.median(sdc[pk])))[0])
            ci = q(bs)
            to_lv = (lambda x: -x) if unit == 'dB SNR' else (lambda x: float(np.exp(x)))
            ci_lv = sorted(to_lv(x) for x in ci)
            sv = sev(L)
            out[f'{c}|{p}'] = dict(comparator=c, parameter=p, unit=unit, grid=levels, slope_logSD_per_severity=slope,
                                   level_extrapolated=to_lv(s0) if np.isfinite(s0) else np.nan, ci=ci_lv,
                                   within_grid=bool(np.isfinite(s0) and sv.min() <= s0 <= sv.max()),
                                   n_boot_finite=int(np.isfinite(bs).sum()), n_hearts=int(len(hearts)))
    return out


R['E3_extrapolated_DESCRIPTIVE'] = dict(
    label=('EXTRAPOLATED, descriptive, NOT pre-registered (added 2026-10-28): noise level at which the median noise SD '
           'would equal the median construction SD, from an OLS line of log median noise SD on severity (-dB, or ln '
           'level) over all tested levels; percentile 95% CI from the same heart resampling (seed 20261028). The '
           'pre-registered break-even (E3) is unchanged and stays "beyond grid" where it is.'),
    primary=extrap_block(CVT['primary'], NSD[('corr', False)]), amp_loss=extrap_block(CVT['amp_loss'], NSD[('amp', False)]))
print(f'E3 done {time.time() - T0:.0f}s', flush=True)


# ================================================================== E4: three-length rule (logic of analyse_gates.g4_block)
def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def lsfit(X, y):
    Am = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(Am, y, rcond=None)[0]
    return b, Am @ b


def g4_block(e, params=P):
    rng = np.random.default_rng(SEED)
    hearts = sorted(e.patient.unique())
    rows = []
    for h in hearts:
        eh = e[e.patient == h]
        refc = eh[eh.is_reference].set_index('istar')
        cr = CON[(CON.patient == h) & CON.is_reference].iloc[0]
        for c, g in eh[~eh.is_reference].groupby('construction'):
            g = g.set_index('istar')
            r = [float((g[f'{p}_hat'] / refc.loc[g.index, f'{p}_hat']).mean()) for p in P]
            cc = CON[CON.construction == c].iloc[0]
            rows.append(dict(h=h, c=c, r=r, l=[cc.L_scale / cr.L_scale, cc.L_wall / cr.L_wall, cc.L_base / cr.L_base]))
    H = np.array([r['h'] for r in rows]); Y = np.array([r['r'] for r in rows]); X = np.array([r['l'] for r in rows])
    out = dict(n_constructions=len(rows), n_hearts=len(hearts))
    for j, p in enumerate(P):
        if p not in params:
            continue
        rng = np.random.default_rng(SEED)
        y = Y[:, j]

        def st(idx):
            yy, XX, HH = y[idx], X[idx], H[idx]
            b, yh = lsfit(XX, yy)
            s = dict(R2=r2(yy, yh))
            pred = np.empty_like(yy)
            for hh in np.unique(HH):
                tr = HH != hh
                bb, _ = lsfit(XX[tr], yy[tr])
                pred[~tr] = np.column_stack([np.ones((~tr).sum()), XX[~tr]]) @ bb
            s['R2_LOHO'] = r2(yy, pred)
            return s
        full = st(np.arange(len(y)))
        bs = []
        for _ in range(NB):
            pick = rng.choice(hearts, len(hearts), replace=True)
            bs.append(st(np.concatenate([np.nonzero(H == hh)[0] for hh in pick])))
        out[p] = {k: dict(value=full[k], ci=q([b[k] for b in bs])) for k in ['R2', 'R2_LOHO']}
    return out


def _e4_task(lp):
    return lp, g4_block(SETS[{'corr': 'primary', 'amp': 'amp_loss'}[lp[0]]], [lp[1]])


with CTX.Pool(min(NPROC, 4)) as pool:
    e4 = dict(pool.map(_e4_task, [(l, p) for l in ['corr', 'amp'] for p in P]))
R['E4_three_length'] = {}
for l in ['corr', 'amp']:
    o = dict(e4[(l, P[0])]); o.update({p: e4[(l, p)][p] for p in P})
    o['flag_in_sample_R2_ge_0.9_both'] = bool(all(o[p]['R2']['value'] >= 0.9 for p in P))
    o['flag_LOHO_R2_ge_0.9_both'] = bool(all(o[p]['R2_LOHO']['value'] >= 0.9 for p in P))
    R['E4_three_length'][l] = o
print(f'E4 done {time.time() - T0:.0f}s', flush=True)


# ================================================================== E5: arm T and arm F
def boot_mean(v):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float)
    return float(v.mean()), q(v[rng.integers(0, len(v), (NB, len(v)))].mean(1))


Tf = TT[TT['round'] == 2]  # v2 report round
armT = []
for (loss, ref, reg), g in Tf[Tf.ref.isin(['09', '17'])].groupby(['loss', 'ref', 'reg']):
    for setname, gg in [('pooled_HCM_heavy', g), ('HCM', g[g.group == 'HCM']), ('healthy', g[g.group == 'healthy'])]:
        d = dict(loss=loss, ref=ref, reg=reg, set=setname, n_hearts=int(gg.heart.nunique()), n_estimates=int(len(gg)),
                 bound_hits=int(gg.bound_hit.astype(bool).sum()), fval_median=float(gg.fval.median()),
                 fval_iqr=[float(gg.fval.quantile(.25)), float(gg.fval.quantile(.75))])
        for p in P:
            ph = gg.groupby('heart')[f'rel_{p}']
            d[f'{p}_signed'] = boot_mean(ph.mean().values)
            d[f'{p}_abs'] = boot_mean(ph.apply(lambda s: s.abs().mean()).values)
        armT.append(d)
R['E5_armT'] = armT
# check against the arm T owner's analysis.json
ja = json.load(open(INPUTS['armT_json']))['cells']
mx = 0.0
for d in armT:
    for c in ja:
        if (c['loss'], c['ref'], c['reg'], c['set']) == (d['loss'], d['ref'], d['reg'], d['set']):
            for p in P:
                for kind in ['signed', 'abs']:
                    mx = max(mx, abs(c[f'{p}_{kind}'][0] - d[f'{p}_{kind}'][0]))
R['E5_armT_max_abs_diff_vs_owner'] = mx

armF = []
for (st_, vox, loss, win), g in FF.groupby(['set', 'vox', 'loss', 'window']):
    d = dict(set=st_, vox=float(vox), loss=loss, window=win, n_hearts=int(g.patient.nunique()), n_estimates=int(len(g)),
             n_censored=int(g.censored.astype(bool).sum()), fval_median=float(g.fval.median()))
    for p in P:
        ph = g.groupby('patient')[f'ratio_{p}'].apply(lambda s: np.log(s).mean()).values
        m = ph.mean()
        if len(ph) > 1:
            se = ph.std(ddof=1) / np.sqrt(len(ph)); tq = stats.t.ppf(0.975, len(ph) - 1)
            tci = [float(np.exp(m - tq * se)), float(np.exp(m + tq * se))]
            bci = [float(np.exp(x)) for x in boot_mean(ph)[1]]
        else:
            tci = bci = [np.nan, np.nan]
        d[p] = dict(gm=float(np.exp(m)), t_ci=tci, boot_ci=bci, per_estimate=ipr(g[f'ratio_{p}']))
    armF.append(d)
R['E5_armF'] = armF
jf = json.load(open(INPUTS['armF_json']))
R['E5_armF_owner_primary'] = {p: jf['estimates'].get('re|1.5|corr|qrs|keep_censored', {}).get(p, {}).get('gm_ratio') for p in P}

# side-by-side comparison of factor magnitudes (correlation loss)
cmp_rows = []
for p in P:
    pr = R['E1_primary'][p]
    cmp_rows.append(dict(factor='construction (arm C, primary)', parameter=p, metric='median over hearts of between-construction CV',
                         value=pr['cv_median'], ci=pr['ci'], n='50 hearts'))
    cmp_rows.append(dict(factor='construction (arm C, primary)', parameter=p, metric='median over hearts of between-construction SD of theta_hat/theta*',
                         value=pr['sd_median'], ci=pr['sd_ci'], n='50 hearts'))
    for d in armT:
        if d['loss'] == 'corr' and d['reg'] == 'trans' and d['set'] == 'pooled_HCM_heavy':
            cmp_rows.append(dict(factor=f'torso (arm T, reference {d["ref"]}, translation only)', parameter=p,
                                 metric='mean absolute relative bias', value=d[f'{p}_abs'][0], ci=d[f'{p}_abs'][1], n=f'{d["n_hearts"]} hearts'))
            cmp_rows.append(dict(factor=f'torso (arm T, reference {d["ref"]}, translation only)', parameter=p,
                                 metric='mean signed relative bias', value=d[f'{p}_signed'][0], ci=d[f'{p}_signed'][1], n=f'{d["n_hearts"]} hearts'))
    for d in armF:
        if (d['set'], d['vox'], d['loss'], d['window']) == ('re', 1.5, 'corr', 'qrs'):
            cmp_rows.append(dict(factor='model form (arm F, reaction-eikonal targets, QRS window)', parameter=p,
                                 metric='geometric-mean theta_hat/theta* minus 1', value=d[p]['gm'] - 1,
                                 ci=[d[p]['t_ci'][0] - 1, d[p]['t_ci'][1] - 1], n=f'{d["n_hearts"]} hearts'))
R['E5_comparison'] = cmp_rows
pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, 'tables', 'factor_comparison.csv'), index=False)


# ================================================================== E6: exploratory, ACDC group
def holm(pv):
    pv = np.asarray(pv, float); o = np.argsort(pv); m = len(pv)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * pv[i]); adj[i] = min(1.0, run)
    return adj


cvp = CVT['primary']
groups = sorted(cvp.group.unique())
tests = []
for p in P:
    v = cvp[f'cv_{p}']
    samples = [v[cvp.group == g].values for g in groups]
    H, pk = stats.kruskal(*samples)
    n, k = len(v), len(groups)
    eps = (H - k + 1) / (n - k)
    rng = np.random.default_rng(SEED)
    be, br = [], []
    for _ in range(NB):
        sb = [s[rng.integers(0, len(s), len(s))] for s in samples]
        try:
            Hb = stats.kruskal(*sb)[0]; be.append((Hb - k + 1) / (n - k)); br.append(Hb / (n - 1))
        except ValueError:
            pass
    # C3 (2026-10-28): (H - k + 1)/(n - k) is eta^2_H (label was 'epsilon^2' before); epsilon^2_R = H/(n - 1) added
    tests.append(dict(test='Kruskal-Wallis across 5 groups', parameter=p, effect='eta^2_H', estimate=float(eps), ci=q(be),
                      effect_formula='(H - k + 1)/(n - k)', label_before_2026_10_28_correction='epsilon^2',
                      epsilon2_R=dict(formula='H/(n - 1)', estimate=float(H / (n - 1)), ci=q(br)),
                      statistic=f'H = {H:.2f}, df = {k - 1}', p=float(pk), n=int(n)))
    for g in groups:
        a = v[cvp.group == g].values; b = v[cvp.group != g].values
        U, pu = stats.mannwhitneyu(a, b, alternative='two-sided')
        rng = np.random.default_rng(SEED)
        bd = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(NB)]
        tests.append(dict(test=f'{g} vs other groups', parameter=p, effect='difference in median CV (group minus rest)',
                          estimate=float(np.median(a) - np.median(b)), ci=q(bd), statistic=f'U = {U:.1f}', p=float(pu),
                          n=f'{len(a)} vs {len(b)}'))
adj = holm([t['p'] for t in tests])
for t, a_ in zip(tests, adj):
    t['p_holm'] = float(a_)
R['E6_group'] = dict(label='EXPLORATORY', family_size=len(tests), tests=tests,
                     group_cv={g: {p: ipr(cvp[cvp.group == g][f'cv_{p}']) for p in P} for g in groups})


# ================================================================== fit quality, exclusions, root flags
fq = []
for k in ['primary', 'amp_loss', 'ukbb_added_all', 'rv4_10hearts_SxM', 'rv2_10hearts_1.0mm']:
    e = SETS[k]
    fq.append(dict(set=k, n_estimates=int(len(e)), n_censored=int(e.censored.sum()),
                   fval=ipr(e.fval_emu), by_source={s: dict(n=int(len(g)), censored=int(g.censored.sum()), fval_median=float(g.fval_emu.median()))
                                                    for s, g in e.groupby('source')}))
R['fit_quality'] = fq
mp = MAN[MAN.source.isin(PRIMARY_SOURCES) & (MAN.rv_wall_mm == 3.0) & ~MAN.is_reference.astype(bool)]
exc = MAN[MAN.source.isin(PRIMARY_SOURCES) & (MAN.rv_wall_mm == 3.0) & ~MAN.include_primary.astype(bool)]
R['exclusions_primary'] = exc[['heart', 'source', 'resolution_mm', 'exclusion_reason']].astype(str).to_dict('records')
R['n_primary_constructions'] = int(SETS['primary'].construction.nunique())
pc = SETS['primary'].drop_duplicates('construction')
R['root_flag'] = {f'{r:.1f}': dict(flagged=int(pc[(pc.resolution_mm == r) & ~pc.is_reference].root_flag.sum()),
                                   denominator=int((~pc[pc.resolution_mm == r].is_reference).sum())) for r in [1.5, 1.0]}
# C4 (2026-10-28): A4 denominator read literally, plus the reconciliation with mesh2 and calibC
_mn = MAN[~MAN.is_reference.astype(bool) & MAN.root_shift_mm.notna()]
_cn = CON[~CON.is_reference.astype(bool) & CON.root_shift_mm.notna()]
R['root_flag_A4'] = dict(
    rule='root shift > 10 mm; A4 denominator = non-reference constructions of the same resolution (every analysis set) '
         'with a mapped root, from data/processed/acdc_mesh_v2/MANIFEST.tsv',
    by_resolution={})
for r in [1.5, 1.0]:
    m_ = _mn[_mn.resolution_mm == r]; c_ = _cn[_cn.resolution_mm == r]; m3 = m_[m_.rv_wall_mm == 3.0]
    R['root_flag_A4']['by_resolution'][f'{r:.1f}'] = dict(
        A4_literal=dict(flagged=int(m_.root_flag.astype(bool).sum()), denominator=int(len(m_)),
                        by_set={s: dict(flagged=int(g.root_flag.astype(bool).sum()), n=int(len(g))) for s, g in m_.groupby('analysis_set')}),
        primary_set=R['root_flag'][f'{r:.1f}'],
        mesh2_rv3_only=dict(flagged=int(m3.root_flag.astype(bool).sum()), denominator=int(len(m3)),
                            note='mesh2 README denominator: RV wall 3 mm constructions only (all sources incl. ukbb_cardiac and G1-excluded)'),
        calibC_calibrated=dict(flagged=int(c_.root_flag.astype(bool).sum()), denominator=int(len(c_)),
                               note='calibC README denominator: constructions that were calibrated (constructions.csv)'),
        mapped_not_calibrated=sorted(set(m_.construction_id) - set(c_.construction)),
        flagged_constructions=[[x, float(v)] for x, v in m_[m_.root_flag.astype(bool)][['construction_id', 'root_shift_mm']].values])
# ================================================================== B1: smooth-surface fast layer, Amendment 2 B1 (integration owner)
# Declared sensitivity, added after the full-arm data. Arm C rerun with the B1 layer (results/2026-10-28_layer_sdf/
# estimates_sdf.csv, round 2, 494 primary constructions). E1 CV with the same cv_per_heart / summarise_cv rule as primary;
# paired difference B1 minus A2 = median over hearts of B1 CV minus median of A2 CV, hearts resampled jointly
# (boot_paired_diff, 2,000, default_rng(20261028)); the median of per-heart differences is reported beside it. The primary
# estimand stays on A2 and is not changed.
INPUTS_B1 = dict(estimates_sdf='results/2026-10-28_layer_sdf/estimates_sdf.csv',
                 layer_share='results/2026-10-28_layer_sdf/layer_share.csv',
                 layer_share_summary='results/2026-10-28_layer_sdf/layer_share_summary.csv',
                 unit_test_json='results/2026-10-28_layer_sdf/unit_test_sdf.json',
                 unit_test_table='results/2026-10-28_layer_sdf/unit_test_table.csv')
_b1_ok = all(os.path.exists(v) for v in INPUTS_B1.values())
if _b1_ok:
    ES = pd.read_csv(INPUTS_B1['estimates_sdf'])
    _qc = dict(n_rows=int(len(ES)), n_hearts=int(ES.patient.nunique()), n_constructions=int(ES.construction.nunique()),
               all_round2=bool((ES['round'] == 2).all()), all_include_primary=bool(ES.include_primary.all()),
               layer_values=sorted(ES.layer.astype(str).unique()),
               n_nan_ratio=int(ES[['ratio_v_endo', 'ratio_v_t']].isna().sum().sum()))
    # same constructions, theta* and keys as the A2 primary rows
    _a2 = E[E.include_primary]
    _k = ['construction', 'istar', 'loss']
    _mm = ES.merge(_a2[_k + ['v_endo_star', 'v_t_star']], on=_k, how='outer', suffixes=('_b1', '_a2'), indicator=True)
    _qc['n_keys_unmatched'] = int((_mm['_merge'] != 'both').sum())
    _qc['max_abs_theta_star_diff'] = float(np.nanmax(np.abs(np.r_[_mm.v_endo_star_b1 - _mm.v_endo_star_a2,
                                                                     _mm.v_t_star_b1 - _mm.v_t_star_a2])))
    _qc['n_censored'] = {l_: int(ES[ES.loss == l_].censored.sum()) for l_ in ['corr', 'amp']}
    _qc['n_censored_A2'] = {l_: int(_a2[_a2.loss == l_].censored.sum()) for l_ in ['corr', 'amp']}
    _ref = ES[ES.is_reference.astype(bool) & (ES.loss == 'corr')]
    _qc['reference_recovery_max_abs'] = {p: float(np.max(np.abs(_ref[f'ratio_{p}'] - 1))) for p in P}
    CVT_B1 = {'corr': cv_per_heart(ES[ES.loss == 'corr']), 'amp': cv_per_heart(ES[ES.loss == 'amp'])}
    CVT_B1['corr'].to_csv(os.path.join(OUT, 'tables', 'B1_per_heart_cv_sdf_corr.csv'))
    b1e1 = {}
    for l_, a2k, lab in [('corr', 'primary', 'correlation loss (primary)'), ('amp', 'amp_loss', 'amplitude loss (secondary)')]:
        d = summarise_cv(CVT_B1[l_], f'B1 smooth-surface layer, {lab}', INPUTS_B1['estimates_sdf'])
        d['A2_comparison_set'] = a2k
        for p in P:
            diff, ci, n = boot_paired_diff(CVT_B1[l_][f'cv_{p}'], CVT[a2k][f'cv_{p}'])
            d[p]['A2'] = dict(cv_median=R['E1_primary'][p]['cv_median'] if a2k == 'primary' else
                              next(x for x in sens if x['set'] == 'amp_loss')[p]['cv_median'],
                              ci=R['E1_primary'][p]['ci'] if a2k == 'primary' else next(x for x in sens if x['set'] == 'amp_loss')[p]['ci'])
            d[p]['diff_vs_A2'] = dict(diff=diff, ci=ci, n_common=n,
                                      definition='median over hearts of B1 CV minus median over hearts of A2 CV, hearts resampled jointly')
            # beside it: median of the per-heart paired differences, same seed
            dd = (CVT_B1[l_][f'cv_{p}'] - CVT[a2k][f'cv_{p}']).dropna()
            m_, ci_ = boot_median(dd.values)
            d[p]['median_per_heart_diff'] = dict(diff=m_, ci=ci_, n=int(len(dd)),
                                                 definition='median over hearts of (B1 CV - A2 CV) per heart; descriptive, beside the pre-declared difference')
        b1e1[l_] = d
    # layer share of myocardial volume (from the owner's per-construction table)
    LS = pd.read_csv(INPUTS_B1['layer_share'])
    ls = {}
    for r_, g_ in [('1.5', LS[LS.resolution_mm == 1.5]), ('1.0', LS[LS.resolution_mm == 1.0]), ('both', LS)]:
        ls[r_] = dict(n_constructions=int(len(g_)), B1=ipr(g_.vol_share_sdf), A2=ipr(g_.vol_share_a2),
                      ratio_B1_over_A2=ipr(g_.vol_share_sdf / g_.vol_share_a2), A2_only=ipr(g_.a2_only_share),
                      B1_only=dict(**ipr(g_.sdf_only_share), max=float(g_.sdf_only_share.max())))
    LSS = pd.read_csv(INPUTS_B1['layer_share_summary'])
    ls_src = {f'{r_.resolution_mm}|{r_.source}': dict(n=int(r_.n), B1_median=float(r_.vol_share_sdf_median),
                                                     B1_p2_5=float(r_['vol_share_sdf_p2.5']), B1_p97_5=float(r_['vol_share_sdf_p97.5']),
                                                     A2_median=float(r_.vol_share_a2_median), ratio_median=float(r_.ratio_sdf_a2_median))
              for _, r_ in LSS.iterrows()}
    # unit test against its pre-registered lines (ellipsoid within +-15% of the analytic shell; 1.0/1.5 ratio within +-15%)
    UT = json.load(open(INPUTS_B1['unit_test_json'])); UTT = pd.read_csv(INPUTS_B1['unit_test_table'])
    el = UTT[UTT.case == 'ellipsoid']
    ut = dict(lines='ellipsoid (20, 25, 40) mm, 8 poses: layer volume within +-15% of the analytic shell; median volume ratio '
                    '1.0/1.5 mm within +-15% (A2 unit-test lines, unchanged by Amendment 2 B1)',
              by_resolution={f'{v:.1f}': dict(n_poses=int(len(g_)), err_B1_median=float(g_.err_sdf.median()),
                                              err_B1_min=float(g_.err_sdf.min()), err_B1_max=float(g_.err_sdf.max()),
                                              err_A2_median=float(g_.err_a2.median()),
                                              n_within_15pct=int((g_.err_sdf.abs() <= 0.15).sum()))
                             for v, g_ in el.groupby('vox_mm')},
              ratio_1p0_over_1p5=float(UT['acceptance']['median_ratio_1p0_over_1p5']))
    _r = el.groupby('vox_mm').v_sdf_mL.median()
    ut['ratio_1p0_over_1p5_recomputed'] = float(_r.loc[1.0] / _r.loc[1.5])
    ut['ellipsoid_within_15pct'] = bool((el.err_sdf.abs() <= 0.15).all())
    ut['ratio_within_15pct'] = bool(abs(ut['ratio_1p0_over_1p5_recomputed'] - 1) <= 0.15)
    ut['result'] = 'PASS' if (ut['ellipsoid_within_15pct'] and ut['ratio_within_15pct']) else 'FAIL'
    ut['owner_json_PASS'] = bool(UT['acceptance']['PASS'])
    ut['other_cases'] = [dict(case=r_.case, vox_mm=float(r_.vox_mm), err_B1=float(r_.err_sdf), err_A2=float(r_.err_a2))
                         for _, r_ in UTT[UTT.case != 'ellipsoid'].iterrows()]
    ut['note'] = 'sphere and plane reported, not acceptance lines; plane whole-face deficit from rounded cavity edges (owner README)'
    R['B1_smooth_layer_SENSITIVITY'] = dict(
        label=('Amendment 2 B1, declared sensitivity, added after the full-arm data: smooth-surface (signed-distance) fast '
               'layer; the primary estimand stays on the A2 layer'),
        inputs=INPUTS_B1, qc=_qc, E1=b1e1, layer_share=ls, layer_share_by_source=ls_src, unit_test=ut)
else:
    R['B1_smooth_layer_SENSITIVITY'] = dict(skipped='input missing', inputs=INPUTS_B1)
print(f'B1 done {time.time() - T0:.0f}s', flush=True)

# ================================================================== B4: E3 on the extended grid, Amendment 2 B4 (integration owner)
# Declared extension of E3, added after the full-arm data. Noise estimates = arm C noise_estimates.csv concatenated with
# results/2026-10-28_noise_ext/noise_estimates_ext.csv; same noise_sd and crossing rule (A3), "beyond grid" kept. COMP
# (A3 grid, used by E3, E3x and the figures) is not changed: the extended grid is a separate dict COMP_EXT, and the block
# function below is the E3 breakeven_block with the grid passed in. Self-check: run with COMP on the A3 data it must
# reproduce R['E3_breakeven'] exactly.
INPUTS_B4 = dict(armN='results/2026-10-28_armC/noise_estimates.csv',
                 armN_ext='results/2026-10-28_noise_ext/noise_estimates_ext.csv')
COMP_EXT = dict(white=('dB SNR', [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30], lambda L: -np.asarray(L, float)),
                lp40=('dB SNR', [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30], lambda L: -np.asarray(L, float)),
                wander=('% of lead QRS peak-to-peak', [5, 10, 20, 40, 80], lambda L: np.log(np.asarray(L, float))),
                disp=('mm SD', [5, 10, 20, 30, 40], lambda L: np.log(np.asarray(L, float))))


def sev_to_level_g(comp, c, s):
    unit, levels = comp[c][0], comp[c][1]
    if np.isposinf(s):
        return f'beyond grid: < {min(levels)} dB' if unit == 'dB SNR' else f'beyond grid: > {max(levels)}'
    if np.isneginf(s):
        return f'beyond grid: > {max(levels)} dB' if unit == 'dB SNR' else f'beyond grid: < {min(levels)}'
    if not np.isfinite(s):
        return 'nan'
    return float(-s) if unit == 'dB SNR' else float(np.exp(s))


def breakeven_block_g(cvtab, nsd, loss_label, comp, keep_boot=False):
    """E3 breakeven_block with the grid as an argument (same rule, same seed and resampling order)."""
    out, boots = {}, {}
    for c, (unit, levels, sev) in comp.items():
        for p in P:
            sub = nsd[nsd.comparator == c]
            piv = sub.pivot(index='patient', columns='level', values=f'sd_{p}')[[float(x) for x in levels]]
            hearts = np.array(sorted(set(piv.index) & set(cvtab.index)))
            sdc = cvtab.loc[hearts, f'sd_{p}'].values; M = piv.loc[hearts].values
            med_n = np.median(M, axis=0); med_c = float(np.median(sdc))
            s0, flag = crossing(np.array(levels, float), sev, med_n, med_c)
            rng = np.random.default_rng(SEED)
            bs, pks = [], []
            for _ in range(NB):
                pk = rng.integers(0, len(hearts), len(hearts)); pks.append(pk)
                bs.append(crossing(np.array(levels, float), sev, np.median(M[pk], 0), float(np.median(sdc[pk])))[0])
            bs = np.array(bs)
            lo, hi = np.percentile(bs, [2.5, 97.5], method='lower'), np.percentile(bs, [2.5, 97.5], method='higher')
            ci_sev = [float(lo[0]), float(hi[1])]
            ph = [crossing(np.array(levels, float), sev, M[j], sdc[j]) for j in range(len(hearts))]
            n_in = sum(f == 'interp' for _, f in ph)
            n_above = sum(np.isposinf(s) for s, _ in ph); n_below = sum(np.isneginf(s) for s, _ in ph)
            out[f'{c}|{p}'] = dict(comparator=c, parameter=p, unit=unit, levels=levels, n_hearts=int(len(hearts)),
                                   sd_cons_median=med_c, sd_noise_median={str(l): float(v) for l, v in zip(levels, med_n)},
                                   ratio_cons_to_noise={str(l): float(med_c / v) for l, v in zip(levels, med_n)},
                                   breakeven=sev_to_level_g(comp, c, s0), flag=flag,
                                   ci=[sev_to_level_g(comp, c, x) for x in (ci_sev[::-1] if unit == 'dB SNR' else ci_sev)], ci_severity=ci_sev,
                                   boot_beyond_most_severe=int(np.isposinf(bs).sum()), boot_beyond_least_severe=int(np.isneginf(bs).sum()),
                                   per_heart_interp=int(n_in), per_heart_beyond_most_severe=int(n_above),
                                   per_heart_beyond_least_severe=int(n_below), loss=loss_label)
            if keep_boot:
                boots[f'{c}|{p}'] = dict(s0=s0, bs=bs, pks=pks, hearts=hearts, sdc=sdc, M=M, levels=np.array(levels, float), sev=sev)
    return (out, boots) if keep_boot else out


def _leaves_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_leaves_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_leaves_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


_b4_ok = all(os.path.exists(v) for v in INPUTS_B4.values())
if _b4_ok:
    NZE = pd.read_csv(INPUTS_B4['armN_ext'])
    _b4qc = dict(header_identical=list(NZE.columns) == list(NZ.columns), n_rows_ext=int(len(NZE)), n_rows_A3=int(len(NZ)),
                 n_nan_ratio_ext=int(NZE[['ratio_v_endo', 'ratio_v_t']].isna().sum().sum()))
    NZX = pd.concat([NZ, NZE], ignore_index=True)
    _kk = ['patient', 'istar', 'comparator', 'level', 'rep', 'loss']
    _b4qc['n_duplicate_keys'] = int(NZX.duplicated(_kk).sum())
    _b4qc['levels_by_comparator'] = {c: sorted(float(x) for x in g.level.unique()) for c, g in NZX.groupby('comparator')}
    _b4qc['grid_matches_COMP_EXT'] = all(_b4qc['levels_by_comparator'][c] == sorted(float(x) for x in COMP_EXT[c][1]) for c in COMP_EXT)
    _b4qc['n_rows_per_level_loss'] = int(NZE.groupby(['comparator', 'level', 'loss']).size().min())
    NSDX = {('corr', False): noise_sd(NZX, 'corr'), ('amp', False): noise_sd(NZX, 'amp'), ('corr', True): noise_sd(NZX, 'corr', True)}
    # self-check: generalised block on the A3 grid and A3 data reproduces E3 exactly
    _b4qc['selfcheck_reproduces_E3_primary'] = _leaves_equal(json.loads(json.dumps(breakeven_block_g(CVT['primary'], NSD[('corr', False)], 'corr', COMP),
                                                                                   default=lambda o: o.item() if hasattr(o, 'item') else str(o))),
                                                             json.loads(json.dumps(R['E3_breakeven']['primary'],
                                                                                   default=lambda o: o.item() if hasattr(o, 'item') else str(o))))
    b4 = {}
    BOOT4 = {}
    for k_, cvk, nk, lab in [('primary', 'primary', ('corr', False), 'corr, extended grid, Amendment 2 B4'),
                             ('amp_loss', 'amp_loss', ('amp', False), 'amp, extended grid, Amendment 2 B4'),
                             ('censored_dropped', 'censored_dropped', ('corr', True), 'corr, censored dropped, extended grid, Amendment 2 B4')]:
        b4[k_], BOOT4[k_] = breakeven_block_g(CVT[cvk], NSDX[nk], lab, COMP_EXT, keep_boot=True)
    NSDX[('corr', False)][NSDX[('corr', False)].level.isin([-15, -10, -5, 0, 5]) & NSDX[('corr', False)].comparator.isin(['white', 'lp40'])
                          | (NSDX[('corr', False)].comparator.isin(['wander']) & NSDX[('corr', False)].level.isin([40, 80]))
                          | (NSDX[('corr', False)].comparator.isin(['disp']) & NSDX[('corr', False)].level.isin([30, 40]))
                          ].to_csv(os.path.join(OUT, 'tables', 'B4_noise_sd_per_heart_corr_newlevels.csv'), index=False)
    # measured (extended grid) against E3x (A3 grid, log-linear extrapolation), same resamples: E3x's extrap_block and this
    # block both draw rng.integers(0, n, n) from a fresh default_rng(20261028) per cell over the same 50 hearts, so the
    # resample sequences are identical and the difference can be resampled jointly.
    cmpx = []
    for k_, xk in [('primary', 'primary'), ('amp_loss', 'amp_loss')]:
        for c, (unit, levels, sev) in COMP_EXT.items():
            for p in P:
                bt = BOOT4[k_][f'{c}|{p}']; ex = R['E3_extrapolated_DESCRIPTIVE'][xk][f'{c}|{p}']; a3 = R['E3_breakeven'][xk][f'{c}|{p}']
                cvtab = CVT[k_]
                piv = (NSD[('corr', False)] if k_ == 'primary' else NSD[('amp', False)])
                piv = piv[piv.comparator == c].pivot(index='patient', columns='level', values=f'sd_{p}')[[float(x) for x in COMP[c][1]]]
                assert np.array_equal(np.array(sorted(set(piv.index) & set(cvtab.index))), bt['hearts'])
                MA3 = piv.loc[bt['hearts']].values; LA3 = np.array(COMP[c][1], float)
                to_lv = (lambda x: -x) if unit == 'dB SNR' else (lambda x: float(np.exp(x)) if np.isfinite(x) else x)
                xs0, _ = loglin_cross(LA3, sev, np.median(MA3, 0), float(np.median(bt['sdc'])))
                dif_b = []
                for pk in bt['pks']:
                    xb, _ = loglin_cross(LA3, sev, np.median(MA3[pk], 0), float(np.median(bt['sdc'][pk])))
                    mb = bt['bs'][len(dif_b)]
                    dif_b.append(to_lv(mb) - to_lv(xb) if (np.isfinite(mb) and np.isfinite(xb)) else np.nan)
                dif_b = np.array(dif_b, float)
                meas = sev_to_level_g(COMP_EXT, c, bt['s0'])
                d0 = (to_lv(bt['s0']) - to_lv(xs0)) if (np.isfinite(bt['s0']) and np.isfinite(xs0)) else np.nan
                cmpx.append(dict(loss='corr' if k_ == 'primary' else 'amp', comparator=c, parameter=p, unit=unit,
                                 measured_extended=meas, measured_ci=b4[k_][f'{c}|{p}']['ci'],
                                 E3x_extrapolated=ex['level_extrapolated'], E3x_ci=ex['ci'],
                                 E3x_recomputed=(to_lv(xs0) if np.isfinite(xs0) else np.nan),
                                 E3_A3_grid=a3['breakeven'],
                                 diff_measured_minus_E3x=float(d0), diff_ci=q(dif_b), n_boot_diff_finite=int(np.isfinite(dif_b).sum()),
                                 measured_inside_A3_grid=isinstance(a3['breakeven'], float)))
    R['B4_E3_extended_grid'] = dict(
        label='extended grid, Amendment 2 B4 (declared extension of E3, added after the full-arm data)',
        rule=('A3 rule unchanged (median over hearts of SD_noise against median SD_cons, log SD linearly interpolated in '
              'severity, percentile 95% CI over 2,000 heart resamples, default_rng(20261028) per cell); "beyond grid" kept, no '
              'extrapolated point. Grid = A3 levels plus B4 levels.'),
        grid={c: v[1] for c, v in COMP_EXT.items()}, inputs=INPUTS_B4, qc=_b4qc,
        primary=b4['primary'], amp_loss=b4['amp_loss'], censored_dropped=b4['censored_dropped'],
        noise_censored_ext=NZE.groupby(['loss', 'comparator', 'level']).censored.sum().reset_index().to_dict('records'),
        comparison_with_E3x=dict(
            label=('measured break-even on the extended grid minus the E3x log-linear extrapolation from the A3 grid (native '
                   'units: dB, percentage points of peak-to-peak, mm); 95% CI from the same heart resamples for both, resamples '
                   'with either value beyond grid or undefined left out (count given)'),
            rows=cmpx))
else:
    R['B4_E3_extended_grid'] = dict(skipped='input missing', inputs=INPUTS_B4)
print(f'B4 done {time.time() - T0:.0f}s', flush=True)

R['runtime_s_before_figures'] = time.time() - T0
print(f'numbers done {time.time() - T0:.0f}s', flush=True)

# ================================================================== figures (DRAFT)
if not A.nofig:
    import figures_draft
    R['figures'] = figures_draft.make_all(A.figs, SETS['primary'], CVT['primary'], R, NSD[('corr', False)], COMP)

R['runtime_s'] = time.time() - T0
json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
import write_results_md  # noqa: E402  (same directory)
write_results_md.write(R, os.path.join(OUT, 'RESULTS.md'))
print(f'runtime {time.time() - T0:.1f} s')
