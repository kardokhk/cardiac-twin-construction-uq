"""v03 manuscript analyses (owner analysis-v03, 2026-10-29).

Usage (project root, login node):
    /project/home/p201509/envs/cvri-sim/bin/python -I code/manuscript/v03_analyses.py
Writes results/2026-10-29_v03_analyses/ (results.json, CSV tables, README.md is static and hand-written).

Inputs (read only):
    results/2026-10-28_armC/estimates.csv          round 2, loss corr unless stated
    results/2026-10-28_G1_segmentation/metrics_test50.tsv
    results/2026-10-28_armC/constructions.csv       grid_wall_median_s (eikonal run time)
    results/2026-10-28_armC/refinement.csv          wall_s of refinement runs
    data/processed/acdc_mesh_v2/MANIFEST.tsv         node_count, mesh_path (tets read from HDF5 shape only)
    results/2026-10-28_analysis/results.json         E1 regression target

Definitions (all functions below; constructions are rows of estimates.csv with distinct `construction`):
    ratio            theta_hat / theta*  (columns ratio_v_t, ratio_v_endo)
    per heart-network recovery ratio r_hn = median over the 3 theta* and the 2 resolutions (6 estimates) of ratio
    per heart-network error e_hn = median over the same 6 estimates of |ratio - 1|  (primary error definition)
    alternative error a_hn = |r_hn - 1|  (reported beside e_hn as a check)
    per-heart CV     median over theta* of SD/mean across constructions (ddof 1)  (manuscript E1 estimand)
    cluster bootstrap 2,000 resamples of hearts with replacement, numpy default_rng(20261028) created afresh
                     for every bootstrap call, percentile 2.5 and 97.5.
"""
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261028
NB = 2000
PARAMS = ('v_t', 'v_endo')
PRIMARY_NETS = ('nnunet_task027_acdc', 'nnunet_task114_mnms', 'nnunet_ukb_sax', 'cinema_acdc_sax')
WEAK = 'ukbb_cardiac_fcn_sa'
GROUP_ORDER = ('DCM', 'HCM', 'MINF', 'NOR', 'RV')
OUT = 'results/2026-10-29_v03_analyses'
INPUTS = dict(estimates='results/2026-10-28_armC/estimates.csv',
              metrics='results/2026-10-28_G1_segmentation/metrics_test50.tsv',
              constructions='results/2026-10-28_armC/constructions.csv',
              refinement='results/2026-10-28_armC/refinement.csv',
              manifest='data/processed/acdc_mesh_v2/MANIFEST.tsv',
              e1_results='results/2026-10-28_analysis/results.json')


# ------------------------------------------------------------------ generic helpers
def pct_ci(bs):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    if len(bs) == 0:
        return [float('nan'), float('nan')]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_indices(n, nb=NB, seed=SEED):
    """(nb, n) matrix of resampled unit indices; fresh generator per call."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, (nb, n))


def boot_median(v, nb=NB, seed=SEED):
    """Median of per-heart values with cluster-bootstrap percentile CI (one value per heart)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float('nan'), [float('nan'), float('nan')]
    idx = boot_indices(len(v), nb, seed)
    return float(np.median(v)), pct_ci(np.median(v[idx], axis=1))


def iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 25)), float(np.percentile(v, 75))]


# ------------------------------------------------------------------ data selection
def load_estimates(path=INPUTS['estimates'], loss='corr'):
    e = pd.read_csv(path)
    e = e[(e['round'] == 2) & (e.loss == loss)].copy()
    for c in ('is_reference', 'include_primary', 'g1_usable', 'censored'):
        e[c] = e[c].astype(bool)
    return e


def network_rows(e, source):
    """Rows used for a network's per-heart summaries: primary set for the four primary networks,
    ukbb_sensitivity & g1_usable for ukbb_cardiac, expert at 1.0 mm (non-reference) for 'expert_1.0'."""
    if source == 'expert_1.0':
        return e[e.include_primary & (e.source == 'expert') & (e.resolution_mm == 1.0)]
    if source == WEAK:
        return e[(e.analysis_set == 'ukbb_sensitivity') & e.g1_usable & (e.source == WEAK)]
    return e[e.include_primary & (e.source == source)]


def per_heart_network(rows):
    """rows: estimates for one source. Returns per-heart DataFrame with r_hn, e_hn, a_hn for both params."""
    out = []
    for h, g in rows.groupby('patient'):
        d = dict(patient=int(h), group=g.group.iloc[0], n_est=int(len(g)))
        for p in PARAMS:
            r = g[f'ratio_{p}'].values
            d[f'ratio_{p}'] = float(np.median(r))
            d[f'err_{p}'] = float(np.median(np.abs(r - 1)))
            d[f'abserr_med_{p}'] = float(abs(np.median(r) - 1))
        out.append(d)
    return pd.DataFrame(out).set_index('patient')


def heart_cv(rows, value_col):
    """Per-heart CV: median over theta* of SD/mean across constructions (ddof 1). Series by patient."""
    res = {}
    for h, g in rows.groupby('patient'):
        cvs = []
        for _, gi in g.groupby('istar'):
            v = gi[value_col].values
            if len(v) >= 2:
                cvs.append(np.std(v, ddof=1) / np.mean(v))
        res[int(h)] = float(np.median(cvs)) if cvs else float('nan')
    return pd.Series(res, dtype=float)


def heart_sd(rows, value_col):
    """Per-heart spread in native units: median over theta* of SD across constructions (ddof 1)."""
    res = {}
    for h, g in rows.groupby('patient'):
        sds = [np.std(gi[value_col].values, ddof=1) for _, gi in g.groupby('istar') if len(gi) >= 2]
        res[int(h)] = float(np.median(sds)) if sds else float('nan')
    return pd.Series(res, dtype=float)


# ------------------------------------------------------------------ item 2: ranks and Kendall W
def kendall_w(ranks):
    """Kendall coefficient of concordance with tie correction.
    ranks: (m raters = hearts, n objects = networks) matrix of within-row ranks (average ranks for ties)."""
    R = np.asarray(ranks, float)
    m, n = R.shape
    if m < 2 or n < 2:
        return float('nan')
    S = float(((R.sum(axis=0) - m * (n + 1) / 2.0) ** 2).sum())
    T = 0.0
    for row in R:
        _, cnt = np.unique(row, return_counts=True)
        T += float((cnt ** 3 - cnt).sum())
    den = m ** 2 * (n ** 3 - n) - m * T
    return float(12.0 * S / den) if den > 0 else float('nan')


def rank_largest_first(mat):
    """Within-row ranks, 1 = largest value; ties get average ranks."""
    return np.vstack([stats.rankdata(-np.asarray(r, float), method='average') for r in np.asarray(mat, float)])


def first_counts(mat, names):
    """Number of rows in which each column holds the strict or tied maximum (ties credit every tied column)."""
    mat = np.asarray(mat, float)
    mx = mat.max(axis=1, keepdims=True)
    hit = np.isclose(mat, mx, rtol=0, atol=0)
    return {nm: int(hit[:, j].sum()) for j, nm in enumerate(names)}


def friedman_from_w(w, m, n):
    chi2 = m * (n - 1) * w
    return float(chi2), float(stats.chi2.sf(chi2, n - 1))


def w_permutation_p(ranks, nperm=NB, seed=SEED):
    """Permutation null for W: shuffle ranks independently within each heart."""
    R = np.asarray(ranks, float)
    rng = np.random.default_rng(seed)
    w0 = kendall_w(R)
    null = np.empty(nperm)
    for b in range(nperm):
        null[b] = kendall_w(np.vstack([rng.permutation(r) for r in R]))
    return float((1 + np.sum(null >= w0 - 1e-12)) / (nperm + 1)), float(np.mean(null))


def boot_w(ranks, nb=NB, seed=SEED):
    R = np.asarray(ranks, float)
    idx = boot_indices(len(R), nb, seed)
    return pct_ci([kendall_w(R[i]) for i in idx])


# ------------------------------------------------------------------ item 3: Dice vs error
def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float('nan')
    return float(stats.spearmanr(x[ok], y[ok])[0])


def pooled_spearman_boot(df, xcol, ycol, nb=NB, seed=SEED):
    """Spearman over heart-network pairs; CI by resampling hearts (all pairs of a drawn heart enter)."""
    hearts = np.array(sorted(df.patient.unique()))
    by = {h: df[df.patient == h] for h in hearts}
    est = spearman(df[xcol], df[ycol])
    idx = boot_indices(len(hearts), nb, seed)
    bs = []
    for row in idx:
        s = pd.concat([by[hearts[i]] for i in row])
        bs.append(spearman(s[xcol], s[ycol]))
    return est, pct_ci(bs), int(len(hearts)), int(len(df))


def within_heart_spearman(df, xcol, ycol):
    """Per-heart Spearman across that heart's networks (NaN when either variable is constant or < 3 networks)."""
    return pd.Series({int(h): spearman(g[xcol], g[ycol]) for h, g in df.groupby('patient')}, dtype=float)


def boot_mean(v, nb=NB, seed=SEED):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    idx = boot_indices(len(v), nb, seed)
    return float(np.mean(v)), pct_ci(np.mean(v[idx], axis=1)), int(len(v))


# ------------------------------------------------------------------ item 4: ensemble size
def subset_cv_matrix(rows, nets, k, value_col):
    """Hearts x subsets matrix of per-heart CV computed over the constructions of each k-subset of nets."""
    subs = list(itertools.combinations(nets, k))
    cols = {}
    for s in subs:
        cols['+'.join(s)] = heart_cv(rows[rows.source.isin(s)], value_col)
    return pd.DataFrame(cols), subs


def ensemble_stat(mat):
    """Mean over subsets of the median over hearts. mat: hearts x subsets array."""
    return float(np.mean(np.median(np.asarray(mat, float), axis=0)))


def boot_ensemble(mats, nb=NB, seed=SEED):
    """mats: dict k -> hearts x subsets array (same heart order). Returns point, CI per k and ratio to k=4."""
    ks = sorted(mats)
    n = len(next(iter(mats.values())))
    idx = boot_indices(n, nb, seed)
    pt = {k: ensemble_stat(mats[k]) for k in ks}
    bs = {k: np.array([ensemble_stat(np.asarray(mats[k])[i]) for i in idx]) for k in ks}
    kmax = max(ks)
    out = {}
    for k in ks:
        out[k] = dict(value=pt[k], ci=pct_ci(bs[k]), ratio_to_k4=pt[k] / pt[kmax], ratio_ci=pct_ci(bs[k] / bs[kmax]),
                      n_subsets=int(np.asarray(mats[k]).shape[1]))
    return out


def half_spread_fraction(pair_mat, full):
    """pair_mat: hearts x pairs CV; full: per-heart 4-network CV. Indicator CV_pair >= 0.5 * CV_full.
    Returns per-heart fraction of pairs meeting it, and the all-pairs and any-pair heart fractions."""
    P = np.asarray(pair_mat, float)
    f = np.asarray(full, float)[:, None]
    ind = P >= 0.5 * f
    per_heart = ind.mean(axis=1)
    return per_heart, float(np.mean(ind.all(axis=1))), float(np.mean(ind.any(axis=1)))


# ------------------------------------------------------------------ item 6: mesh size
TETS_TABLE = 'data/processed/acdc_mesh_v2/MANIFEST_tets.tsv'  # element counts read from the mesh files (public release)


def tets_count(mesh_path):
    fn = mesh_path.split(':')[0]
    if os.path.exists(fn):
        import h5py
        with h5py.File(fn, 'r') as f:
            return int(f['tets'].shape[0])
    t = pd.read_csv(TETS_TABLE, sep='\t')  # meshes are not redistributed; use the table written from them
    return int(t.loc[t.mesh_file == fn, 'tets'].iloc[0])


# ================================================================== main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    R = dict(inputs=INPUTS, seed=SEED, n_boot=NB, date='2026-10-29',
             code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
             versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                           scipy=__import__('scipy').__version__),
             input_sha256={k: hashlib.sha256(open(v, 'rb').read()).hexdigest() for k, v in INPUTS.items()})
    E = load_estimates()
    prim = E[E.include_primary]

    # ---- regression check: E1
    e1 = {p: boot_median(heart_cv(prim, f'ratio_{p}'))[0] for p in PARAMS}
    ref = json.load(open(INPUTS['e1_results']))['E1_primary']
    assert round(100 * e1['v_t'], 1) == 16.1 and round(100 * e1['v_endo'], 1) == 2.6, e1
    for p in PARAMS:
        assert abs(e1[p] - ref[p]['cv_median']) < 1e-12, (p, e1[p], ref[p]['cv_median'])
    R['E1_regression'] = dict(v_t=e1['v_t'], v_endo=e1['v_endo'], n_hearts=int(prim.patient.nunique()),
                              n_constructions=int(prim.construction.nunique()), n_estimates=int(len(prim)),
                              matches_results_json=True)

    # ---- metrics
    M = pd.read_csv(INPUTS['metrics'], sep='\t')
    M['pid'] = M.patient.str.replace('patient', '').astype(int)
    M = M.set_index(['pid', 'network'])

    # ---- item 1: per-network table
    sources = list(PRIMARY_NETS) + [WEAK, 'expert_1.0']
    PH = {}
    rows_t = []
    for s in sources:
        ph = per_heart_network(network_rows(E, s))
        if s != 'expert_1.0':
            for c, mc in (('myo_dice', 'MYO_dice'), ('lv_dice', 'LV_dice'), ('rv_dice', 'RV_dice')):
                ph[c] = [float(M.loc[(h, s), mc]) for h in ph.index]
        PH[s] = ph
        d = dict(source=s, role=('primary network' if s in PRIMARY_NETS else
                                 'sensitivity (G1-usable rows)' if s == WEAK else 'expert labels, 1.0 mm'),
                 n_hearts=int(len(ph)), n_estimates=int(ph.n_est.sum()))
        if s != 'expert_1.0':
            ms = M.xs(s, level='network')
            for c, mc in (('myo', 'MYO_dice'), ('lv', 'LV_dice'), ('rv', 'RV_dice')):
                d[f'{c}_dice_median_used_hearts'] = float(ph[f'{c}_dice'].median())
                d[f'{c}_dice_median_all50'] = float(ms[mc].median())
        for p in PARAMS:
            d[f'ratio_{p}_median'] = float(ph[f'ratio_{p}'].median())
            d[f'ratio_{p}_q25'], d[f'ratio_{p}_q75'] = iqr(ph[f'ratio_{p}'])
            m_, ci = boot_median(ph[f'err_{p}'])
            d[f'abs_err_{p}_median'], d[f'abs_err_{p}_ci_lo'], d[f'abs_err_{p}_ci_hi'] = m_, ci[0], ci[1]
            m2, ci2 = boot_median(ph[f'abserr_med_{p}'])
            d[f'alt_abs_err_med_{p}_median'], d[f'alt_abs_err_med_{p}_ci_lo'], d[f'alt_abs_err_med_{p}_ci_hi'] = m2, ci2[0], ci2[1]
        rows_t.append(d)
    T1 = pd.DataFrame(rows_t)
    T1.to_csv(f'{OUT}/table1_per_network.csv', index=False, float_format='%.10g')
    R['item1_per_network'] = T1.to_dict(orient='records')
    long = pd.concat([PH[s].assign(source=s).reset_index() for s in sources], ignore_index=True)
    long.to_csv(f'{OUT}/per_heart_network_long.csv', index=False, float_format='%.10g')

    # ---- item 2: heatmap matrix and rank concordance
    mat = pd.DataFrame({s: PH[s]['ratio_v_t'] for s in sources})
    grp = E.groupby('patient').group.first()
    mat.insert(0, 'group', grp.reindex(mat.index))
    mat['group_order'] = mat.group.map({g: i for i, g in enumerate(GROUP_ORDER)})
    mat.index.name = 'patient'
    mat = mat.reset_index().sort_values(['group_order', 'patient'])
    mat.drop(columns='group_order').to_csv(f'{OUT}/heatmap_ratio_v_t_heart_x_network.csv', index=False,
                                           float_format='%.10g')
    item2 = {}
    for errname in ('err', 'abserr_med'):
        for p in PARAMS:
            em = pd.DataFrame({s: PH[s][f'{errname}_{p}'] for s in PRIMARY_NETS}).dropna()
            rk = rank_largest_first(em.values)
            w = kendall_w(rk)
            chi2, pf = friedman_from_w(w, *rk.shape)
            pp, wnull = w_permutation_p(rk)
            fc = first_counts(em.values, PRIMARY_NETS)
            item2[f'{errname}_{p}'] = dict(
                n_hearts_complete=int(len(em)), first_counts=fc, n_networks_first_in_ge1_heart=int(sum(v > 0 for v in fc.values())),
                kendall_w=w, kendall_w_ci=boot_w(rk), friedman_chi2=chi2, friedman_df=3, friedman_p=pf,
                permutation_p=pp, w_null_mean=wnull, mean_rank_by_network=dict(zip(PRIMARY_NETS, rk.mean(axis=0).tolist())))
    R['item2_rank_concordance'] = item2

    # ---- item 3: Dice vs error (non-reference rows: networks only; expert has no Dice)
    item3 = {}
    for label, nets in (('primary4', PRIMARY_NETS), ('primary4_plus_ukbb', PRIMARY_NETS + (WEAK,))):
        df = long[long.source.isin(nets)].copy()
        for dice in ('myo_dice', 'rv_dice'):
            for p in PARAMS:
                est, ci, nh, npair = pooled_spearman_boot(df, dice, f'err_{p}')
                est2, ci2, _, _ = pooled_spearman_boot(df, dice, f'abserr_med_{p}')
                wh = within_heart_spearman(df, dice, f'err_{p}')
                mwh, ciwh, nwh = boot_mean(wh.values)
                item3[f'{label}__{dice}__{p}'] = dict(
                    pooled_spearman=est, pooled_ci=ci, n_hearts=nh, n_pairs=npair,
                    pooled_spearman_alt_abserr_med=est2, pooled_ci_alt=ci2,
                    within_heart_mean_spearman=mwh, within_heart_ci=ciwh, within_heart_n_hearts=nwh,
                    within_heart_n_positive=int((wh > 0).sum()), within_heart_n_negative=int((wh < 0).sum()))
    R['item3_dice_vs_error'] = item3

    # ---- item 4: ensemble size (networks only, both resolutions, expert excluded)
    net_rows = prim[prim.source.isin(PRIMARY_NETS)]
    ncons = net_rows.groupby('patient').source.nunique()
    complete = sorted(ncons[ncons == 4].index)
    item4 = {}
    for setname, hearts in (('complete_hearts', complete), ('all_hearts_available', sorted(ncons.index))):
        rows = net_rows[net_rows.patient.isin(hearts)]
        for p in PARAMS:
            mats = {}
            for k in (2, 3, 4):
                mk, _ = subset_cv_matrix(rows, PRIMARY_NETS, k, f'ratio_{p}')
                mats[k] = mk.reindex(hearts)
                mats[k].to_csv(f'{OUT}/ensemble_cv_{p}_k{k}_{setname}.csv', index_label='patient', float_format='%.10g')
            if setname == 'all_hearts_available':
                # drop subset cells that collapse to a single network (missing networks); per-subset median skips NaN
                for k in mats:
                    m_ = mats[k].copy()
                    for h in hearts:
                        have = set(rows[rows.patient == h].source.unique())
                        for c in m_.columns:
                            if len(have & set(c.split('+'))) < 2:
                                m_.loc[h, c] = np.nan
                    mats[k] = m_
                pt = {k: float(np.mean(np.nanmedian(mats[k].values, axis=0))) for k in mats}
                item4[f'{setname}_{p}'] = dict(n_hearts=len(hearts), value={k: pt[k] for k in pt},
                                               ratio_to_k4={k: pt[k] / pt[4] for k in pt},
                                               note='point estimates only; subsets with fewer than 2 available networks set to NaN')
                continue
            bo = boot_ensemble({k: mats[k].values for k in mats})
            ph_frac, all_f, any_f = half_spread_fraction(mats[2].values, mats[4].values[:, 0])
            mf, cif, _ = boot_mean(ph_frac)
            item4[f'{setname}_{p}'] = dict(n_hearts=len(hearts), by_k={str(k): v for k, v in bo.items()},
                                           half_spread_mean_fraction_of_pairs=mf, half_spread_ci=cif,
                                           half_spread_hearts_all_pairs=all_f, half_spread_hearts_any_pair=any_f)
    R['item4_ensemble_size'] = item4
    R['item4_excluded_from_complete'] = [int(h) for h in ncons.index if h not in complete]

    # ---- item 5: physical units
    item5 = {}
    for setname, rows in (('primary', prim), ('primary_no_ref', prim[~prim.is_reference])):
        for p in PARAMS:
            sd = heart_sd(rows, f'{p}_hat')
            m_, ci = boot_median(sd.values)
            item5[f'{setname}_{p}'] = dict(sd_m_per_s_median=m_, ci=ci, per_heart_iqr=iqr(sd.values), n_hearts=int(sd.notna().sum()))
            sd.to_csv(f'{OUT}/sd_{p}_hat_m_per_s_{setname}.csv', header=[f'sd_{p}_hat'], index_label='patient')
    ts = prim.drop_duplicates(['patient', 'istar'])
    for p in PARAMS:
        item5[f'true_{p}'] = dict(median_over_theta_star=float(ts[f'{p}_star'].median()),
                                  iqr=iqr(ts[f'{p}_star']), n_theta_star=int(len(ts)),
                                  median_of_per_heart_median=float(ts.groupby('patient')[f'{p}_star'].median().median()))
    R['item5_physical_units'] = item5

    # ---- item 6: mesh size and run time
    MF = pd.read_csv(INPUTS['manifest'], sep='\t')
    MF = MF[MF.include_primary.astype(bool) & MF.construction_id.isin(prim.construction.unique())].copy()
    MF['tets'] = [tets_count(mp) for mp in MF.mesh_path]
    C = pd.read_csv(INPUTS['constructions'])
    C = C[C.construction.isin(prim.construction.unique())]
    RF = pd.read_csv(INPUTS['refinement'])
    RF = RF[RF.construction.isin(prim.construction.unique())]  # refinement runs of both calibration rounds
    item6 = {}
    for res in (1.5, 1.0):
        mf = MF[MF.resolution_mm == res]
        c = C[C.resolution_mm == res]
        rf = RF[RF.resolution_mm == res]
        item6[str(res)] = dict(n_constructions=int(len(mf)), node_median=float(mf.node_count.median()),
                               node_iqr=iqr(mf.node_count), tet_median=float(mf.tets.median()), tet_iqr=iqr(mf.tets),
                               grid_run_wall_s_median_over_constructions=float(c.grid_wall_median_s.median()),
                               grid_run_wall_s_iqr=iqr(c.grid_wall_median_s),
                               refinement_run_wall_s_median=float(rf.wall_s.median()) if len(rf) else None,
                               n_refinement_runs=int(len(rf)))
    # paired 1.0/1.5 ratios per heart-source
    MF['key'] = MF.heart.astype(str) + '_' + MF.source
    piv = MF.pivot_table(index='key', columns='resolution_mm', values=['node_count', 'tets'])
    C['key'] = C.patient.astype(str) + '_' + C.source
    pc = C.pivot_table(index='key', columns='resolution_mm', values='grid_wall_median_s')
    item6['paired_ratio_1.0_over_1.5'] = dict(
        nodes_median=float((piv['node_count'][1.0] / piv['node_count'][1.5]).median()),
        tets_median=float((piv['tets'][1.0] / piv['tets'][1.5]).median()),
        grid_wall_median=float((pc[1.0] / pc[1.5]).median()), n_pairs=int(pc.dropna().shape[0]))
    item6['run_time_source'] = ('constructions.csv grid_wall_median_s = median over the 81 grid eikonal runs (9 x 9 theta grid, hearts/*.h5 grid/theta) of that '
                                'construction of per-run wall time (run_armC.py, one thread per run, runs packed '
                                'on a node); refinement.csv wall_s per refinement run, rounds 1 and 2 pooled')
    R['item6_resolution'] = item6
    MF[['construction_id', 'heart', 'source', 'resolution_mm', 'node_count', 'tets']].to_csv(
        f'{OUT}/mesh_size_primary.csv', index=False)

    R['runtime_s'] = round(time.time() - t0, 1)
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, default=float)
    print(json.dumps(dict(E1=R['E1_regression'], runtime_s=R['runtime_s'])))


if __name__ == '__main__':
    main()
