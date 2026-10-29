"""Independent recompute of results/2026-10-29_v03_analyses/results.json (does not import their code)."""
import itertools, json, sys
import numpy as np, pandas as pd, h5py
from scipy import stats

ROOT = '/project/home/p201509/work/cvri/simulation/'
SEED, NB = 20261028, 2000
NETS = ['nnunet_task027_acdc', 'nnunet_task114_mnms', 'nnunet_ukb_sax', 'cinema_acdc_sax']
WEAK = 'ukbb_cardiac_fcn_sa'
P = ['v_t', 'v_endo']

def idx(n):
    return np.random.default_rng(SEED).integers(0, n, (NB, n))

def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return np.percentile(a, [2.5, 97.5])

E = pd.read_csv(ROOT + 'results/2026-10-28_armC/estimates.csv')
E = E[(E['round'] == 2) & (E.loss == 'corr')]
for c in ['include_primary', 'is_reference', 'g1_usable']:
    E[c] = E[c].astype(str).str.lower().eq('true')
prim = E[E.include_primary]
out = {}

def cv_per_heart(df, col):
    g = df.groupby(['patient', 'istar'])[col].agg(lambda v: v.std(ddof=1) / v.mean() if len(v) > 1 else np.nan)
    return g.groupby(level=0).median()

for p in P:
    out[f'E1_{p}'] = cv_per_heart(prim, f'ratio_{p}').median()

M = pd.read_csv(ROOT + 'results/2026-10-28_G1_segmentation/metrics_test50.tsv', sep='\t')
M['patient'] = M.patient.str[7:].astype(int)

def rows_for(s):
    if s == 'expert_1.0':
        return prim[(prim.source == 'expert') & (prim.resolution_mm == 1.0)]
    if s == WEAK:
        return E[(E.analysis_set == 'ukbb_sensitivity') & E.g1_usable & (E.source == WEAK)]
    return prim[prim.source == s]

PH = {}
for s in NETS + [WEAK, 'expert_1.0']:
    r = rows_for(s).copy()
    d = {}
    for p in P:
        r[f'e_{p}'] = (r[f'ratio_{p}'] - 1).abs()
        d[f'r_{p}'] = r.groupby('patient')[f'ratio_{p}'].median()
        d[f'e_{p}'] = r.groupby('patient')[f'e_{p}'].median()
        d[f'a_{p}'] = (d[f'r_{p}'] - 1).abs()
    ph = pd.DataFrame(d).sort_index()
    if s != 'expert_1.0':
        ms = M[M.network == s].set_index('patient')
        ph['myo'] = ms.MYO_dice.reindex(ph.index); ph['rv'] = ms.RV_dice.reindex(ph.index)
        out[f'{s}_myo_used'] = ph.myo.median(); out[f'{s}_myo_all'] = ms.MYO_dice.median()
        out[f'{s}_rv_used'] = ph.rv.median()
    out[f'{s}_n'] = len(ph); out[f'{s}_nest'] = len(r)
    for p in P:
        v = ph[f'r_{p}']
        out[f'{s}_ratio_{p}'] = [v.median(), v.quantile(.25), v.quantile(.75)]
        for kind in 'ea':
            v = ph[f'{kind}_{p}'].values
            out[f'{s}_{kind}_{p}'] = [np.median(v), *ci(np.median(v[idx(len(v))], axis=1))]
    PH[s] = ph

# item 2: Kendall W via scipy Friedman (W = chi2 / (m (k - 1))), ranks 1 = largest
def W_of(X):
    if X.shape[0] < 2: return np.nan
    chi = stats.friedmanchisquare(*X.T).statistic
    return chi / (X.shape[0] * (X.shape[1] - 1))
for kind in 'ea':
    for p in P:
        X = pd.concat([PH[s][f'{kind}_{p}'].rename(s) for s in NETS], axis=1).dropna()
        A = X.values
        Rk = np.vstack([stats.rankdata(-row) for row in A])
        w = W_of(Rk)
        fr = stats.friedmanchisquare(*A.T)
        bw = [W_of(Rk[i]) for i in idx(len(Rk))]
        # permutation, same scheme as the analysis (fresh rng, row-wise permutation)
        rng = np.random.default_rng(SEED)
        null = np.array([W_of(np.vstack([rng.permutation(r) for r in Rk])) for _ in range(NB)])
        # alternative scheme with rng.permuted (different draws)
        rng2 = np.random.default_rng(SEED)
        null2 = np.array([W_of(rng2.permuted(Rk, axis=1)) for _ in range(NB)])
        first = (A == A.max(axis=1, keepdims=True)).sum(axis=0)
        out[f'item2_{kind}_{p}'] = dict(n=len(A), first=dict(zip(NETS, first.tolist())), W=w, Wci=ci(bw).tolist(),
            chi2=fr.statistic, friedman_p=fr.pvalue, perm_p=(1 + (null >= w - 1e-12).sum()) / (NB + 1),
            perm_p_alt=(1 + (null2 >= w - 1e-12).sum()) / (NB + 1), null_mean=null.mean(), null_mean_alt=null2.mean())

# item 3
L = pd.concat([PH[s].assign(source=s) for s in NETS + [WEAK]]).reset_index()
def sp(df, x, y):
    return stats.spearmanr(df[x], df[y]).statistic
for lab, nets in [('p4', NETS), ('p4u', NETS + [WEAK])]:
    df = L[L.source.isin(nets)]
    H = np.array(sorted(df.patient.unique()))
    groups = {h: df[df.patient == h] for h in H}
    I = idx(len(H))
    for dice in ['myo', 'rv']:
        for p in P:
            y = f'e_{p}'
            bs = [sp(pd.concat([groups[H[i]] for i in row]), dice, y) for row in I]
            wh = np.array([sp(groups[h], dice, y) if groups[h][dice].nunique() > 1 and len(groups[h]) >= 3 else np.nan for h in H])
            wh = wh[np.isfinite(wh)]
            out[f'item3_{lab}_{dice}_{p}'] = dict(rho=sp(df, dice, y), ci=ci(bs).tolist(), npairs=len(df), nh=len(H),
                rho_alt=sp(df, dice, f'a_{p}'), wh=wh.mean(), whci=ci(wh[idx(len(wh))].mean(axis=1)).tolist(),
                npos=int((wh > 0).sum()), nneg=int((wh < 0).sum()))

# item 4
nr = prim[prim.source.isin(NETS)]
cnt = nr.groupby('patient').source.nunique()
comp = sorted(cnt[cnt == 4].index)
out['item4_excluded'] = sorted(cnt[cnt < 4].index.tolist())
for p in P:
    rows = nr[nr.patient.isin(comp)]
    mats = {}
    for k in (2, 3, 4):
        mats[k] = np.column_stack([cv_per_heart(rows[rows.source.isin(sub)], f'ratio_{p}').reindex(comp).values
                                   for sub in itertools.combinations(NETS, k)])
    stat = lambda m: np.mean(np.median(m, axis=0))
    I = idx(len(comp))
    bs = {k: np.array([stat(mats[k][i]) for i in I]) for k in mats}
    res = {}
    for k in mats:
        res[k] = dict(v=stat(mats[k]), ci=ci(bs[k]).tolist(), ratio=stat(mats[k]) / stat(mats[4]), rci=ci(bs[k] / bs[4]).tolist(),
                      alt_median_of_mean=np.median(mats[k].mean(axis=1)))
    ind = mats[2] >= 0.5 * mats[4]
    ph = ind.mean(axis=1)
    res['half'] = dict(frac=ph.mean(), ci=ci(ph[I].mean(axis=1)).tolist(), all=ind.all(axis=1).mean(), any=ind.any(axis=1).mean())
    # all-hearts variant
    allh = sorted(cnt.index); rows = nr
    have = nr.groupby('patient').source.unique()
    va = {}
    for k in (2, 3, 4):
        cols = []
        for sub in itertools.combinations(NETS, k):
            c = cv_per_heart(rows[rows.source.isin(sub)], f'ratio_{p}').reindex(allh)
            c[[len(set(have[h]) & set(sub)) < 2 for h in allh]] = np.nan
            cols.append(np.nanmedian(c.values))
        va[k] = np.mean(cols)
    res['allhearts'] = va
    out[f'item4_{p}'] = res

# item 5
for lab, rows in [('prim', prim), ('noref', prim[~prim.is_reference])]:
    for p in P:
        sd = rows.groupby(['patient', 'istar'])[f'{p}_hat'].std(ddof=1).groupby(level=0).median().values
        out[f'item5_{lab}_{p}'] = [np.median(sd), *ci(np.median(sd[idx(len(sd))], axis=1)), np.percentile(sd, 25), np.percentile(sd, 75)]
ts = prim.groupby(['patient', 'istar'])[['v_t_star', 'v_endo_star']].first()
out['item5_true'] = [ts.v_t_star.median(), ts.v_endo_star.median(), len(ts)]

# item 6
MF = pd.read_csv(ROOT + 'data/processed/acdc_mesh_v2/MANIFEST.tsv', sep='\t')
cons = set(prim.construction)
MF = MF[MF.construction_id.isin(cons)].copy()
def ntet(mp):
    with h5py.File(ROOT + mp.split(':')[0], 'r') as f:
        return f['tets'].shape[0]
MF['tets'] = MF.mesh_path.map(ntet)
C = pd.read_csv(ROOT + 'results/2026-10-28_armC/constructions.csv'); C = C[C.construction.isin(cons)]
RF = pd.read_csv(ROOT + 'results/2026-10-28_armC/refinement.csv'); RF = RF[RF.construction.isin(cons)]
for res in (1.5, 1.0):
    m = MF[MF.resolution_mm == res]; c = C[C.resolution_mm == res]
    out[f'item6_{res}'] = dict(n=len(m), nodes=m.node_count.median(), tets=m.tets.median(), wall=c.grid_wall_median_s.median(),
                               wall_iqr=c.grid_wall_median_s.quantile([.25, .75]).tolist(),
                               ref_wall=RF[RF.resolution_mm == res].wall_s.median(), nref=int((RF.resolution_mm == res).sum()),
                               nodes_csv_vs_manifest_maxdiff=None)
pm = MF.pivot_table(index=['heart', 'source'], columns='resolution_mm', values=['node_count', 'tets'])
pc = C.pivot_table(index=['patient', 'source'], columns='resolution_mm', values='grid_wall_median_s')
out['item6_ratio'] = dict(nodes=(pm['node_count'][1.0] / pm['node_count'][1.5]).median(), tets=(pm['tets'][1.0] / pm['tets'][1.5]).median(),
                          wall=(pc[1.0] / pc[1.5]).median(), n=int(pc.dropna().shape[0]))
# cross-check node counts between MANIFEST and constructions.csv
mm = MF.set_index('construction_id').node_count.reindex(C.construction).values
out['item6_nodes_manifest_vs_constructions_maxabsdiff'] = float(np.nanmax(np.abs(mm - C.n_nodes.values)))
# grid size check from one heart h5
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
print('ok')
