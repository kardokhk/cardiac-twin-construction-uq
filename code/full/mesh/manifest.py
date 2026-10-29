"""Full-arm construction manifest and summaries (owner mesh2, 2026-10-28).
Usage: python -I -B manifest.py <results_run_dir> [<results_run_dir> ...]   (later dirs override earlier ones)
Reads <run>/qc/*.json; writes data/processed/acdc_mesh_v2/MANIFEST.tsv, <last run>/constructions.tsv and
<last run>/summary.json.

Inclusion (Amendment A1; assumptions in results/2026-10-28_mesh_v2/README.md):
  g1_usable  network masks: usable_all in data/processed/acdc_seg/MANIFEST.tsv; expert labels: True (reference).
  g2_pass    all failure criteria F1 to F9 of results/2026-10-28_G2_meshing/G2_criteria.md pass (status 'pass').
  include_primary      primary source (expert + 4 networks), RV wall 3 mm, g1_usable and g2_pass.
  include_sensitivity  the varied level of a declared sensitivity analysis: ukbb_cardiac_fcn_sa at RV wall 3 mm
                       with g2_pass ("run on every heart where it meshes", A1; G1 status reported beside it), or
                       an RV-wall 2 or 4 mm construction with g1_usable and g2_pass.
Root shift: maximum over the 4 roots of |mapped root node - reference rule position|; flag > 10 mm (A4); the
denominator is the non-reference constructions of the same resolution (RV wall 3 mm) with a mapped root.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import acdc_lib as A          # noqa: E402
from lists import PRIMARY, SENS   # noqa: E402

runs = sys.argv[1:]
rows = {}
for r in runs:
    for f in sorted(glob.glob(os.path.join(r, 'qc', '*.json'))):
        q = json.load(open(f)); q['qc_run'] = os.path.basename(os.path.normpath(r)); rows[q['name']] = q
df = pd.DataFrame(list(rows.values()))
for c in ('mesh_ok', 'smoke_ok'):
    df[c] = df[c].fillna(False).astype(bool) if c in df else False
seg = pd.read_csv(os.path.join(A.ROOT, 'data', 'processed', 'acdc_seg', 'MANIFEST.tsv'), sep='\t')
seg['pid'] = seg.patient.str.replace('patient', '').astype(int)
usable = {(int(p), n): bool(u) for p, n, u in zip(seg.pid, seg.network, seg.usable_all)}
df['g1_usable'] = [True if s == 'expert' else usable.get((int(p), s), False) for p, s in zip(df.patient, df.source)]
df['g2_pass'] = df.status == 'pass'
df['g2_fail_code'] = df.fail_code.where(~df.g2_pass, '')
rv = df.rv_wall_mm.astype(float)
main = np.isclose(rv, 3.0)
df['include_primary'] = df.source.isin(PRIMARY) & main & df.g1_usable & df.g2_pass
df['include_sensitivity'] = ((df.source.isin(SENS) & main & df.g2_pass) |
                             (~main & df.source.isin(PRIMARY) & df.g1_usable & df.g2_pass))
df['analysis_set'] = np.where(~main, np.where(np.isclose(df.vox, 1.0), 'rv_wall_extra_1.0mm', 'rv_wall_sensitivity'),
                             np.where(df.source.isin(SENS), 'ukbb_sensitivity', 'primary'))


def reason(r):
    if r.include_primary:
        return ''
    out = []
    if not r.g1_usable:
        out.append('G1 mask not usable')
    if not r.g2_pass:
        out.append(f'G2 {r.fail_code}: {str(r.msg)[:80]}')
    if r.analysis_set == 'ukbb_sensitivity':
        out.append('ukbb_cardiac: sensitivity level only (A1)')
    if r.analysis_set == 'rv_wall_sensitivity':
        out.append(f'RV wall {r.rv_wall_mm:.0f} mm sensitivity (plan section 3)')
    if r.analysis_set == 'rv_wall_extra_1.0mm':
        out.append(f'RV wall {r.rv_wall_mm:.0f} mm at 1.0 mm: extra, not pre-registered')
    return '; '.join(out)


df['exclusion_reason'] = df.apply(reason, axis=1)
df['is_reference'] = (df.source == 'expert') & np.isclose(df.vox, 1.5) & main
rs = df.get('root_shift_mm')
df['root_shift_each_mm'] = [json.dumps(x) if isinstance(x, list) else '' for x in rs]
df['root_shift_max_mm'] = [max(x) if isinstance(x, list) else np.nan for x in rs]
df['root_flag'] = (df.root_shift_max_mm > 10.0) & ~df.is_reference
df['construction_id'] = df.name
df['resolution_mm'] = df.vox
df['mesh_path'] = np.where(df.mesh_ok, df.get('h5', pd.Series([''] * len(df))).fillna('') + ':/pts,/tets,/etag,/fib', '')
dp = os.path.join('data', 'processed', 'acdc_mesh_v2', 'displaced')
df['displaced_leadfield_path'] = [os.path.join(dp, f'patient{int(p):03d}_W_displaced.h5')
                                  if ref and os.path.exists(os.path.join(A.ROOT, dp, f'patient{int(p):03d}_W_displaced.h5')) else ''
                                  for p, ref in zip(df.patient, df.is_reference)]
df = df.rename(columns={'patient': 'heart', 'n_nodes': 'node_count'})
df = df.sort_values(['heart', 'rv_wall_mm', 'resolution_mm', 'source'])
cols = ['heart', 'group', 'construction_id', 'source', 'resolution_mm', 'rv_wall_mm', 'include_primary',
        'include_sensitivity', 'exclusion_reason', 'mesh_path', 'node_count', 'root_shift_max_mm', 'root_flag',
        'smoke_ok', 'displaced_leadfield_path']
extra = ['analysis_set', 'is_reference', 'g1_usable', 'g2_pass', 'g2_fail_code', 'root_shift_each_mm', 'endo_frac',
         'endo_vol_frac', 'endo_frac_onevoxel', 'endo_vol_frac_onevoxel', 'err_lv_myo', 'err_lv_cav',
         'smoke_act_max_ms', 'gates_same_mesh', 'gates_W_rel_maxdiff', 'gates_etag_equal_onevoxel', 'gates_ref_maxdiff',
         'maxrss_mb', 'total_s', 'qc_run']
extra = [c for c in extra if c in df.columns]
man = df[cols + extra].rename(columns={'root_shift_max_mm': 'root_shift_mm'})
man.to_csv(os.path.join(A.ROOT, 'data', 'processed', 'acdc_mesh_v2', 'MANIFEST.tsv'), sep='\t', index=False)
last = runs[-1]
df.drop(columns=[c for c in ('traceback',) if c in df]).to_csv(os.path.join(last, 'constructions.tsv'), sep='\t', index=False)


def q(x):
    x = pd.Series(x).dropna().astype(float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=float(x.median()), p2_5=float(x.quantile(0.025)), p97_5=float(x.quantile(0.975)),
                min=float(x.min()), max=float(x.max()))


S = {}
S['n_constructions'] = int(len(df))
S['by_set'] = {k: dict(attempted=int(len(g)), g2_pass=int(g.g2_pass.sum()), smoke_ok=int(g.smoke_ok.sum()),
                       include_primary=int(g.include_primary.sum()), include_sensitivity=int(g.include_sensitivity.sum()))
               for k, g in df.groupby('analysis_set')}
tab = {}
for (s, v, w), g in df.groupby(['source', 'resolution_mm', 'rv_wall_mm']):
    tab[f'{s}|{v}|{w}'] = dict(attempted=int(len(g)), g1_usable=int(g.g1_usable.sum()), g2_pass=int(g.g2_pass.sum()),
                               included=int((g.include_primary | g.include_sensitivity).sum()),
                               fail_codes=g[~g.g2_pass].fail_code.value_counts().to_dict(),
                               excluded_hearts=sorted(int(h) for h in g[~(g.include_primary | g.include_sensitivity)].heart))
S['by_source_resolution_rvwall'] = tab
# root shifts (A4)
rt = {}
for v in (1.5, 1.0):
    g = df[np.isclose(df.resolution_mm, v) & np.isclose(df.rv_wall_mm, 3.0) & ~df.is_reference & df.root_shift_max_mm.notna()]
    gp = g[g.source.isin(PRIMARY)]
    rt[str(v)] = dict(denominator_all=int(len(g)), flagged_all=int(g.root_flag.sum()),
                      denominator_primary_sources=int(len(gp)), flagged_primary_sources=int(gp.root_flag.sum()),
                      flagged_included_primary=int(g[g.include_primary].root_flag.sum()),
                      denominator_included_primary=int(g.include_primary.sum()),
                      flagged_by_source=g[g.root_flag].source.value_counts().to_dict(),
                      flagged_list=g[g.root_flag][['construction_id', 'root_shift_max_mm']].values.tolist(),
                      shift_mm=q(g.root_shift_max_mm))
S['root_shift'] = rt
# fast layer share (A2) against the one-voxel definition, mesh share of myocardial volume
ly = {}
for v in (1.5, 1.0):
    g = df[np.isclose(df.resolution_mm, v) & np.isclose(df.rv_wall_mm, 3.0) & df.mesh_ok]
    ge = g[g.source == 'expert']
    ly[str(v)] = dict(all_vol_frac_a2=q(g.endo_vol_frac), all_vol_frac_onevoxel=q(g.endo_vol_frac_onevoxel),
                      expert_vol_frac_a2=q(ge.endo_vol_frac), expert_vol_frac_onevoxel=q(ge.endo_vol_frac_onevoxel),
                      trio=g[g.heart.isin([101, 102, 104])][['construction_id', 'endo_vol_frac', 'endo_vol_frac_onevoxel']]
                      .values.tolist())
S['endo_layer'] = ly
S['node_count'] = {f'{v}|{w}': q(g.node_count) for (v, w), g in df[df.mesh_ok].groupby(['resolution_mm', 'rv_wall_mm'])}
S['smoke_act_max_ms'] = {str(v): q(g.smoke_act_max_ms) for v, g in df[df.smoke_ok].groupby('resolution_mm')}
S['included_but_smoke_fail'] = int(((df.include_primary | df.include_sensitivity) & ~df.smoke_ok).sum())
for c in ('gates_same_mesh', 'gates_etag_equal_onevoxel'):
    if c in df:
        x = df[c].dropna()
        S[c] = dict(n=int(len(x)), true=int(x.astype(bool).sum()))
for c in ('gates_W_rel_maxdiff', 'gates_ref_maxdiff', 'maxrss_mb', 'total_s'):
    if c in df:
        S[c] = q(df[c])
S['maxrss_mb_by_res'] = {str(v): q(g.maxrss_mb) for v, g in df.groupby('resolution_mm')} if 'maxrss_mb' in df else None
json.dump(S, open(os.path.join(last, 'summary.json'), 'w'), indent=1, default=str)
print(json.dumps({k: S[k] for k in ('n_constructions', 'by_set', 'root_shift', 'included_but_smoke_fail')}, indent=1, default=str))
