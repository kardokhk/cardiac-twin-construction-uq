"""Aggregate construction QC into data/processed/acdc_mesh/MANIFEST.tsv and G2 / bonus summaries (owner mesh).
Reads every QC JSON under results/2026-10-28_G2_meshing/*/qc (a later run overrides an earlier one for the same
construction; 'full' sorts after 'test'). Usage: python -I manifest.py <results_mode_dir>"""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import acdc_lib as A

rdir = sys.argv[1]
base = os.path.join(A.ROOT, 'results', '2026-10-28_G2_meshing')
rows = {}
for mode in ('test', 'full'):
    for f in sorted(glob.glob(os.path.join(base, mode, 'qc', '*.json'))):
        q = json.load(open(f)); q['qc_run'] = mode; rows[q['name']] = q
df = pd.DataFrame(list(rows.values()))
df['mesh_ok'] = df.get('mesh_ok', False).fillna(False).astype(bool)
df['smoke_ok'] = df.get('smoke_ok', False).fillna(False).astype(bool)
keep = [c for c in ['name', 'patient', 'group', 'source', 'vox', 'status', 'fail_code', 'mesh_ok', 'smoke_ok', 'msg',
                    'n_lv_slices', 'slice_mm', 'direction', 'auto_direction', 'n_nodes', 'n_tets', 'kept_frac',
                    'n_components', 'myo_mesh_mL', 'lv_myo_mL', 'lv_cav_mL', 'rv_wall_mL', 'rv_cav_mL',
                    'lv_myo_label_mL', 'lv_cav_label_mL', 'rv_cav_label_mL', 'err_lv_myo', 'err_lv_cav', 'L_scale',
                    'L_wall', 'L_base', 'endo_frac', 'fib_nonfinite', 'nonmanifold_edges', 'boundary_edges',
                    'root_shift_mm', 'n_stim', 'outside_nodes', 'outside_nodes_icp', 'outside_min_bary', 'icp_rot_deg', 'icp_rms_mm',
                    'frame_rms_mm', 'icp_shift_mm', 'smoke_act_max_ms', 'smoke_unactivated', 'timing_s', 'total_s',
                    'h5', 'label_path', 'qc_run'] if c in df.columns]
df = df[keep].sort_values(['patient', 'source', 'vox'])
df.to_csv(os.path.join(base, 'constructions.tsv'), sep='\t', index=False)
# MANIFEST (constructions with a mesh file)
m = df[df['mesh_ok']].copy()
m['construction_id'] = m['name']
m['network'] = m['source']
m['resolution_mm'] = m['vox']
m['mesh_path'] = m['h5'] + ':/pts,/tets,/etag,/fib'
m['lead_field_path'] = m['h5'] + ':/W'
m['root_node_path'] = m['h5'] + ':/roots,/stim'
m['smoke_test'] = np.where(m['smoke_ok'], 'pass', 'fail')
m = m.rename(columns={'patient': 'heart', 'n_nodes': 'node_count'})
cols = ['heart', 'group', 'construction_id', 'network', 'resolution_mm', 'mesh_path', 'node_count', 'lead_field_path',
        'root_node_path', 'smoke_test', 'smoke_act_max_ms', 'err_lv_myo', 'err_lv_cav']
m[cols].to_csv(os.path.join(A.ROOT, 'data', 'processed', 'acdc_mesh', 'MANIFEST.tsv'), sep='\t', index=False)
print(f'MANIFEST: {len(m)} constructions with meshes ({int(m.smoke_ok.sum())} smoke pass)')
# G2: expert labels, 1.5 mm, 50 hearts
g = df[(df.source == 'expert') & (df.vox == 1.5)]
n = len(g); nm = int(g.mesh_ok.sum()); ns = int((g.status == 'pass').sum())
out = [f'G2 (expert, 1.5 mm): {n} hearts attempted; meshing-only pass {nm}/{n}; meshing + smoke pass {ns}/{n}; '
       f'line >= 35 of 50 -> {"PASS" if nm >= 35 and n == 50 else ("FAIL" if n == 50 else "INCOMPLETE")}']
out.append('fail codes: ' + json.dumps(g[g.status != 'pass'].groupby('fail_code').size().to_dict()))
for c in ['n_nodes', 'err_lv_myo', 'err_lv_cav', 'smoke_act_max_ms', 'icp_rot_deg', 'icp_rms_mm', 'frame_rms_mm',
          'nonmanifold_edges', 'endo_frac', 'outside_nodes', 'total_s']:
    if c in g and g[c].notna().any():
        x = g[c].dropna().astype(float)
        out.append(f'  {c}: median {x.median():.4g} [2.5th {x.quantile(0.025):.4g}, 97.5th {x.quantile(0.975):.4g}] '
                   f'min {x.min():.4g} max {x.max():.4g} (n={len(x)})')
out.append('by group (meshing-only pass): ' + json.dumps(g.groupby('group').mesh_ok.agg(['sum', 'count']).to_dict()))
# bonus: networks at 1.5 mm
b = df[(df.source != 'expert') & (df.vox == 1.5)]
if len(b):
    t = b.groupby('source').agg(attempted=('name', 'size'), mesh_ok=('mesh_ok', 'sum'),
                                mesh_and_smoke=('status', lambda s: int((s == 'pass').sum())))
    out.append('networks at 1.5 mm (all hearts attempted):\n' + t.to_string())
    out.append('network fail codes: ' + json.dumps(b[b.status != 'pass'].groupby(['source', 'fail_code']).size()
                                                   .reset_index().values.tolist()))
tri = df[df.patient.isin([101, 102, 104])]
out.append('trio S x M:\n' + tri[['name', 'status', 'fail_code', 'n_nodes', 'err_lv_myo', 'err_lv_cav', 'smoke_act_max_ms',
                                  'total_s']].to_string(index=False))
txt = '\n'.join(out)
open(os.path.join(base, 'summary.txt'), 'w').write(txt + '\n')
print(txt)
