"""Assign the 50 hearts to K node shards by estimated eikonal cost (owner calibC, 2026-10-28).
Cost per construction = node_count x (gates median packed wall per node: 157.6 us at 1.5 mm, 207.2 us at 1.0 mm,
results/2026-10-28_G3G4_ministudy/constructions.csv) x 92 runs. Greedy longest-processing-time assignment; deterministic.
Usage: python -I -B plan_shards.py K > shards.json
"""
import json, os, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
K = int(sys.argv[1])
M = pd.read_csv(os.path.join(ROOT, 'data/processed/acdc_mesh_v2/MANIFEST.tsv'), sep='\t')
M = M[M.include_primary.astype(bool) | M.include_sensitivity.astype(bool)]
us = {1.5: 157.6e-6, 1.0: 207.2e-6}
M['cost_s'] = M.node_count * M.resolution_mm.map(us) * 92
h = M.groupby('heart').agg(cost_s=('cost_s', 'sum'), n=('heart', 'size')).sort_values('cost_s', ascending=False)
bins = [[0.0, []] for _ in range(K)]
for p, r in h.iterrows():
    b = min(bins, key=lambda x: x[0]); b[0] += r.cost_s; b[1].append(int(p))
out = dict(K=K, n_constructions=int(len(M)), shards=[sorted(b[1]) for b in bins],
           est_core_h=[round(b[0] / 3600, 1) for b in bins], total_est_core_h=round(h.cost_s.sum() / 3600, 1))
print(json.dumps(out, indent=1))
