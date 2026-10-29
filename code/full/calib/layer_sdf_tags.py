"""Amendment 2 B1 layer tags for the primary constructions (owner layer-sdf, 2026-10-28).
Usage: python -I -B layer_sdf_tags.py <out.h5> <share.csv> [--hearts 102 ...] [--workers 64]

For every MANIFEST.tsv row with include_primary (RV wall 3 mm, 1.5 and 1.0 mm), reads the construction file
(data/processed/acdc_mesh_v2/cons/<id>.h5; read only, its `etag` is never touched) and computes
endo_layer.sdf_layer on the stored label grid `lab` with the stored grid map (grid_lo, grid_R, grid_c, vox).
Writes one HDF5 file: /cons/<id>/etag_sdf (int8, 1 myocardium, 2 B1 fast layer; same element order as the
construction's tets) and /cons/<id>/endo_dist_sdf_mm (float32, inf beyond 3 mm), with attributes; and one CSV with the
layer share of myocardial volume (B1 and A2) per construction. No random numbers are drawn.
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'full', 'mesh'))
import endo_layer          # noqa: E402
import carpio              # noqa: E402

MESH_DIR = os.path.join(ROOT, 'data', 'processed', 'acdc_mesh_v2')


def one(row):
    name = row['construction_id']
    t0 = time.time()
    with h5py.File(os.path.join(MESH_DIR, 'cons', name + '.h5'), 'r') as f:
        L = f['lab'][:]; pts = f['pts'][:]; tets = f['tets'][:].astype(np.int64); etag = f['etag'][:]
        lo = f['grid_lo'][:]; R = f['grid_R'][:]; c = f['grid_c'][:]; vox = float(f.attrs['vox'])
    vol = carpio.tet_volumes(pts, tets)
    lay, dist, info = endo_layer.sdf_layer(L, pts, tets, vox, lo, R, c)
    a2 = etag == 2
    share = dict(construction=name, patient=int(row['heart']), source=row['source'], resolution_mm=vox,
                 rv_wall_mm=float(row['rv_wall_mm']), is_reference=bool(row['is_reference']), n_tets=int(len(tets)),
                 myo_mL=float(vol.sum() / 1000), vol_share_sdf=float(vol[lay].sum() / vol.sum()),
                 vol_share_a2=float(vol[a2].sum() / vol.sum()), tet_share_sdf=float(lay.mean()),
                 tet_share_a2=float(a2.mean()), both_share=float(vol[lay & a2].sum() / vol.sum()),
                 sdf_only_share=float(vol[lay & ~a2].sum() / vol.sum()),
                 a2_only_share=float(vol[~lay & a2].sum() / vol.sum()),
                 n_tri=int(info['n_tri_kept']), n_tri_dropped_basal=int(info['n_tri_all'] - info['n_tri_kept']),
                 n_ball_queries=int(info.get('n_ball_queries', 0)), wall_s=time.time() - t0)
    return name, np.where(lay, 2, 1).astype(np.int8), dist.astype(np.float32), share, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_h5'); ap.add_argument('share_csv')
    ap.add_argument('--hearts', type=int, nargs='*'); ap.add_argument('--workers', type=int, default=64)
    a = ap.parse_args()
    M = pd.read_csv(os.path.join(MESH_DIR, 'MANIFEST.tsv'), sep='\t')
    M = M[M.include_primary.astype(bool)]
    if a.hearts:
        M = M[M.heart.isin(a.hearts)]
    rows = [r for _, r in M.iterrows()]
    rows.sort(key=lambda r: r['resolution_mm'])            # 1.0 mm (larger) first
    code_sha = hashlib.sha256(open(endo_layer.__file__, 'rb').read()).hexdigest()
    shares = []
    tmp = a.out_h5 + '.part'
    with h5py.File(tmp, 'w') as H, ProcessPoolExecutor(a.workers) as ex:
        H.attrs.update(definition='Amendment 2 B1: myocardial elements whose centroid lies within 1.5 mm of the smooth '
                                  'LV or RV endocardial surface (0.5 level set of the blood-pool label upsampled to '
                                  '0.25 mm, Gaussian sigma = one native voxel, marching cubes); cavity side only',
                       function='code/full/mesh/endo_layer.py:sdf_layer', endo_layer_sha256=code_sha,
                       tags='1 myocardium, 2 B1 fast layer', created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                       manifest='data/processed/acdc_mesh_v2/MANIFEST.tsv (include_primary)')
        for k, (name, tag, dist, share, info) in enumerate(ex.map(one, rows)):
            g = H.create_group(f'cons/{name}')
            g.create_dataset('etag_sdf', data=tag, compression='gzip', compression_opts=4)
            g.create_dataset('endo_dist_sdf_mm', data=dist, compression='gzip', compression_opts=4)
            g.attrs.update({k_: v for k_, v in info.items()})
            g.attrs.update(vol_share_sdf=share['vol_share_sdf'], vol_share_a2=share['vol_share_a2'])
            shares.append(share)
            if k % 25 == 0:
                print(k, name, round(share['vol_share_sdf'], 3), round(share['vol_share_a2'], 3),
                      round(share['wall_s'], 1), flush=True)
    os.replace(tmp, a.out_h5)
    S = pd.DataFrame(shares).sort_values(['patient', 'resolution_mm', 'source'])
    S.to_csv(a.share_csv, index=False)
    for v in (1.5, 1.0):
        s = S[S.resolution_mm == v]
        print(v, 'n', len(s), 'B1 share median', s.vol_share_sdf.median().round(3),
              np.percentile(s.vol_share_sdf, [2.5, 97.5]).round(3), 'A2', s.vol_share_a2.median().round(3))


if __name__ == '__main__':
    main()
