"""Unit test of the Amendment 2 B1 smooth-surface fast layer (owner layer-sdf, 2026-10-28).
Run: python -I -B test_endo_layer_sdf.py <out.json> [workers]

Same cases, geometry, seeds and acceptance lines as test_endo_layer.py (A2 test, owner mesh2), which is imported and not
modified: plane, sphere r 25 mm (LV and RV label), triaxial ellipsoid (20, 25, 40) mm in 8 poses
(numpy default_rng(20261028)), at 1.5 and 1.0 mm voxels, threshold 1.5 mm. The layer is endo_layer.sdf_layer; the A2
layer (endo_layer.a2_layer) is recomputed beside it on the identical mesh for a paired comparison.
Acceptance (B1: unchanged from A2): ellipsoid layer volume within +/-15% of the analytic shell in every pose at both
resolutions, and the median 1.0/1.5 mm volume ratio within +/-15%. The A2 file also declared "plane exact to 1e-9"; for
the smooth surface the plane case is a finite box cavity whose edges and corners are rounded by the smoothing, so it is
reported (error and thickness over the flat face) but is not part of the B1 acceptance, which names the ellipsoid lines
only. Reported against the line whatever the result.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sax_route        # noqa: E402
import carpio           # noqa: E402
import endo_layer       # noqa: E402
import test_endo_layer as T0      # noqa: E402  (A2 test: steiner, grid_labels, constants)

D = T0.D
SEED = T0.SEED


def layer_volumes(L, vox):
    lo = np.zeros(3); R = np.eye(3); c0 = np.zeros(3)
    pts, tets, iface, lv_if = sax_route.mesh_from_labels(L, vox, lo, R, c0, qc={})
    vol = carpio.tet_volumes(pts, tets)
    t = time.time()
    sdf, dist, info = endo_layer.sdf_layer(L, pts, tets, vox, lo, R, c0)
    ts = time.time() - t
    a2, _, _ = endo_layer.a2_layer(L, pts, tets, vox, lo, R, c0)
    cen = endo_layer.tet_centroids_index(pts, tets, vox, lo, R, c0)
    return dict(v_sdf=float(vol[sdf].sum()), v_a2=float(vol[a2].sum()), v_myo=float(vol.sum()), n_tets=int(len(tets)),
                frac_sdf=float(sdf.mean()), frac_a2=float(a2.mean()), vol_frac_sdf=float(vol[sdf].sum() / vol.sum()),
                sdf_s=ts, n_tri=int(info['n_tri_kept']), n_tri_all=int(info['n_tri_all'])), (vol, sdf, cen)


def case(args):
    kind, vox, extra = args
    if kind == 'plane':
        n = 24
        L = np.zeros((n, n, n), np.int8); L[2:-2, 2:-2, 2:-2] = 1; L[2:-2, 2:-2, 2:8] = 3
        r, (vol, sdf, cen) = layer_volumes(L, vox)
        face_area = (n - 4) ** 2 * vox ** 2
        rows = 1.0 if vox == 1.5 else 5.0 / 3.0
        expected = face_area * rows * vox
        # flat-face check: tets whose centroid lies over the central part of the cavity face (>= 4 voxels from its
        # edge in x and y), compared with the same footprint of the analytic slab
        # voxel index of each tet (centroids lie within +/-0.25 voxel of their voxel centre); fixed 2026-10-28 after
        # job 5395849, which selected by centroid and so counted the border voxels of the footprint only in part
        vi = np.floor(cen[:, :2] + 0.5)
        core = (vi[:, 0] >= 6) & (vi[:, 0] <= n - 7) & (vi[:, 1] >= 6) & (vi[:, 1] <= n - 7)
        core_area = (n - 12) ** 2 * vox ** 2
        r.update(case='plane', vox=vox, analytic=expected, err_sdf=r['v_sdf'] / expected - 1,
                 err_a2=r['v_a2'] / expected - 1, thickness_sdf_mm=r['v_sdf'] / face_area,
                 core_thickness_sdf_mm=float(vol[sdf & core].sum() / core_area), core_expected_mm=rows * vox)
        return r
    if kind == 'sphere':
        lab = extra
        v_sph = 4 / 3 * np.pi * ((25 + D) ** 3 - 25 ** 3)
        L = T0.grid_labels(vox, 40, lambda P: (P ** 2).sum(-1) < 25 ** 2, lambda P: (P ** 2).sum(-1) < 34 ** 2, lab)
        r, _ = layer_volumes(L, vox)
        r.update(case=f'sphere_label{lab}', vox=vox, analytic=v_sph, err_sdf=r['v_sdf'] / v_sph - 1,
                 err_a2=r['v_a2'] / v_sph - 1)
        return r
    ip, Rm, off, ax_c, ax_e, v_ell = extra
    sh = off * vox

    def ins(P, ax):
        Q = (P - sh) @ Rm
        return ((Q / ax) ** 2).sum(-1) < 1
    L = T0.grid_labels(vox, 52, lambda P: ins(P, ax_c), lambda P: ins(P, ax_e))
    r, _ = layer_volumes(L, vox)
    r.update(case='ellipsoid', pose=ip, vox=vox, analytic=v_ell, err_sdf=r['v_sdf'] / v_ell - 1,
             err_a2=r['v_a2'] / v_ell - 1)
    return r


def main(out, workers):
    res = dict(threshold_mm=D, seed=SEED, layer='endo_layer.sdf_layer (Amendment 2 B1)', up_mm=endo_layer.SDF_UP_MM,
               sigma='one native voxel (vox mm)', cases=[])
    ax_c = np.array([20.0, 25.0, 40.0]); ax_e = ax_c + 9.0
    A, M = T0.steiner(*ax_c)
    v_ell = A * D + M * D ** 2 + 4 / 3 * np.pi * D ** 3
    res['ellipsoid'] = dict(cavity_semi_axes_mm=ax_c.tolist(), epi_semi_axes_mm=ax_e.tolist(), analytic_layer_mL=v_ell / 1000)
    rng = np.random.default_rng(SEED)        # identical pose draws to test_endo_layer.py
    poses = [(Rotation.random(random_state=int(rng.integers(2**31))).as_matrix(), rng.uniform(0, 1, 3)) for _ in range(8)]
    jobs = []
    for vox in (1.5, 1.0):
        for ip, (Rm, off) in enumerate(poses):
            jobs.append(('ellipsoid', vox, (ip, Rm, off, ax_c, ax_e, v_ell)))
        jobs += [('sphere', vox, 3), ('sphere', vox, 4), ('plane', vox, None)]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(case, jobs):
            res['cases'].append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    summ = {}
    for vox in (1.5, 1.0):
        e = [c for c in res['cases'] if c['case'] == 'ellipsoid' and c['vox'] == vox]
        s = np.array([c['err_sdf'] for c in e]); a = np.array([c['err_a2'] for c in e])
        summ[str(vox)] = dict(sdf_err_median=float(np.median(s)), sdf_err_min=float(s.min()), sdf_err_max=float(s.max()),
                              a2_err_median=float(np.median(a)), a2_err_min=float(a.min()), a2_err_max=float(a.max()),
                              sdf_vol_frac_median=float(np.median([c['vol_frac_sdf'] for c in e])),
                              v_sdf_median_mL=float(np.median([c['v_sdf'] for c in e])) / 1000)
    res['summary'] = summ
    ell_ok = all(abs(c['err_sdf']) <= 0.15 for c in res['cases'] if c['case'] == 'ellipsoid')
    ratio = summ['1.0']['v_sdf_median_mL'] / summ['1.5']['v_sdf_median_mL']
    res['acceptance'] = dict(ellipsoid_within_15pct=ell_ok, median_ratio_1p0_over_1p5=ratio,
                             ratio_within_15pct=abs(ratio - 1) <= 0.15, PASS=bool(ell_ok and abs(ratio - 1) <= 0.15),
                             note='plane case reported, not an acceptance line for B1')
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(summary=summ, acceptance=res['acceptance']), indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'plane':        # plane cases only (login-node rerun after the fix)
        out = [case(('plane', v, None)) for v in (1.5, 1.0)]
        json.dump(out, open(sys.argv[1], 'w'), indent=1); print(out)
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
