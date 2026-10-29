"""Unit test of the A2 fast endocardial layer (owner mesh2, 2026-10-28). Run: python -I -B test_endo_layer.py <out.json>

Cases (voxel sizes 1.5 and 1.0 mm, mesh by sax_route.mesh_from_labels, layer by endo_layer.a2_layer, threshold 1.5 mm):
  1. plane: an axis-aligned cavity half-space. Known answer from the centroid rule: the 6 tets of a voxel have
     centroids at -0.25, 0, +0.25 voxel (two each) along the normal, so the layer is exactly 1 voxel row at 1.5 mm
     (1.5 mm thick) and 1 + 2/3 rows at 1.0 mm (1.667 mm thick; 1.25 and 1.5 mm centroids are within 1.5 mm).
  2. sphere: cavity radius 25 mm, wall 9 mm, cavity label 3 (LV) and label 4 (RV). Analytic shell volume
     4/3 pi ((r + d)^3 - r^3), d = 1.5 mm.
  3. ellipsoid: triaxial cavity (20, 25, 40) mm, epicardium (29, 34, 49) mm, 8 random poses (rotation and sub-voxel
     centre offset; numpy default_rng(20261028)). Analytic layer volume by the Steiner formula for a convex body,
     V(d) = A d + M d^2 + 4/3 pi d^3 (A area, M integral of mean curvature), A and M by quadrature (checked on the
     sphere to 1e-10).
Reported: layer volume / analytic - 1 for the A2 layer and for the gates one-voxel layer. Acceptance, declared here
before the first run: A2 |error| <= 15% in every ellipsoid pose at both resolutions, and the median A2 volume at
1.0 mm within 15% of that at 1.5 mm; plane case exact to 1e-9 relative."""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sax_route      # noqa: E402
import carpio         # noqa: E402
import endo_layer     # noqa: E402

D = 1.5
SEED = 20261028


def steiner(a, b, c, nth=400, nph=800):
    x, w = np.polynomial.legendre.leggauss(nth)
    th = (x + 1) * np.pi / 2; wt = w * np.pi / 2
    ph = np.arange(nph) * 2 * np.pi / nph; wp = 2 * np.pi / nph
    T, P = np.meshgrid(th, ph, indexing='ij')
    X = a * np.sin(T) * np.cos(P); Y = b * np.sin(T) * np.sin(P); Z = c * np.cos(T)
    dA = np.sin(T) * np.sqrt(b**2 * c**2 * np.sin(T)**2 * np.cos(P)**2 + a**2 * c**2 * np.sin(T)**2 * np.sin(P)**2
                             + a**2 * b**2 * np.cos(T)**2)
    h = np.sqrt(X**2 / a**4 + Y**2 / b**4 + Z**2 / c**4)
    H = (a**2 + b**2 + c**2 - (X**2 + Y**2 + Z**2)) / (2 * a**2 * b**2 * c**2 * h**3)
    W = wt[:, None] * wp
    A = float((dA * W).sum()); M = float((H * dA * W).sum())
    return A, M


def layer_volumes(L, vox):
    lo = np.zeros(3); R = np.eye(3); c0 = np.zeros(3)
    qc = {}
    pts, tets, iface, lv_if = sax_route.mesh_from_labels(L, vox, lo, R, c0, qc=qc)
    vol = carpio.tet_volumes(pts, tets)
    a2, dist, nf = endo_layer.a2_layer(L, pts, tets, vox, lo, R, c0)
    one = endo_layer.onevoxel_layer(tets, iface)
    return dict(v_a2=float(vol[a2].sum()), v_one=float(vol[one].sum()), v_myo=float(vol.sum()), n_tets=int(len(tets)),
                frac_a2=float(a2.mean()), frac_one=float(one.mean()), n_faces=int(nf))


def grid_labels(vox, half_mm, inside_cav, inside_epi, cav_label=3):
    n = int(np.ceil(2 * half_mm / vox)) + 1
    g = (np.arange(n) - (n - 1) / 2) * vox
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([X, Y, Z], -1)
    L = np.zeros((n, n, n), np.int8)
    L[inside_epi(P)] = 1
    L[inside_cav(P)] = cav_label
    return L


def main(out):
    res = dict(threshold_mm=D, seed=SEED, cases=[])
    # 1. plane
    for vox in (1.5, 1.0):
        n = 24
        L = np.zeros((n, n, n), np.int8); L[2:-2, 2:-2, 2:-2] = 1; L[2:-2, 2:-2, 2:8] = 3
        r = layer_volumes(L, vox)
        face_area = (n - 4) ** 2 * vox ** 2          # only the cavity face z = 8 - 0.5 in index units bounds myocardium
        rows = 1.0 if vox == 1.5 else 5.0 / 3.0
        expected = face_area * rows * vox
        # the cavity also touches nothing else: its other faces are background or grid edge (L = 0 there)
        r.update(case='plane', vox=vox, analytic=expected, err_a2=r['v_a2'] / expected - 1,
                 thickness_a2_mm=r['v_a2'] / face_area)
        res['cases'].append(r)
    # 2. sphere, LV and RV label
    A_s, M_s = steiner(25.0, 25.0, 25.0)
    v_sph = 4 / 3 * np.pi * ((25 + D) ** 3 - 25 ** 3)
    res['steiner_check_sphere_rel'] = abs(A_s * D + M_s * D ** 2 + 4 / 3 * np.pi * D ** 3 - v_sph) / v_sph
    for vox in (1.5, 1.0):
        for lab in (3, 4):
            L = grid_labels(vox, 40, lambda P: (P ** 2).sum(-1) < 25 ** 2, lambda P: (P ** 2).sum(-1) < 34 ** 2, lab)
            r = layer_volumes(L, vox)
            r.update(case=f'sphere_label{lab}', vox=vox, analytic=v_sph, err_a2=r['v_a2'] / v_sph - 1,
                     err_one=r['v_one'] / v_sph - 1)
            res['cases'].append(r)
    # 3. ellipsoid, random poses
    ax_c = np.array([20.0, 25.0, 40.0]); ax_e = ax_c + 9.0
    A, M = steiner(*ax_c)
    v_ell = A * D + M * D ** 2 + 4 / 3 * np.pi * D ** 3
    res['ellipsoid'] = dict(cavity_semi_axes_mm=ax_c.tolist(), epi_semi_axes_mm=ax_e.tolist(), area_mm2=A,
                            mean_curv_integral_mm=M, analytic_layer_mL=v_ell / 1000)
    rng = np.random.default_rng(SEED)
    poses = [(Rotation.random(random_state=int(rng.integers(2**31))).as_matrix(), rng.uniform(0, 1, 3)) for _ in range(8)]
    for vox in (1.5, 1.0):
        for ip, (Rm, off) in enumerate(poses):
            sh = off * vox

            def ins(P, ax):
                Q = (P - sh) @ Rm          # body coordinates
                return ((Q / ax) ** 2).sum(-1) < 1
            L = grid_labels(vox, 52, lambda P: ins(P, ax_c), lambda P: ins(P, ax_e))
            r = layer_volumes(L, vox)
            r.update(case='ellipsoid', pose=ip, vox=vox, analytic=v_ell, err_a2=r['v_a2'] / v_ell - 1,
                     err_one=r['v_one'] / v_ell - 1)
            res['cases'].append(r)
            print(r, flush=True)
    # summary and acceptance
    summ = {}
    for vox in (1.5, 1.0):
        e = [c for c in res['cases'] if c['case'] == 'ellipsoid' and c['vox'] == vox]
        a2 = np.array([c['err_a2'] for c in e]); one = np.array([c['err_one'] for c in e])
        summ[str(vox)] = dict(a2_err_median=float(np.median(a2)), a2_err_min=float(a2.min()), a2_err_max=float(a2.max()),
                              one_err_median=float(np.median(one)), one_err_min=float(one.min()), one_err_max=float(one.max()),
                              a2_frac_median=float(np.median([c['frac_a2'] for c in e])),
                              one_frac_median=float(np.median([c['frac_one'] for c in e])),
                              v_a2_median_mL=float(np.median([c['v_a2'] for c in e])) / 1000)
    res['summary'] = summ
    plane_ok = all(abs(c['err_a2']) < 1e-9 for c in res['cases'] if c['case'] == 'plane')
    ell_ok = all(abs(c['err_a2']) <= 0.15 for c in res['cases'] if c['case'] == 'ellipsoid')
    ratio = summ['1.0']['v_a2_median_mL'] / summ['1.5']['v_a2_median_mL']
    res['acceptance'] = dict(plane_exact=plane_ok, ellipsoid_within_15pct=ell_ok, median_ratio_1p0_over_1p5=ratio,
                             ratio_within_15pct=abs(ratio - 1) <= 0.15,
                             PASS=bool(plane_ok and ell_ok and abs(ratio - 1) <= 0.15))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(dict(summary=summ, acceptance=res['acceptance'],
                          plane=[(c['vox'], c['err_a2'], c['thickness_a2_mm']) for c in res['cases'] if c['case'] == 'plane'],
                          sphere=[(c['case'], c['vox'], round(c['err_a2'], 4), round(c['err_one'], 4))
                                  for c in res['cases'] if c['case'].startswith('sphere')],
                          steiner_check=res['steiner_check_sphere_rel']), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
