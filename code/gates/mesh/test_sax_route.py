"""Regression test: the pilot path of sax_route.sax_construction / mesh_from_labels reproduces the pilot route
(code/pilot/I11/06_constructions.py) bit for bit on the ka owner's analytic biventricular shell, with and without
observer perturbations. Uses the ka owner's extraction (code/gates/ka/ellipsoid_sax.py). Run: python -I test_sax_route.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ka'))
import ellipsoid_sax as E

PILOT = os.path.join(HERE, '..', '..', 'pilot', 'I11', '06_constructions.py')
MINE = os.path.join(HERE, 'sax_route.py')


def build(route, vox, ovr, seed):
    segs, carpio, _ = E.load_route(route)
    lo_mm, hi_mm = E.bbox('biv')
    lo = np.floor(lo_mm / vox) - 4; hi = np.ceil(hi_mm / vox) + 4
    n = tuple((hi - lo).astype(int) + 1)
    lab = E.label_grid('biv', lo * vox, n, vox)
    ns = E.make_namespace(segs, carpio, vox, lo, ovr)
    new, p = ns['sax_construction'](lab, np.random.default_rng(seed))
    pts, tets, iface, lv_if = ns['mesh_from_labels'](new)
    return new, p, pts, tets, iface, lv_if, ns['lengths'](new)


for vox in (1.5, 1.0):
    for ovr in (dict(spacing=10.0, offset=3.0, inplane=1.25, d_endo=0.0, d_epi=0.0, d_rv=0.0, drop_basal=0, field_sd=0.0),
                dict()):                        # {} = pilot's own random perturbations (seeded)
        a = build(PILOT, vox, ovr, 7); b = build(MINE, vox, ovr, 7)
        assert a[1] == b[1], (a[1], b[1])
        nd = int((a[0] != b[0]).sum())
        assert nd == 0, f'labels differ in {nd} voxels'
        for x, y in zip(a[2:6], b[2:6]):
            assert x.shape == y.shape and np.array_equal(x, y)
        assert a[6] == b[6]
        print(f'vox {vox} ovr {bool(ovr)}: identical labels ({a[0].size} voxels), {len(a[3])} tets, lengths equal')
print('PASS')
