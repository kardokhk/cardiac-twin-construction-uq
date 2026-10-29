"""Boundary-set conflict audit for KA-FIB (S6), I11 gates, owner ka, 2026-10-28.

Regenerates the mesh and boundary classification of fibres_compare.py (same build/boundary_faces/classify, no ldrb
solve) and counts nodes that belong to both an endocardial set (LV 2, RV 3) and the epicardial set (4). Then, from
the stored per-element fields in fibres_<mesh>.h5 (same element order), reports the |helix difference| and fibre
angle difference in elements touching / not touching conflict nodes, and the activation-time difference.
Deterministic (no random draws). Usage: python -I conflict_nodes.py <fibres_dir> <out_json>
"""
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fibres_compare as fc  # noqa: E402


def q(x, p):
    return float(np.quantile(x, p)) if x.size else float("nan")


def summ(x):
    return dict(n=int(x.size), median=q(x, 0.5), p95=q(x, 0.95), p99=q(x, 0.99), max=float(x.max()) if x.size else float("nan"))


def main():
    fdir, out = sys.argv[1:3]
    res = {}
    for mesh in ("ellipsoid", "heart09"):
        lab, lo, Rf, cvec, pts, tets, etag, stim, info, meta = fc.build(mesh)
        F, fcen, nrm = fc.boundary_faces(tets, pts)
        cls = fc.classify(F, fcen, nrm, lab, lo, Rf, cvec, fc.VOX)
        nodes = {k: np.unique(F[cls == v]) for k, v in dict(base=1, lv=2, rv=3, epi=4).items()}
        lv_c = np.intersect1d(nodes["lv"], nodes["epi"]); rv_c = np.intersect1d(nodes["rv"], nodes["epi"])
        H = h5py.File(os.path.join(fdir, f"fibres_{mesh}.h5"), "r")
        dh = H["dhelix"][:].astype(float); da = H["dangle"][:].astype(float)
        dAT = H["at_i09"][:].astype(float) - H["at_ldrb"][:].astype(float)
        assert dh.shape[0] == len(tets)
        r = dict(n_nodes=int(len(pts)), n_tets=int(len(tets)),
                 nodes_per_set={k: int(v.size) for k, v in nodes.items()},
                 lv_endo_also_epi=int(lv_c.size), rv_endo_also_epi=int(rv_c.size),
                 lv_endo_also_rv_endo=int(np.intersect1d(nodes["lv"], nodes["rv"]).size))
        for name, cset in (("lv_or_rv_conflict", np.union1d(lv_c, rv_c)), ("lv_conflict_only", lv_c)):
            m = np.zeros(len(pts), bool); m[cset] = True
            touch = m[tets].any(1)
            r[name] = dict(n_conflict_nodes=int(cset.size), n_tets_touching=int(touch.sum()),
                           n_tets_away=int((~touch).sum()),
                           dhelix_touching=summ(dh[touch]), dhelix_away=summ(dh[~touch]),
                           dangle_touching=summ(da[touch]), dangle_away=summ(da[~touch]))
        ad = np.abs(dAT)
        r["dAT_abs_ms"] = dict(summ(ad), n_over_0p01=int((ad > 0.01).sum()), n_over_0p1=int((ad > 0.1).sum()),
                               n_over_1=int((ad > 1).sum()), argmax_node=int(ad.argmax()))
        r["dangle_deg_all"] = dict(summ(da), n_over_1=int((da > 1).sum()), n_over_10=int((da > 10).sum()),
                                   n_over_45=int((da > 45).sum()))
        r["dhelix_deg_all"] = dict(summ(dh), n_over_1=int((dh > 1).sum()), n_over_10=int((dh > 10).sum()))
        r["info"] = info
        res[mesh] = r
        print(mesh, json.dumps({k: v for k, v in r.items() if k != "info"}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
