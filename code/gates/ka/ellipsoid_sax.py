"""Known-answer test KA-SAX (I11 gates, owner ka, written 2026-10-28).

An analytic truncated-ellipsoid LV shell (optionally with an RV crescent cavity and an assumed 3 mm RV free
wall) is voxelised at the mesh resolution VOX, sampled on short-axis slices at ACDC-like spacing, and passed
through the short-axis reconstruction route of a reconstruction module (default: the pilot route
code/pilot/I11/06_constructions.py). Myocardial and cavity volumes of the reconstruction are compared with the
analytic (LV) or fine-grid quadrature (RV) values.

How the route is used, without modifying it:
  * The functions sdf, smooth_field, sax_construction, mesh_from_labels and lengths are extracted from the
    module source with `ast` and executed verbatim in a namespace that supplies the module globals they read
    (np, ndi, carpio, VOX, lo, Rf, c). The SHA-256 of the module file and of each function's source text is
    recorded in the output JSON.
  * Observer perturbations are switched off and slice geometry is set by intercepting the parameter `dict(...)`
    call inside sax_construction (the namespace-level `dict` overrides spacing, offset, inplane, d_endo, d_epi,
    d_rv, drop_basal, field_sd; rv_wall stays at the module value). smooth_field is replaced by zeros when
    field_sd == 0 (the original divides by the field SD). The RNG passed in is numpy default_rng(SEED); its draws
    are overwritten by the override, so the reconstruction is deterministic.
Heart frame: axis 0 = long axis, base (u = 0) to apex, Rf = identity, c = 0.

Usage: python -I ellipsoid_sax.py --route <module.py> --out <dir> [--vox 1.5 1.0] [--spacing 10 5 8]
                                  [--inplane 1.25 1.6] [--noff 8] [--shape lv biv] [--truth-h 0.2]
"""
import argparse
import ast
import builtins
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

SEED = 20261028
# geometry (mm). LV: truncated prolate ellipsoids, equator at u = H_B (base plane u = 0, apex towards +u)
H_B = 10.0
A_EN, C_EN = 25.0, 65.0      # endocardium: short (radial) and long semi-axis
A_EP, C_EP = 33.0, 73.0      # epicardium (8 mm wall at equator and apex)
# RV cavity: ellipsoid centre (u = H_B, y = Y_RV, x = 0), semi-axes (C_RV along u, B_RV along y, X_RV along x),
# minus the LV epicardial body, u >= 0. RV wall: within RV_WALL of the RV cavity, outside LV, within RV u-range.
Y_RV, C_RV, B_RV, X_RV = 25.0, 55.0, 30.0, 40.0
RV_WALL = 3.0
FUNCS = ("sdf", "smooth_field", "sax_construction", "mesh_from_labels", "lengths")


def trunc_ellipsoid_vol(a, cc, hb):
    """Volume of x^2/a^2 + y^2/a^2 + z^2/cc^2 <= 1 with z <= hb (z towards base), hb < cc."""
    return np.pi * a * a * ((hb + cc) - (hb ** 3 + cc ** 3) / (3 * cc * cc))


def inside_lv(u, y, x, a, cc):
    z = H_B - u
    return (u >= 0) & ((y * y + x * x) / (a * a) + z * z / (cc * cc) <= 1.0)


def inside_rvcav(u, y, x):
    z = H_B - u
    e = (u >= 0) & (z * z / C_RV ** 2 + (y - Y_RV) ** 2 / B_RV ** 2 + x * x / X_RV ** 2 <= 1.0)
    return e & ~inside_lv(u, y, x, A_EP, C_EP)


def bbox(shape):
    umax = H_B + C_EP + 1.0
    ymin, ymax = -A_EP - 1.0, (Y_RV + B_RV + RV_WALL + 1.0) if shape == "biv" else A_EP + 1.0
    xm = max(A_EP, X_RV + RV_WALL) + 1.0 if shape == "biv" else A_EP + 1.0
    return np.array([0.0, ymin, -xm]), np.array([umax, ymax, xm])


def label_grid(shape, origin, n, h, rv_wall=RV_WALL):
    """Labels at voxel centres origin + i*h: 1 LV myo, 2 RV wall, 3 LV blood, 4 RV blood."""
    u = origin[0] + np.arange(n[0]) * h
    y = origin[1] + np.arange(n[1]) * h
    x = origin[2] + np.arange(n[2]) * h
    U, Y, X = np.meshgrid(u, y, x, indexing="ij")
    lab = np.zeros(n, np.int8)
    epi = inside_lv(U, Y, X, A_EP, C_EP)
    cav = inside_lv(U, Y, X, A_EN, C_EN)
    lab[epi] = 1
    lab[cav] = 3
    if shape == "biv":
        rvc = inside_rvcav(U, Y, X)
        lab[rvc] = 4
        d = ndi.distance_transform_edt(~rvc, sampling=h)
        zr = np.nonzero(rvc.any((1, 2)))[0]
        inrange = np.zeros(n[0], bool)
        inrange[zr.min():zr.max() + 1] = True
        rvw = (d <= rv_wall) & ~rvc & ~epi & inrange[:, None, None]
        lab[rvw] = 2
    return lab


def truth_volumes(shape, h):
    t = dict(lv_myo_mL=(trunc_ellipsoid_vol(A_EP, C_EP, H_B) - trunc_ellipsoid_vol(A_EN, C_EN, H_B)) / 1000,
             lv_cav_mL=trunc_ellipsoid_vol(A_EN, C_EN, H_B) / 1000, truth_method_lv="analytic")
    if shape == "biv":
        lo, hi = bbox(shape)
        n = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
        lab = label_grid(shape, lo + h / 2, n, h)
        t["rv_wall_mL"] = float((lab == 2).sum()) * h ** 3 / 1000
        t["rv_cav_mL"] = float((lab == 4).sum()) * h ** 3 / 1000
        t["lv_myo_quad_mL"] = float((lab == 1).sum()) * h ** 3 / 1000
        t["lv_cav_quad_mL"] = float((lab == 3).sum()) * h ** 3 / 1000
        t["truth_method_rv"] = f"midpoint quadrature on a {h} mm grid"
    else:
        t["rv_wall_mL"] = 0.0
        t["rv_cav_mL"] = 0.0
    t["myo_mL"] = t["lv_myo_mL"] + t["rv_wall_mL"]
    return t


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def load_route(path):
    src = open(path, "rb").read()
    tree = ast.parse(src.decode())
    text = src.decode()
    segs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            segs[node.name] = ast.get_source_segment(text, node)
    missing = [f for f in FUNCS if f not in segs]
    if missing:
        raise SystemExit(f"route {path} lacks functions {missing}")
    # carpio from the route's own directory (as the pilot imports it)
    rdir = os.path.dirname(os.path.abspath(path))
    spec = importlib.util.spec_from_file_location("carpio_route", os.path.join(rdir, "carpio.py"))
    carpio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(carpio)
    return segs, carpio, dict(route_file=os.path.abspath(path), route_sha256=sha256(src),
                              carpio_sha256=sha256(open(os.path.join(rdir, "carpio.py"), "rb").read()),
                              function_sha256={k: sha256(v.encode()) for k, v in segs.items()})


def make_namespace(segs, carpio, vox, lo, ovr):
    ns = dict(np=np, ndi=ndi, carpio=carpio, VOX=vox, lo=lo, Rf=np.eye(3), c=np.zeros(3), __builtins__=builtins)
    bdict = builtins.dict

    def dict_(*a, **k):
        if "spacing" in k and "rv_wall" in k:
            k.update({kk: v for kk, v in ovr.items() if kk in k})
        return bdict(*a, **k)

    ns["dict"] = dict_
    for name in FUNCS:
        exec(compile(segs[name], f"<route:{name}>", "exec"), ns)
    orig = ns["smooth_field"]

    def smooth_zero(shape, sd, corr, rng):
        if sd == 0:
            return np.zeros(shape, np.float32)
        return orig(shape, sd, corr, rng)

    ns["smooth_field"] = smooth_zero
    return ns


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def run_one(segs, carpio, shape, vox, spacing, inplane, offset, truth):
    lo_mm, hi_mm = bbox(shape)
    lo = np.floor(lo_mm / vox) - 4
    hi = np.ceil(hi_mm / vox) + 4
    n = tuple((hi - lo).astype(int) + 1)
    lab = label_grid(shape, lo * vox, n, vox)       # voxel centres at (i + lo) * VOX, as in the pilot grid
    ovr = dict(spacing=float(spacing), offset=float(offset), inplane=float(inplane), d_endo=0.0, d_epi=0.0,
               d_rv=0.0, drop_basal=0, field_sd=0.0)
    ns = make_namespace(segs, carpio, vox, lo, ovr)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    new, p = ns["sax_construction"](lab, rng)
    pts, tets, iface, lv_if = ns["mesh_from_labels"](new)
    vol_mesh = float(np.abs(carpio.tet_volumes(pts, tets)).sum()) / 1000
    L = ns["lengths"](new)
    Lref = ns["lengths"](lab)
    myo_new = (new == 1) | (new == 2)
    myo_ref = (lab == 1) | (lab == 2)
    v3 = vox ** 3 / 1000
    r = dict(shape=shape, vox_mm=vox, spacing_mm=spacing, inplane_req_mm=inplane,
             inplane_eff_mm=max(1.0, inplane / vox) * vox, offset_mm=offset, n_slices=None,
             myo_mesh_mL=vol_mesh, myo_lab_mL=L["myo_mL"], lv_myo_mL=float((new == 1).sum()) * v3,
             rv_wall_mL=float((new == 2).sum()) * v3, lv_cav_mL=float((new == 3).sum()) * v3,
             rv_cav_mL=float((new == 4).sum()) * v3,
             ref_myo_mL=Lref["myo_mL"], ref_lv_myo_mL=float((lab == 1).sum()) * v3,
             ref_lv_cav_mL=float((lab == 3).sum()) * v3, ref_rv_cav_mL=float((lab == 4).sum()) * v3,
             ref_rv_wall_mL=float((lab == 2).sum()) * v3,
             dice_myo_vs_voxtruth=dice(myo_new, myo_ref), n_tets=int(len(tets)), n_nodes=int(len(pts)),
             L_base_mm=L["L_base"], ref_L_base_mm=Lref["L_base"], t_s=time.time() - t0,
             p_used={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in p.items()})
    # slices actually used (recomputed exactly as in the route, for reporting)
    n0 = lab.shape[0]
    zs = np.arange(offset / vox, n0 - 1, spacing / vox)
    r["n_slices"] = int(sum((lab[int(round(z))] == 1).any() for z in zs))
    for k, tk in (("myo_mesh_mL", "myo_mL"), ("lv_myo_mL", "lv_myo_mL"), ("lv_cav_mL", "lv_cav_mL"),
                  ("rv_cav_mL", "rv_cav_mL"), ("rv_wall_mL", "rv_wall_mL"), ("ref_myo_mL", "myo_mL"),
                  ("ref_lv_cav_mL", "lv_cav_mL"), ("ref_rv_cav_mL", "rv_cav_mL")):
        if truth[tk] > 0:
            r["err_" + k] = (r[k] - truth[tk]) / truth[tk]
    return r


def _worker(args):
    route, j, truth = args
    segs, carpio, _ = load_route(route)
    return run_one(segs, carpio, *j, truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                    "pilot", "I11", "06_constructions.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--vox", type=float, nargs="+", default=[1.5, 1.0])
    ap.add_argument("--spacing", type=float, nargs="+", default=[10.0, 5.0, 8.0])
    ap.add_argument("--inplane", type=float, nargs="+", default=[1.25, 1.6])
    ap.add_argument("--noff", type=int, default=8)
    ap.add_argument("--shape", nargs="+", default=["biv"])  # "lv" alone crashes the pilot route (no RV cavity)
    ap.add_argument("--truth-h", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    segs, carpio, meta = load_route(a.route)
    meta.update(seed=SEED, numpy=np.__version__, python=sys.version.split()[0],
                geometry=dict(H_B=H_B, A_EN=A_EN, C_EN=C_EN, A_EP=A_EP, C_EP=C_EP, Y_RV=Y_RV, C_RV=C_RV,
                              B_RV=B_RV, X_RV=X_RV, RV_WALL=RV_WALL))
    truths = {s: truth_volumes(s, a.truth_h) for s in a.shape}
    meta["truth"] = truths
    print(json.dumps(meta, indent=1), flush=True)
    jobs = [(s, v, sp, ip, sp * k / a.noff) for s in a.shape for v in a.vox for sp in a.spacing
            for ip in a.inplane for k in range(a.noff)]
    rows = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            it = ex.map(_worker, [(a.route, j, truths[j[0]]) for j in jobs])
            rows = list(it)
    else:
        rows = [run_one(segs, carpio, *j, truths[j[0]]) for j in jobs]
    for r in rows:
        print(f"{r['shape']} vox {r['vox_mm']} sp {r['spacing_mm']} ip {r['inplane_req_mm']} off {r['offset_mm']:.2f}"
              f" n_sl {r['n_slices']}: myo err {r['err_myo_mesh_mL']*100:+.2f}% lvcav err {r['err_lv_cav_mL']*100:+.2f}%"
              f" ref myo err {r['err_ref_myo_mL']*100:+.2f}% ({r['t_s']:.1f}s)", flush=True)
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(a.out, "ellipsoid_sax.json"), "w"), indent=1,
              default=float)
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "p_used"} for r in rows])
    df.to_csv(os.path.join(a.out, "ellipsoid_sax.csv"), index=False)
    cols = [c for c in df.columns if c.startswith("err_")]
    summ = df.groupby(["shape", "vox_mm", "spacing_mm", "inplane_req_mm"])[cols].agg(["median", "min", "max"])
    summ.to_csv(os.path.join(a.out, "ellipsoid_sax_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()
