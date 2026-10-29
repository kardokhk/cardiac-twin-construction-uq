"""Known-answer test KA-SHELL (S7), I11 gates, owner ka, 2026-10-28.

Repeats the I09 pilot thick-shell eikonal test (code/pilot/I09/00_known_answer.py): spherical shell a = 10,
b = 15 mm, isotropic 600 mm/s, point stimulus at the outer pole, exact time = geodesic around the inner ball / v.
Acceptance line (red team, carried into STUDY_PLAN section 5): max |error| / max analytic T < 2%.

Mesh families
  delaunay : the I09 mesh generator (layered Fibonacci points + Delaunay), source text extracted verbatim from
             00_known_answer.py with `ast` (fib_sphere, geod) and the mesh loop copied line for line.
  voxel    : the study's own mesh type: Kuhn 6-tet split of a cubic grid (as in mesh_from_labels of the I11
             route), voxels whose centre lies in the shell; exact time evaluated at the mesh nodes, i.e. on the
             staircase geometry, so part of the error is geometric (the study meshes have the same staircase).
  delaunay_snap : Delaunay mesh with inner-surface chords removed by projecting every node with r < a to r = a
             (diagnostic for the polyhedral-inner-boundary short-cut).
Solver settings: two variants: (i) I09 settings (carpio.write_par: dream.fim.tol 1e-4, max_iter 5000);
             (ii) the study settings of code/pilot/I11/carpio.run_eikonal (max_iter 500, default tol).
Correction 2026-10-28 (owner ka): the I09 `geod` marks a node "blocked" when the closest point of the straight
             segment lies at distance < a. Inner-surface Fibonacci nodes sit at r = a(1 - ~1e-16) and the segment's
             closest point to the origin is then the node itself, so lit-side inner nodes were wrongly sent to the wrap
             formula, which there is shorter than the straight chord (analytic time too small). `geod_fixed` below
             uses closest < a(1 - 1e-9) and, as a geometric bound, never returns less than the chord. The I09 `geod`
             is still loaded and evaluated so that both errors are written (columns *_i09geod). Note: "snap" moves
             nodes by ~1e-15 mm only; it changes results solely by flipping the blocked test, so it is not a test of
             the polyhedral short-cut.
Usage: python -I thick_shell.py <workdir> <out_csv> [--h 1.5 1.0 0.75 0.5] [--workers N]
"""
import argparse
import ast
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
I09 = os.path.join(HERE, "..", "..", "pilot", "I09")
I11 = os.path.join(HERE, "..", "..", "pilot", "I11")
SRC = os.path.join(I09, "00_known_answer.py")
A_IN, B_OUT, V = 10.0, 15.0, 600.0


def _load_i09():
    text = open(SRC).read()
    tree = ast.parse(text)
    ns = dict(np=np)
    segs = {n.name: ast.get_source_segment(text, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("fib_sphere", "geod")}
    for k, s in segs.items():
        exec(compile(s, f"<I09:{k}>", "exec"), ns)
    return ns, hashlib.sha256(text.encode()).hexdigest()


def geod_fixed(P, s, a, tol=1e-9):
    """Exact geodesic length around the ball of radius a (I09 geod with the blocked-test tolerance fix)."""
    rp = np.linalg.norm(P, axis=1); rs = np.linalg.norm(s)
    d = P - s
    t = np.clip(-(s @ d.T) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-30), 0, 1)
    closest = np.linalg.norm(s + t[:, None] * d, axis=1)
    chord = np.linalg.norm(d, axis=1)
    blocked = closest < a * (1 - tol)
    ang = np.arccos(np.clip((P @ s) / (np.maximum(rp, 1e-30) * rs), -1, 1))
    wrap = np.sqrt(rs ** 2 - a ** 2) + np.sqrt(np.maximum(rp ** 2 - a ** 2, 0)) + \
        a * (ang - np.arccos(a / rs) - np.arccos(np.minimum(a / rp, 1)))
    out = chord.copy()
    out[blocked] = np.maximum(wrap[blocked], chord[blocked])   # a geodesic is never shorter than the chord
    return out, blocked


def delaunay_shell(h, snap=False):
    ns, _ = _load_i09()
    fib_sphere = ns["fib_sphere"]
    # --- copied from 00_known_answer.py (thick shell loop body)
    layers = np.linspace(A_IN, B_OUT, int(round((B_OUT - A_IN) / h)) + 1)
    pts = [np.array([[0.0, 0.0, B_OUT]])]
    for r in layers:
        m = int(4 * np.pi * r * r / (h * h * np.sqrt(3) / 2))
        P = fib_sphere(m, r)
        if r == B_OUT:
            P = P[np.linalg.norm(P - [0, 0, B_OUT], axis=1) > 0.5 * h]
        pts.append(P)
    X = np.concatenate(pts)
    tri = Delaunay(X)
    T = tri.simplices.astype(np.int64)
    cen = X[T].mean(1)
    keep = np.linalg.norm(cen, axis=1) > A_IN
    a, b, c, e = (X[T[:, k]] for k in range(4))
    vv = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a)) / 6
    keep &= vv > 1e-6 * h ** 3
    T = T[keep]
    used = np.unique(T)
    remap = -np.ones(X.shape[0], np.int64); remap[used] = np.arange(used.size)
    X = X[used]; T = remap[T]
    src = int(remap[0])
    # --- end of copy
    if snap:
        r = np.linalg.norm(X, axis=1)
        X = np.where((r < A_IN)[:, None], X * (A_IN / np.maximum(r, 1e-12))[:, None], X)
    return X, T, src


def voxel_shell(h):
    n = int(np.ceil(B_OUT / h)) + 2
    g = (np.arange(-n, n + 1)) * h
    C = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
    r = np.linalg.norm(C, axis=-1)
    vox = (r >= A_IN) & (r <= B_OUT)
    vi = np.argwhere(vox)
    shp = np.array(vox.shape) + 1
    corner = lambda a, b, cc: ((vi[:, 0] + a) * shp[1] + (vi[:, 1] + b)) * shp[2] + (vi[:, 2] + cc)
    cn = [corner(a, b, cc) for a in (0, 1) for b in (0, 1) for cc in (0, 1)]
    paths = [(4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3)]
    T = np.concatenate([np.column_stack([cn[0], cn[p], cn[q], cn[7]]) for p, q in paths])
    used, inv = np.unique(T, return_inverse=True)
    T = inv.reshape(T.shape)
    ijk = np.column_stack(np.unravel_index(used, tuple(shp))).astype(float) - 0.5
    X = (ijk - n) * h
    # source: node closest to the outer pole (0, 0, B_OUT) on the staircase
    src = int(np.argmin(np.linalg.norm(X - [0, 0, B_OUT], axis=1)))
    return X, T, src


def run(args):
    fam, h, solver, work = args
    sys.path.insert(0, I09)
    import carpio as c9  # I09 helpers (write_par, run)
    ns, _ = _load_i09()
    geod = ns["geod"]
    t0 = time.time()
    if fam == "voxel":
        X, T, src = voxel_shell(h)
    else:
        X, T, src = delaunay_shell(h, snap=(fam == "delaunay_snap"))
    a, b, c, e = (X[T[:, k]] for k in range(4))
    v = np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a)
    T = T.copy(); T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    s = X[src]
    an_old = geod(X, s, A_IN) / V * 1e3          # I09 reference (bugged blocked test), kept for comparison
    g, blocked = geod_fixed(X, s, A_IN)
    an = g / V * 1e3
    # voxel nodes inside the inner ball (staircase): blocked nodes use the wrap formula with clamped radius,
    # bounded below by the chord; this is approximate on the staircase (documented)
    d = os.path.join(work, f"{fam}_h{h}_{solver}")
    os.makedirs(d, exist_ok=True)
    if solver == "i09":
        c9.write_mesh(os.path.join(d, "m"), X, T, fib=np.tile([1.0, 0, 0], (T.shape[0], 1)))
        c9.write_vtx(os.path.join(d, "stim.vtx"), [src])
        c9.write_par(os.path.join(d, "p.par"), "m", "out", ["stim"], vel=(V, V, V), tend=1)
        ts = time.time()
        rc = c9.run("p.par", d, omp=1)
        wall = time.time() - ts
        at = c9.read_igb(os.path.join(d, "out", "act.igb"))[-1].astype(float)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("carpio11", os.path.join(I11, "carpio.py"))
        c11 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c11)
        c11.write_mesh(os.path.join(d, "m"), X * 1000, T, np.ones(len(T), int))
        c11.write_vtx(os.path.join(d, "stim.vtx"), [src])
        at, wall = c11.run_eikonal(os.path.join(d, "m"), os.path.join(d, "stim.vtx"), [(1, V / 1000, V / 1000, V / 1000)],
                                   os.path.join(d, "sim"), threads=1, tend=200)
        rc = 0
    err = at - an
    err_old = at - an_old
    r_node = np.linalg.norm(X, axis=1)
    far = np.linalg.norm(X - s, axis=1) > 3 * h
    ang = np.degrees(np.arccos(np.clip((X @ s) / (np.maximum(r_node, 1e-12) * np.linalg.norm(s)), -1, 1)))
    imax = int(np.argmax(np.abs(err)))
    q = lambda m: float(np.abs(err[m]).max() / an.max()) if m.any() else float("nan")
    out = dict(family=fam, h_mm=h, solver=solver, nodes=len(X), tets=len(T), rc=rc, wall_s=wall,
               Tmax_analytic_ms=float(an.max()), max_abs_err_ms=float(np.abs(err).max()),
               max_err_frac_of_Tmax=float(np.abs(err).max() / an.max()),
               p99_abs_err_frac=float(np.quantile(np.abs(err), 0.99) / an.max()),
               max_err_frac_far3h=q(far), TAT_err_frac=float((at.max() - an.max()) / an.max()),
               mean_signed_err_ms=float(err.mean()), median_signed_err_ms=float(np.median(err)),
               argmax_angle_deg=float(ang[imax]), argmax_r_mm=float(r_node[imax]),
               frac_nodes_over_2pct=float((np.abs(err) / an.max() > 0.02).mean()),
               max_err_frac_shadow=q(ang > 120), max_err_frac_lit=q(ang <= 60),
               n_nodes_inside_inner=int((r_node < A_IN - 1e-9).sum()), n_nodes_r_lt_a_any=int((r_node < A_IN).sum()),
               min_r_minus_a_mm=float(r_node.min() - A_IN), n_blocked_fixed=int(blocked.sum()),
               n_nodes_analytic_changed=int((np.abs(an - an_old) > 1e-9).sum()),
               max_analytic_change_ms=float(np.abs(an - an_old).max()),
               max_err_frac_of_Tmax_i09geod=float(np.abs(err_old).max() / an_old.max()),
               argmax_angle_deg_i09geod=float(ang[int(np.argmax(np.abs(err_old)))]),
               argmax_r_mm_i09geod=float(r_node[int(np.argmax(np.abs(err_old)))]),
               mean_signed_err_ms_i09geod=float(err_old.mean()),
               frac_nodes_over_2pct_i09geod=float((np.abs(err_old) / an_old.max() > 0.02).mean()),
               p99_abs_err_frac_i09geod=float(np.quantile(np.abs(err_old), 0.99) / an_old.max()),
               t_total_s=time.time() - t0)
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p) and f.split(".")[-1] in ("pts", "elem", "lon"):
            os.remove(p)
    print(out, flush=True)
    arrs = dict(X=X.astype(np.float32), at=np.asarray(at, np.float32), an=an.astype(np.float32),
                an_i09geod=an_old.astype(np.float32))
    return out, arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("out_csv")
    ap.add_argument("--h", type=float, nargs="+", default=[1.5, 1.0, 0.75, 0.5])
    ap.add_argument("--families", nargs="+", default=["delaunay", "delaunay_snap", "voxel"])
    ap.add_argument("--solvers", nargs="+", default=["i09", "study"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    _, sha = _load_i09()
    print("I09 00_known_answer.py sha256", sha, flush=True)
    jobs = [(f, h, s, a.work) for f in a.families for h in a.h for s in a.solvers]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    rows = [r[0] for r in res]
    np.savez_compressed(os.path.splitext(a.out_csv)[0] + "_nodes.npz",
                        **{f"{r[0]['family']}_h{r[0]['h_mm']}_{r[0]['solver']}_{k}": v for r in res for k, v in r[1].items()})
    import pandas as pd
    df = pd.DataFrame(rows)
    df["src_sha256"] = sha
    df.to_csv(a.out_csv, index=False)
    print(df[["family", "h_mm", "solver", "nodes", "max_err_frac_of_Tmax", "max_err_frac_far3h", "TAT_err_frac",
              "mean_signed_err_ms", "argmax_angle_deg", "max_err_frac_of_Tmax_i09geod", "n_nodes_analytic_changed"]].to_string())


if __name__ == "__main__":
    main()
