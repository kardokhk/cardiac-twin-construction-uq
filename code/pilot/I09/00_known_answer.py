"""I09 pilot known-answer tests for the openCARP v20.0 eikonal solver (pure eikonal, dream.solve = 0).

KA-slab : 10 mm cube, Kuhn tetrahedra at h = 1, 0.5, 0.25 mm, corner point stimulus.
          (a) isotropic v = 600 mm/s, T = |x|/v;
          (b) transversely isotropic, fibre (1,1,0)/sqrt2, v_f 600, v_t 300 mm/s,
              T = sqrt((x.f)^2/v_f^2 + (|x|^2-(x.f)^2)/v_t^2).
KA-shell: thick spherical shell a = 10, b = 15 mm (Delaunay of layered Fibonacci points, h = 1.5, 1.0, 0.75 mm),
          isotropic 600 mm/s, point stimulus at the outer pole; T = geodesic distance around the inner ball / v.
Metrics : max |error| / max analytic T (the red-team 2% criterion), max relative error at nodes farther than 3h
          from the source, error in total activation time (max T).
Usage   : python -I 00_known_answer.py <workdir> <out_csv>
"""
import os
import sys

import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carpio  # noqa: E402

work, out_csv = sys.argv[1:3]
os.makedirs(work, exist_ok=True)
rows = []


def run_case(name, X, T, src, fib, vel, analytic, h):
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    a, b, c, e = (X[T[:, k]] for k in range(4))
    v = np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a)
    T = T.copy(); T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    carpio.write_mesh(os.path.join(d, "m"), X, T, fib=np.tile(fib, (T.shape[0], 1)))
    carpio.write_vtx(os.path.join(d, "stim.vtx"), [src])
    carpio.write_par(os.path.join(d, "p.par"), "m", "out", ["stim"], vel=vel, tend=1)
    rc = carpio.run("p.par", d, omp=4)
    at = carpio.read_igb(os.path.join(d, "out", "act.igb"))[-1].astype(float)
    err = at - analytic
    far = np.linalg.norm(X - X[src], axis=1) > 3 * h
    r = dict(test=name, h_mm=h, nodes=X.shape[0], tets=T.shape[0], rc=rc,
             max_abs_err_ms=float(np.abs(err).max()), Tmax_analytic_ms=float(analytic.max()),
             max_err_frac_of_Tmax=float(np.abs(err).max() / analytic.max()),
             max_rel_err_far=float((np.abs(err[far]) / analytic[far]).max()),
             TAT_err_frac=float((at.max() - analytic.max()) / analytic.max()),
             mean_signed_err_ms=float(err.mean()))
    print(r, flush=True)
    rows.append(r)
    for f in ("m.pts", "m.elem", "m.lon"):
        os.remove(os.path.join(d, f))


# --- slab
for h in (1.0, 0.5, 0.25):
    n = int(round(10 / h))
    g = np.arange(n + 1) * h
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    cc = [idx[i:i + n, j:j + n, k:k + n].ravel() for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    paths = [(1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
    T = np.concatenate([np.stack([cc[0], cc[p], cc[q], cc[7]], 1) for p, q in paths])
    run_case(f"slab_iso_h{h}", X, T, 0, [1.0, 0.0, 0.0], (600, 600, 600), np.linalg.norm(X, axis=1) / 600 * 1e3, h)
    f = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    pf = X @ f
    an = np.sqrt(pf ** 2 / 600 ** 2 + (np.einsum("ij,ij->i", X, X) - pf ** 2) / 300 ** 2) * 1e3
    run_case(f"slab_aniso_h{h}", X, T, 0, f, (600, 300, 300), an, h)

# --- thick shell
A_IN, B_OUT = 10.0, 15.0


def fib_sphere(m, r):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def geod(P, s, a):
    rp = np.linalg.norm(P, axis=1); rs = np.linalg.norm(s)
    d = P - s
    t = np.clip(-(s @ d.T) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-30), 0, 1)
    closest = np.linalg.norm(s + t[:, None] * d, axis=1)
    out = np.linalg.norm(d, axis=1)
    blocked = closest < a
    cosang = np.clip((P @ s) / (rp * rs), -1, 1)
    ang = np.arccos(cosang)
    wrap = np.sqrt(rs ** 2 - a ** 2) + np.sqrt(np.maximum(rp ** 2 - a ** 2, 0)) + \
        a * (ang - np.arccos(a / rs) - np.arccos(np.minimum(a / rp, 1)))
    out[blocked] = wrap[blocked]
    return out


for h in (1.5, 1.0, 0.75):
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
    an = geod(X, np.array([0, 0, B_OUT]), A_IN) / 600 * 1e3
    run_case(f"shell_iso_h{h}", X, T, src, [1.0, 0.0, 0.0], (600, 600, 600), an, h)

import pandas as pd  # noqa: E402
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("written", out_csv)
