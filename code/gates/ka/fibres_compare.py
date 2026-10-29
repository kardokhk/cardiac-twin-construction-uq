"""Known-answer test KA-FIB (S6), I11 gates, owner ka, 2026-10-28.

Bayer 2012 fibres from the I09 pilot implementation (code/gates/ka/bayer_i09.py, verbatim rule functions)
against the reference LDRB implementation fenicsx-ldrb 0.2.0 (github.com/finsberg/fenicsx-ldrb, successor of
finsberg/ldrb), with identical inputs: the same tetrahedral mesh, the same boundary sets (LV endo, RV endo, epi,
base) and the same apex node set (taken from ldrb's own apex search and passed to I09). Angles +40/-50 helix,
-65/+25 sheet (I09 values; ldrb defaults are identical). ldrb is evaluated in DG_0, i.e. per element from the
exact element gradients of the P1 Laplace solutions and the element mean of the scalars, which is what I09 does.

Meshes: "ellipsoid" = analytic biventricular ellipsoid of ellipsoid_sax.py voxelised at 1.5 mm (reference
voxelisation, no SAX step) and meshed with the verbatim I11 mesh_from_labels; "heart09" = pilot I11 reference
construction c0ref of Malik heart 09 at 1.5 mm (data/processed/pilot_I13/cons/cons09.h5).

Comparison: helix angle of both fields in one frame (I09 epicardial axis frame: circumferential e0,
apicobasal e1), |difference| (wrapped to [0, 90]); angle between fibre lines acos|f1.f2|; transmural profile of
the median helix angle in the LV free wall (depth < 0.1) over 10 bins of phi_epi; one openCARP eikonal run with
each fibre field (study settings: tag 1 v_l 0.6 m/s, v_t = v_n 0.29 m/s; tag 2 fast endocardial layer
isotropic 1.55 m/s; pilot root nodes), activation-time difference.
Usage: python -I fibres_compare.py <mesh: ellipsoid|heart09> <outdir> <workdir>
"""
import json
import os
import sys
import time

import numpy as np
import h5py
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # code/
sys.path.insert(0, HERE)
import bayer_i09  # noqa: E402
import ellipsoid_sax as es  # noqa: E402

VOX = 1.5
THETA = (1.55, 0.29)   # (v_endo, v_t) m/s, geometric middle of the pilot theta* range
V_FIB = 0.6


def boundary_faces(T, X):
    F = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 3, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 1]]])
    opp = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 3]])
    key = np.sort(F, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    F, opp = F[b], opp[b]
    c = X[F].mean(1)
    nrm = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    s = np.sign(np.einsum("ij,ij->i", nrm, c - X[opp]))   # make outward
    return F, c, nrm * s[:, None]


def classify(F, fc, nrm, lab, lo, Rf, cvec, vox, base_frac=0.15):
    """1 base, 2 LV endo, 3 RV endo, 4 epi; by the label of the voxel across each boundary face."""
    probe = fc + 0.5 * vox * nrm
    U = (probe - cvec) @ Rf.T
    idx = np.round(U / vox - lo).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(lab.shape)), axis=1)
    L = np.zeros(len(F), np.int8)
    L[ok] = lab[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    cls = np.full(len(F), 4, np.int32)
    cls[L == 3] = 2
    cls[L == 4] = 3
    u = ((fc - cvec) @ Rf.T)[:, 0]
    umin, umax = u.min(), u.max()
    basal = (L == 0) & ((nrm @ Rf[0]) < -0.7) & (u < umin + base_frac * (umax - umin))
    cls[basal] = 1
    return cls


def build(mesh):
    segs, carpio, meta = es.load_route(os.path.join(ROOT, "pilot", "I11", "06_constructions.py"))
    if mesh == "ellipsoid":
        lo_mm, hi_mm = es.bbox("biv")
        lo = np.floor(lo_mm / VOX) - 4
        hi = np.ceil(hi_mm / VOX) + 4
        n = tuple((hi - lo).astype(int) + 1)
        lab = es.label_grid("biv", lo * VOX, n, VOX)
        Rf, cvec = np.eye(3), np.zeros(3)
        ns = es.make_namespace(segs, carpio, VOX, lo, {})
        pts, tets, iface, lv_if = ns["mesh_from_labels"](lab)
        dist = cKDTree(pts[iface]).query(pts)[0]
        etag = np.where((dist <= 1e-6)[tets].any(1), 2, 1)
        # root nodes: LV mid-septal, anterior and posterior paraseptal, RV free wall (mid-ventricle, u = 40 mm)
        lvn = np.nonzero(lv_if)[0]; rvn = np.nonzero(iface & ~lv_if)[0]
        um = 40.0
        targets = [(um, es.A_EN, 0.0), (um, es.A_EN * 0.5, es.A_EN * 0.87), (um, es.A_EN * 0.5, -es.A_EN * 0.87)]
        roots = [lvn[cKDTree(pts[lvn]).query(t)[1]] for t in targets]
        roots.append(rvn[cKDTree(pts[rvn]).query((um, es.Y_RV + es.B_RV, 0.0))[1]])
        stim = np.unique(np.concatenate([cKDTree(pts).query_ball_point(pts[r], 2.5) for r in roots]))
        info = dict(source="analytic BiV ellipsoid, reference voxelisation", roots=[int(r) for r in roots])
    else:
        d = os.path.join(ROOT, "..", "data", "processed", "pilot_I13")
        H = h5py.File(os.path.join(d, "heart09.h5"), "r")
        frame = H["frame"][:]; cvec = frame[0]; Rf = np.vstack(frame[1:4])
        z = np.load(os.path.join(d, "malik09.npz"))
        tp = z["pts"].astype(np.float64); tt = z["tets"]; tg = z["tags"]
        ht = tt[np.isin(tg, [20, 21, 22, 23])]
        U = (tp[np.unique(ht)] - cvec) @ Rf.T
        lo = np.floor(U.min(0) / VOX) - 4
        del tp, tt, tg, ht, U
        C = h5py.File(os.path.join(d, "cons", "cons09.h5"), "r")["c0ref"]
        lab = C["lab"][:]; pts = C["pts"][:]; tets = C["tets"][:].astype(np.int64)
        etag = C["etag"][:].astype(int); stim = C["stim"][:]
        ns = es.make_namespace(segs, carpio, VOX, lo, {})
        ns["Rf"], ns["c"] = Rf, cvec
        p2, t2, _, _ = ns["mesh_from_labels"](lab)
        same = bool(p2.shape == pts.shape and np.allclose(p2, pts) and np.array_equal(t2, tets))
        info = dict(source="cons09.h5 c0ref (pilot I11, Malik heart 09, 1.5 mm)", regenerated_mesh_identical=same,
                    cons_sha256=None)
        if not same:
            print("WARNING: regenerated mesh differs from stored c0ref mesh", flush=True)
    return lab, lo, Rf, cvec, pts, tets, etag, stim, info, meta


def run_ldrb(pts, tets, F, cls, workdir):
    from mpi4py import MPI
    import basix.ufl
    import dolfinx
    import ufl
    import ldrb
    el = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, tets.astype(np.int64), ufl.Mesh(el), pts.astype(np.float64))
    msh.topology.create_connectivity(2, 0)
    msh.topology.create_connectivity(2, 3)
    ent, val = dolfinx.io.distribute_entity_data(msh, 2, F.astype(np.int64), cls.astype(np.int32))
    ffun = dolfinx.mesh.meshtags_from_entities(msh, 2, dolfinx.graph.adjacencylist(ent), val.astype(np.int32))
    t0 = time.time()
    out = ldrb.dolfinx_ldrb(mesh=msh, ffun=ffun, markers=dict(base=[1], lv=[2], rv=[3], epi=[4]),
                            fiber_space="DG_0", alpha_endo_lv=40, alpha_epi_lv=-50, beta_endo_lv=-65,
                            beta_epi_lv=25)
    t_ldrb = time.time() - t0
    oc = msh.topology.original_cell_index
    V0 = out.f0.function_space
    dofs = V0.dofmap.list[:, 0]
    f = out.f0.x.array.reshape(-1, 3)[dofs]
    fib = np.zeros((len(tets), 3)); fib[oc] = f
    # apex nodes (psi = 0) from ldrb's P1 apex field, mapped to input node indices by coordinates
    ap = out.apex
    xa = ap.function_space.tabulate_dof_coordinates()
    zero = np.abs(ap.x.array) < 1e-10
    apex_nodes = np.unique(cKDTree(pts).query(xa[zero])[1])
    return fib, apex_nodes, t_ldrb, dolfinx.__version__, ldrb.__version__ if hasattr(ldrb, "__version__") else "0.2.0"


def helix(fib, Q):
    ha = np.degrees(np.arctan2(np.einsum("ij,ij->i", fib, Q[:, :, 1]), np.einsum("ij,ij->i", fib, Q[:, :, 0])))
    return (ha + 90) % 180 - 90


def eikonal(pts, tets, etag, fib, stim, work, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("carpio11", os.path.join(ROOT, "pilot", "I11", "carpio.py"))
    c11 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c11)
    d = os.path.join(work, name); os.makedirs(d, exist_ok=True)
    base = os.path.join(d, "m")
    c11.write_mesh(base, pts * 1000, tets, etag, fib.astype(np.float64))
    c11.write_vtx(base + "_stim.vtx", stim)
    at, wall = c11.run_eikonal(base, base + "_stim.vtx", [(1, V_FIB, THETA[1], THETA[1]),
                                                          (2, THETA[0], THETA[0], THETA[0])],
                               os.path.join(d, "sim"), threads=1, tend=400)
    return at, wall


def main():
    mesh, outdir, work = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True); os.makedirs(work, exist_ok=True)
    t0 = time.time()
    lab, lo, Rf, cvec, pts, tets, etag, stim, info, meta = build(mesh)
    F, fc, nrm = boundary_faces(tets, pts)
    cls = classify(F, fc, nrm, lab, lo, Rf, cvec, VOX)
    counts = {k: int((cls == v).sum()) for k, v in dict(base=1, lv=2, rv=3, epi=4).items()}
    print(mesh, "nodes", len(pts), "tets", len(tets), "faces", counts, flush=True)
    fib_l, apex, t_ldrb, dv, lv_ = run_ldrb(pts, tets, F, cls, work)
    print("ldrb done", t_ldrb, "apex nodes", len(apex), flush=True)
    nodes = lambda k: np.unique(F[cls == k])
    t1 = time.time()
    res, bmeta = bayer_i09.bayer_fibres(pts, tets, nodes(2), nodes(3), nodes(4), apex, nodes(1))
    t_i09 = time.time() - t1
    fib_b = res["fib"]
    Q = res["frame"]
    ha_b, ha_l = helix(fib_b, Q), helix(fib_l, Q)
    dha = np.abs((ha_b - ha_l + 90) % 180 - 90)
    dang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", fib_b, fib_l)), 0, 1)))
    depth, eepi = res["depth"], res["e_epi"]
    regions = dict(all=np.ones(len(tets), bool), lv_free_wall=depth < 0.1, septum=(depth >= 0.1) & (depth <= 0.9),
                   rv=depth > 0.9)
    stats = {}
    for k, m in regions.items():
        stats[k] = dict(n=int(m.sum()), dhelix_median=float(np.median(dha[m])), dhelix_p95=float(np.quantile(dha[m], 0.95)),
                        dangle_median=float(np.median(dang[m])), dangle_p95=float(np.quantile(dang[m], 0.95)),
                        dangle_max=float(dang[m].max()))
    bins = np.linspace(0, 1, 11)
    prof = []
    m0 = regions["lv_free_wall"]
    for i in range(10):
        m = m0 & (eepi >= bins[i]) & (eepi < bins[i + 1] + (1e-9 if i == 9 else 0))
        prof.append(dict(phi_epi_lo=float(bins[i]), phi_epi_hi=float(bins[i + 1]), n=int(m.sum()),
                         helix_i09_median=float(np.median(ha_b[m])) if m.any() else None,
                         helix_ldrb_median=float(np.median(ha_l[m])) if m.any() else None,
                         dhelix_median=float(np.median(dha[m])) if m.any() else None))
    # eikonal with each field
    at_b, w_b = eikonal(pts, tets, etag, fib_b, stim, work, f"{mesh}_i09")
    at_l, w_l = eikonal(pts, tets, etag, fib_l, stim, work, f"{mesh}_ldrb")
    dAT = at_b - at_l
    eik = dict(theta=THETA, v_fib=V_FIB, TAT_i09_ms=float(at_b.max()), TAT_ldrb_ms=float(at_l.max()),
               dTAT_ms=float(at_b.max() - at_l.max()), dAT_abs_median_ms=float(np.median(np.abs(dAT))),
               dAT_abs_p95_ms=float(np.quantile(np.abs(dAT), 0.95)), dAT_abs_max_ms=float(np.abs(dAT).max()),
               dAT_mean_ms=float(dAT.mean()), ATmean_i09_ms=float(at_b.mean()), ATmean_ldrb_ms=float(at_l.mean()),
               wall_i09_s=w_b, wall_ldrb_s=w_l)
    out = dict(mesh=mesh, info=info, n_nodes=len(pts), n_tets=len(tets), face_counts=counts, n_apex_nodes=int(len(apex)),
               i09_nonfinite=res["nonfinite"], ldrb_nonfinite=int((~np.isfinite(fib_l)).any(1).sum()),
               ldrb_zero=int((np.linalg.norm(fib_l, axis=1) < 0.5).sum()),
               helix_lvfw_endo_i09=float(np.median(ha_b[m0 & (eepi < 0.1)])),
               helix_lvfw_epi_i09=float(np.median(ha_b[m0 & (eepi > 0.9)])),
               helix_lvfw_endo_ldrb=float(np.median(ha_l[m0 & (eepi < 0.1)])),
               helix_lvfw_epi_ldrb=float(np.median(ha_l[m0 & (eepi > 0.9)])),
               stats=stats, profile=prof, eikonal=eik, t_ldrb_s=t_ldrb, t_i09_s=t_i09, t_total_s=time.time() - t0,
               versions=dict(dolfinx=dv, fenicsx_ldrb=lv_, numpy=np.__version__), route_meta=meta, i09_meta=bmeta)
    json.dump(out, open(os.path.join(outdir, f"fibres_{mesh}.json"), "w"), indent=1, default=float)
    with h5py.File(os.path.join(outdir, f"fibres_{mesh}.h5"), "w") as f:
        for k, v in dict(fib_i09=fib_b, fib_ldrb=fib_l, helix_i09=ha_b, helix_ldrb=ha_l, dhelix=dha, dangle=dang,
                         depth=depth, phi_epi=eepi, at_i09=at_b, at_ldrb=at_l).items():
            f.create_dataset(k, data=np.asarray(v, np.float32), compression="gzip", compression_opts=1)
    print(json.dumps({k: out[k] for k in ("stats", "eikonal", "helix_lvfw_endo_i09", "helix_lvfw_epi_i09",
                                          "helix_lvfw_endo_ldrb", "helix_lvfw_epi_ldrb")}, indent=1, default=float))


if __name__ == "__main__":
    main()
