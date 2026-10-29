"""Electrode-displacement lead fields for Amendment A3.3 (owner mesh2, I11 full arm, 2026-10-28).

For each heart's reference construction (expert, 1.5 mm, RV wall 3 mm; data/processed/acdc_mesh_v2/cons), the
precordial electrodes V1 to V6 of torso 09 (electrode rule v2, heart09.h5) are displaced on the torso surface by an
isotropic tangential Gaussian offset with SD 5, 10 and 20 mm, 20 replicates each; limb electrodes (RA, LA, LL) and
the reference RL stay fixed. Lead fields are recomputed for each replicate and turned into ECG weights W exactly as
in the construction build (W = lead_matrix @ (K_i Z_elec)^T, pilot kernel eplib).

Displacement (assumptions stated in results/2026-10-28_mesh_v2/README.md):
  * seed = 20261028 + 1000 * patient + 100 * sd_index + rep (sd_index 0, 1, 2 for 5, 10, 20 mm; rep 0..19);
    rng = numpy.random.default_rng(seed); g = rng.standard_normal((6, 2)) for V1..V6 in that order.
  * Tangent basis at the electrode (outward normal n of the closest outer-surface triangle): t1 = superior axis
    (LPS +z) projected onto the tangent plane, t2 = n x t1. Offset vector SD * (g1 t1 + g2 t2).
  * The offset is applied as a geodesic-like walk on the outer torso surface: steps of at most 0.5 mm along the
    direction, re-projected onto the tangent plane of the current triangle, each step snapped to the closest point
    of the surface; walked length = SD * |g|.
  * Electrode node = nearest torso boundary node to the end point (the rule v2 convention for undisplaced
    electrodes, so a zero offset returns the original node).
Lead field per electrode node s: K Z = e_s - e_RL with Z[RL] = 0, homogeneous 0.2 S/m on the full Malik heart 09
torso (15,208,101 tets), pyamg smoothed aggregation (max_coarse 500, numpy global seed 20261028 before setup), CG to
relative residual 1e-10 (as eplib.solve_leadfields). Each distinct node is solved once.

Usage (all with OMP_NUM_THREADS=1):
  python -I -B displace.py prep <scratch> <pid,...>
  python -I -B displace.py solve <scratch> <worker> <nworkers>
  python -I -B displace.py assemble <scratch> <pid>
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acdc_lib as A        # noqa: E402
import eplib                # noqa: E402  (code/pilot/I13/eplib.py via acdc_lib's path)

SDS = (5.0, 10.0, 20.0)
NREP = 20
SEED0 = 20261028
AMG_SEED = 20261028
STEP_MM = 0.5
VIDX = [3, 4, 5, 6, 7, 8]                   # V1..V6 in eplib.ELECS
PILOT = os.path.join(A.ROOT, 'data', 'processed', 'pilot_I13')
OUT = os.path.join(A.ROOT, 'data', 'processed', 'acdc_mesh_v2')


def seed_of(pid, sdi, rep):
    return SEED0 + 1000 * pid + 100 * sdi + rep


# ------------------------------------------------------------------------------------------------ surface geometry
def outer_surface(pts, tets):
    """Boundary faces with outward unit normals; keeps the connected component of the largest face count."""
    f4 = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)]
    F = np.concatenate([tets[:, list(a[:3])] for a in f4])
    O = np.concatenate([tets[:, a[3]] for a in f4])
    Fs = np.sort(F, axis=1)
    _, idx, cnt = np.unique(Fs, axis=0, return_index=True, return_counts=True)
    b = idx[cnt == 1]
    F, O = F[b], O[b]
    nrm = np.cross(pts[F[:, 1]] - pts[F[:, 0]], pts[F[:, 2]] - pts[F[:, 0]])
    flip = np.einsum('ij,ij->i', nrm, pts[O] - pts[F[:, 0]]) > 0
    F[flip] = F[flip][:, [0, 2, 1]]; nrm[flip] *= -1
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # connected components of the surface (shared vertices)
    from scipy.sparse.csgraph import connected_components
    nf = len(F)
    rows = np.repeat(np.arange(nf), 3); cols = F.ravel()
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nf, len(pts)))
    adj = M @ M.T
    ncomp, lab = connected_components(adj, directed=False)
    big = np.argmax(np.bincount(lab))
    keep = lab == big
    return F[keep], nrm[keep], np.unique(F), dict(n_boundary_faces=int(nf), n_components=int(ncomp),
                                                   n_outer_faces=int(keep.sum()))


def closest_on_tris(X, A_, B_, C_):
    """Exact closest point on triangles (row-wise). Returns points (m, 3)."""
    def seg(P, U, V):
        d = V - U
        t = np.clip(np.einsum('ij,ij->i', P - U, d) / np.maximum(np.einsum('ij,ij->i', d, d), 1e-30), 0, 1)
        return U + t[:, None] * d
    n = np.cross(B_ - A_, C_ - A_); n2 = np.einsum('ij,ij->i', n, n)
    w = X - np.einsum('ij,ij->i', X - A_, n)[:, None] / np.maximum(n2, 1e-30)[:, None] * n
    # barycentric inside test
    v0, v1, v2 = B_ - A_, C_ - A_, w - A_
    d00 = np.einsum('ij,ij->i', v0, v0); d01 = np.einsum('ij,ij->i', v0, v1); d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0); d21 = np.einsum('ij,ij->i', v2, v1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; bu = 1 - bv - bw
    inside = (bu >= 0) & (bv >= 0) & (bw >= 0)
    cands = [seg(X, A_, B_), seg(X, B_, C_), seg(X, C_, A_)]
    best = cands[0]; bd = ((X - best) ** 2).sum(1)
    for cnd in cands[1:]:
        dd = ((X - cnd) ** 2).sum(1); u = dd < bd; best[u] = cnd[u]; bd[u] = dd[u]
    best[inside] = w[inside]
    return best


class Surface:
    def __init__(self, pts, F, nrm, k=16):
        self.P, self.F, self.N, self.k = pts, F, nrm, k
        self.tree = cKDTree(pts[F].mean(1))

    def closest(self, X):
        _, cand = self.tree.query(X, k=self.k)
        best = np.empty_like(X); bd = np.full(len(X), np.inf); bf = np.zeros(len(X), np.int64)
        for j in range(self.k):
            f = cand[:, j]; T = self.F[f]
            q = closest_on_tris(X, self.P[T[:, 0]], self.P[T[:, 1]], self.P[T[:, 2]])
            d = ((X - q) ** 2).sum(1); u = d < bd
            best[u] = q[u]; bd[u] = d[u]; bf[u] = f[u]
        return best, bf, np.sqrt(bd)


def walk(surf, X0, offsets_2d, step=STEP_MM):
    """X0 (m, 3) start points on the surface; offsets_2d (m, 2) tangential offsets (mm) in the (t1, t2) basis."""
    X, f, _ = surf.closest(X0.copy())
    n = surf.N[f]
    ez = np.array([0.0, 0.0, 1.0])
    t1 = ez[None] - (n @ ez)[:, None] * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    u = offsets_2d[:, :1] * t1 + offsets_2d[:, 1:] * t2
    Lw = np.linalg.norm(offsets_2d, axis=1)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30)
    nst = np.maximum(1, np.ceil(Lw / step)).astype(int)
    h = Lw / nst
    for s in range(int(nst.max())):
        a = nst > s
        if not a.any():
            break
        na = surf.N[f[a]]
        ua = u[a] - np.einsum('ij,ij->i', u[a], na)[:, None] * na
        ua /= np.maximum(np.linalg.norm(ua, axis=1, keepdims=True), 1e-30)
        Xa, fa, _ = surf.closest(X[a] + h[a, None] * ua)
        X[a] = Xa; f[a] = fa; u[a] = ua
    return X, Lw


# ------------------------------------------------------------------------------------------------ helpers
def locate(src_pts, src_tets, query, k=12, tree=None):
    """eplib.locate_and_interp, returning (tet index, barycentric lambdas, score) instead of values."""
    if tree is None:
        tree = cKDTree(src_pts[src_tets].mean(1))
    _, cand = tree.query(query, k=k)
    best_lam = np.full((len(query), 4), np.nan); best_score = np.full(len(query), -np.inf)
    best_tet = np.zeros(len(query), dtype=np.int64)
    for c in range(k):
        T = src_tets[cand[:, c]]
        X = src_pts[T]
        D = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)
        l123 = np.linalg.solve(D, (query - X[:, 0])[:, :, None])[:, :, 0]
        lam = np.column_stack([1 - l123.sum(1), l123])
        score = lam.min(1)
        upd = score > best_score
        best_score[upd] = score[upd]; best_lam[upd] = lam[upd]; best_tet[upd] = cand[upd, c]
    return best_tet, best_lam, best_score


def ref_h5(pid):
    return os.path.join(OUT, 'cons', f'patient{pid:03d}__expert__1.5__rv3.0.h5')


# ------------------------------------------------------------------------------------------------ stages
def prep(scr, pids):
    import h5py
    t0 = time.time()
    os.makedirs(scr, exist_ok=True)
    H = h5py.File(os.path.join(PILOT, 'heart09.h5'), 'r')
    assert H.attrs['electrode_rule'] == 'v2'
    elec_xyz = H['elec_xyz'][:]; elec_nodes = H['elec_nodes'][:]; frame = H['frame'][:]
    d = np.load(os.path.join(PILOT, 'malik09.npz'))
    pts = d['pts'].astype(np.float64); tets = d['tets'].astype(np.int64)
    # near subset (identical construction to code/gates/mesh/build.py prep_torso)
    cen = pts[tets].mean(1)
    near = np.all(np.abs(cen - frame[0]) < 140.0, axis=1)
    sub = tets[near]; del cen
    nodes, inv = np.unique(sub, return_inverse=True)
    T = A.load_torso()
    assert len(nodes) == len(T['tp']) and np.array_equal(pts[nodes], T['tp']), 'near-subset mapping mismatch'
    assert np.array_equal(inv.reshape(sub.shape).astype(np.int32), T['tt'])
    print(f'near subset verified: {len(nodes)} nodes {time.time()-t0:.0f}s', flush=True)
    # outer surface and all boundary nodes
    F, N, bn_all, sinfo = outer_surface(pts, tets)        # bn_all: all boundary nodes (as eplib.boundary_faces)
    assert np.array_equal(bn_all[cKDTree(pts[bn_all]).query(elec_xyz)[1]], elec_nodes), 'electrode node rule mismatch'
    print('surface', sinfo, f'{time.time()-t0:.0f}s', flush=True)
    surf = Surface(pts, F, N)
    btree = cKDTree(pts[bn_all])
    # edge length on the surface (node spacing, reported)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    elen = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
    sinfo.update(edge_mm_median=float(np.median(elen)), edge_mm_p2_5=float(np.percentile(elen, 2.5)),
                 edge_mm_p97_5=float(np.percentile(elen, 97.5)))
    # draws
    rows = []
    for pid in pids:
        for sdi, sd in enumerate(SDS):
            for rep in range(NREP):
                g = np.random.default_rng(seed_of(pid, sdi, rep)).standard_normal((6, 2))
                for j in range(6):
                    rows.append((pid, sdi, rep, j, sd * g[j, 0], sd * g[j, 1]))
    R = np.array(rows)
    X0 = elec_xyz[np.array(VIDX)[R[:, 3].astype(int)]]
    Xe, Lw = walk(surf, X0, R[:, 4:6])
    dn, ni = btree.query(Xe)
    enode = bn_all[ni]
    # zero-offset check
    Xz, _ = walk(surf, elec_xyz[VIDX], np.zeros((6, 2)))
    zn = bn_all[btree.query(Xz)[1]]
    assert np.array_equal(zn, elec_nodes[VIDX]), (zn, elec_nodes[VIDX])
    euclid_node = np.linalg.norm(pts[enode] - X0, axis=1)
    euclid_walk = np.linalg.norm(Xe - X0, axis=1)
    print(f'draws {len(R)} walked {time.time()-t0:.0f}s; node snap distance median {np.median(dn):.2f} max {dn.max():.2f} mm',
          flush=True)
    uniq = np.unique(np.concatenate([enode, elec_nodes[VIDX]]))
    # heart-node locations in the near subset, for every requested heart
    ttree = cKDTree(T['tp'][T['tt']].mean(1))
    used = []
    for pid in pids:
        with h5py.File(ref_h5(pid), 'r') as f:
            hp = f['pts'][:]
        bt, lam, sc = locate(T['tp'], T['tt'].astype(np.int64), hp, tree=ttree)
        np.savez(os.path.join(scr, f'loc_{pid}.npz'), tet=bt.astype(np.int32), lam=lam, score=sc)
        used.append(np.unique(T['tt'][bt]))
    U = np.unique(np.concatenate(used))                       # near-subset indices
    print(f'U: {len(U)} near-subset nodes; {len(uniq)} distinct electrode nodes {time.time()-t0:.0f}s', flush=True)
    # stiffness, pinned at RL
    K = eplib.stiffness(pts, tets, sigma=eplib.SIG_BULK)
    ref = int(elec_nodes[-1])
    keep = np.ones(K.shape[0]); keep[ref] = 0
    Dk = sp.diags(keep)
    Ap = (Dk @ K @ Dk + sp.diags(1 - keep)).tocsr()
    sp.save_npz(os.path.join(scr, 'A.npz'), Ap, compressed=False)
    np.savez(os.path.join(scr, 'meta.npz'), R=R, Xe=Xe, Lw=Lw, enode=enode, snap_mm=dn, euclid_node=euclid_node,
             euclid_walk=euclid_walk, uniq=uniq, U=U, Ufull=nodes[U], ref=ref, elec_nodes=elec_nodes, elec_xyz=elec_xyz,
             pids=np.array(pids))
    json.dump(dict(surface=sinfo, n_draws=int(len(R)), n_unique_nodes=int(len(uniq)), n_U=int(len(U)),
                   snap_mm_median=float(np.median(dn)), snap_mm_max=float(dn.max()),
                   walk_vs_target_ratio_median=float(np.median(euclid_walk[Lw > 0] / Lw[Lw > 0])),
                   prep_s=time.time() - t0), open(os.path.join(scr, 'prep.json'), 'w'), indent=1)
    print(f'prep done {time.time()-t0:.0f}s', flush=True)


def solve(scr, wid, nw):
    import pyamg
    t0 = time.time()
    m = np.load(os.path.join(scr, 'meta.npz'))
    uniq = m['uniq']; Ufull = m['Ufull']; ref = int(m['ref'])
    mine = uniq[wid::nw]
    if len(mine) == 0:
        return
    Ap = sp.load_npz(os.path.join(scr, 'A.npz')).tocsr()
    np.random.seed(AMG_SEED)
    ml = pyamg.smoothed_aggregation_solver(Ap, symmetry='symmetric', max_coarse=500)
    tset = time.time() - t0
    out = np.zeros((len(mine), len(Ufull)), np.float32); its = []; rr = []
    n = Ap.shape[0]
    for j, s in enumerate(mine):
        b = np.zeros(n); b[s] = 1.0; b[ref] = 0.0
        res = []
        x = ml.solve(b, tol=1e-10, accel='cg', residuals=res, maxiter=500)
        out[j] = x[Ufull]; its.append(len(res)); rr.append(res[-1] / res[0])
    np.save(os.path.join(scr, f'Z_{wid:04d}.npy'), out)
    np.savez(os.path.join(scr, f'Zmeta_{wid:04d}.npz'), nodes=mine, its=np.array(its), relres=np.array(rr))
    print(f'worker {wid}: {len(mine)} solves, setup {tset:.0f}s, total {time.time()-t0:.0f}s, its {min(its)}-{max(its)}, '
          f'max relres {max(rr):.1e}', flush=True)


def assemble(scr, pid):
    import glob
    import h5py
    t0 = time.time()
    m = np.load(os.path.join(scr, 'meta.npz'))
    U = m['U']; R = m['R']
    sel = R[:, 0].astype(int) == pid
    Rp = R[sel]; enode = m['enode'][sel]
    # column lookup
    col = {}; Zs = []
    for k, f in enumerate(sorted(glob.glob(os.path.join(scr, 'Zmeta_*.npz')))):
        zm = np.load(f)
        for j, s in enumerate(zm['nodes']):
            col[int(s)] = (k, j)
        Zs.append(np.load(f.replace('Zmeta_', 'Z_').replace('.npz', '.npy'), mmap_mode='r'))

    def zcol(s):
        k, j = col[int(s)]; return np.asarray(Zs[k][j], np.float64)
    upos = -np.ones(int(U.max()) + 1, np.int64); upos[U] = np.arange(len(U))
    T = A.load_torso()
    loc = np.load(os.path.join(scr, f'loc_{pid}.npz'))
    tet = T['tt'][loc['tet']].astype(np.int64); lam = loc['lam']
    P = upos[tet]; assert (P >= 0).all()
    Zbase = np.einsum('qi,qim->qm', lam, T['Z'].astype(np.float64)[tet])          # (N, 10), as acdc_lib.ecg_weights
    with h5py.File(ref_h5(pid), 'r') as f:
        pts = f['pts'][:]; tets = f['tets'][:].astype(np.int64); fib = f['fib'][:].astype(np.float64); W0 = f['W'][:]
    Ki = eplib.stiffness(pts, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
    LM = eplib.lead_matrix()

    def weights(vnodes):
        Z = Zbase.copy()
        for j, s in enumerate(vnodes):
            Z[:, VIDX[j]] = np.einsum('qi,qi->q', lam, zcol(s)[P])
        return LM @ (Ki @ Z).T
    # zero-offset check: new solves at the original nodes against the stored W
    Wz = weights(m['elec_nodes'][VIDX])
    Wb = LM @ (Ki @ Zbase).T
    chk = dict(W_zero_vs_W_rel_maxdiff=float(np.abs(Wz - W0).max() / np.abs(W0).max()),
               W_base_vs_W_rel_maxdiff=float(np.abs(Wb - W0).max() / np.abs(W0).max()))
    Wd = np.zeros((len(SDS), NREP, 8, len(pts)), np.float32)
    nodes = np.zeros((len(SDS), NREP, 6), np.int64); xyz = np.zeros((len(SDS), NREP, 6, 3))
    geo = np.zeros((len(SDS), NREP, 6)); eu = np.zeros((len(SDS), NREP, 6)); seeds = np.zeros((len(SDS), NREP), np.int64)
    Xe = m['Xe'][sel]; Lw = m['Lw'][sel]; en = m['euclid_node'][sel]
    for sdi in range(len(SDS)):
        for rep in range(NREP):
            r = (Rp[:, 1].astype(int) == sdi) & (Rp[:, 2].astype(int) == rep)
            o = np.argsort(Rp[r, 3])
            vn = enode[r][o]
            Wd[sdi, rep] = weights(vn)
            nodes[sdi, rep] = vn; xyz[sdi, rep] = Xe[r][o]; geo[sdi, rep] = Lw[r][o]; eu[sdi, rep] = en[r][o]
            seeds[sdi, rep] = seed_of(pid, sdi, rep)
    os.makedirs(os.path.join(OUT, 'displaced'), exist_ok=True)
    fn = os.path.join(OUT, 'displaced', f'patient{pid:03d}_W_displaced.h5')
    rel = np.array([[np.linalg.norm(Wd[a, b] - W0) / np.linalg.norm(W0) for b in range(NREP)] for a in range(len(SDS))])
    with h5py.File(fn, 'w') as f:
        f.create_dataset('W_disp', data=Wd, compression='gzip', compression_opts=4)
        f['W_zero'] = Wz.astype(np.float32)
        f['sd_mm'] = np.array(SDS); f['seed'] = seeds; f['elec_node'] = nodes; f['elec_xyz_walk'] = xyz
        f['walk_mm'] = geo; f['euclid_from_original_node_mm'] = eu; f['W_rel_change'] = rel
        f.attrs.update(patient=pid, construction=os.path.relpath(ref_h5(pid), A.ROOT),
                       dims='W_disp[sd_index, rep, lead (I, II, V1-V6), node of the construction]',
                       seed_rule='20261028 + 1000*patient + 100*sd_index + rep; default_rng(seed).standard_normal((6,2)) for V1..V6',
                       displaced='V1-V6 (limb RA, LA, LL and reference RL fixed)',
                       solver='pyamg SA max_coarse 500, numpy seed 20261028 before setup, CG rel res 1e-10, pinned RL',
                       torso='Malik 2026 heart 09 full mesh, homogeneous 0.2 S/m, electrode rule v2', **chk)
    print(f'patient{pid}: {json.dumps(chk)} rel |W_disp - W| median by SD '
          f'{[round(float(np.median(rel[a])), 4) for a in range(len(SDS))]} {time.time()-t0:.0f}s', flush=True)
    return fn


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
    elif cmd == 'solve':
        solve(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == 'assemble':
        assemble(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
