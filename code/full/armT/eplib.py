"""Shared library for pilots I13 and I11 (2026-10-22).

Forward model: openCARP v20.0 eikonal activation times (computed elsewhere) -> transmembrane potential
template Vm(t) = V0 + A * sigmoid((t - ta)/tau) -> lead-field ECG
    phi_e(t) - phi_ref(t) = - sum_n w_en Vm_n(t),   w_e = K_i Z_e,
where K_i is the intracellular (anisotropic) stiffness matrix over myocardial elements and Z_e the
lead field of electrode e (unit current injected at e, withdrawn at the reference electrode RL) in a
homogeneous isotropic volume conductor (the whole torso mesh). Because sum_n w_en = 0, the ECG is a
convolution of the template derivative with a weighted histogram of activation times, so a forward
ECG costs milliseconds once activation times are known.

All lengths in mm, velocities in m/s (= mm/ms), times in ms.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

SIG_IL, SIG_IT = 0.17, 0.019   # intracellular conductivities (S/m), Niederer 2011 values
SIG_BULK = 0.2                 # homogeneous torso conductivity (S/m), assumption
LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ELECS = ['RA', 'LA', 'LL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'RL']


# ----------------------------------------------------------------------------- FEM
def tet_grads(pts, tets):
    """Return (vol (M,), G (M,4,3)) gradients of P1 barycentric functions."""
    X = pts[tets]                                   # (M,4,3)
    D = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)  # columns
    det = np.linalg.det(D)
    Dinv = np.linalg.inv(D)                          # rows are grads of lambda1..3
    G = np.empty((len(tets), 4, 3))
    G[:, 1:, :] = Dinv
    G[:, 0, :] = -Dinv.sum(axis=1)
    return np.abs(det) / 6.0, G


def stiffness(pts, tets, sigma=1.0, fib=None, sig_l=None, sig_t=None, chunk=2_000_000):
    """Assemble P1 stiffness. Isotropic sigma (scalar or per-element) or transversely isotropic with fib."""
    n = len(pts)
    rows, cols, vals = [], [], []
    for s in range(0, len(tets), chunk):
        T = tets[s:s + chunk]
        vol, G = tet_grads(pts, T)
        if fib is None:
            sg = sigma if np.isscalar(sigma) else sigma[s:s + chunk]
            Ke = np.einsum('mik,mjk->mij', G, G) * (vol * sg)[:, None, None]
        else:
            f = fib[s:s + chunk]
            f = f / np.linalg.norm(f, axis=1, keepdims=True)
            Gf = np.einsum('mik,mk->mi', G, f)
            Ke = (sig_t * np.einsum('mik,mjk->mij', G, G) + (sig_l - sig_t) * Gf[:, :, None] * Gf[:, None, :])
            Ke *= vol[:, None, None]
        r = np.repeat(T, 4, axis=1); c = np.tile(T, (1, 4))
        rows.append(r.ravel().astype(np.int32)); cols.append(c.ravel().astype(np.int32)); vals.append(Ke.ravel())
        del Ke, G
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    K.sum_duplicates()
    return K


def boundary_faces(tets):
    f = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    fs = np.sort(f, axis=1)
    _, idx, cnt = np.unique(fs, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def solve_leadfields(K, src_nodes, ref_node, tol=1e-10, verbose=True):
    """Z[:, e] solves K Z = e_src - e_ref with Z[ref]=0. Uses pyamg SA-preconditioned CG."""
    import pyamg
    n = K.shape[0]
    keep = np.ones(n); keep[ref_node] = 0
    Dk = sp.diags(keep)
    A = (Dk @ K @ Dk + sp.diags(1 - keep)).tocsr()
    ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric', max_coarse=500)
    Z = np.zeros((n, len(src_nodes)))
    for j, s in enumerate(src_nodes):
        b = np.zeros(n); b[s] = 1.0; b[ref_node] = 0.0
        res = []
        x = ml.solve(b, tol=tol, accel='cg', residuals=res, maxiter=500)
        Z[:, j] = x
        if verbose:
            print(f'   leadfield {j}: {len(res)} it, rel res {res[-1]/res[0]:.1e}', flush=True)
    return Z


# ----------------------------------------------------------------------------- electrodes
def _ray_hit(orig, d, tri_pts, last=False):
    """Nearest positive intersection of ray orig + t d with triangles (Moller-Trumbore). Returns point or None."""
    v0, v1, v2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2); det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    tv = orig - v0
    u = np.einsum('ij,ij->i', tv, p) * inv
    q = np.cross(tv, e1)
    v = (q @ d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    if not m.any():
        return None
    return orig + (t[m].max() if last else t[m].min()) * d


def place_electrodes_v2(pts, bfaces, heart_centroid, spine_pts=None):
    """Rule v2 (heart-anchored; adopted 2026-10-22 after rule v1 was found to tie limb electrodes to the CT field of
    view and to send V6 to the arm or the wrong side in some torsos). LPS coordinates, z_h = ventricular centroid.
      V1, V2: anterior rays at x_m -/+ 25 mm, z_h + 20 mm; V3: x_m + 57, z_h + 10; V4: x_m + 90, z_h
      V5: first body-surface exit of the ray from the ventricular centroid toward (+x, -y) at 45 deg, z_h
      V6: first body-surface exit of the ray from the ventricular centroid toward +x (patient left), z_h
      RA, LA: anterior rays at x_m -/+ 70 mm, z = min(z_h + 150, z_top - 30)
      LL, RL: anterior rays at x_m +/- 70 mm, z = max(z_h - 180, z_bot + 30)
    x_m = x of the vertebral column (tag 4 nodes within 20 mm of that height; mean of tag 4 if none), else the
    mid-point of the torso's x extent at that height."""
    tri = pts[bfaces]
    bn = np.unique(bfaces); P = pts[bn]
    zc = heart_centroid[2]
    ztop, zbot = P[:, 2].max(), P[:, 2].min()

    def trunk_centre(z):
        s = P[np.abs(P[:, 2] - z) < 3]
        c = np.array([(s[:, 0].min() + s[:, 0].max()) / 2, (s[:, 1].min() + s[:, 1].max()) / 2])
        if spine_pts is not None:
            sp = spine_pts[np.abs(spine_pts[:, 2] - z) < 20]
            sp = sp if len(sp) else spine_pts
            c = np.array([sp[:, 0].mean(), c[1]])     # x from the spine, y from the slice bounding box
        return c

    def ant_retry(dx, z, zdir):
        for k in range(12):
            for j in range(4):
                zz = z + zdir * 10 * k
                c = trunk_centre(zz)
                # ray from behind the body toward anterior; the LAST crossing is the anterior skin (robust to
                # internal mesh boundaries and to the start point)
                h = _ray_hit(np.array([c[0] + np.sign(dx) * max(abs(dx) - 10 * j, 0), P[:, 1].max() + 10, zz]),
                             np.array([0, -1.0, 0]), tri, last=True)
                if h is not None:
                    return h
        raise RuntimeError('electrode placement failed')

    def from_heart(d):
        o = np.array(heart_centroid, dtype=float); d = np.asarray(d, float) / np.linalg.norm(d)
        h = _ray_hit(o, d, tri)
        if h is None:
            raise RuntimeError('electrode placement failed')
        return h
    E = {}
    E['V1'] = ant_retry(-25, zc + 20, -1); E['V2'] = ant_retry(25, zc + 20, -1)
    E['V3'] = ant_retry(57, zc + 10, -1); E['V4'] = ant_retry(90, zc, -1)
    E['V5'] = from_heart([1, -1, 0]); E['V6'] = from_heart([1, 0, 0])
    zt = min(zc + 150, ztop - 30); zb = max(zc - 180, zbot + 30)
    E['RA'] = ant_retry(-70, zt, -1); E['LA'] = ant_retry(70, zt, -1)
    E['LL'] = ant_retry(70, zb, +1); E['RL'] = ant_retry(-70, zb, +1)
    xyz = np.array([E[k] for k in ELECS])
    node = bn[cKDTree(P).query(xyz)[1]]
    return xyz, node


def place_electrodes(pts, bfaces, heart_centroid):
    """Rule-based 12-lead (Mason-Likar torso-limb) placement on a torso in LPS coordinates (x left, y posterior,
    z superior). Convention (assumption, not anatomy-recorded):
      precordial heights: V1,V2 at z_h+20 mm; V3 at z_h+10; V4-V6 at z_h (z_h = ventricular centroid height)
      x positions from the midline x_m (centre of the trunk bounding box at that height):
         V1 x_m-25, V2 x_m+25, V3 x_m+57, V4 x_m+90 (anterior rays, +y -> -y direction),
         V5 ray from trunk centre at 60 deg (anterior->left), V6 at 90 deg (left lateral)
      RA, LA: z_top-40 mm, x_m -/+ 80 mm, anterior rays; LL, RL: z_bottom+40 mm, x_m +/- 80 mm, anterior rays.
    """
    tri = pts[bfaces]
    zc = heart_centroid[2]
    bn = np.unique(bfaces); P = pts[bn]
    ztop, zbot = P[:, 2].max(), P[:, 2].min()

    def trunk_centre(z):
        s = P[np.abs(P[:, 2] - z) < 3]
        return np.array([(s[:, 0].min() + s[:, 0].max()) / 2, (s[:, 1].min() + s[:, 1].max()) / 2])

    def anterior(x, z):
        c = trunk_centre(z)
        o = np.array([x, c[1], z]); h = _ray_hit(o, np.array([0, -1.0, 0]), tri)
        return h

    def radial(ang_deg, z):
        c = trunk_centre(z); a = np.radians(ang_deg)
        d = np.array([np.sin(a), -np.cos(a), 0.0])
        return _ray_hit(np.array([c[0], c[1], z]), d, tri)

    E = {}
    xm = lambda z: trunk_centre(z)[0]

    def ant_retry(dx, z, zdir):
        # if the anterior ray misses (narrow neck/pelvis, cropped field of view), move 10 mm toward the midline,
        # then 10 mm toward the heart, until it hits (fallback logged in the returned flag)
        for k in range(12):
            for j in range(4):
                zz = z + zdir * 10 * k
                h = anterior(xm(zz) + np.sign(dx) * max(abs(dx) - 10 * j, 0), zz)
                if h is not None:
                    return h, (k, j)
        raise RuntimeError('electrode placement failed')

    def rad_retry(ang, z):
        for k in range(10):
            h = radial(ang - 5 * k, z)
            if h is not None:
                return h, k
        raise RuntimeError('electrode placement failed')
    fb = {}
    E['V1'], fb['V1'] = ant_retry(-25, zc + 20, -1)
    E['V2'], fb['V2'] = ant_retry(25, zc + 20, -1)
    E['V3'], fb['V3'] = ant_retry(57, zc + 10, -1)
    E['V4'], fb['V4'] = ant_retry(90, zc, -1)
    E['V5'], fb['V5'] = rad_retry(60, zc)
    E['V6'], fb['V6'] = rad_retry(90, zc)
    zt, zb = ztop - 40, zbot + 40
    E['RA'], fb['RA'] = ant_retry(-80, zt, -1); E['LA'], fb['LA'] = ant_retry(80, zt, -1)
    E['LL'], fb['LL'] = ant_retry(80, zb, +1); E['RL'], fb['RL'] = ant_retry(-80, zb, +1)
    print('   electrode fallbacks (steps):', {k: v for k, v in fb.items() if v not in (0, (0, 0))}, flush=True)
    xyz = np.array([E[k] for k in ELECS])
    node = bn[cKDTree(P).query(xyz)[1]]
    return xyz, node


# ----------------------------------------------------------------------------- ECG
def lead_matrix():
    """8 x 10 matrix mapping electrode potentials (ELECS order) to LEADS."""
    M = np.zeros((8, 10)); ix = {k: i for i, k in enumerate(ELECS)}
    M[0, ix['LA']] = 1; M[0, ix['RA']] = -1
    M[1, ix['LL']] = 1; M[1, ix['RA']] = -1
    for j, v in enumerate(['V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
        M[2 + j, ix[v]] = 1
        for w in ('RA', 'LA', 'LL'):
            M[2 + j, ix[w]] -= 1 / 3
    return M


def ecg_weights(Ki, Z_elec):
    """Ki: intracellular stiffness on heart nodes (N x N); Z_elec: (N, 10) lead fields at heart nodes
    (RL column zero). Returns W (8, N) lead weights so that ECG_l(t) = -sum_n W_ln Vm_n(t)."""
    Wel = (Ki @ Z_elec).T          # (10, N)
    return lead_matrix() @ Wel


class ECGModel:
    """ECG from activation times via weighted histogram convolved with the upstroke template derivative."""

    def __init__(self, W, tmax=200.0, dt=0.5, tau=1.0, amp=110.0):
        self.W = np.asarray(W, dtype=np.float64)
        self.dt, self.nt = dt, int(round(tmax / dt)) + 1
        self.t = np.arange(self.nt) * dt
        k = np.arange(-int(8 * tau / dt), int(8 * tau / dt) + 1) * dt
        s = 1 / (1 + np.exp(-k / tau))
        self.kern = amp * s * (1 - s) / tau * dt   # derivative of sigmoid template, integrated over a bin

    def __call__(self, ta):
        ta = np.asarray(ta, dtype=np.float64)
        x = np.clip(ta / self.dt, 0, self.nt - 2)
        i0 = np.floor(x).astype(np.int64); f = x - i0
        out = np.empty((self.W.shape[0], self.nt))
        for l in range(self.W.shape[0]):
            h = np.bincount(i0, self.W[l] * (1 - f), minlength=self.nt) + \
                np.bincount(i0 + 1, self.W[l] * f, minlength=self.nt)
            # ECG = -sum w Vm ; dVm/dt contributes via h * kern ; integrate in time
            out[l] = -np.cumsum(np.convolve(h[: self.nt], self.kern, mode='same'))
        return out


def loss_corr(sim, tgt):
    a = sim - sim.mean(1, keepdims=True); b = tgt - tgt.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-30)
    return 1.0 - r.mean()


def loss_amp(sim, tgt):
    return ((sim - tgt) ** 2).sum() / (tgt ** 2).sum()


# ----------------------------------------------------------------------------- emulator
class GridEmulator:
    """Tensor-product cubic-spline interpolation of nodal activation times over a 2D log-parameter grid."""

    def __init__(self, g1, g2, T):
        from scipy.interpolate import CubicSpline
        self.g1, self.g2 = np.log(g1), np.log(g2)
        self.T = T.reshape(len(g1) * len(g2), -1).astype(np.float32)   # row index i*len(g2)+j
        self.S1 = CubicSpline(self.g1, np.eye(len(g1)), bc_type='not-a-knot')
        self.S2 = CubicSpline(self.g2, np.eye(len(g2)), bc_type='not-a-knot')

    def __call__(self, th):
        a = self.S1(np.log(th[0])); b = self.S2(np.log(th[1]))
        w = np.outer(a, b).ravel().astype(np.float32)
        return w @ self.T


def calibrate(emu, ecg, target, loss, bounds, ngrid=25):
    """Grid search then Nelder-Mead in log space. Returns theta_hat, loss."""
    from scipy.optimize import minimize
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    g1 = np.linspace(lo[0], hi[0], ngrid); g2 = np.linspace(lo[1], hi[1], ngrid)
    best = (np.inf, None)
    vals = np.empty((ngrid, ngrid))
    for i, a in enumerate(g1):
        for j, b in enumerate(g2):
            v = loss(ecg(emu(np.exp([a, b]))), target); vals[i, j] = v
            if v < best[0]:
                best = (v, np.array([a, b]))
    f = lambda x: loss(ecg(emu(np.exp(np.clip(x, lo, hi)))), target) + 1e3 * np.sum(np.maximum(0, x - hi) + np.maximum(0, lo - x))
    r = minimize(f, best[1], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-10, maxiter=400))
    return np.exp(np.clip(r.x, lo, hi)), r.fun


# ----------------------------------------------------------------------------- geometry helpers
def locate_and_interp(src_pts, src_tets, values, query, k=12):
    """Linear (barycentric) interpolation of nodal `values` (Nsrc, m) at `query` points. Points outside
    all candidate tets fall back to the candidate with the least-negative barycentric coordinate."""
    cent = src_pts[src_tets].mean(1)
    tree = cKDTree(cent)
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
    T = src_tets[best_tet]
    out = np.einsum('qi,qim->qm', best_lam, values[T])
    return out, best_score


def kabsch(A, B):
    """Rigid R,t minimising |R A + t - B| for paired points."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def icp(src, dst, R0, t0, iters=50):
    tree = cKDTree(dst); R, t = R0, t0
    for _ in range(iters):
        moved = src @ R.T + t
        d, j = tree.query(moved)
        keep = d < np.percentile(d, 90)
        R, t = kabsch(src[keep], dst[j[keep]])
    moved = src @ R.T + t
    d, _ = tree.query(moved)
    return R, t, float(np.sqrt((d ** 2).mean()))
