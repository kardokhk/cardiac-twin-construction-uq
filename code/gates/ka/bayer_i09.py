"""I09 pilot Bayer 2012 LDRB fibres as a callable function (I11 gates, owner ka, 2026-10-28).

The rule functions (egrad/emean are re-bound per mesh; unit, axis, orient, rot2quat, quat2rot, bislerp) are
extracted verbatim with `ast` from code/pilot/I09/02_case.py and executed unchanged. The glue that the pilot runs
at module level (P1 stiffness, the three Laplace solves with one AMG hierarchy, psi, fibre assembly) is copied
line for line into `bayer_fibres` with the pilot's mesh variables turned into arguments. SHA-256 of the source
file and of every extracted function is returned in the metadata.
"""
import ast
import hashlib
import os

import numpy as np
import pyamg
import scipy.sparse as sp

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pilot", "I09", "02_case.py")
NAMES = ("unit", "axis", "orient", "rot2quat", "quat2rot", "bislerp")
ALPHA_ENDO, ALPHA_EPI, BETA_ENDO, BETA_EPI = 40.0, -50.0, -65.0, 25.0   # as in 02_case.py


def _load(src=SRC):
    text = open(src).read()
    tree = ast.parse(text)
    segs = {n.name: ast.get_source_segment(text, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    ns = dict(np=np)
    for k in NAMES:
        exec(compile(segs[k], f"<I09:{k}>", "exec"), ns)
    meta = dict(i09_source=os.path.abspath(src),
                i09_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                i09_function_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in segs.items()})
    return ns, meta


def bayer_fibres(X, T, lv, rv, epi, apex, base, angles=(ALPHA_ENDO, ALPHA_EPI, BETA_ENDO, BETA_EPI)):
    """X (n,3) mm, T (ne,4) positively oriented; node index arrays. Returns per-element fibre (ne,3), fields."""
    ns, meta = _load()
    unit, axis, orient, bislerp = ns["unit"], ns["axis"], ns["orient"], ns["bislerp"]
    A_EN, A_EP, B_EN, B_EP = angles
    x0 = X[T[:, 0]]
    J = np.stack([X[T[:, 1]] - x0, X[T[:, 2]] - x0, X[T[:, 3]] - x0], 1)
    vol = np.linalg.det(J) / 6.0
    Ji = np.linalg.inv(J)
    G = np.empty((T.shape[0], 4, 3))
    G[:, 1:, :] = np.transpose(Ji, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(1)
    avol = np.abs(vol)
    n = X.shape[0]
    Ke = avol[:, None, None] * np.einsum("eik,ejk->eij", G, G)
    rows = np.repeat(T, 4, axis=1).ravel()
    cols = np.tile(T, (1, 4)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    del Ke, rows, cols

    def solve(dn, gv):
        mask = np.zeros(n, bool); mask[dn] = True
        I = np.where(~mask)[0]
        u = np.zeros(n); u[dn] = gv
        Kii = K[I][:, I].tocsr()
        rhs = -(K[I][:, dn] @ gv)
        ml = pyamg.smoothed_aggregation_solver(Kii, symmetry="symmetric")
        u[I] = ml.solve(rhs, tol=1e-10, accel="cg", maxiter=500)
        return u

    epi = np.setdiff1d(epi, np.union1d(lv, rv))
    dn = np.concatenate([lv, rv, epi])
    mask = np.zeros(n, bool); mask[dn] = True
    I = np.where(~mask)[0]
    Kii = K[I][:, I].tocsr(); Kid = K[I][:, dn]
    ml = pyamg.smoothed_aggregation_solver(Kii, symmetry="symmetric")
    fields = {}
    for name, g in (("lv", np.r_[np.ones(lv.size), np.zeros(rv.size + epi.size)]),
                    ("rv", np.r_[np.zeros(lv.size), np.ones(rv.size), np.zeros(epi.size)])):
        u = np.zeros(n); u[dn] = g
        u[I] = ml.solve(-(Kid @ g), tol=1e-10, accel="cg", maxiter=500)
        fields[name] = u
    fields["epi"] = 1.0 - fields["lv"] - fields["rv"]
    psi = solve(np.concatenate([apex, base]), np.r_[np.zeros(apex.size), np.ones(base.size)])

    def egrad(u):
        return np.einsum("ei,eij->ej", u[T], G)

    def emean(u):
        return u[T].mean(1)

    g_psi, g_lv, g_rv, g_epi = egrad(psi), egrad(fields["lv"]), egrad(fields["rv"]), egrad(fields["epi"])
    e_lv, e_rv, e_epi = emean(fields["lv"]), emean(fields["rv"]), emean(fields["epi"])
    depth = e_rv / np.maximum(e_lv + e_rv, 1e-12)
    a_s = A_EN * (1 - depth) - A_EN * depth
    b_s = B_EN * (1 - depth) - B_EN * depth
    a_w = A_EN * (1 - e_epi) + A_EP * e_epi
    b_w = B_EN * (1 - e_epi) + B_EP * e_epi
    Q_lv = orient(axis(g_psi, -g_lv), a_s, b_s)
    Q_rv = orient(axis(g_psi, g_rv), a_s, b_s)
    Q_endo = bislerp(Q_lv, Q_rv, depth)
    Q_epi = orient(axis(g_psi, g_epi), a_w, b_w)
    Q = bislerp(Q_endo, Q_epi, e_epi)
    fib = Q[:, :, 0]
    nonfinite = int((~np.isfinite(fib)).any(1).sum())
    fib = unit(np.nan_to_num(fib, nan=1.0))
    Qe = axis(g_psi, g_epi)
    out = dict(fib=fib, psi=psi, phi_lv=fields["lv"], phi_rv=fields["rv"], phi_epi=fields["epi"],
               depth=depth, e_epi=e_epi, frame=Qe, g_psi=g_psi, g_epi=g_epi, nonfinite=nonfinite)
    return out, meta
