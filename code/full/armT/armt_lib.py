"""Arm T (torso and electrode geometry) library, I11 full-arm run, owner armT, 2026-10-28.

Governing documents: final/AMENDMENT_2026-10-28.md (A2 fast layer, A4 prior, theta* and fibres), final/STUDY_PLAN.md
section 3 arm T and section 4 item 5. Reused code (copied, not edited): eplib.py from code/pilot/I13 (SHA-256 035c9368...),
carpio.py from code/gates/mesh (032f2524...), bayer_i09.py from code/gates/ka (c61bd95d...; it reads the I09 rule from
code/pilot/I09/02_case.py).

Hearts: 17 Malik 2026 HCM heart-in-torso models (ids '01'..'17', converted by the I13 pilot) and 5 KCL healthy torsos
(ids 'K1'..'K5', Zenodo 7253863, converted here). Patient id used in seeds: Malik NN -> NN, KCL k -> 200 + k.

Construction per heart (identical for all 22): 1.5 mm voxelisation of the deposited ventricles in the heart frame (pilot
code/pilot/I11/06_constructions.py, reference construction c0ref, no perturbation), voxel -> 6 tets, largest component;
Bayer 2012 fibres (I09 implementation) with numpy global seed 20261028 before the call (seeded pyamg); fast endocardial
layer per Amendment A2 (elements whose centroid lies within 1.5 mm of the LV or RV cavity surface); four root nodes by the
pilot rule on the native mesh, mapped to the nearest LV/RV endocardial node; own torso, electrode rule v2, homogeneous
0.2 S/m lead fields.
Units: mm, ms, m/s (= mm/ms).
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import eplib      # noqa: E402  (copy of code/pilot/I13/eplib.py)
import carpio     # noqa: E402  (copy of code/gates/mesh/carpio.py)

OUT = os.path.join(ROOT, 'results', '2026-10-28_armT')
DATA = os.path.join(OUT, 'data')
PILOT = os.path.join(ROOT, 'data', 'processed', 'pilot_I13')
MALIK = [f'{i:02d}' for i in range(1, 18)]
KCL = [f'K{i}' for i in range(1, 6)]
HEARTS = MALIK + KCL
REFS = ['09', '17']                  # plan section 3: Malik 09 (median torso volume), Malik 17 (75th percentile)
REGS = ['icp', 'trans']              # full rigid ICP; orientation-preserving translation only (PRIMARY)

VOX = 1.5
V_FIB = 0.6
D_FAST = 1.5                         # Amendment A2
R_ROOT = 2.5
SEED_BASE = 20261028
FIB_SEED = 20261028
B_LO, B_HI = np.array([0.5, 0.1]), np.array([4.0, 0.8])       # A4 prior
T_LO, T_HI = np.array([0.8, 0.15]), np.array([2.6, 0.45])     # A4 theta* box
NG = 9
G1 = np.exp(np.linspace(np.log(B_LO[0]), np.log(B_HI[0]), NG))
G2 = np.exp(np.linspace(np.log(B_LO[1]), np.log(B_HI[1]), NG))
STEP = np.array([np.log(B_HI[0] / B_LO[0]), np.log(B_HI[1] / B_LO[1])]) / (NG - 1)
TEND = 1000.0
LOSSES = [('corr', eplib.loss_corr), ('amp', eplib.loss_amp)]
N_HELD = 6


def pid_of(hid):
    return 200 + int(hid[1:]) if hid.startswith('K') else int(hid)


def group_of(hid):
    return 'healthy' if hid.startswith('K') else 'HCM'


def heart_file(hid):
    return os.path.join(DATA, f'heart_{hid}.h5')


def npz_of(hid):
    if hid.startswith('K'):
        return os.path.join(DATA, f'kcl{hid[1:]}.npz')
    return os.path.join(PILOT, f'malik{hid}.npz')


def theta_star(hid, n=3):
    """A4: Latin hypercube in log space over T_LO..T_HI, seed 20261028 + patient id (same algebra as the gates)."""
    rng = np.random.default_rng(SEED_BASE + pid_of(hid))
    u = np.empty((n, 2))
    for d in range(2):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return np.exp(np.log(T_LO) + u * (np.log(T_HI) - np.log(T_LO)))


def held_out(hid, n=N_HELD):
    rng = np.random.default_rng([SEED_BASE, pid_of(hid), 1])
    lo, hi = np.log(B_LO * 1.05), np.log(B_HI / 1.05)
    return np.exp(lo + rng.random((n, 2)) * (hi - lo))


def grid_thetas():
    return np.array([[a, b] for a in G1 for b in G2])


def bound_hit(th, rtol=1e-3):
    th = np.asarray(th)
    return bool(np.isclose(th, B_LO, rtol=rtol).any() or np.isclose(th, B_HI, rtol=rtol).any())


class CorrectedEmulator:
    """eplib.GridEmulator plus simple kriging of the nodal residual at refinement points (gates calib_lib, copied)."""

    def __init__(self, base, th_ref=None, resid=None, nugget=1e-6):
        self.base = base
        self.X = None
        if th_ref is not None and len(th_ref):
            self.X = np.log(np.asarray(th_ref, float)) / STEP
            d2 = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
            K = np.exp(-0.5 * d2) + nugget * np.eye(len(self.X))
            self.A = np.linalg.solve(K, np.asarray(resid, np.float64)).astype(np.float32)

    def __call__(self, th):
        t = self.base(th)
        if self.X is None:
            return t
        x = np.log(np.asarray(th, float)) / STEP
        k = np.exp(-0.5 * ((self.X - x) ** 2).sum(1)).astype(np.float32)
        return t + k @ self.A


def calibrate(emu, ecg, target, loss):
    return eplib.calibrate(emu, ecg, target, loss, (B_LO, B_HI))


# ------------------------------------------------------------------------------------------------ geometry
def closest_on_triangles(p, a, b, c):
    """Vectorised closest point on triangles (a, b, c) to points p (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = p - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den; w = vc / den
    out = a + ab * v[:, None] + ac * w[:, None]                     # interior
    # edge regions
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(m, d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0); out[m] = (a + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(m, d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0); out[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    den2 = (d4 - d3) + (d5 - d6)
    t = np.where(m, (d4 - d3) / np.where(den2 == 0, 1, den2), 0); out[m] = (b + (c - b) * t[:, None])[m]
    # vertex regions
    m = (d1 <= 0) & (d2 <= 0); out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3); out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6); out[m] = c[m]
    return out


def dist_to_surface(q, pts, tris, k=12):
    """Exact distance from points q to a triangle soup (pts[tris]), searching the k triangles with nearest centroids."""
    cen = pts[tris].mean(1)
    _, cand = cKDTree(cen).query(q, k=k)
    best = np.full(len(q), np.inf)
    for j in range(k):
        T = tris[cand[:, j]]
        cp = closest_on_triangles(q, pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]])
        best = np.minimum(best, np.linalg.norm(q - cp, axis=1))
    return best


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
