"""Full arm C and arm N library (owner calibC, I11 full arm, 2026-10-28). Copied from code/gates/calib/calib_lib.py
(gates owner calib) and changed only where Amendment 1 (final/AMENDMENT_2026-10-28.md) or the v2 constructions require:
  * constructions from data/processed/acdc_mesh_v2 (MANIFEST.tsv); reference = expert, 1.5 mm, RV wall 3 mm; A2 fast
    layer ('etag') via code/full/mesh/acdc_lib.py;
  * arm N comparators (A3): white and lp40 at 10-30 dB (unchanged definitions), baseline wander 5/10/20% and
    electrode displacement 5/10/20 mm (definitions below); per-replicate seeds keyed by
    (20261028, patient, theta* index, comparator index, level, replicate).
Everything else (prior, theta*, grid, emulator, refinement, losses, bound rule, noise SNR rule) is unchanged.
BASELINE WANDER (A3.2, assumption where A3 is silent): window t in [0, max target activation + 10 ms] (the QRS window
    of the SNR rule); s = t / t_w in [0, 1]; per lead d(s) = c0 + c1 s + c2 s^2, c ~ N(0, 1) i.i.d.; d rescaled so that
    max |d| over the window = level x (max - min of that lead over the window); after the window the drift is held at
    its end value (continuous); the whole drift is added to the target.
ELECTRODE DISPLACEMENT (A3.3): target = ECGModel(W_disp[sd, rep], tmax)(T*), W_disp from
    data/processed/acdc_mesh_v2/displaced (owner mesh2; replicate seed 20261028 + 1000 patient + 100 sd_index + rep);
    calibrated with the undisplaced stored W and the undisplaced reference emulator. Replicate rep uses the same
    displaced lead field for all three theta*.

Gates text follows.

Parameters theta = (v_endo, v_t) in m/s; fibre velocity fixed at 0.6 m/s (acdc_lib.V_FIB).

PRIOR (declared before any run of this script; widened from the pilot's v_endo 0.8-3.0, v_t 0.15-0.5 m/s, because 19
of 102 pilot correlation-loss estimates and 18 of 102 amplitude-loss estimates sat on the v_t upper bound 0.5):
    v_endo in [0.5, 4.0] m/s, v_t in [0.1, 0.8] m/s (box in log space; 9 x 9 log grid).
    v_t > 0.6 m/s (= v_fib) means a fibre/transverse ratio below 1. It is allowed so that bound hits reflect the data
    rather than a box edge; such estimates are flagged (flag_vt_gt_vfib).
THETA* REGION (Latin hypercube, 3 draws per heart, in log space, seed 20261028 + patient id):
    v_endo in [0.8, 2.6] m/s, v_t in [0.15, 0.45] m/s (inside the prior with log margins of 0.41 to 0.58 on every side;
    the pilot drew theta* from 1.0-2.4 and 0.2-0.42, with a v_t upper margin of only 0.17).
CALIBRATION: pilot logic (code/pilot/I11/07_calibrate_cons.py, eplib.calibrate unchanged: 25 x 25 log grid search then
    Nelder-Mead in log space, clipped to the prior). Emulator: eplib.GridEmulator (tensor cubic spline over the 9 x 9
    log grid of nodal activation times) plus a local discrepancy term fitted to up to 9 refinement runs (simple kriging
    of the nodal residual T_sim - T_emu, Gaussian kernel with length scale one grid step per log-parameter, nugget 1e-6).
    Refinement schedule per construction: round 1 = one run at each distinct theta-hat of round 0 (3 targets x 2 losses,
    up to 6); round 2 = one run at each distinct primary-loss theta-hat of round 1 (up to 3). Points closer than 0.01 in
    log units to an existing refinement point are not rerun.
LOSSES: 'corr' = 1 - mean Pearson r over 8 leads (PRIMARY); 'amp' = relative L2 (eplib.loss_amp).
BOUND HIT: theta-hat within relative 1e-3 of a prior bound in either parameter; such estimates are censored.
NOISE (arm N): SNR_dB per lead = 10 log10(P_l / sigma_l^2), P_l = mean of x_l(t)^2 over the QRS window
    t in [0, max target activation + 10 ms] (samples every 0.5 ms). Zero-mean Gaussian noise with that sigma_l is added to
    every sample of lead l (the whole target vector). 'white' (primary): i.i.d. per sample. 'lp40' (sensitivity): white
    noise low-pass filtered (4th-order Butterworth, 40 Hz, zero phase) and rescaled to the same per-lead variance.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'full', 'mesh'))
import acdc_lib as A          # noqa: E402
import sax_route              # noqa: E402
eplib = A.eplib

B_LO, B_HI = np.array([0.5, 0.1]), np.array([4.0, 0.8])
T_LO, T_HI = np.array([0.8, 0.15]), np.array([2.6, 0.45])
NG = 9
G1 = np.exp(np.linspace(np.log(B_LO[0]), np.log(B_HI[0]), NG))
G2 = np.exp(np.linspace(np.log(B_LO[1]), np.log(B_HI[1]), NG))
STEP = np.array([np.log(B_HI[0] / B_LO[0]), np.log(B_HI[1] / B_LO[1])]) / (NG - 1)
SEED_BASE = 20261028
SNRS = [10, 15, 20, 25, 30]
NREP = 20
NOISE_TYPES = ['white', 'lp40']
# arm N comparators: (name, levels, unit); index in this list is the seed key 'comparator'
COMPARATORS = [('white', [10, 15, 20, 25, 30], 'dB SNR'), ('lp40', [10, 15, 20, 25, 30], 'dB SNR'),
               ('wander', [5, 10, 20], '% of lead QRS peak-to-peak'), ('disp', [5, 10, 20], 'mm SD')]
MESH_DIR = os.path.join(ROOT, 'data', 'processed', 'acdc_mesh_v2')
LOSSES = [('corr', eplib.loss_corr), ('amp', eplib.loss_amp)]
SOURCES = A.SOURCES
VOXES = [1.5, 1.0]
TEND = 1000.0
DT = 0.5


def cons_name(pid, source, vox, rv=3.0):
    return f'patient{pid:03d}__{source}__{vox:.1f}__rv{rv:.1f}'


def cons_path(name):
    return os.path.join(MESH_DIR, 'cons', name + '.h5')


def disp_path(pid):
    return os.path.join(MESH_DIR, 'displaced', f'patient{pid:03d}_W_displaced.h5')


def ref_name(pid):
    return cons_name(pid, 'expert', 1.5, 3.0)


def rep_seed(pid, istar, comp_idx, level, rep):
    return [SEED_BASE, int(pid), int(istar), int(comp_idx), int(level), int(rep)]


def grid_thetas():
    return np.array([[a, b] for a in G1 for b in G2])


def theta_star(pid, n=3):
    """Latin hypercube in log space over the theta* region; seed SEED_BASE + pid."""
    rng = np.random.default_rng(SEED_BASE + pid)
    u = np.empty((n, 2))
    for d in range(2):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return np.exp(np.log(T_LO) + u * (np.log(T_HI) - np.log(T_LO)))


def bound_hit(th, rtol=1e-3):
    th = np.asarray(th)
    lo = np.isclose(th, B_LO, rtol=rtol); hi = np.isclose(th, B_HI, rtol=rtol)
    return lo, hi


class CorrectedEmulator:
    """GridEmulator plus simple kriging of the nodal residual at refinement points (exact at those points up to the
    nugget; decays to the plain emulator one to two grid steps away)."""

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


def qrs_window(act_target):
    return float(np.max(act_target)) + 10.0


def add_noise(target, act_max_ms, snr_db, kind, rng):
    """target (8, nt) at DT ms; returns noisy copy and per-lead sigma."""
    nt = target.shape[1]
    t = np.arange(nt) * DT
    w = t <= act_max_ms + 10.0
    P = (target[:, w] ** 2).mean(1)
    sig = np.sqrt(P / 10 ** (snr_db / 10))
    z = rng.standard_normal(target.shape)
    if kind == 'lp40':
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, 40.0, btype='low', fs=1000.0 / DT, output='sos')
        z = sosfiltfilt(sos, z, axis=1)
        z = z / z.std(1, keepdims=True)
    elif kind != 'white':
        raise ValueError(kind)
    return target + sig[:, None] * z, sig


def add_wander(target, act_max_ms, frac, rng):
    """Quadratic baseline drift per lead (see docstring). frac = peak deviation / lead peak-to-peak over the window.
    Returns noisy copy and the drift."""
    nt = target.shape[1]
    t = np.arange(nt) * DT
    tw = act_max_ms + 10.0
    w = t <= tw
    s = np.minimum(t, tw) / tw
    c = rng.standard_normal((target.shape[0], 3))
    d = c[:, :1] + c[:, 1:2] * s[None] + c[:, 2:3] * s[None] ** 2
    ptp = target[:, w].max(1) - target[:, w].min(1)
    pk = np.abs(d[:, w]).max(1)
    d = d * (frac * ptp / pk)[:, None]
    return target + d, d


def lengths_of(h5path):
    import h5py
    with h5py.File(h5path, 'r') as f:
        L = f['lab'][:]; vox = float(f.attrs['vox'])
    return sax_route.lengths(L, vox)
