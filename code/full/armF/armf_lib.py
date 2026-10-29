"""Arm F (model-form comparator) library, owner armF, I11 full run, 2026-10-28.

Copied, not imported, from upstream (SHA-256 of the sources at copy time):
  acdc_lib.py, carpio.py, sax_route.py  <- code/full/mesh/ (55457320..., 032f2524..., e3eecec4...), unchanged.
  Constants and functions marked [calib_lib] <- code/gates/calib/calib_lib.py (bf9d937a...), unchanged.
eplib comes from code/pilot/I13/eplib.py through acdc_lib's sys.path entry (read only).

Design (Amendment 1, A4; study plan section 3, arm F):
  * Hearts: 10 ACDC test hearts, 2 per Info.cfg group, lowest patient ids whose expert 1.5 mm rv 3.0 reference
    construction passes G2 (data/processed/acdc_mesh_v2/MANIFEST.tsv). Construction: expert labels, 1.5 mm, A2 layer
    (dataset 'etag'), RV wall 3 mm.
  * theta*: 3 per heart, Latin hypercube in log space, region v_endo 0.8-2.6, v_t 0.15-0.45 m/s, seed 20261028 + id
    (the same draws as arm C).
  * Targets ('re'): openCARP v20.0 reaction-eikonal RE+ (dream.solve 1), monodomain, tenTusscherPanfilov, default
    conductivities, dt 25 us, Vm every 0.5 ms, layout default0, renumbering 0, run to the unit's tmax.
    ECG_l(t) = -sum_n W_ln Vm_n(t) with the construction's own W, minus its value at t = 0 (resting offset).
  * Control targets ('tmpl'): pure eikonal at theta* through the activation-template ECG (exactly the arm C target).
    theta_hat/theta* of the control measures emulator plus optimiser error; the RE minus control contrast isolates
    model form.
  * Emulator: 9 x 9 log grid of nodal activation times (pure eikonal, tend 1000 ms) on the same construction, cubic
    spline plus kriging of refinement-run residuals [calib_lib]. Refinement: round 1 at every distinct theta_hat of
    round 0 (all target sets, both losses, QRS window), round 2 at every distinct primary-loss theta_hat of round 1
    (QRS window); points within 0.01 log units of an existing point are not rerun.
  * Losses: 'corr' 1 - mean Pearson r over 8 leads (primary), 'amp' relative L2.
  * Windows: 'qrs' (primary for arm F): samples t <= max target activation + 10 ms (the A3 QRS window rule);
    for RE targets the activation is the later of openCARP's act.igb and the Vm upstroke (-40 mV crossing).
    'full': 0 to tmax (the arm C ECG vector), as a sensitivity: it includes early repolarisation of the RE target,
    which the template ECG cannot represent.
  * tmax of a unit: max over its grid runs and control-target runs of finite activation + 20 ms, rounded up to 0.5 ms.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import acdc_lib as A          # noqa: E402  (code/full/armF/acdc_lib.py, copy of code/full/mesh/acdc_lib.py)
carpio = A.carpio
eplib = A.eplib

# ---------------------------------------------------------------- [calib_lib] constants
B_LO, B_HI = np.array([0.5, 0.1]), np.array([4.0, 0.8])
T_LO, T_HI = np.array([0.8, 0.15]), np.array([2.6, 0.45])
NG = 9
G1 = np.exp(np.linspace(np.log(B_LO[0]), np.log(B_HI[0]), NG))
G2 = np.exp(np.linspace(np.log(B_LO[1]), np.log(B_HI[1]), NG))
STEP = np.array([np.log(B_HI[0] / B_LO[0]), np.log(B_HI[1] / B_LO[1])]) / (NG - 1)
SEED_BASE = 20261028
LOSSES = [('corr', eplib.loss_corr), ('amp', eplib.loss_amp)]
WINDOWS = ['qrs', 'full']
TEND_EIK = 1000.0
DT = 0.5
RE_DT_US = 25.0
MANIFEST = os.path.join(ROOT, 'data', 'processed', 'acdc_mesh_v2', 'MANIFEST.tsv')


def grid_thetas():
    return np.array([[a, b] for a in G1 for b in G2])


def theta_star(pid, n=3):
    """[calib_lib] Latin hypercube in log space over the theta* region; seed SEED_BASE + pid."""
    rng = np.random.default_rng(SEED_BASE + pid)
    u = np.empty((n, 2))
    for d in range(2):
        u[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return np.exp(np.log(T_LO) + u * (np.log(T_HI) - np.log(T_LO)))


def bound_hit(th, rtol=1e-3):
    """[calib_lib]"""
    th = np.asarray(th)
    lo = np.isclose(th, B_LO, rtol=rtol); hi = np.isclose(th, B_HI, rtol=rtol)
    return lo, hi


class CorrectedEmulator:
    """[calib_lib] GridEmulator plus simple kriging of the nodal residual at refinement points."""

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
    """[calib_lib]"""
    return eplib.calibrate(emu, ecg, target, loss, (B_LO, B_HI))


# ---------------------------------------------------------------- arm F additions
def select_hearts(manifest=MANIFEST, per_group=2):
    """Lowest patient ids per Info.cfg group whose expert 1.5 mm rv 3.0 reference passes G2."""
    import csv
    rows = list(csv.DictReader(open(manifest), delimiter='\t'))
    ok = {}
    for r in rows:
        if (r['is_reference'] == 'True' and r['g2_pass'] == 'True' and r['source'] == 'expert'
                and float(r['resolution_mm']) == 1.5 and float(r['rv_wall_mm']) == 3.0 and r['smoke_ok'] == 'True'):
            ok.setdefault(r['group'], []).append(int(r['heart']))
    out = []
    for g in sorted(ok):
        out += sorted(ok[g])[:per_group]
    return out, {g: sorted(ok[g])[:per_group] for g in sorted(ok)}


def cons_path(pid, vox):
    return os.path.join(ROOT, 'data', 'processed', 'acdc_mesh_v2', 'cons',
                        f'patient{pid:03d}__expert__{vox:.1f}__rv3.0.h5')


def regions(th):
    return [(1, A.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])]


def tmax_of(acts):
    a = np.asarray(acts, np.float64)
    a = a[(a >= 0) & (a < 1e6)]
    return float(np.ceil((a.max() + 20.0) / DT) * DT)


def re_ecg(vm, W):
    """vm (T, N) -> ECG (8, T), resting offset (value at t = 0) removed."""
    e = -(np.asarray(W, np.float64) @ vm.T.astype(np.float64))
    return e - e[:, :1]


def upstroke_act(vm, thr=-40.0):
    up = vm > thr
    ok = up.any(0)
    first = up.argmax(0)
    t = np.where(ok, first * DT, np.nan)
    return t, int((~ok).sum())


def qrs_nw(act_max):
    """Number of samples with t <= act_max + 10 ms (A3 QRS window rule)."""
    return int(np.floor((float(act_max) + 10.0) / DT + 1e-9)) + 1


class WinECG:
    """ECGModel truncated to the first nw samples (the template ECG is causal: a cumulative sum)."""

    def __init__(self, ecg, nw):
        self.ecg, self.nw = ecg, int(nw)

    def __call__(self, ta):
        return self.ecg(ta)[:, :self.nw]


def lead_r(a, b):
    return [float(np.corrcoef(a[l], b[l])[0, 1]) for l in range(a.shape[0])]
