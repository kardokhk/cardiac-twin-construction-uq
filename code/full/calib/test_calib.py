"""Known-answer tests for calib_lib (gates owner calib; extended by owner calibC for A3, 2026-10-28). Run: python -I -B test_calib.py"""
import os, sys, importlib.util
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_lib as C

# 1. theta* inside the theta* region, LHS strata, deterministic
for p in (101, 102, 104):
    t = C.theta_star(p)
    assert np.all(t >= C.T_LO) and np.all(t <= C.T_HI)
    u = (np.log(t) - np.log(C.T_LO)) / (np.log(C.T_HI) - np.log(C.T_LO))
    for d in range(2):
        assert sorted(np.floor(u[:, d] * 3).astype(int)) == [0, 1, 2]
    assert np.allclose(t, C.theta_star(p))
# 2. corrected emulator: exact at refinement points (to nugget), unchanged far away
class Lin:
    def __call__(self, th): return np.array([np.log(th[0]), np.log(th[1]), 1.0], np.float32)
base = Lin(); pts = np.array([[1.0, 0.3], [2.0, 0.2]]); res = np.array([[0.5, -0.2, 0.1], [0.0, 0.3, -0.4]])
E = C.CorrectedEmulator(base, pts, res)
for k in range(2):
    assert np.allclose(E(pts[k]) - base(pts[k]), res[k], atol=1e-4)
far = np.array([3.9, 0.79]); assert np.allclose(E(far), base(far), atol=1e-3)
# 3. noise: realised per-lead SNR close to nominal over the QRS window, both noise types
rng = np.random.default_rng(1)
x = np.zeros((8, 2001)); t = np.arange(2001) * C.DT
x[:, :400] = np.sin(2 * np.pi * t[:400] / 100.0)[None] * np.arange(1, 9)[:, None]
for kind in C.NOISE_TYPES:
    y, sig = C.add_noise(x, 190.0, 20, kind, rng)
    P = (x[:, t <= 200] ** 2).mean(1)
    assert np.allclose(10 * np.log10(P / sig ** 2), 20)
    snr = 10 * np.log10(P / ((y - x) ** 2).mean(1))
    assert np.all(np.abs(snr - 20) < 0.5), (kind, snr)
# 4. bound flags
lo, hi = C.bound_hit(C.B_LO); assert lo.all() and not hi.any()
# 5. baseline wander: peak deviation over the QRS window = level x lead peak-to-peak; held constant after the window
rng = np.random.default_rng(2)
for frac in (0.05, 0.10, 0.20):
    y, d = C.add_wander(x, 190.0, frac, rng)
    w = t <= 200.0
    ptp = x[:, w].max(1) - x[:, w].min(1)
    assert np.allclose(np.abs(d[:, w]).max(1), frac * ptp)
    assert np.allclose(d[:, ~w], d[:, w][:, -1:])
    assert np.allclose(y - x, d)
# 6. seeds keyed by (patient, istar, comparator, level, rep): deterministic and distinct
a1 = np.random.default_rng(C.rep_seed(101, 0, 2, 10, 3)).standard_normal(4)
a2 = np.random.default_rng(C.rep_seed(101, 0, 2, 10, 3)).standard_normal(4)
a3 = np.random.default_rng(C.rep_seed(101, 0, 2, 10, 4)).standard_normal(4)
assert np.array_equal(a1, a2) and not np.allclose(a1, a3)
# 7. paths and reference name for the v2 constructions
assert C.ref_name(101) == 'patient101__expert__1.5__rv3.0'
assert os.path.exists(C.cons_path(C.ref_name(101))) and os.path.exists(C.disp_path(101))
assert [c[0] for c in C.COMPARATORS] == ['white', 'lp40', 'wander', 'disp']
print('all calib tests passed')
