"""Known-answer tests for calib_lib and analyse_gates helpers (owner calib, 2026-10-28). Run: python -I test_calib.py"""
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
# 5. break-even interpolation (analyse_gates.breakeven) on a known log-linear curve
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyse_gates.py')).read()
ns = {}; exec(src.split('def g3_block')[0].split("t_start = time.time()")[0] + "\nSNRS=[10,15,20,25,30]\n" +
              src[src.index('def breakeven'):src.index('def g3_block')], ns)
sdn = 10 ** (-np.array([10, 15, 20, 25, 30]) / 20.0)      # SD halves per 6 dB
for s0 in (12.0, 22.5, 30.0):
    be, fl = ns['breakeven'](10 ** (-s0 / 20), sdn); assert abs(be - s0) < 1e-9 and fl == 'interp', (be, fl)
be, fl = ns['breakeven'](10 ** (-4 / 20), sdn); assert abs(be - 4) < 1e-9 and fl == 'extrap<10'
print('all calib tests passed')
