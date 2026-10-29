"""Known-answer tests for armf_lib (owner armF, 2026-10-28). Run: python -I -B test_armf.py"""
import os
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armf_lib as F  # noqa: E402

ok = True


def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)
    ok &= bool(cond)


# 1. heart selection rule (A4) gives 2 per group, lowest ids
hearts, groups = F.select_hearts()
check('select_hearts', hearts == [101, 106, 104, 105, 103, 112, 102, 107, 109, 119], str(groups))

# 2. theta* equals the gates draws stored by calib (same seed rule as arm C)
for p in (101, 102, 104):
    fn = os.path.join(F.ROOT, 'results', '2026-10-28_G3G4_ministudy', f'heart_patient{p:03d}.h5')
    with h5py.File(fn, 'r') as H:
        ref = H['targets/theta_star'][:]
    check(f'theta_star {p}', np.array_equal(ref, F.theta_star(p)), str(F.theta_star(p).round(4).tolist()))

# 3. qrs_nw arithmetic: t <= amax + 10 at 0.5 ms
check('qrs_nw', F.qrs_nw(60.0) == 141 and F.qrs_nw(60.2) == 141 and F.qrs_nw(60.5) == 142)

# 4. WinECG truncation equals a shorter ECGModel (causal cumsum)
rng = np.random.default_rng(1)
W = rng.standard_normal((8, 500)); W -= W.mean(1, keepdims=True)
ta = rng.uniform(0, 80, 500)
e_long = F.eplib.ECGModel(W, tmax=200.0)
e_short = F.eplib.ECGModel(W, tmax=100.0)
d = np.abs(F.WinECG(e_long, e_short.nt)(ta) - e_short(ta)).max()
check('WinECG == shorter ECGModel', d < 1e-9 * np.abs(e_short(ta)).max() + 1e-12, f'maxdiff {d:.2e}')

# 5. re_ecg: (a) constant Vm gives a zero ECG; (b) a step of +110 mV at one node from t = 1 ms gives -110 W[:, n];
#    (c) on a synthetic sigmoid Vm (tau 1 ms, 110 mV) it agrees with the activation-template ECG up to the template's
#    own 0.5 ms binning error (threshold r > 0.995, rel L2 < 0.06; observed values printed)
t = np.arange(e_long.nt) * F.DT
z = F.re_ecg(np.full((e_long.nt, 500), -86.0, np.float32), W)
step = np.full((e_long.nt, 500), -86.0, np.float32); step[2:, 7] += 110.0
es = F.re_ecg(step, W)
check('re_ecg constant and step', np.abs(z).max() == 0 and np.allclose(es[:, -1], -110.0 * W[:, 7], rtol=1e-6)
      and np.abs(es[:, :2]).max() == 0)
vm = -86.0 + 110.0 / (1 + np.exp(-(t[:, None] - ta[None, :]) / 1.0))
ev = F.re_ecg(vm.astype(np.float32), W)
et = e_long(ta)
r = np.min(F.lead_r(ev, et))
rel = np.linalg.norm(ev - et) / np.linalg.norm(et)
check('re_ecg vs template (synthetic sigmoid)', r > 0.995 and rel < 0.06, f'min r {r:.6f}, rel L2 {rel:.4f}')

# 6. upstroke_act on the same synthetic Vm: -40 mV crossing at ta + tau*ln(46/64) rounded up to the 0.5 ms grid
up, un = F.upstroke_act(vm)
lag = up - ta
check('upstroke_act', un == 0 and lag.min() > -0.33 - 1e-6 and lag.max() < 0.17 + 1e-6,
      f'lag range {lag.min():.3f} to {lag.max():.3f} ms')

# 7. tmax_of ignores unactivated (> 1e6, < 0) and rounds up to 0.5 ms
check('tmax_of', F.tmax_of([10.0, 99.2, 2e6, -1]) == 119.5)

# 8. bound_hit
lo, hi = F.bound_hit([0.5, 0.8])
check('bound_hit', lo[0] and hi[1] and not lo[1] and not hi[0])

print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
