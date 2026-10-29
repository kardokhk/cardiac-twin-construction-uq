"""I11 pilot: calibrate (v_endo, v_t) on every construction of one heart to the fixed synthetic 12-lead target
generated on the reference construction (c0ref) at theta* (3 draws; direct openCARP runs), in the heart's own
torso.  Emulator per construction from its own 9 x 9 design grid.  Both losses.
Usage: python -I 07_calibrate_cons.py <cons.h5> <runsdir> <hid> <out.json>
"""
import sys, os, json, time
import numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I13'))
import eplib
B_LO, B_HI = np.array([0.8, 0.15]), np.array([3.0, 0.5])
consf, runsdir, hid, out = sys.argv[1:5]
t0 = time.time()
C = h5py.File(consf, 'r')
names = sorted(C.keys())
R = {n: h5py.File(os.path.join(runsdir, f'runs_h{hid}_{n}.h5'), 'r') for n in names}
ref = R['c0ref']; kind = ref['kind'][:]; th = ref['theta'][:]
tmax = max(float(R[n]['T'][:].max()) for n in names) + 20
ecg_ref = eplib.ECGModel(C['c0ref']['W'][:], tmax=tmax)
targets = [(th[i], ecg_ref(ref['T'][i])) for i in np.nonzero(kind == 1)[0]]
res = dict(hid=hid, constructions={}, wall_s=None)
for n in names:
    T = R[n]['T'][:].astype(np.float32); k = R[n]['kind'][:]
    assert (T < 1e6).all() and (T >= 0).all(), f'unactivated nodes in {n}'
    emu = eplib.GridEmulator(R[n]['G1'][:], R[n]['G2'][:], T[k == 0])
    ecg = eplib.ECGModel(C[n]['W'][:], tmax=tmax)
    he = []
    for i in np.nonzero(k == 2)[0]:
        te = emu(R[n]['theta'][i])
        he.append(float(np.sqrt(((te - T[i]) ** 2).mean())))
    cal = []
    for tstar, tgt in targets:
        for lname, lf in [('corr', eplib.loss_corr), ('amp', eplib.loss_amp)]:
            thh, fval = eplib.calibrate(emu, ecg, tgt, lf, (B_LO, B_HI))
            cal.append(dict(theta_star=tstar.tolist(), loss=lname, theta_hat=thh.tolist(), fval=float(fval)))
            print(f'  h{hid} {n:6s} theta* {tstar.round(3)} {lname:4s} -> {thh.round(3)} loss {fval:.2e}', flush=True)
    res['constructions'][n] = dict(attrs={a: (v.item() if hasattr(v, 'item') else v) for a, v in C[n].attrs.items()},
                                   heldout_rms_ms=he, calib=cal, qrs_max_ms=float(T[k == 0].max()))
res['wall_s'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1, default=float)
print('done', out, f'{time.time()-t0:.0f}s')
