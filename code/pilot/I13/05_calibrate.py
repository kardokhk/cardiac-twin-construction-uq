"""Calibrate (v_endo, v_t) for one heart: targets are 12-lead (8 independent leads) QRS ECGs computed in the
heart's OWN torso from direct openCARP runs at theta*; calibration uses a cubic-spline emulator of nodal
activation times built from the 9 x 9 design grid, with ECG weights of arm A (own torso, control) or arm B
(reference torso).  Two losses: correlation-based (1 - mean Pearson r over leads) and raw amplitude
(relative L2).  Also reports emulator error on 6 held-out direct runs.
Usage: python -I 05_calibrate.py <consNN.h5> <runs.h5> <out.json> [--ref name=path.npz ...]
"""
import sys, os, json, time, argparse
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eplib
B_LO, B_HI = np.array([0.8, 0.15]), np.array([3.0, 0.5])

ap = argparse.ArgumentParser()
ap.add_argument('heart'); ap.add_argument('runs'); ap.add_argument('out')
ap.add_argument('--ref', nargs='*', default=[])
ap.add_argument('--Wkey', default='W_own')
a = ap.parse_args()
t0 = time.time()
H = h5py.File(a.heart, 'r'); Rn = h5py.File(a.runs, 'r')
th = Rn['theta'][:]; kind = Rn['kind'][:]; T = Rn['T'][:].astype(np.float32)
G1, G2 = Rn['G1'][:], Rn['G2'][:]
assert (T < 1e6).all() and (T >= 0).all(), 'unactivated nodes in design runs'
emu = eplib.GridEmulator(G1, G2, T[kind == 0])
arms = {'A_own': H['c0ref']['W'][:]}
for r in a.ref:
    k, p = r.split('=')
    arms[k] = np.load(p)['W']
tmax = float(T.max()) + 20
ecg = {k: eplib.ECGModel(W, tmax=tmax) for k, W in arms.items()}
res = {'heart': a.heart, 'runs': a.runs, 'n_nodes': int(T.shape[1]), 'qrs_max_ms': float(T.max()), 'arms': list(arms)}

# emulator error on held-out direct runs
he = []
for i in np.nonzero(kind == 2)[0]:
    te = emu(th[i]); td = T[i]
    sim_e, sim_d = ecg['A_own'](te), ecg['A_own'](td)
    he.append(dict(theta=th[i].tolist(), rms_ms=float(np.sqrt(((te - td) ** 2).mean())), max_ms=float(np.abs(te - td).max()),
                   ecg_corr_loss=float(eplib.loss_corr(sim_e, sim_d)), ecg_amp_loss=float(eplib.loss_amp(sim_e, sim_d))))
res['heldout'] = he
# calibrations
cal = []
for i in np.nonzero(kind == 1)[0]:
    tstar = th[i]
    target = ecg['A_own'](T[i])                         # target always in own torso, direct run
    for arm in arms:
        for lname, lf in [('corr', eplib.loss_corr), ('amp', eplib.loss_amp)]:
            thh, fval = eplib.calibrate(emu, ecg[arm], target, lf, (B_LO, B_HI))
            cal.append(dict(theta_star=tstar.tolist(), arm=arm, loss=lname, theta_hat=thh.tolist(), fval=float(fval),
                            rel_err=((thh - tstar) / tstar).tolist()))
            print(f'  theta* {tstar.round(3)} {arm:8s} {lname:4s} -> {thh.round(3)} rel {((thh-tstar)/tstar*100).round(2)}% '
                  f'loss {fval:.2e}', flush=True)
res['calib'] = cal
res['wall_s'] = time.time() - t0
json.dump(res, open(a.out, 'w'), indent=1)
print('done', a.out, f'{time.time()-t0:.0f}s')
