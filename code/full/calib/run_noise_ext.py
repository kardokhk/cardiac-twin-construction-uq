"""Amendment 2 B4 extended noise grids (owner noise-ext, I11, 2026-10-28). Declared extension of E3.

Reuses calib_lib.py (noise, wander and calibration definitions unchanged) and the arm C per-heart HDF5 files
(results/2026-10-28_armC/hearts/heart_patientNNN.h5, opened read-only): the reference construction's final corrected
emulator (round 2 refinement points), its stored W, the targets and theta*. Nothing in arm C is rewritten.
New levels (A2 B4): white and lp40 at 5, 0, -5, -10, -15 dB SNR; wander 40% and 80%; displacement 30 and 40 mm SD;
20 replicates; both losses (same draw for both); 3 theta* per heart.
Seeds: white, lp40, wander: numpy.random.default_rng([20261028, patient, istar, comparator_index, level_key, rep]),
    comparator_index white 0, lp40 1, wander 2, disp 3 (A3 scheme, calib_lib.rep_seed). level_key = level for
    level >= 0; numpy SeedSequence rejects negative entropy, so level_key = 1000 + |level| for negative SNR
    (-5 -> 1005, -10 -> 1010, -15 -> 1015). No key collides with an A3 or another B4 level.
    disp: displaced lead fields from data/processed/acdc_mesh_v2/displaced_ext (code/full/mesh/displace_ext.py), seed
    20261028 + 1000 patient + 100 sd_index + rep, sd_index 3 (30 mm) and 4 (40 mm); target = ECG of theta* activation
    through W_disp, calibration with the undisplaced stored W and emulator; one displaced field per replicate shared by
    the three theta* (as A3).
Usage: python -I -B run_noise_ext.py --out <file.h5> --hearts 101 ... --workers 126 [--nrep 20] [--comparators ...]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_lib as C          # noqa: E402
import run_armC as RA          # noqa: E402  (reuse _load_emulator; importing runs no driver code)

eplib = C.eplib
T0 = time.time()
ARMC_HEARTS = os.path.join(C.ROOT, 'results', '2026-10-28_armC', 'hearts')
DISP_EXT = os.environ.get('NOISE_EXT_DISP_DIR', os.path.join(C.MESH_DIR, 'displaced_ext'))   # env override for the test job only
CI = {n: i for i, (n, _, _) in enumerate(C.COMPARATORS)}      # white 0, lp40 1, wander 2, disp 3
UNIT = {n: u for n, _, u in C.COMPARATORS}
LEVELS_EXT = {'white': [5, 0, -5, -10, -15], 'lp40': [5, 0, -5, -10, -15], 'wander': [40, 80], 'disp': [30, 40]}


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def level_key(lev):
    lev = int(lev)
    return lev if lev >= 0 else 1000 + abs(lev)


def rep_seed_ext(pid, istar, ci, lev, rep):
    return C.rep_seed(pid, istar, ci, level_key(lev), rep)


def heart_file(pid):
    return os.path.join(ARMC_HEARTS, f'heart_patient{pid:03d}.h5')


def disp_ext_path(pid):
    return os.path.join(DISP_EXT, f'patient{pid:03d}_W_displaced_ext.h5')


def w_noise_ext(args):
    pid, istar, cn, lev, nrep = args
    hfile = heart_file(pid); ref = C.ref_name(pid)
    _, emu, ecg, tgt = RA._load_emulator(hfile, ref)
    with h5py.File(hfile, 'r') as H:
        Tt = H['targets/T'][istar]; amax = float(Tt.max()); tmax = float(H.attrs['tmax_ms'])
        assert int(H.attrs['patient']) == pid
    ci = CI[cn]
    if cn == 'disp':
        with h5py.File(disp_ext_path(pid), 'r') as f:
            si = list(np.round(f['sd_mm'][:]).astype(int)).index(int(lev))
            Wd = f['W_disp'][si].astype(np.float64); dseed = f['seed'][si]
            assert Wd.shape[-1] == len(Tt), (Wd.shape, Tt.shape)
    out = np.full((nrep, len(C.LOSSES), 3), np.nan)
    seeds = np.zeros(nrep, np.int64)
    for r in range(nrep):
        if cn in ('white', 'lp40'):
            rng = np.random.default_rng(rep_seed_ext(pid, istar, ci, lev, r))
            noisy, _ = C.add_noise(tgt[istar], amax, lev, cn, rng)
            seeds[r] = -1
        elif cn == 'wander':
            rng = np.random.default_rng(rep_seed_ext(pid, istar, ci, lev, r))
            noisy, _ = C.add_wander(tgt[istar], amax, lev / 100.0, rng)
            seeds[r] = -1
        else:
            noisy = eplib.ECGModel(Wd[r], tmax=tmax)(Tt)
            seeds[r] = int(dseed[r])
        for j, (ln, lf) in enumerate(C.LOSSES):
            th, fv = C.calibrate(emu, ecg, noisy, lf)
            out[r, j] = [th[0], th[1], fv]
    return (pid, istar, cn, lev), out, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--hearts', type=int, nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=126); ap.add_argument('--nrep', type=int, default=C.NREP)
    ap.add_argument('--comparators', nargs='+', default=list(LEVELS_EXT))
    a = ap.parse_args()
    assert not os.path.exists(a.out), f'refusing to overwrite {a.out}'
    tasks = [(p, i, cn, lev, a.nrep) for p in a.hearts for i in range(3) for cn in a.comparators
             for lev in LEVELS_EXT[cn]]
    tasks.sort(key=lambda t: t[2] == 'disp', reverse=True)
    log('tasks', len(tasks), 'hearts', len(a.hearts), 'comparators', a.comparators)
    res = {}
    with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
        for k, (key, out, seeds) in enumerate(ex.map(w_noise_ext, tasks)):
            res[key] = (out, seeds)
            if k % 200 == 0:
                log(f'{k}/{len(tasks)}')
    tmp = a.out + '.part'
    with h5py.File(tmp, 'w') as O:
        O.attrs.update(owner='noise-ext', amendment='AMENDMENT_2 B4', created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                       nrep=a.nrep, emulator='reference construction, final corrected emulator (round 2 points), '
                       'from results/2026-10-28_armC/hearts', level_key_rule='level if >= 0 else 1000 + |level|')
        for p in a.hearts:
            with h5py.File(heart_file(p), 'r') as H:
                g = O.create_group(f'patient{p:03d}')
                g.attrs.update(patient=p, group=H.attrs['group'], reference=H.attrs['reference'])
                g['theta_star'] = H['targets/theta_star'][:]
            for cn in a.comparators:
                levs = LEVELS_EXT[cn]
                arr = np.full((3, len(levs), a.nrep, len(C.LOSSES), 3), np.nan)
                sd = np.zeros((3, len(levs), a.nrep), np.int64)
                for i in range(3):
                    for li, lev in enumerate(levs):
                        arr[i, li], sd[i, li] = res[(p, i, cn, lev)]
                d = g.create_dataset(f'{cn}/theta_hat', data=arr)
                g[f'{cn}/rng_seed'] = sd
                d.attrs.update(axes='theta* x level x replicate x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)',
                               levels=levs, unit=UNIT[cn], comparator_index=CI[cn],
                               seed_rule=('default_rng([20261028, patient, istar, comparator_index, level_key, rep]); '
                                          'level_key = level if >= 0 else 1000 + |level|; same draw for both losses'
                                          if cn != 'disp' else 'displaced_ext lead field: seed 20261028 + 1000 '
                                          'patient + 100 sd_index + rep, sd_index 3 (30 mm), 4 (40 mm)'))
    os.replace(tmp, a.out)
    log('written', a.out, 'total', f'{time.time()-T0:.0f}s')


if __name__ == '__main__':
    main()
