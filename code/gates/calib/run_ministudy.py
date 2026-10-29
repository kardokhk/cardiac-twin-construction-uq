"""G3/G4 mini-study driver (owner calib, I11 gates, 2026-10-28). Definitions, prior and seeds: calib_lib.py docstring.

Phases (the driver is the only writer of the per-heart HDF5; workers only read):
  1  eikonal runs: 3 theta* targets on the reference construction (expert, 1.5 mm) + 81 grid runs per construction
  2  calibration rounds 0, 1, 2 with refinement runs between rounds (up to 9 per construction)
  3  noise arm N on the reference construction's final emulator (5 SNR x 20 replicates x 2 noise types x 2 losses)
  4  tidy CSVs (estimates.csv, noise_estimates.csv, refinement.csv, constructions.csv)
Usage: python -I run_ministudy.py --out <dir> --hearts 101 102 104 --workers 126 [--nrep 20] [--vox 1.5 1.0]
Writes <out>/heart_patientNNN.h5 per heart and the CSVs. Phase 1 is skipped for a heart whose HDF5 already holds it.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_lib as C          # noqa: E402

A = C.A
eplib = C.eplib
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ workers
def _mat(name, tmp):
    return name, A.materialize(C.cons_path(name), os.path.join(tmp, 'm_' + name))


def _load_emulator(hfile, name):
    with h5py.File(hfile, 'r') as H:
        g = H['cons'][name]
        base = eplib.GridEmulator(C.G1, C.G2, g['grid_T'][:])
        if 'ref_theta' in g and len(g['ref_theta']):
            emu = C.CorrectedEmulator(base, g['ref_theta'][:], g['ref_T'][:] - g['ref_Temu0'][:])
        else:
            emu = C.CorrectedEmulator(base)
        tmax = float(H.attrs['tmax_ms']); tgt = H['targets/ecg'][:]
    with h5py.File(C.cons_path(name), 'r') as f:
        W = f['W'][:]
    return base, emu, eplib.ECGModel(W, tmax=tmax), tgt


def w_calibrate(args):
    """All 3 targets x 2 losses on one construction with its current (corrected) emulator."""
    hfile, name = args
    t0 = time.time()
    _, emu, ecg, tgt = _load_emulator(hfile, name)
    out = np.full((len(tgt), len(C.LOSSES), 3), np.nan)       # theta_hat (2), fval
    for i in range(len(tgt)):
        for j, (ln, lf) in enumerate(C.LOSSES):
            th, fv = C.calibrate(emu, ecg, tgt[i], lf)
            out[i, j] = [th[0], th[1], fv]
    return name, out, time.time() - t0


def w_noise(args):
    hfile, ref, istar, snr, kidx, nrep = args
    _, emu, ecg, tgt = _load_emulator(hfile, ref)
    with h5py.File(hfile, 'r') as H:
        amax = float(H['targets/T'][istar].max()); pid = int(H.attrs['patient'])
    seed = [C.SEED_BASE, pid, istar, snr, kidx]
    rng = np.random.default_rng(seed)
    out = np.full((nrep, len(C.LOSSES), 3), np.nan)
    for r in range(nrep):
        noisy, sig = C.add_noise(tgt[istar], amax, snr, C.NOISE_TYPES[kidx], rng)
        for j, (ln, lf) in enumerate(C.LOSSES):
            th, fv = C.calibrate(emu, ecg, noisy, lf)
            out[r, j] = [th[0], th[1], fv]
    return (istar, snr, kidx), out, seed


# ------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--hearts', type=int, nargs='+', default=[101, 102, 104])
    ap.add_argument('--workers', type=int, default=126); ap.add_argument('--nrep', type=int, default=C.NREP)
    ap.add_argument('--vox', type=float, nargs='+', default=C.VOXES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'calib')
    os.makedirs(tmp, exist_ok=True)
    hf = {p: os.path.join(a.out, f'heart_patient{p:03d}.h5') for p in a.hearts}
    names = {}
    for p in a.hearts:
        L = []
        for v in a.vox:
            for s in C.SOURCES:
                n = C.cons_name(p, s, v)
                if os.path.exists(C.cons_path(n)) and bool(h5py.File(C.cons_path(n), 'r').attrs.get('smoke_ok', False)):
                    L.append(n)
                else:
                    log('MISSING or failed construction', n)
        assert C.ref_name(p) in L
        names[p] = L
    allnames = [n for p in a.hearts for n in names[p]]
    log('constructions', {p: len(names[p]) for p in a.hearts})

    # ---------------------------------------------------------------- phase 1
    need1 = [p for p in a.hearts if not (os.path.exists(hf[p]) and h5py.File(hf[p], 'r').attrs.get('phase1_done', False))]
    mesh = {}
    with ProcessPoolExecutor(min(a.workers, len(allnames))) as ex:
        for n, mv in ex.map(_mat, allnames, [tmp] * len(allnames)):
            mesh[n] = mv
    log('materialized', len(mesh))
    if need1:
        TH = C.grid_thetas()
        jobs = []
        for p in need1:
            ts = C.theta_star(p)
            for i in range(len(ts)):
                jobs.append((p, '__target__', i, ts[i]))
            for n in names[p]:
                for g in range(len(TH)):
                    jobs.append((p, n, g, TH[g]))
        # longest first: 1.0 mm, then slow corners
        jobs.sort(key=lambda j: (('__1.0' in j[1]), -np.log(j[3]).sum()), reverse=True)

        def one(j):
            p, n, g, th = j
            mn = C.ref_name(p) if n == '__target__' else n
            mb, vtx = mesh[mn]
            act, wall = A.carpio.run_eikonal0(mb, vtx, [(1, A.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                              os.path.join(tmp, f'r_{mn}_{n[:2]}_{g}'), threads=1, tend=C.TEND)
            return j, act.astype(np.float32), wall
        res = {}
        with ThreadPoolExecutor(a.workers) as ex:
            for k, (j, act, wall) in enumerate(ex.map(one, jobs)):
                res[(j[0], j[1], j[2])] = (act, wall)
                if k % 500 == 0:
                    log(f'phase1 {k}/{len(jobs)}')
        log('phase 1 runs done', len(jobs))
        for p in need1:
            ts = C.theta_star(p)
            Tt = np.vstack([res[(p, '__target__', i)][0] for i in range(len(ts))])
            tw = np.array([res[(p, '__target__', i)][1] for i in range(len(ts))])
            tmax = 0.0
            with h5py.File(hf[p], 'w') as H:
                H.attrs.update(patient=p, group=A.info_cfg(p).get('Group'), prior_lo=C.B_LO, prior_hi=C.B_HI,
                               tstar_lo=C.T_LO, tstar_hi=C.T_HI, seed_theta_star=C.SEED_BASE + p, v_fib=A.V_FIB,
                               tend_ms=C.TEND, reference=C.ref_name(p), renumbering=0,
                               opencarp=A.carpio.OPENCARP, created=time.strftime('%Y-%m-%dT%H:%M:%S'))
                H['targets/theta_star'] = ts; H['targets/T'] = Tt; H['targets/wall_s'] = tw
                H['grid/G1'] = C.G1; H['grid/G2'] = C.G2; H['grid/theta'] = TH
                for n in names[p]:
                    T = np.vstack([res[(p, n, g)][0] for g in range(len(TH))])
                    w = np.array([res[(p, n, g)][1] for g in range(len(TH))])
                    unact = int(((T > 1e6) | (T < 0)).sum())
                    tmax = max(tmax, float(T[T < 1e6].max()))
                    g_ = H.create_group(f'cons/{n}')
                    g_.create_dataset('grid_T', data=T, compression='gzip', compression_opts=1)
                    g_['grid_wall_s'] = w
                    Ls = C.lengths_of(C.cons_path(n))
                    with h5py.File(C.cons_path(n), 'r') as f:
                        src, vox = f.attrs['source'], float(f.attrs['vox'])
                    g_.attrs.update(source=src, vox=vox, n_nodes=T.shape[1], unactivated=unact, **Ls)
                    if unact:
                        log('WARNING unactivated nodes', n, unact)
                tmax = max(tmax, float(Tt.max())) + 20.0
                H.attrs['tmax_ms'] = tmax
                with h5py.File(C.cons_path(C.ref_name(p)), 'r') as f:
                    Wr = f['W'][:]
                ecgm = eplib.ECGModel(Wr, tmax=tmax)
                H['targets/ecg'] = np.stack([ecgm(Tt[i]) for i in range(len(ts))])
                H.attrs['phase1_done'] = True
            log(f'heart {p}: tmax {tmax:.1f} ms, target max act {Tt.max(1).round(1)}, theta* {ts.round(3).tolist()}')

    # ---------------------------------------------------------------- phase 2: rounds 0, 1, 2
    work = [(hf[p], n) for p in a.hearts for n in names[p]]
    # reset refinement (rerun phase 2 from scratch)
    for p in a.hearts:
        with h5py.File(hf[p], 'a') as H:
            for n in names[p]:
                g = H['cons'][n]
                for k in ['ref_theta', 'ref_T', 'ref_Temu0', 'ref_round', 'ref_wall_s', 'calib']:
                    if k in g:
                        del g[k]
    cal = {}
    for rnd in range(3):
        t1 = time.time()
        work.sort(key=lambda w: '__1.0' in w[1], reverse=True)
        with ProcessPoolExecutor(min(a.workers, len(work))) as ex:
            for name, out, wall in ex.map(w_calibrate, work):
                cal[(rnd, name)] = out
        log(f'calibration round {rnd}: {len(work)} constructions, {time.time()-t1:.0f}s')
        if rnd == 2:
            break
        # refinement points
        newpts = {}
        for hfile, name in work:
            with h5py.File(hfile, 'r') as H:
                old = H['cons'][name]['ref_theta'][:] if 'ref_theta' in H['cons'][name] else np.zeros((0, 2))
            out = cal[(rnd, name)]
            cand = out[:, :, :2].reshape(-1, 2) if rnd == 0 else out[:, 0, :2]
            pts = []
            for th in cand:
                allp = np.vstack([old] + ([np.array(pts)] if pts else []))
                if len(allp) and np.min(np.abs(np.log(allp) - np.log(th)).max(1)) < 0.01:
                    continue
                pts.append(th)
            newpts[(hfile, name)] = np.array(pts).reshape(-1, 2)
        jobs = [(k, i, th) for k, v in newpts.items() for i, th in enumerate(v)]
        jobs.sort(key=lambda j: '__1.0' in j[0][1], reverse=True)

        def one2(j):
            (hfile, name), i, th = j
            mb, vtx = mesh[name]
            act, wall = A.carpio.run_eikonal0(mb, vtx, [(1, A.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                              os.path.join(tmp, f'f_{name}_{rnd}_{i}'), threads=1, tend=C.TEND)
            return j, act.astype(np.float32), wall
        t1 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rr = list(ex.map(one2, jobs))
        log(f'refinement round {rnd + 1}: {len(jobs)} runs, {time.time()-t1:.0f}s')
        for (hfile, name), v in newpts.items():
            if not len(v):
                continue
            acts = np.vstack([x[1] for x in rr if x[0][0] == (hfile, name)])
            walls = np.array([x[2] for x in rr if x[0][0] == (hfile, name)])
            with h5py.File(hfile, 'a') as H:
                g = H['cons'][name]
                base = eplib.GridEmulator(C.G1, C.G2, g['grid_T'][:])
                e0 = np.vstack([base(th) for th in v]).astype(np.float32)
                for key, val in [('ref_theta', v), ('ref_T', acts), ('ref_Temu0', e0), ('ref_wall_s', walls),
                                 ('ref_round', np.full(len(v), rnd + 1))]:
                    if key in g:
                        val = np.concatenate([g[key][:], val]); del g[key]
                    g.create_dataset(key, data=val)
    for hfile, name in work:
        with h5py.File(hfile, 'a') as H:
            g = H['cons'][name]
            g['calib'] = np.stack([cal[(r, name)] for r in range(3)])      # (round, istar, loss, [v_endo, v_t, fval])
            g['calib'].attrs['axes'] = 'round(0,1,2) x theta* x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)'

    # ---------------------------------------------------------------- phase 3: noise arm on the reference emulator
    t1 = time.time()
    tasks = [(hf[p], C.ref_name(p), i, s, k, a.nrep) for p in a.hearts for i in range(3) for s in C.SNRS
             for k in range(len(C.NOISE_TYPES))]
    noise = {}
    with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
        for t, (key, out, seed) in zip(tasks, ex.map(w_noise, tasks)):
            noise[(t[0],) + key] = (out, seed)
    for p in a.hearts:
        arr = np.full((3, len(C.SNRS), len(C.NOISE_TYPES), a.nrep, len(C.LOSSES), 3), np.nan)
        for i in range(3):
            for si, s in enumerate(C.SNRS):
                for k in range(len(C.NOISE_TYPES)):
                    arr[i, si, k] = noise[(hf[p], i, s, k)][0]
        with h5py.File(hf[p], 'a') as H:
            if 'noise' in H:
                del H['noise']
            d = H.create_dataset('noise/theta_hat', data=arr)
            d.attrs.update(axes='theta* x SNR x noise type x replicate x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)',
                           snr_db=C.SNRS, noise_types=json.dumps(C.NOISE_TYPES),
                           seed_rule='np.random.default_rng([20261028, patient, istar, snr_db, noise_type_index]); '
                                     'the same noise draws are used for both losses',
                           emulator='reference construction, final corrected emulator (round 2 refinement points)')
    log(f'noise arm: {len(tasks)} tasks, {time.time()-t1:.0f}s')

    # ---------------------------------------------------------------- phase 4: CSVs
    import csv_export
    csv_export.export(a.out, a.hearts)
    log('done; total', f'{time.time()-T0:.0f}s')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
