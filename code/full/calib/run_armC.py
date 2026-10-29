"""Full arm C and arm N driver (owner calibC, I11 full arm, 2026-10-28). Copied from code/gates/calib/run_ministudy.py;
changes: construction list from data/processed/acdc_mesh_v2/MANIFEST.tsv (rows with include_primary or
include_sensitivity), names carry the RV wall, arm N has the four Amendment A3 comparators with per-replicate seeds,
the per-heart HDF5 carries the manifest flags. Phases 1 and 2 (eikonal grid, targets, calibration rounds 0 to 2 with
refinement) are the gates logic unchanged. Definitions, prior and seeds: calib_lib.py docstring.
Usage: python -I -B run_armC.py --out <dir> --hearts 101 102 ... --workers 126 [--nrep 20]
Writes <out>/hearts/heart_patientNNN.h5 per heart (CSV export is a separate step: csv_export.py).
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
    """One (heart, theta*, comparator, level) cell, nrep replicates, both losses (same draw for both)."""
    hfile, ref, istar, ci, lev, nrep = args
    _, emu, ecg, tgt = _load_emulator(hfile, ref)
    with h5py.File(hfile, 'r') as H:
        Tt = H['targets/T'][istar]; amax = float(Tt.max()); pid = int(H.attrs['patient'])
        tmax = float(H.attrs['tmax_ms'])
    cn = C.COMPARATORS[ci][0]
    if cn == 'disp':
        with h5py.File(C.disp_path(pid), 'r') as f:
            si = list(np.round(f['sd_mm'][:]).astype(int)).index(int(lev))
            Wd = f['W_disp'][si].astype(np.float64); dseed = f['seed'][si]
            assert Wd.shape[-1] == len(Tt), (Wd.shape, Tt.shape)
    out = np.full((nrep, len(C.LOSSES), 3), np.nan)
    seeds = np.zeros(nrep, np.int64)
    for r in range(nrep):
        if cn in ('white', 'lp40'):
            rng = np.random.default_rng(C.rep_seed(pid, istar, ci, lev, r))
            noisy, _ = C.add_noise(tgt[istar], amax, lev, cn, rng)
            seeds[r] = -1
        elif cn == 'wander':
            rng = np.random.default_rng(C.rep_seed(pid, istar, ci, lev, r))
            noisy, _ = C.add_wander(tgt[istar], amax, lev / 100.0, rng)
            seeds[r] = -1
        else:
            noisy = eplib.ECGModel(Wd[r], tmax=tmax)(Tt)
            seeds[r] = int(dseed[r])
        for j, (ln, lf) in enumerate(C.LOSSES):
            th, fv = C.calibrate(emu, ecg, noisy, lf)
            out[r, j] = [th[0], th[1], fv]
    return (istar, ci, lev), out, seeds


# ------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--hearts', type=int, nargs='+', default=[101, 102, 104])
    ap.add_argument('--workers', type=int, default=126); ap.add_argument('--nrep', type=int, default=C.NREP)
    ap.add_argument('--manifest', default=os.path.join(C.MESH_DIR, 'MANIFEST.tsv'))
    ap.add_argument('--skip-noise', action='store_true')
    ap.add_argument('--cons-filter', default=None, help='regex on construction_id (test runs only)')
    a = ap.parse_args()
    import pandas as pd
    M = pd.read_csv(a.manifest, sep='\t')
    M = M[M.include_primary.astype(bool) | M.include_sensitivity.astype(bool)]
    if a.cons_filter:
        M = M[M.construction_id.str.contains(a.cons_filter) | M.is_reference.astype(bool)]
    a.out = os.path.join(a.out, 'hearts')
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'calib')
    os.makedirs(tmp, exist_ok=True)
    hf = {p: os.path.join(a.out, f'heart_patient{p:03d}.h5') for p in a.hearts}
    names = {}
    meta = {}
    for p in a.hearts:
        L = []
        for _, r in M[M.heart == p].iterrows():
            n = r.construction_id
            if os.path.exists(C.cons_path(n)) and bool(r.smoke_ok):
                L.append(n); meta[n] = r
            else:
                log('MISSING or failed construction (manifest says included)', n)
        assert C.ref_name(p) in L
        names[p] = L
    allnames = [n for p in a.hearts for n in names[p]]
    log('constructions', {p: len(names[p]) for p in a.hearts})

    # ---------------------------------------------------------------- phase 1
    need1 = [p for p in a.hearts if not (os.path.exists(hf[p]) and h5py.File(hf[p], 'r').attrs.get('phase1_done', False))]
    mesh = {}
    lens = {}
    with ProcessPoolExecutor(min(a.workers, len(allnames))) as ex:
        for n, mv in ex.map(_mat, allnames, [tmp] * len(allnames)):
            mesh[n] = mv
        for n, Ls in zip(allnames, ex.map(C.lengths_of, [C.cons_path(n) for n in allnames])):
            lens[n] = Ls
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
                               tend_ms=C.TEND, reference=C.ref_name(p), renumbering=0, etag_key='etag (A2)',
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
                    Ls = lens[n]
                    with h5py.File(C.cons_path(n), 'r') as f:
                        src, vox = f.attrs['source'], float(f.attrs['vox'])
                    r = meta[n]
                    g_.attrs.update(source=src, vox=vox, n_nodes=T.shape[1], unactivated=unact, **Ls,
                                    rv_wall_mm=float(r.rv_wall_mm), analysis_set=str(r.analysis_set),
                                    include_primary=bool(r.include_primary),
                                    include_sensitivity=bool(r.include_sensitivity),
                                    is_reference=bool(r.is_reference), g1_usable=bool(r.g1_usable),
                                    root_flag=bool(r.root_flag) if r.root_flag == r.root_flag else False,
                                    root_shift_mm=float(r.root_shift_mm), etag='etag (A2)')
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
        byk = {}
        for x in rr:
            byk.setdefault(x[0][0], []).append(x)
        for hfile in sorted({k[0] for k in newpts}):
          with h5py.File(hfile, 'a') as H:
            for (hf_, name), v in newpts.items():
                if hf_ != hfile or not len(v):
                    continue
                acts = np.vstack([x[1] for x in byk[(hfile, name)]])
                walls = np.array([x[2] for x in byk[(hfile, name)]])
                g = H['cons'][name]
                base = eplib.GridEmulator(C.G1, C.G2, g['grid_T'][:])
                e0 = np.vstack([base(th) for th in v]).astype(np.float32)
                for key, val in [('ref_theta', v), ('ref_T', acts), ('ref_Temu0', e0), ('ref_wall_s', walls),
                                 ('ref_round', np.full(len(v), rnd + 1))]:
                    if key in g:
                        val = np.concatenate([g[key][:], val]); del g[key]
                    g.create_dataset(key, data=val)
    for hfile in sorted({w[0] for w in work}):
      with h5py.File(hfile, 'a') as H:
        for hf_, name in work:
            if hf_ != hfile:
                continue
            g = H['cons'][name]
            g['calib'] = np.stack([cal[(r, name)] for r in range(3)])      # (round, istar, loss, [v_endo, v_t, fval])
            g['calib'].attrs['axes'] = 'round(0,1,2) x theta* x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)'

    # ---------------------------------------------------------------- phase 3: arm N on the reference emulator
    if not a.skip_noise:
        t1 = time.time()
        tasks = [(hf[p], C.ref_name(p), i, ci, lev, a.nrep) for p in a.hearts for i in range(3)
                 for ci, (cn, levs, _) in enumerate(C.COMPARATORS) for lev in levs]
        tasks.sort(key=lambda t: t[3] == 3, reverse=True)       # displacement tasks first (slightly longer)
        noise = {}
        with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
            for t, (key, out, seeds) in zip(tasks, ex.map(w_noise, tasks)):
                noise[(t[0],) + key] = (out, seeds)
        for p in a.hearts:
            with h5py.File(hf[p], 'a') as H:
                if 'noise' in H:
                    del H['noise']
                for ci, (cn, levs, unit) in enumerate(C.COMPARATORS):
                    arr = np.full((3, len(levs), a.nrep, len(C.LOSSES), 3), np.nan)
                    sd = np.zeros((3, len(levs), a.nrep), np.int64)
                    for i in range(3):
                        for li, lev in enumerate(levs):
                            arr[i, li], sd[i, li] = noise[(hf[p], i, ci, lev)]
                    d = H.create_dataset(f'noise/{cn}/theta_hat', data=arr)
                    H[f'noise/{cn}/rng_seed'] = sd
                    d.attrs.update(axes='theta* x level x replicate x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)',
                                   levels=levs, unit=unit, comparator_index=ci,
                                   seed_rule=('np.random.default_rng([20261028, patient, istar, comparator_index, '
                                              'level, replicate]); same draw for both losses' if cn != 'disp' else
                                              'displaced lead field replicate from mesh2: seed 20261028 + 1000 '
                                              'patient + 100 sd_index + rep (rng_seed holds it)'),
                                   emulator='reference construction, final corrected emulator (round 2 points)')
        log(f'noise arm: {len(tasks)} tasks, {time.time()-t1:.0f}s')

    log('done; total', f'{time.time()-T0:.0f}s')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
