"""Arm T eikonal runs, calibration and forward similarity (owner armT, 2026-10-28).

Per heart (own construction, fibre velocity 0.6 m/s, renumbering 0, tend 1000 ms, one openCARP process per core):
  81 grid runs (9 x 9 log grid over the A4 prior), 3 theta* target runs (A4 Latin hypercube, seed 20261028 + id),
  6 held-out runs (emulator check, seed [20261028, id, 1]).
Targets: 8-lead activation-template ECG of the direct theta* run in the heart's OWN torso (W_own).
Arms: 'own' (control, own torso) and, for each reference torso RR in {09, 17} with RR != heart, 'RR_icp' and 'RR_trans'.
Calibration: eplib.calibrate (25 x 25 log grid search, Nelder-Mead, prior box) on a cubic-spline emulator; round 0 on the
grid emulator, then one refinement run at every distinct round-0 theta-hat of the heart (all arms, both losses; points
within 0.01 log units merged), round 1 on the corrected emulator (gates CorrectedEmulator). Round 1 is reported.
Every concurrent openCARP run has its own directory under $TMPDIR/<job>/.
Usage: python -I t04_runs.py --workers 126 [--hearts ...]
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402
eplib, carpio = L.eplib, L.carpio
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def materialize(hid, tmp):
    out = os.path.join(tmp, f'm_{hid}')
    os.makedirs(out, exist_ok=True)
    with h5py.File(L.heart_file(hid), 'r') as f:
        C = f['cons']
        base = os.path.join(out, f'h{hid}')
        carpio.write_mesh(base, C['pts'][:] * 1000.0, C['tets'][:].astype(np.int64), C['etag'][:].astype(np.int64),
                          C['fib'][:].astype(np.float64))
        carpio.write_vtx(base + '_stim.vtx', C['stim'][:])
    return hid, (base, base + '_stim.vtx')


def arms_of(hid):
    with h5py.File(L.heart_file(hid), 'r') as f:
        return ['own'] + [k for k in f['reg'] if not k.endswith('_self')]


def W_of(f, arm):
    return f['cons/W_own'][:] if arm == 'own' else f[f'reg/{arm}/W'][:]


def load_emu(f, rnd):
    base = eplib.GridEmulator(L.G1, L.G2, f['runs/grid_T'][:])
    if rnd == 0 or 'runs/ref_theta' not in f:
        return base, L.CorrectedEmulator(base)
    return base, L.CorrectedEmulator(base, f['runs/ref_theta'][:], f['runs/ref_T'][:] - f['runs/ref_Temu0'][:])


def w_cal(args):
    hid, arm, rnd = args
    t0 = time.time()
    with h5py.File(L.heart_file(hid), 'r') as f:
        _, emu = load_emu(f, rnd)
        tmax = float(f['runs'].attrs['tmax_ms']); tgt = f['runs/target_ecg'][:]
        ecg = eplib.ECGModel(W_of(f, arm), tmax=tmax)
    out = np.full((len(tgt), len(L.LOSSES), 3), np.nan)
    for i in range(len(tgt)):
        for j, (ln, lf) in enumerate(L.LOSSES):
            th, fv = L.calibrate(emu, ecg, tgt[i], lf)
            out[i, j] = [th[0], th[1], fv]
    return (hid, arm, rnd), out, time.time() - t0


def w_post(hid):
    """Held-out emulator error (grid emulator, own torso) and forward similarity of ECGs own vs each arm at theta*."""
    with h5py.File(L.heart_file(hid), 'r') as f:
        base, _ = load_emu(f, 0)
        tmax = float(f['runs'].attrs['tmax_ms'])
        Wo = f['cons/W_own'][:]
        eo = eplib.ECGModel(Wo, tmax=tmax)
        held = []
        for th, td in zip(f['runs/held_theta'][:], f['runs/held_T'][:]):
            te = base(th)
            held.append([float(np.sqrt(((te - td) ** 2).mean())), float(np.abs(te - td).max()),
                         float(eplib.loss_corr(eo(te), eo(td))), float(eplib.loss_amp(eo(te), eo(td)))])
        sim = {}
        Tt = f['runs/target_T'][:]
        for arm in [k for k in f['reg']]:
            ea = eplib.ECGModel(f[f'reg/{arm}/W'][:], tmax=tmax)
            rows = []
            for i in range(len(Tt)):
                a, b = eo(Tt[i]), ea(Tt[i])
                rows.append([1 - float(eplib.loss_corr(b, a)), float(np.sqrt((b ** 2).mean() / (a ** 2).mean()))])
            sim[arm] = rows
    return hid, np.array(held), sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=126); ap.add_argument('--hearts', nargs='+', default=L.HEARTS)
    a = ap.parse_args()
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'armT_runs_{os.environ.get("SLURM_JOB_ID", "x")}')
    os.makedirs(tmp, exist_ok=True)
    hearts = a.hearts
    mesh = {}
    with ProcessPoolExecutor(min(a.workers, len(hearts))) as ex:
        for h, mv in ex.map(materialize, hearts, [tmp] * len(hearts)):
            mesh[h] = mv
    log('materialized', len(mesh))

    def run(hid, th, tag):
        mb, vtx = mesh[hid]
        return carpio.run_eikonal0(mb, vtx, [(1, L.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                   os.path.join(tmp, f'r_{hid}_{tag}'), threads=1, tend=L.TEND)

    # ------------------------------------------------------------------ phase 1
    need = []
    for h in hearts:
        with h5py.File(L.heart_file(h), 'r') as f:
            if 'runs' not in f or not f['runs'].attrs.get('phase1_done', False):
                need.append(h)
    TH = L.grid_thetas()
    jobs = []
    for h in need:
        for i, th in enumerate(TH):
            jobs.append((h, 'g', i, th))
        for i, th in enumerate(L.theta_star(h)):
            jobs.append((h, 't', i, th))
        for i, th in enumerate(L.held_out(h)):
            jobs.append((h, 'h', i, th))
    jobs.sort(key=lambda j: -np.log(j[3]).sum(), reverse=True)    # slowest (low velocity) first

    def one(j):
        h, k, i, th = j
        try:
            act, wall = run(h, th, f'{k}{i}')
        except Exception as e:      # noqa: BLE001
            log('FAILED', h, k, i, str(e)[:300]); act, wall = np.full(1, -1.0), -1.0
        return j, act.astype(np.float32), wall
    res = {}
    t1 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for n, (j, act, wall) in enumerate(ex.map(one, jobs)):
            res[(j[0], j[1], j[2])] = (act, wall)
            if n % 300 == 0:
                log(f'phase 1 {n}/{len(jobs)}')
    log(f'phase 1: {len(jobs)} runs, {time.time()-t1:.0f}s')
    for h in need:
        G = np.vstack([res[(h, 'g', i)][0] for i in range(len(TH))])
        Tt = np.vstack([res[(h, 't', i)][0] for i in range(3)])
        Th = np.vstack([res[(h, 'h', i)][0] for i in range(L.N_HELD)])
        walls = np.array([res[(h, k, i)][1] for k, n_ in (('g', len(TH)), ('t', 3), ('h', L.N_HELD)) for i in range(n_)])
        allT = np.vstack([G, Tt, Th])
        unact = int(((allT > 1e6) | (allT < 0)).sum())
        tmax = float(allT[allT < 1e6].max()) + 20.0
        with h5py.File(L.heart_file(h), 'a') as f:
            if 'runs' in f:
                del f['runs']
            r = f.create_group('runs')
            r.create_dataset('grid_T', data=G, compression='gzip', compression_opts=1)
            r['grid_theta'] = TH; r['target_theta'] = L.theta_star(h); r['target_T'] = Tt
            r['held_theta'] = L.held_out(h); r['held_T'] = Th; r['wall_s'] = walls
            W = f['cons/W_own'][:]
            ecg = eplib.ECGModel(W, tmax=tmax)
            r['target_ecg'] = np.stack([ecg(Tt[i]) for i in range(3)])
            r.attrs.update(tmax_ms=tmax, unactivated=unact, seed_theta_star=L.SEED_BASE + L.pid_of(h), tend_ms=L.TEND,
                           renumbering=0, opencarp=carpio.OPENCARP, v_fib=L.V_FIB, phase1_done=bool(unact == 0),
                           median_wall_s=float(np.median(walls)))
        log(f'heart {h}: tmax {tmax:.1f} ms, unactivated {unact}, median wall {np.median(walls):.1f}s, '
            f'theta* {L.theta_star(h).round(3).tolist()}')

    # ------------------------------------------------------------------ phase 2: calibration rounds
    arms = {h: arms_of(h) for h in hearts}
    work = [(h, arm) for h in hearts for arm in arms[h]]
    for h in hearts:
        with h5py.File(L.heart_file(h), 'a') as f:
            for k in ['ref_theta', 'ref_T', 'ref_Temu0', 'ref_wall_s']:
                if f'runs/{k}' in f:
                    del f[f'runs/{k}']
            if 'calib' in f:
                del f['calib']
    cal = {}
    for rnd in (0, 1):
        t1 = time.time()
        with ProcessPoolExecutor(min(a.workers, len(work))) as ex:
            for key, out, wall in ex.map(w_cal, [(h, arm, rnd) for h, arm in work]):
                cal[key] = out
        log(f'calibration round {rnd}: {len(work)} heart-arms, {time.time()-t1:.0f}s')
        if rnd == 1:
            break
        newpts = {}
        for h in hearts:
            cand = np.vstack([cal[(h, arm, 0)][:, :, :2].reshape(-1, 2) for arm in arms[h]])
            pts = []
            for th in cand:
                if pts and np.min(np.abs(np.log(np.array(pts)) - np.log(th)).max(1)) < 0.01:
                    continue
                pts.append(th)
            newpts[h] = np.array(pts)
        rj = [(h, i, th) for h in hearts for i, th in enumerate(newpts[h])]

        def one2(j):
            h, i, th = j
            try:
                act, wall = run(h, th, f'f{i}')
            except Exception as e:      # noqa: BLE001
                log('FAILED refinement', h, i, str(e)[:300]); act, wall = None, -1.0
            return j, act, wall
        t1 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rr = list(ex.map(one2, rj))
        log(f'refinement: {len(rj)} runs, {time.time()-t1:.0f}s')
        for h in hearts:
            ok = [(x[0][2], x[1], x[2]) for x in rr if x[0][0] == h and x[1] is not None and (x[1] < 1e6).all()]
            if not ok:
                log('WARNING no refinement runs for', h)
                continue
            with h5py.File(L.heart_file(h), 'a') as f:
                base = eplib.GridEmulator(L.G1, L.G2, f['runs/grid_T'][:])
                th = np.array([o[0] for o in ok]); acts = np.vstack([o[1] for o in ok]).astype(np.float32)
                f['runs/ref_theta'] = th; f['runs/ref_T'] = acts
                f['runs/ref_Temu0'] = np.vstack([base(t) for t in th]).astype(np.float32)
                f['runs/ref_wall_s'] = np.array([o[2] for o in ok])
    rows = []
    for h in hearts:
        ts = L.theta_star(h)
        with h5py.File(L.heart_file(h), 'a') as f:
            for arm in arms[h]:
                arr = np.stack([cal[(h, arm, r)] for r in (0, 1)])
                d = f.create_dataset(f'calib/{arm}', data=arr)
                d.attrs['axes'] = 'round(0,1) x theta* x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)'
                for r in (0, 1):
                    for i in range(3):
                        for j, (ln, _) in enumerate(L.LOSSES):
                            th = arr[r, i, j, :2]
                            ref, reg = ('own', 'own') if arm == 'own' else arm.split('_')
                            rows.append(dict(heart=h, group=L.group_of(h), ref=ref, reg=reg, loss=ln, round=r, istar=i,
                                             v_endo_star=ts[i, 0], v_t_star=ts[i, 1], v_endo_hat=th[0], v_t_hat=th[1],
                                             fval=arr[r, i, j, 2], bound_hit=int(L.bound_hit(th)),
                                             rel_v_endo=th[0] / ts[i, 0] - 1, rel_v_t=th[1] / ts[i, 1] - 1))
    with open(os.path.join(L.OUT, 'estimates.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    log('estimates.csv rows', len(rows))

    # ------------------------------------------------------------------ phase 3: held-out error, forward similarity
    post = {}
    with ProcessPoolExecutor(min(a.workers, len(hearts))) as ex:
        for h, held, sim in ex.map(w_post, hearts):
            post[h] = dict(held=held.tolist(), sim=sim)
    json.dump(post, open(os.path.join(L.OUT, 'post.json'), 'w'), indent=0)
    shutil.rmtree(tmp, ignore_errors=True)
    log('done')


if __name__ == '__main__':
    main()
