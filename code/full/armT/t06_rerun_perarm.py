"""Arm T follow-up 2026-10-28 (owner armT): calibration rerun with PER-ARM refinement and two refinement rounds.

Why: t04_runs.py pooled the round-0 estimates of ALL arms of a heart (including 'own', whose estimate is about theta*)
into one refinement set shared by every arm, so every theta* had a refinement run within about 0.004 log units and the
kriged correction (nugget 1e-6) made the emulator near-exact at the truth for the reference-torso arms too.

What this script does (t04 outputs are kept; nothing of t04 is overwritten):
  * reuses the phase-1 runs stored by t04 (81 grid runs, 3 theta* targets per heart; targets in the own torso);
  * for every heart-arm (own, RR_icp, RR_trans) a SEPARATE refinement set, filled only from that arm's own estimates:
      round 0 on the grid emulator -> refinement at that arm's round-0 estimates (both losses, 6 candidates);
      round 1 on the arm's corrected emulator -> refinement at that arm's round-1 corr-loss estimates (3 candidates);
      round 2 reported.  Same candidate rule as the gates (code/gates/calib/run_ministudy.py lines 197-210: round 0 both
      losses, round 1 primary loss, points within 0.01 log units of an existing point of the same set merged);
  * no arm ever sees another arm's points, so no reference arm has a point near theta* unless it proposed it;
  * every refinement point is logged with its distance to the nearest theta* in grid half-steps (half-step = STEP/2,
    0.13 log units for both parameters).
Writes groups refv2/<arm>/{theta,T,Temu0,round,wall_s} and calib_v2/<arm> (round 0,1,2 x theta* x loss x 3) into each
heart HDF5, results/2026-10-28_armT/estimates_v2.csv and refinement_v2.csv.
Usage: python -I t06_rerun_perarm.py --workers 126 [--hearts ...]
"""
import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402
import t04_runs as R4         # noqa: E402  (materialize, arms_of, W_of reused unchanged)
eplib, carpio = L.eplib, L.carpio
T0 = time.time()
NROUND = 3                    # rounds 0, 1, 2
HALF = L.STEP / 2


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def load_emu_arm(f, arm, rnd):
    """Grid emulator plus kriging of this arm's own refinement residuals with round index < rnd + 1 (i.e. added before
    calibration round rnd)."""
    base = eplib.GridEmulator(L.G1, L.G2, f['runs/grid_T'][:])
    g = f.get(f'refv2/{arm}')
    if rnd == 0 or g is None:
        return L.CorrectedEmulator(base)
    m = g['round'][:] <= rnd
    if not m.any():
        return L.CorrectedEmulator(base)
    return L.CorrectedEmulator(base, g['theta'][:][m], g['T'][:][m] - g['Temu0'][:][m])


def w_cal(args):
    hid, arm, rnd = args
    t0 = time.time()
    with h5py.File(L.heart_file(hid), 'r') as f:
        emu = load_emu_arm(f, arm, rnd)
        tmax = float(f['runs'].attrs['tmax_ms']); tgt = f['runs/target_ecg'][:]
        ecg = eplib.ECGModel(R4.W_of(f, arm), tmax=tmax)
        npts = int((f[f'refv2/{arm}/round'][:] <= rnd).sum()) if (rnd > 0 and f'refv2/{arm}' in f) else 0
    out = np.full((len(tgt), len(L.LOSSES), 3), np.nan)
    for i in range(len(tgt)):
        for j, (ln, lf) in enumerate(L.LOSSES):
            th, fv = L.calibrate(emu, ecg, tgt[i], lf)
            out[i, j] = [th[0], th[1], fv]
    return (hid, arm, rnd), out, npts, time.time() - t0


def halfsteps_to_star(th, ts):
    """Chebyshev distance from th to the nearest theta* in grid half-steps (log space)."""
    return float(np.min(np.max(np.abs(np.log(ts) - np.log(th)) / HALF, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=126); ap.add_argument('--hearts', nargs='+', default=L.HEARTS)
    a = ap.parse_args()
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'armT_v2_{os.environ.get("SLURM_JOB_ID", "x")}')
    os.makedirs(tmp, exist_ok=True)
    hearts = a.hearts
    for h in hearts:
        with h5py.File(L.heart_file(h), 'r') as f:
            assert f['runs'].attrs.get('phase1_done', False), f'phase 1 missing for {h}'
    mesh = {}
    with ProcessPoolExecutor(min(a.workers, len(hearts))) as ex:
        for h, mv in ex.map(R4.materialize, hearts, [tmp] * len(hearts)):
            mesh[h] = mv
    log('materialized', len(mesh))
    arms = {h: R4.arms_of(h) for h in hearts}
    work = [(h, arm) for h in hearts for arm in arms[h]]
    for h in hearts:                                   # reset only the v2 groups
        with h5py.File(L.heart_file(h), 'a') as f:
            for k in ('refv2', 'calib_v2'):
                if k in f:
                    del f[k]
    cal, npts_used, refrows = {}, {}, []
    for rnd in range(NROUND):
        t1 = time.time()
        with ProcessPoolExecutor(min(a.workers, len(work))) as ex:
            for key, out, npts, wall in ex.map(w_cal, [(h, arm, rnd) for h, arm in work]):
                cal[key] = out; npts_used[key] = npts
        log(f'calibration round {rnd}: {len(work)} heart-arms, {time.time()-t1:.0f}s')
        if rnd == NROUND - 1:
            break
        newpts = {}
        for h, arm in work:
            with h5py.File(L.heart_file(h), 'r') as f:
                old = f[f'refv2/{arm}/theta'][:] if f'refv2/{arm}' in f else np.zeros((0, 2))
            out = cal[(h, arm, rnd)]
            cand = out[:, :, :2].reshape(-1, 2) if rnd == 0 else out[:, 0, :2]     # gates rule
            pts = []
            for th in cand:
                allp = np.vstack([old] + ([np.array(pts)] if pts else []))
                if len(allp) and np.min(np.abs(np.log(allp) - np.log(th)).max(1)) < 0.01:
                    continue
                pts.append(th)
            newpts[(h, arm)] = np.array(pts).reshape(-1, 2)
        jobs = [(k, i, th) for k, v in newpts.items() for i, th in enumerate(v)]
        jobs.sort(key=lambda j: -np.log(j[2]).sum(), reverse=True)              # slow (low velocity) first

        def one2(j):
            (h, arm), i, th = j
            mb, vtx = mesh[h]
            try:
                act, wall = carpio.run_eikonal0(mb, vtx, [(1, L.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                                os.path.join(tmp, f'f_{h}_{arm}_{rnd}_{i}'), threads=1, tend=L.TEND)
            except Exception as e:      # noqa: BLE001
                log('FAILED refinement', h, arm, rnd, i, str(e)[:300]); act, wall = None, -1.0
            return j, act, wall
        t1 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rr = list(ex.map(one2, jobs))
        log(f'refinement round {rnd + 1}: {len(jobs)} runs, {time.time()-t1:.0f}s')
        for (h, arm), v in newpts.items():
            ok = [x for x in rr if x[0][0] == (h, arm) and x[1] is not None and (x[1] < 1e6).all() and (x[1] >= 0).all()]
            nfail = len(v) - len(ok)
            if nfail:
                log('WARNING', nfail, 'failed/unactivated refinement runs', h, arm, rnd)
            if not ok:
                continue
            th = np.array([x[0][2] for x in ok]); acts = np.vstack([x[1] for x in ok]).astype(np.float32)
            walls = np.array([x[2] for x in ok])
            ts = L.theta_star(h)
            with h5py.File(L.heart_file(h), 'a') as f:
                base = eplib.GridEmulator(L.G1, L.G2, f['runs/grid_T'][:])
                e0 = np.vstack([base(t) for t in th]).astype(np.float32)
                gname = f'refv2/{arm}'
                for key, val in [('theta', th), ('T', acts), ('Temu0', e0), ('wall_s', walls),
                                 ('round', np.full(len(th), rnd + 1))]:
                    if f'{gname}/{key}' in f:
                        val = np.concatenate([f[f'{gname}/{key}'][:], val]); del f[f'{gname}/{key}']
                    f.create_dataset(f'{gname}/{key}', data=val)
                f[gname].attrs['source_arm'] = arm
            for t_, w_ in zip(th, walls):
                refrows.append(dict(heart=h, arm=arm, added_before_round=rnd + 1, v_endo=t_[0], v_t=t_[1],
                                    proposed_by=arm, halfsteps_to_nearest_theta_star=halfsteps_to_star(t_, ts),
                                    within_half_step=int(halfsteps_to_star(t_, ts) < 1), wall_s=w_))
    rows = []
    for h in hearts:
        ts = L.theta_star(h)
        with h5py.File(L.heart_file(h), 'a') as f:
            for arm in arms[h]:
                arr = np.stack([cal[(h, arm, r)] for r in range(NROUND)])
                d = f.create_dataset(f'calib_v2/{arm}', data=arr)
                d.attrs['axes'] = 'round(0,1,2) x theta* x loss(corr, amp) x (v_endo_hat, v_t_hat, fval)'
                d.attrs['refinement'] = 'per arm, own estimates only (t06_rerun_perarm.py)'
                for r in range(NROUND):
                    for i in range(3):
                        for j, (ln, _) in enumerate(L.LOSSES):
                            th = arr[r, i, j, :2]
                            ref, reg = ('own', 'own') if arm == 'own' else arm.split('_')
                            rows.append(dict(heart=h, group=L.group_of(h), ref=ref, reg=reg, loss=ln, round=r, istar=i,
                                             v_endo_star=ts[i, 0], v_t_star=ts[i, 1], v_endo_hat=th[0], v_t_hat=th[1],
                                             fval=arr[r, i, j, 2], bound_hit=int(L.bound_hit(th)),
                                             rel_v_endo=th[0] / ts[i, 0] - 1, rel_v_t=th[1] / ts[i, 1] - 1,
                                             n_refinement_points=npts_used[(h, arm, r)]))
    with open(os.path.join(L.OUT, 'estimates_v2.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(L.OUT, 'refinement_v2.csv'), 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(refrows[0])); w.writeheader(); w.writerows(refrows)
    log('estimates_v2.csv rows', len(rows), 'refinement_v2.csv rows', len(refrows))
    shutil.rmtree(tmp, ignore_errors=True)
    log('done')


if __name__ == '__main__':
    main()
