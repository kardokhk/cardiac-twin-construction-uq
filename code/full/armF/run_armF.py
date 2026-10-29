"""Arm F driver (owner armF, I11 full run, 2026-10-28). Definitions: armf_lib.py docstring.

Phases (the driver is the only writer of the per-heart HDF5; calibration workers only read):
  1  pure eikonal: 81 grid runs + 3 control-target runs per unit (unit = heart x resolution)
  2  reaction-eikonal targets (one openCARP RE+ per core, own scratch directory per run), ECG from Vm
  3  calibration rounds 0, 1, 2 with refinement runs between rounds
Usage: python -I -B run_armF.py --out <dir> --hearts 101 ... --workers 124 [--conv-heart 101] [--conv-mesh 1.0]
  --conv-heart: also runs RE at dt 12.5 us on that heart's 1.5 mm unit (set 're_dt12p5'); with --conv-mesh 1.0 it
  also builds the 1.0 mm unit of that heart (emulator, control and RE targets) for the mesh check.
Writes <out>/heart_patientNNN.h5.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armf_lib as F          # noqa: E402

A, eplib, carpio = F.A, F.eplib, F.carpio
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def ukey(pid, vox):
    return f'{vox:.1f}'


def load_emulator(hfile, vk):
    with h5py.File(hfile, 'r') as H:
        g = H['units'][vk]
        base = eplib.GridEmulator(F.G1, F.G2, g['grid_T'][:])
        if 'ref_theta' in g and len(g['ref_theta']):
            emu = F.CorrectedEmulator(base, g['ref_theta'][:], g['ref_T'][:] - g['ref_Temu0'][:])
        else:
            emu = F.CorrectedEmulator(base)
        tmax = float(g.attrs['tmax_ms']); cp = g.attrs['cons_path']
    with h5py.File(cp, 'r') as f:
        W = f['W'][:]
    return base, emu, eplib.ECGModel(W, tmax=tmax)


def w_calibrate(args):
    """One target (unit, set, istar): both losses x both windows on the unit's current emulator."""
    hfile, vk, sname, i = args
    t0 = time.time()
    _, emu, ecg = load_emulator(hfile, vk)
    with h5py.File(hfile, 'r') as H:
        g = H['units'][vk]['targets'][sname]
        tgt = g['ecg'][i]; amax = float(g['act_max_ms'][i])
        ts = H['theta_star'][i]
    out = np.full((len(F.LOSSES), len(F.WINDOWS), 4), np.nan)  # v_endo_hat, v_t_hat, fval, loss at theta*
    for w, wn in enumerate(F.WINDOWS):
        nw = F.qrs_nw(amax) if wn == 'qrs' else tgt.shape[1]
        nw = min(nw, tgt.shape[1])
        e = F.WinECG(ecg, nw); tg = tgt[:, :nw]
        for j, (ln, lf) in enumerate(F.LOSSES):
            th, fv = F.calibrate(emu, e, tg, lf)
            out[j, w] = [th[0], th[1], fv, lf(e(emu(ts)), tg)]
    return (hfile, vk, sname, i), out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--hearts', type=int, nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=124)
    ap.add_argument('--conv-heart', type=int, default=None); ap.add_argument('--conv-mesh', type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'armF_{os.getpid()}')
    os.makedirs(tmp, exist_ok=True)
    sel, groups = F.select_hearts()
    log('selection rule gives', groups)
    hf = {p: os.path.join(a.out, f'heart_patient{p:03d}.h5') for p in a.hearts}
    units = [(p, 1.5) for p in a.hearts]
    if a.conv_heart is not None and a.conv_mesh is not None:
        units.append((a.conv_heart, a.conv_mesh))
    TS = {p: F.theta_star(p) for p in a.hearts}

    # ---------------------------------------------------------------- materialize
    mesh = {}
    for (p, v) in units:
        mesh[(p, v)] = A.materialize(F.cons_path(p, v), os.path.join(tmp, f'm_{p}_{v:.1f}'), etag_key='etag')
    W = {}
    for (p, v) in units:
        with h5py.File(F.cons_path(p, v), 'r') as f:
            W[(p, v)] = f['W'][:]
            assert f.attrs['source'] == 'expert' and abs(float(f.attrs['vox']) - v) < 1e-9
    log('materialized', len(mesh))

    # ---------------------------------------------------------------- phase 1: pure eikonal
    TH = F.grid_thetas()
    jobs = []
    for (p, v) in units:
        for i in range(3):
            jobs.append(((p, v), 'tmpl', i, TS[p][i]))
        for g in range(len(TH)):
            jobs.append(((p, v), 'grid', g, TH[g]))
    jobs.sort(key=lambda j: (j[0][1] < 1.25, -np.log(j[3]).sum()), reverse=True)

    def one(j):
        u, kind, g, th = j
        mb, vtx = mesh[u]
        act, wall = carpio.run_eikonal0(mb, vtx, F.regions(th), os.path.join(tmp, f'e_{u[0]}_{u[1]:.1f}_{kind}_{g}'),
                                        threads=1, tend=F.TEND_EIK)
        return j, act.astype(np.float32), wall
    res = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (j, act, wall) in enumerate(ex.map(one, jobs)):
            res[(j[0], j[1], j[2])] = (act, wall)
    log('phase 1 eikonal runs', len(jobs))
    tmax = {}
    for u in units:
        acts = [res[(u, 'grid', g)][0] for g in range(len(TH))] + [res[(u, 'tmpl', i)][0] for i in range(3)]
        tmax[u] = F.tmax_of(np.concatenate(acts))
    log('tmax', {f'{u[0]}_{u[1]}': tmax[u] for u in units})

    # ---------------------------------------------------------------- phase 2: reaction-eikonal targets
    rejobs = []
    for u in units:
        for i in range(3):
            rejobs.append((u, 're', i, F.RE_DT_US))
    if a.conv_heart is not None:
        for i in range(3):
            rejobs.append(((a.conv_heart, 1.5), 're_dt12p5', i, F.RE_DT_US / 2))
    rejobs.sort(key=lambda j: (j[0][1] > 1.25, j[3] > 20))   # 1.0 mm and dt 12.5 first

    def one_re(j):
        u, sname, i, dt_us = j
        mb, vtx = mesh[u]
        sd = os.path.join(tmp, f're_{u[0]}_{u[1]:.1f}_{sname}_{i}')
        vm, act, wall, lg = carpio.run_reaction_eikonal(mb, vtx, F.regions(TS[u[0]][i]), sd, mode=1, np_mpi=1,
                                                       tend=tmax[u], dt_us=dt_us, spacedt=F.DT, keep=False,
                                                       layout='default0')
        nt = int(round(tmax[u] / F.DT)) + 1
        frames = vm.shape[0]
        if frames < nt:
            vm = np.vstack([vm, np.repeat(vm[-1:], nt - frames, 0)])
        vm = vm[:nt]
        sha = hashlib.sha256(np.ascontiguousarray(vm).tobytes()).hexdigest()
        ecg = F.re_ecg(vm, W[u])
        up, unact = F.upstroke_act(vm)
        warn = [l for l in lg.splitlines() if l.startswith('WARNING-LINE')]
        del vm
        return j, dict(ecg=ecg.astype(np.float64), act=act.astype(np.float32), act_up=up.astype(np.float32),
                       unact=unact, wall=wall, sha=sha, frames=frames, warn=len(warn))
    t1 = time.time()
    rres = {}
    with ThreadPoolExecutor(min(a.workers, len(rejobs))) as ex:
        for j, r in ex.map(one_re, rejobs):
            rres[(j[0], j[1], j[2])] = (j, r)
            log(f'RE {j[0]} {j[1]} {j[2]}: wall {r["wall"]:.0f}s frames {r["frames"]} unact {r["unact"]} warn {r["warn"]}')
    log(f'phase 2: {len(rejobs)} RE runs, {time.time() - t1:.0f}s')

    # ---------------------------------------------------------------- write per-heart HDF5
    for p in a.hearts:
        with h5py.File(hf[p], 'w') as H:
            H.attrs.update(patient=p, group=A.info_cfg(p).get('Group'), prior_lo=F.B_LO, prior_hi=F.B_HI,
                           tstar_lo=F.T_LO, tstar_hi=F.T_HI, seed_theta_star=F.SEED_BASE + p, v_fib=A.V_FIB,
                           tend_eikonal_ms=F.TEND_EIK, renumbering=0, opencarp=carpio.OPENCARP,
                           re_mode='RE+ (dream.solve 1), monodomain, tenTusscherPanfilov, layout default0',
                           ecg_vm='ECG_l(t) = -sum_n W_ln Vm_n(t) minus its t = 0 value',
                           windows=json.dumps(F.WINDOWS), losses=json.dumps([l for l, _ in F.LOSSES]),
                           created=time.strftime('%Y-%m-%dT%H:%M:%S'))
            H['theta_star'] = TS[p]
            H['grid/theta'] = TH
            for u in [x for x in units if x[0] == p]:
                g = H.create_group(f'units/{u[1]:.1f}')
                g.attrs.update(cons_path=F.cons_path(*u), tmax_ms=tmax[u], vox=u[1], n_nodes=W[u].shape[1])
                T = np.vstack([res[(u, 'grid', k)][0] for k in range(len(TH))])
                g.create_dataset('grid_T', data=T, compression='gzip', compression_opts=1)
                g['grid_wall_s'] = np.array([res[(u, 'grid', k)][1] for k in range(len(TH))])
                g.attrs['grid_unactivated'] = int(((T > 1e6) | (T < 0)).sum())
                ecgm = eplib.ECGModel(W[u], tmax=tmax[u])
                Tt = np.vstack([res[(u, 'tmpl', i)][0] for i in range(3)])
                s = g.create_group('targets/tmpl')
                s['act'] = Tt; s['ecg'] = np.stack([ecgm(Tt[i]) for i in range(3)])
                s['act_max_ms'] = Tt.max(1).astype(np.float64)
                s['wall_s'] = np.array([res[(u, 'tmpl', i)][1] for i in range(3)])
                s.attrs['kind'] = 'pure eikonal at theta*, activation-template ECG (arm C target)'
                for sname in sorted({k[1] for k in rres if k[0] == u}):
                    rr = [rres[(u, sname, i)][1] for i in range(3)]
                    s = g.create_group(f'targets/{sname}')
                    s['ecg'] = np.stack([r['ecg'] for r in rr])
                    s.create_dataset('act', data=np.stack([r['act'] for r in rr]), compression='gzip', compression_opts=1)
                    s.create_dataset('act_up', data=np.stack([r['act_up'] for r in rr]), compression='gzip',
                                     compression_opts=1)
                    am = [max(float(np.nanmax(r['act'][r['act'] >= 0])), float(np.nanmax(r['act_up']))) for r in rr]
                    s['act_max_ms'] = np.array(am)
                    s['wall_s'] = np.array([r['wall'] for r in rr])
                    s['frames'] = np.array([r['frames'] for r in rr]); s['unactivated'] = np.array([r['unact'] for r in rr])
                    s['n_warn_unassigned'] = np.array([r['warn'] for r in rr])
                    s.attrs['vm_sha256'] = json.dumps([r['sha'] for r in rr])
                    s.attrs['dt_us'] = rres[(u, sname, 0)][0][3]
                    s.attrs['kind'] = 'openCARP RE+ full Vm, ECG from Vm via W'
                    # template from the RE run's own activation (separates activation from waveform differences)
                    s['ecg_tmpl_from_re_act'] = np.stack([ecgm(np.where(r['act'] >= 0, r['act'], 1e6)) for r in rr])
    log('HDF5 written')

    # ---------------------------------------------------------------- phase 3: calibration rounds
    tasks = []
    for p in a.hearts:
        with h5py.File(hf[p], 'r') as H:
            for vk in H['units']:
                for sname in H['units'][vk]['targets']:
                    for i in range(3):
                        tasks.append((hf[p], vk, sname, i))
    cal = {}
    for rnd in range(3):
        t1 = time.time()
        with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
            for key, out, wall in ex.map(w_calibrate, tasks):
                cal[(rnd,) + key] = out
        log(f'calibration round {rnd}: {len(tasks)} targets, {time.time() - t1:.0f}s')
        if rnd == 2:
            break
        newpts = {}
        for p in a.hearts:
            with h5py.File(hf[p], 'r') as H:
                for vk in H['units']:
                    g = H['units'][vk]
                    old = g['ref_theta'][:] if 'ref_theta' in g else np.zeros((0, 2))
                    cand = []
                    for (r_, h_, v_, s_, i_), out in cal.items():
                        if r_ == rnd and h_ == hf[p] and v_ == vk:
                            # QRS window (index 0); round 0: both losses; round 1: primary loss only
                            cand += [out[j, 0, :2] for j in ((0, 1) if rnd == 0 else (0,))]
                    pts = []
                    for th in cand:
                        allp = np.vstack([old] + ([np.array(pts)] if pts else []))
                        if len(allp) and np.min(np.abs(np.log(allp) - np.log(th)).max(1)) < 0.01:
                            continue
                        pts.append(th)
                    newpts[(p, vk)] = np.array(pts).reshape(-1, 2)
        jobs = [(k, i, th) for k, v in newpts.items() for i, th in enumerate(v)]
        jobs.sort(key=lambda j: j[0][1] == '1.0', reverse=True)

        def one2(j):
            (p, vk), i, th = j
            mb, vtx = mesh[(p, float(vk))]
            act, wall = carpio.run_eikonal0(mb, vtx, F.regions(th), os.path.join(tmp, f'f_{p}_{vk}_{rnd}_{i}'),
                                            threads=1, tend=F.TEND_EIK)
            return j, act.astype(np.float32), wall
        t1 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rr = list(ex.map(one2, jobs))
        log(f'refinement round {rnd + 1}: {len(jobs)} runs, {time.time() - t1:.0f}s')
        for (p, vk), v in newpts.items():
            if not len(v):
                continue
            acts = np.vstack([x[1] for x in rr if x[0][0] == (p, vk)])
            walls = np.array([x[2] for x in rr if x[0][0] == (p, vk)])
            with h5py.File(hf[p], 'a') as H:
                g = H['units'][vk]
                base = eplib.GridEmulator(F.G1, F.G2, g['grid_T'][:])
                e0 = np.vstack([base(th) for th in v]).astype(np.float32)
                for key, val in [('ref_theta', v), ('ref_T', acts), ('ref_Temu0', e0), ('ref_wall_s', walls),
                                 ('ref_round', np.full(len(v), rnd + 1))]:
                    if key in g:
                        val = np.concatenate([g[key][:], val]); del g[key]
                    g.create_dataset(key, data=val)
    for p in a.hearts:
        with h5py.File(hf[p], 'a') as H:
            for vk in H['units']:
                for sname in H['units'][vk]['targets']:
                    arr = np.stack([np.stack([cal[(r, hf[p], vk, sname, i)] for i in range(3)]) for r in range(3)])
                    d = H['units'][vk]['targets'][sname].create_dataset('calib', data=arr)
                    d.attrs['axes'] = ('round(0,1,2) x theta* x loss(corr, amp) x window(qrs, full) x '
                                       '(v_endo_hat, v_t_hat, fval, loss_at_theta_star)')
    log('done; total', f'{time.time() - T0:.0f}s')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
