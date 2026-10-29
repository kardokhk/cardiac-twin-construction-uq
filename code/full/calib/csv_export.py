"""Tidy CSVs from the per-heart HDF5 files of the full arms C and N (owner calibC, 2026-10-28). Adapted from
code/gates/calib/csv_export.py. Usage: python -I -B csv_export.py <out_dir> [hearts...]  (default: every heart file)
Reads <out_dir>/hearts/heart_patientNNN.h5. Writes into <out_dir>:
  estimates.csv          arm C, final round (2) only, one row per construction x theta* x loss
  estimates_rounds.csv.gz arm C, all calibration rounds 0 to 2
  noise_estimates.csv    arm N, one row per heart x theta* x comparator x level x replicate x loss
  refinement.csv         every refinement run (emulator error, simulated losses against the 3 targets)
  constructions.csv      one row per calibrated construction
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_lib as C        # noqa: E402

LN = [l for l, _ in C.LOSSES]


def flags(th):
    lo, hi = C.bound_hit(th)
    return dict(bound_lo_v_endo=bool(lo[0]), bound_hi_v_endo=bool(hi[0]), bound_lo_v_t=bool(lo[1]),
                bound_hi_v_t=bool(hi[1]), censored=bool(lo.any() or hi.any()),
                flag_vt_gt_vfib=bool(th[1] > C.A.V_FIB))


def export_heart(out, p):
    est, noi, ref, con = [], [], [], []
    if True:
        with h5py.File(os.path.join(out, 'hearts', f'heart_patient{p:03d}.h5'), 'r') as H:
            grp = H.attrs['group']; ts = H['targets/theta_star'][:]; refn = H.attrs['reference']
            tmax = float(H.attrs['tmax_ms']); tgt = H['targets/ecg'][:]
            for n in H['cons']:
                g = H['cons'][n]; at = dict(g.attrs)
                base = dict(patient=p, group=grp, construction=n, source=at['source'], resolution_mm=at['vox'],
                            rv_wall_mm=at['rv_wall_mm'], analysis_set=at['analysis_set'], is_reference=n == refn,
                            include_primary=bool(at['include_primary']),
                            include_sensitivity=bool(at['include_sensitivity']), g1_usable=bool(at['g1_usable']),
                            root_flag=bool(at['root_flag']), root_shift_mm=float(at['root_shift_mm']))
                nref = int(len(g['ref_theta'])) if 'ref_theta' in g else 0
                con.append(dict(base, n_nodes=int(at['n_nodes']), unactivated=int(at['unactivated']),
                                L_scale=at['L_scale'], L_wall=at['L_wall'], L_base=at['L_base'], myo_mL=at['myo_mL'],
                                lv_myo_mL=at['lv_myo_mL'], lv_cav_mL=at['lv_cav_mL'],
                                grid_wall_median_s=float(np.median(g['grid_wall_s'][:])), n_refinement=nref))
                cal = g['calib'][:]
                for r in range(cal.shape[0]):
                    for i in range(cal.shape[1]):
                        for j, l in enumerate(LN):
                            th = cal[r, i, j, :2]
                            est.append(dict(base, istar=i, v_endo_star=ts[i, 0], v_t_star=ts[i, 1], loss=l, round=r,
                                            v_endo_hat=th[0], v_t_hat=th[1], fval_emu=cal[r, i, j, 2],
                                            ratio_v_endo=th[0] / ts[i, 0], ratio_v_t=th[1] / ts[i, 1],
                                            n_refinement=nref, **flags(th)))
                if nref:
                    with h5py.File(C.cons_path(n), 'r') as f:
                        W = f['W'][:]
                    ecg = C.eplib.ECGModel(W, tmax=tmax)
                    RT, RE = g['ref_T'][:], g['ref_Temu0'][:]
                    for k, th in enumerate(g['ref_theta'][:]):
                        e = ecg(RT[k]); d = RT[k].astype(np.float64) - RE[k]
                        row = dict(patient=p, construction=n, source=at['source'], resolution_mm=at['vox'], k=k,
                                   round=int(g['ref_round'][k]), v_endo=th[0], v_t=th[1],
                                   wall_s=float(g['ref_wall_s'][k]), emu0_rms_ms=float(np.sqrt((d ** 2).mean())),
                                   emu0_maxabs_ms=float(np.abs(d).max()))
                        for i in range(len(tgt)):
                            row[f'simloss_corr_t{i}'] = float(C.eplib.loss_corr(e, tgt[i]))
                            row[f'simloss_amp_t{i}'] = float(C.eplib.loss_amp(e, tgt[i]))
                        ref.append(row)
            if 'noise' not in H:
                return est, noi, ref, con
            for ci, (cn, levs, unit) in enumerate(C.COMPARATORS):
                if cn not in H['noise']:
                    continue
                nz = H[f'noise/{cn}/theta_hat'][:]; sd = H[f'noise/{cn}/rng_seed'][:]
                for i in range(nz.shape[0]):
                    for li, lev in enumerate(levs):
                        for rp in range(nz.shape[2]):
                            seed = (int(sd[i, li, rp]) if cn == 'disp' else
                                    '-'.join(map(str, C.rep_seed(p, i, ci, lev, rp))))
                            for j, l in enumerate(LN):
                                th = nz[i, li, rp, j, :2]
                                noi.append(dict(patient=p, group=grp, istar=i, v_endo_star=ts[i, 0],
                                                v_t_star=ts[i, 1], comparator=cn, level=lev, level_unit=unit,
                                                rep=rp, seed=seed, loss=l, v_endo_hat=th[0], v_t_hat=th[1],
                                                fval_emu=nz[i, li, rp, j, 2], ratio_v_endo=th[0] / ts[i, 0],
                                                ratio_v_t=th[1] / ts[i, 1], **flags(th)))
    return est, noi, ref, con


def export(out, hearts, workers=1):
    from concurrent.futures import ProcessPoolExecutor
    est, noi, ref, con = [], [], [], []
    with ProcessPoolExecutor(max(1, min(workers, len(hearts)))) as ex:
        for e, n, r, c in ex.map(export_heart, [out] * len(hearts), hearts):
            est += e; noi += n; ref += r; con += c
    E = pd.DataFrame(est)
    E[E['round'] == 2].to_csv(os.path.join(out, 'estimates.csv'), index=False)
    E.to_csv(os.path.join(out, 'estimates_rounds.csv.gz'), index=False)
    pd.DataFrame(noi).to_csv(os.path.join(out, 'noise_estimates.csv'), index=False)
    pd.DataFrame(ref).to_csv(os.path.join(out, 'refinement.csv'), index=False)
    pd.DataFrame(con).to_csv(os.path.join(out, 'constructions.csv'), index=False)


if __name__ == '__main__':
    o = sys.argv[1]
    hs = [int(x) for x in sys.argv[2:]] or sorted(int(os.path.basename(f)[13:16])
                                                  for f in glob.glob(os.path.join(o, 'hearts', 'heart_patient*.h5')))
    export(o, hs, workers=int(os.environ.get('EXPORT_WORKERS', '1')))
    print('exported', len(hs), 'hearts')
