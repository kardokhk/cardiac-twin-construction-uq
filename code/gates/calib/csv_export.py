"""Tidy CSVs from the per-heart HDF5 files of the G3/G4 mini-study (owner calib, 2026-10-28).
Usage: python -I csv_export.py <out_dir> 101 102 104
Writes estimates.csv (arm C), noise_estimates.csv (arm N), refinement.csv, constructions.csv into <out_dir>.
"""
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


def export(out, hearts):
    est, noi, ref, con = [], [], [], []
    for p in hearts:
        with h5py.File(os.path.join(out, f'heart_patient{p:03d}.h5'), 'r') as H:
            grp = H.attrs['group']; ts = H['targets/theta_star'][:]; refn = H.attrs['reference']
            tmax = float(H.attrs['tmax_ms']); tgt = H['targets/ecg'][:]
            for n in H['cons']:
                g = H['cons'][n]; at = dict(g.attrs)
                con.append(dict(patient=p, group=grp, construction=n, source=at['source'], vox=at['vox'],
                                is_ref=n == refn, n_nodes=int(at['n_nodes']), unactivated=int(at['unactivated']),
                                L_scale=at['L_scale'], L_wall=at['L_wall'], L_base=at['L_base'], myo_mL=at['myo_mL'],
                                lv_myo_mL=at['lv_myo_mL'], lv_cav_mL=at['lv_cav_mL'],
                                grid_wall_median_s=float(np.median(g['grid_wall_s'][:])),
                                n_refinement=int(len(g['ref_theta'])) if 'ref_theta' in g else 0))
                cal = g['calib'][:]
                for r in range(cal.shape[0]):
                    for i in range(cal.shape[1]):
                        for j, l in enumerate(LN):
                            th = cal[r, i, j, :2]
                            est.append(dict(patient=p, group=grp, construction=n, source=at['source'], vox=at['vox'],
                                            is_ref=n == refn, istar=i, v_endo_star=ts[i, 0], v_t_star=ts[i, 1],
                                            loss=l, round=r, v_endo_hat=th[0], v_t_hat=th[1], fval_emu=cal[r, i, j, 2],
                                            ratio_v_endo=th[0] / ts[i, 0], ratio_v_t=th[1] / ts[i, 1], **flags(th)))
                if 'ref_theta' in g:
                    with h5py.File(C.cons_path(n), 'r') as f:
                        W = f['W'][:]
                    ecg = C.eplib.ECGModel(W, tmax=tmax)
                    RT, RE = g['ref_T'][:], g['ref_Temu0'][:]
                    for k, th in enumerate(g['ref_theta'][:]):
                        e = ecg(RT[k]); d = RT[k].astype(np.float64) - RE[k]
                        row = dict(patient=p, construction=n, k=k, round=int(g['ref_round'][k]), v_endo=th[0],
                                   v_t=th[1], wall_s=float(g['ref_wall_s'][k]),
                                   emu0_rms_ms=float(np.sqrt((d ** 2).mean())), emu0_maxabs_ms=float(np.abs(d).max()))
                        for i in range(len(tgt)):
                            row[f'simloss_corr_t{i}'] = float(C.eplib.loss_corr(e, tgt[i]))
                            row[f'simloss_amp_t{i}'] = float(C.eplib.loss_amp(e, tgt[i]))
                        ref.append(row)
            nz = H['noise/theta_hat'][:]
            snrs = list(H['noise/theta_hat'].attrs['snr_db'])
            for i in range(nz.shape[0]):
                for si in range(nz.shape[1]):
                    for k in range(nz.shape[2]):
                        for rp in range(nz.shape[3]):
                            for j, l in enumerate(LN):
                                th = nz[i, si, k, rp, j, :2]
                                noi.append(dict(patient=p, group=grp, istar=i, v_endo_star=ts[i, 0], v_t_star=ts[i, 1],
                                                snr_db=snrs[si], noise_type=C.NOISE_TYPES[k], rep=rp, loss=l,
                                                v_endo_hat=th[0], v_t_hat=th[1], fval_emu=nz[i, si, k, rp, j, 2],
                                                ratio_v_endo=th[0] / ts[i, 0], ratio_v_t=th[1] / ts[i, 1], **flags(th)))
    pd.DataFrame(est).to_csv(os.path.join(out, 'estimates.csv'), index=False)
    pd.DataFrame(noi).to_csv(os.path.join(out, 'noise_estimates.csv'), index=False)
    pd.DataFrame(ref).to_csv(os.path.join(out, 'refinement.csv'), index=False)
    pd.DataFrame(con).to_csv(os.path.join(out, 'constructions.csv'), index=False)


if __name__ == '__main__':
    export(sys.argv[1], [int(x) for x in sys.argv[2:]])
