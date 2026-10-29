"""Build ACDC constructions (owner mesh, I11 gates, 2026-10-28; full-arm copy by owner mesh2, 2026-10-28).
Full-arm changes: output in data/processed/acdc_mesh_v2; RV free-wall thickness is an argument (construction name
gains __rvW.W); the fast endocardial layer `etag` is the Amendment A2 rule (endo_layer.a2_layer, 1.5 mm); the gates
one-voxel layer is stored as `etag_onevoxel`, the centroid distance as `endo_dist_mm`; regression fields compare the
reference transform and, where it exists, the gates construction file (nodes, W) with this build.

Usage:
  python -I build.py prep_torso                      # once: Malik heart 09 torso subset, lead fields, frame
  python -I -B build.py ref  <pid> <qc_dir>                   # expert labels, 1.5 mm, RV wall 3 mm = reference construction
  python -I -B build.py cons <pid> <source> <vox> <rv_wall> <qc_dir>
Outputs: data/processed/acdc_mesh_v2/cons/patientNNN__<source>__<vox>__rv<rv>.h5 (one per construction),
         data/processed/acdc_mesh_v2/ref/patientNNN_ref.npz, and <qc_dir>/<name>.json (status, QC, timings).
Every construction of a heart uses that heart's reference transform, slice direction and root points.
Random numbers: only pyamg inside the Bayer fibre rule, with the numpy global seed 20261028 set before each call.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acdc_lib as A        # noqa: E402
import sax_route            # noqa: E402
import endo_layer           # noqa: E402

OUT = os.path.join(A.ROOT, 'data', 'processed', 'acdc_mesh_v2')
GATES = os.path.join(A.ROOT, 'data', 'processed', 'acdc_mesh')
REF_VOX = 1.5


def prep_torso():
    import h5py
    D = os.path.join(A.ROOT, 'data', 'processed', 'pilot_I13')
    H = h5py.File(os.path.join(D, 'heart09.h5'), 'r')
    assert H.attrs['electrode_rule'] == 'v2'
    frame = H['frame'][:]
    d = np.load(os.path.join(D, 'malik09.npz'))
    tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64)
    cen = tp[tt].mean(1)
    near = np.all(np.abs(cen - frame[0]) < 140.0, axis=1)
    sub = tt[near]
    nodes, inv = np.unique(sub, return_inverse=True)
    Z = H['Z'][:][nodes].astype(np.float32)
    C = h5py.File(os.path.join(D, 'cons', 'cons09.h5'), 'r')['c0ref']
    cp = C['pts'][:]; ctets = C['tets'][:].astype(np.int64)
    surf = cp[A.boundary_nodes(ctets)]
    np.savez(A.TORSO, tp=tp[nodes], tt=inv.reshape(sub.shape).astype(np.int32), Z=Z, frame09=frame,
             elec_xyz=H['elec_xyz'][:], surf09=surf, torso_volume_mL=H.attrs['torso_volume_mL'],
             n_torso_tets_total=len(tt), box_halfwidth_mm=140.0)
    print(f'torso subset: {len(nodes)} nodes, {len(sub)} of {len(tt)} tets; heart09 surface {len(surf)} nodes')


def build(pid, source, vox, qc_dir, ref=None, rv_wall=A.RV_WALL):
    """Build one construction. ref=None means this IS the reference construction (expert, 1.5 mm)."""
    import h5py
    t0 = time.time()
    name = f'patient{pid:03d}__{source}__{vox:.1f}__rv{rv_wall:.1f}'
    qc = dict(name=name, patient=pid, source=source, vox=vox, rv_wall_mm=rv_wall, status='fail', fail_code=None, msg='',
              group=A.info_cfg(pid).get('Group'))
    tm = {}
    try:
        T = A.load_torso()
        path = A.label_path(pid, source)
        qc['label_path'] = os.path.relpath(path, A.ROOT)
        if not os.path.exists(path):
            raise A.GateFail('F1', f'missing label file {path}')
        lab, zooms = A.load_labels(path)
        direction = None if ref is None else int(ref['direction'])
        acdc, lo, Rg, cimg, info = A.prepare_slices(lab, zooms, vox, direction=direction, rv_wall=rv_wall)
        qc.update(info)
        L, p = sax_route.sax_construction(None, None, acdc=acdc)
        tm['recon_s'] = time.time() - t0
        Ls = sax_route.lengths(L, vox)
        qc.update({k: float(v) for k, v in Ls.items()})
        qc['rv_wall_mL'] = float((L == 2).sum()) * vox ** 3 / 1000
        qc['rv_cav_mL'] = float((L == 4).sum()) * vox ** 3 / 1000
        qc['err_lv_myo'] = Ls['lv_myo_mL'] / info['lv_myo_label_mL'] - 1
        qc['err_lv_cav'] = Ls['lv_cav_mL'] / info['lv_cav_label_mL'] - 1
        if ref is None:
            if not (L == 4).any() or not (L == 3).any():
                raise A.GateFail('F2', 'reconstruction lacks LV or RV cavity')
            fr = A.frame_transform(L, vox, lo, Rg, cimg, info['direction'], T)
        else:
            fr = dict(R=ref['R'], t=ref['t'])
        Rt = Rg @ fr['R'].T
        ct = cimg @ fr['R'].T + fr['t']
        mq = {}
        pts, tets, iface, lv_if = sax_route.mesh_from_labels(L, vox, lo, Rt, ct, qc=mq)
        tm['mesh_s'] = time.time() - t0
        qc.update(n_nodes=int(len(pts)), n_tets=int(len(tets)), kept_frac=mq['kept_frac'],
                  n_components=mq['n_components'])
        vol = sax_route.carpio.tet_volumes(pts, tets)
        qc['n_bad_tets'] = int((vol <= 1e-12).sum())
        qc['myo_mesh_mL'] = float(vol.sum()) / 1000
        nm, ne = A.nonmanifold_edges(tets)
        qc.update(nonmanifold_edges=nm, boundary_edges=ne)
        # failure criteria F3 to F7 (G2_criteria.md)
        if mq['kept_frac'] < 0.95:
            raise A.GateFail('F3', f"largest myocardial component holds {mq['kept_frac']:.3f}")
        if len(pts) < 2000:
            raise A.GateFail('F4', f'{len(pts)} nodes')
        if qc['n_bad_tets']:
            raise A.GateFail('F5', f"{qc['n_bad_tets']} inverted or zero-volume tets")
        if abs(qc['err_lv_myo']) > 0.25:
            raise A.GateFail('F6', f"LV myocardial volume error {qc['err_lv_myo']:+.3f}")
        if abs(qc['err_lv_cav']) > 0.25:
            raise A.GateFail('F7', f"LV cavity volume error {qc['err_lv_cav']:+.3f}")
        fib, nonfin, bmeta, bq = A.bayer(pts, tets, lv_if, iface, vox, T)
        tm['fibre_s'] = time.time() - t0
        qc.update(fib_nonfinite=int(nonfin), **bq)
        if nonfin > 0.001 * len(tets):
            raise A.GateFail('F8', f'{nonfin} non-finite fibres')
        etag1 = np.where(endo_layer.onevoxel_layer(tets, iface), 2, 1).astype(np.int8)
        a2, edist, nfaces = endo_layer.a2_layer(L, pts, tets, vox, lo, Rt, ct)
        etag = np.where(a2, 2, 1).astype(np.int8)
        qc.update(endo_frac=float(a2.mean()), endo_vol_frac=float(vol[a2].sum() / vol.sum()),
                  endo_frac_onevoxel=float((etag1 == 2).mean()),
                  endo_vol_frac_onevoxel=float(vol[etag1 == 2].sum() / vol.sum()), n_endo_faces=int(nfaces))
        tm['layer_s'] = time.time() - t0
        if ref is None:
            roots_xyz = A.root_rule(pts, mq['tet_lab'], tets, lv_if, iface, T)
        else:
            roots_xyz = ref['roots_xyz']
        rnodes, stim, shift = A.map_roots(pts, lv_if, iface, roots_xyz)
        qc.update(root_shift_mm=shift.round(3).tolist(), n_stim=int(len(stim)))
        if ref is None:
            src = pts[A.boundary_nodes(tets)]
            R_icp, t_icp, rms = A.eplib.icp(src, T['surf09'], np.eye(3), np.zeros(3))
            rms0 = float(np.sqrt((A.cKDTree(T['surf09']).query(src)[0] ** 2).mean()))
            ang = float(np.degrees(np.arccos(np.clip((np.trace(R_icp) - 1) / 2, -1, 1))))
            icp = dict(R_icp=R_icp, t_icp=t_icp, icp_rms_mm=rms, frame_rms_mm=rms0, icp_rot_deg=ang,
                       icp_shift_mm=float(np.linalg.norm(src.mean(0) @ R_icp.T + t_icp - src.mean(0))))
        else:
            icp = dict(R_icp=ref['R_icp'], t_icp=ref['t_icp'])
        W, outside, smin = A.ecg_weights(pts, tets, fib, T)
        W_icp, outside_icp, smin_icp = A.ecg_weights(pts, tets, fib, T, icp['R_icp'], icp['t_icp'])
        qc.update(outside_min_bary=smin, outside_min_bary_icp=smin_icp)
        tm['leadfield_s'] = time.time() - t0
        qc.update(outside_nodes=outside, outside_nodes_icp=outside_icp)
        os.makedirs(os.path.join(OUT, 'cons'), exist_ok=True)
        h5 = os.path.join(OUT, 'cons', name + '.h5')
        with h5py.File(h5, 'w') as f:
            f['pts'] = pts; f['tets'] = tets.astype(np.int32); f['etag'] = etag; f['etag_onevoxel'] = etag1
            f['endo_dist_mm'] = edist.astype(np.float32); f['fib'] = fib.astype(np.float32)
            f['stim'] = stim.astype(np.int32); f['roots'] = rnodes.astype(np.int32); f['roots_xyz_rule'] = roots_xyz
            f['W'] = W; f['W_icp'] = W_icp
            f['lv_endo'] = np.nonzero(lv_if)[0].astype(np.int32)
            f['rv_endo'] = np.nonzero(iface & ~lv_if)[0].astype(np.int32)
            f['tet_lab'] = mq['tet_lab']
            f.create_dataset('lab', data=L.astype(np.int8), compression='gzip', compression_opts=4)
            f['grid_lo'] = lo; f['grid_R'] = Rt; f['grid_c'] = ct
            f.attrs.update(patient=pid, source=source, vox=vox, v_fib=A.V_FIB, units='mm; W: 8 leads (I, II, V1-V6) x nodes',
                           tags='1 myocardium (v_l = v_fib, v_t = v_n = theta[1]); 2 fast endocardium (isotropic theta[0])',
                           etag='Amendment A2: centroid within 1.5 mm of LV or RV endocardial voxel faces', etag_onevoxel='gates: tet touches a blood-interface node', rv_wall_mm=rv_wall,
                           leads=json.dumps(A.eplib.LEADS), ecg_sign='ECG_l(t) = -sum_n W_ln Vm_n(t)',
                           bayer_meta=json.dumps(bmeta), torso='Malik 2026 heart 09, electrode rule v2, homogeneous 0.2 S/m',
                           placement='frame alignment (W); ICP refinement diagnostic (W_icp)')
        qc['h5'] = os.path.relpath(h5, A.ROOT)
        qc['mesh_ok'] = True
        g5 = os.path.join(GATES, 'cons', f'patient{pid:03d}__{source}__{vox:.1f}.h5')
        if abs(rv_wall - 3.0) < 1e-9 and os.path.exists(g5):
            with h5py.File(g5, 'r') as g:
                same = g['pts'].shape == pts.shape and g['tets'].shape == tets.shape
                qc['gates_same_mesh'] = bool(same and np.array_equal(g['tets'][:], tets.astype(np.int32)))
                if same:
                    qc['gates_W_rel_maxdiff'] = float(np.abs(g['W'][:] - W).max() / np.abs(W).max())
                    qc['gates_etag_equal_onevoxel'] = bool(np.array_equal(g['etag'][:], etag1))
        # smoke test (F9)
        work = os.path.join(os.environ.get('TMPDIR', '/tmp'), name)
        try:
            act, ecg, wall = A.forward(h5, A.THETA_SMOKE, work)
            tm['eikonal_wall_s'] = wall
            unact = int((~np.isfinite(act) | (act < 0) | (act > 1e6)).sum())
            good = act[np.isfinite(act) & (act < 1e6)]
            qc.update(smoke_unactivated=unact, smoke_act_max_ms=float(good.max()) if len(good) else None,
                      smoke_ecg_finite=bool(np.isfinite(ecg).all()), smoke_ecg_maxabs=float(np.abs(ecg).max()))
            ok = unact == 0 and 30 <= good.max() <= 400 and np.isfinite(ecg).all()
            qc['smoke_ok'] = bool(ok)
            with h5py.File(h5, 'a') as f:
                f['smoke_ecg'] = ecg.astype(np.float32); f.attrs['smoke_theta'] = A.THETA_SMOKE
                f.attrs['smoke_act_max_ms'] = qc['smoke_act_max_ms']; f.attrs['smoke_ok'] = qc['smoke_ok']
                f.attrs['smoke_eikonal_wall_s'] = wall
            if not ok:
                raise A.GateFail('F9', f'smoke test: unactivated {unact}, max act {good.max() if len(good) else None}')
        finally:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
        qc['status'] = 'pass'
        if ref is None:
            np.savez(os.path.join(OUT, 'ref', f'patient{pid:03d}_ref.npz'), direction=info['direction'], R=fr['R'],
                     t=fr['t'], roots_xyz=roots_xyz, ax_img=fr['ax_img'], sep_img=fr['sep_img'], ant_img=fr['ant_img'],
                     c_lv_img=fr['c_lv_img'], zooms=np.array(zooms), **icp)
    except A.GateFail as e:
        qc['fail_code'] = e.code; qc['msg'] = str(e)
    except Exception as e:      # noqa: BLE001  (F1)
        qc['fail_code'] = 'F1'; qc['msg'] = repr(e); qc['traceback'] = traceback.format_exc()[-3000:]
    if ref is None and qc['status'] != 'pass' and qc.get('mesh_ok') and qc['fail_code'] == 'F9':
        # the reference passed meshing (F1-F8) but failed the smoke test: keep the reference so other constructions
        # of the heart can still be built and reported (G2 meshing-only line)
        np.savez(os.path.join(OUT, 'ref', f'patient{pid:03d}_ref.npz'), direction=info['direction'], R=fr['R'],
                 t=fr['t'], roots_xyz=roots_xyz, ax_img=fr['ax_img'], sep_img=fr['sep_img'], ant_img=fr['ant_img'],
                 c_lv_img=fr['c_lv_img'], zooms=np.array(zooms), **icp)
    if ref is None and qc.get('mesh_ok'):
        gr = os.path.join(GATES, 'ref', f'patient{pid:03d}_ref.npz')
        if os.path.exists(gr):
            g = np.load(gr)
            qc['gates_ref_maxdiff'] = float(max(np.abs(g['R'] - fr['R']).max(), np.abs(g['t'] - fr['t']).max(),
                                                np.abs(g['roots_xyz'] - roots_xyz).max()))
    qc['timing_s'] = {k: round(v, 1) for k, v in tm.items()}
    qc['total_s'] = round(time.time() - t0, 1)
    import resource
    qc['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    qc['maxrss_children_mb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    if ref is None and qc.get('mesh_ok'):
        qc.update(icp_rot_deg=icp.get('icp_rot_deg'), icp_rms_mm=icp.get('icp_rms_mm'),
                  frame_rms_mm=icp.get('frame_rms_mm'), icp_shift_mm=icp.get('icp_shift_mm'))
    os.makedirs(qc_dir, exist_ok=True)
    json.dump(qc, open(os.path.join(qc_dir, name + '.json'), 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"{name}: {qc['status']} {qc['fail_code'] or ''} {qc['msg'][:200]} nodes {qc.get('n_nodes')} "
          f"errLVmyo {qc.get('err_lv_myo', float('nan')):+.3f} errLVcav {qc.get('err_lv_cav', float('nan')):+.3f} "
          f"act {qc.get('smoke_act_max_ms')} {qc['total_s']}s", flush=True)
    return qc


if __name__ == '__main__':
    cmd = sys.argv[1]
    os.makedirs(os.path.join(OUT, 'ref'), exist_ok=True)
    if cmd == 'prep_torso':
        prep_torso()
    elif cmd == 'ref':
        build(int(sys.argv[2]), 'expert', REF_VOX, sys.argv[3], ref=None)
    elif cmd == 'cons':
        pid, source, vox, rvw, qc_dir = int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
        rf = os.path.join(OUT, 'ref', f'patient{pid:03d}_ref.npz')
        if not os.path.exists(rf):
            name = f'patient{pid:03d}__{source}__{vox:.1f}__rv{rvw:.1f}'
            os.makedirs(qc_dir, exist_ok=True)
            json.dump(dict(name=name, patient=pid, source=source, vox=vox, status='fail', fail_code='NOREF',
                           msg='reference construction (expert 1.5 mm) unavailable'),
                      open(os.path.join(qc_dir, name + '.json'), 'w'))
            print(name, 'NOREF'); sys.exit(0)
        ref = dict(np.load(rf))
        build(pid, source, vox, qc_dir, ref=ref, rv_wall=rvw)
    else:
        raise SystemExit(__doc__)
