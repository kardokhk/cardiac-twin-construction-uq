"""Extended electrode-displacement lead fields for Amendment 2 B4 (owner noise-ext, I11, 2026-10-28).

Same displacement rule, torso, solver and electrode-node convention as code/full/mesh/displace.py (owner mesh2); this
file imports that module unchanged and only changes the SD levels, the sd_index values in the seed, and the output
directory. Differences from displace.py:
  * SD 30 and 40 mm, sd_index 3 and 4 (indices 0, 1, 2 = 5, 10, 20 mm are mesh2's);
    seed = 20261028 + 1000 * patient + 100 * sd_index + rep (displace.seed_of, unchanged), rep 0..19.
  * Output: data/processed/acdc_mesh_v2/displaced_ext/patientNNN_W_displaced_ext.h5 (W_disp[k, rep, lead, node] with
    k = 0, 1 for sd_index 3, 4; dataset sd_index holds 3, 4). The mesh2 files in displaced/ are never opened for write.
  * prep: identical to displace.prep except the draw loop; solve: displace.solve unchanged (reads meta.npz only).
Usage (OMP_NUM_THREADS=1):
  python -I -B displace_ext.py prep <scratch> <pid,...>
  python -I -B displace_ext.py solve <scratch> <worker> <nworkers>
  python -I -B displace_ext.py assemble <scratch> <pid>
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import displace as D        # noqa: E402
A = D.A
eplib = D.eplib

SD_EXT = ((3, 30.0), (4, 40.0))           # (sd_index, SD mm)
NREP = D.NREP
OUT_EXT = os.environ.get('NOISE_EXT_DISP_DIR', os.path.join(D.OUT, 'displaced_ext'))   # env override for the test job only


def prep(scr, pids):
    import h5py
    t0 = time.time()
    os.makedirs(scr, exist_ok=True)
    H = h5py.File(os.path.join(D.PILOT, 'heart09.h5'), 'r')
    assert H.attrs['electrode_rule'] == 'v2'
    elec_xyz = H['elec_xyz'][:]; elec_nodes = H['elec_nodes'][:]; frame = H['frame'][:]
    d = np.load(os.path.join(D.PILOT, 'malik09.npz'))
    pts = d['pts'].astype(np.float64); tets = d['tets'].astype(np.int64)
    cen = pts[tets].mean(1)
    near = np.all(np.abs(cen - frame[0]) < 140.0, axis=1)
    sub = tets[near]; del cen
    nodes, inv = np.unique(sub, return_inverse=True)
    T = A.load_torso()
    assert len(nodes) == len(T['tp']) and np.array_equal(pts[nodes], T['tp']), 'near-subset mapping mismatch'
    assert np.array_equal(inv.reshape(sub.shape).astype(np.int32), T['tt'])
    print(f'near subset verified: {len(nodes)} nodes {time.time()-t0:.0f}s', flush=True)
    F, N, bn_all, sinfo = D.outer_surface(pts, tets)
    assert np.array_equal(bn_all[cKDTree(pts[bn_all]).query(elec_xyz)[1]], elec_nodes), 'electrode node rule mismatch'
    surf = D.Surface(pts, F, N)
    btree = cKDTree(pts[bn_all])
    rows = []
    for pid in pids:
        for sdi, sd in SD_EXT:
            for rep in range(NREP):
                g = np.random.default_rng(D.seed_of(pid, sdi, rep)).standard_normal((6, 2))
                for j in range(6):
                    rows.append((pid, sdi, rep, j, sd * g[j, 0], sd * g[j, 1]))
    R = np.array(rows)
    X0 = elec_xyz[np.array(D.VIDX)[R[:, 3].astype(int)]]
    Xe, Lw = D.walk(surf, X0, R[:, 4:6])
    dn, ni = btree.query(Xe)
    enode = bn_all[ni]
    Xz, _ = D.walk(surf, elec_xyz[D.VIDX], np.zeros((6, 2)))
    zn = bn_all[btree.query(Xz)[1]]
    assert np.array_equal(zn, elec_nodes[D.VIDX]), (zn, elec_nodes[D.VIDX])
    euclid_node = np.linalg.norm(pts[enode] - X0, axis=1)
    euclid_walk = np.linalg.norm(Xe - X0, axis=1)
    print(f'draws {len(R)} walked {time.time()-t0:.0f}s; node snap median {np.median(dn):.2f} max {dn.max():.2f} mm',
          flush=True)
    uniq = np.unique(np.concatenate([enode, elec_nodes[D.VIDX]]))
    ttree = cKDTree(T['tp'][T['tt']].mean(1))
    used = []
    for pid in pids:
        with h5py.File(D.ref_h5(pid), 'r') as f:
            hp = f['pts'][:]
        bt, lam, sc = D.locate(T['tp'], T['tt'].astype(np.int64), hp, tree=ttree)
        np.savez(os.path.join(scr, f'loc_{pid}.npz'), tet=bt.astype(np.int32), lam=lam, score=sc)
        used.append(np.unique(T['tt'][bt]))
    U = np.unique(np.concatenate(used))
    print(f'U: {len(U)} near-subset nodes; {len(uniq)} distinct electrode nodes {time.time()-t0:.0f}s', flush=True)
    K = eplib.stiffness(pts, tets, sigma=eplib.SIG_BULK)
    ref = int(elec_nodes[-1])
    keep = np.ones(K.shape[0]); keep[ref] = 0
    Dk = sp.diags(keep)
    Ap = (Dk @ K @ Dk + sp.diags(1 - keep)).tocsr()
    sp.save_npz(os.path.join(scr, 'A.npz'), Ap, compressed=False)
    np.savez(os.path.join(scr, 'meta.npz'), R=R, Xe=Xe, Lw=Lw, enode=enode, snap_mm=dn, euclid_node=euclid_node,
             euclid_walk=euclid_walk, uniq=uniq, U=U, Ufull=nodes[U], ref=ref, elec_nodes=elec_nodes, elec_xyz=elec_xyz,
             pids=np.array(pids))
    json.dump(dict(surface=sinfo, sd_ext=[list(x) for x in SD_EXT], n_draws=int(len(R)), n_unique_nodes=int(len(uniq)),
                   n_U=int(len(U)), snap_mm_median=float(np.median(dn)), snap_mm_max=float(dn.max()),
                   walk_vs_target_ratio_median=float(np.median(euclid_walk[Lw > 0] / Lw[Lw > 0])),
                   prep_s=time.time() - t0), open(os.path.join(scr, 'prep.json'), 'w'), indent=1)
    print(f'prep done {time.time()-t0:.0f}s', flush=True)


def assemble(scr, pid):
    import glob
    import h5py
    t0 = time.time()
    m = np.load(os.path.join(scr, 'meta.npz'))
    U = m['U']; R = m['R']
    sel = R[:, 0].astype(int) == pid
    Rp = R[sel]; enode = m['enode'][sel]
    col = {}; Zs = []
    for k, f in enumerate(sorted(glob.glob(os.path.join(scr, 'Zmeta_*.npz')))):
        zm = np.load(f)
        for j, s in enumerate(zm['nodes']):
            col[int(s)] = (k, j)
        Zs.append(np.load(f.replace('Zmeta_', 'Z_').replace('.npz', '.npy'), mmap_mode='r'))

    def zcol(s):
        k, j = col[int(s)]; return np.asarray(Zs[k][j], np.float64)
    upos = -np.ones(int(U.max()) + 1, np.int64); upos[U] = np.arange(len(U))
    T = A.load_torso()
    loc = np.load(os.path.join(scr, f'loc_{pid}.npz'))
    tet = T['tt'][loc['tet']].astype(np.int64); lam = loc['lam']
    P = upos[tet]; assert (P >= 0).all()
    Zbase = np.einsum('qi,qim->qm', lam, T['Z'].astype(np.float64)[tet])
    with h5py.File(D.ref_h5(pid), 'r') as f:
        pts = f['pts'][:]; tets = f['tets'][:].astype(np.int64); fib = f['fib'][:].astype(np.float64); W0 = f['W'][:]
    Ki = eplib.stiffness(pts, tets, fib=fib, sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
    LM = eplib.lead_matrix()

    def weights(vnodes):
        Z = Zbase.copy()
        for j, s in enumerate(vnodes):
            Z[:, D.VIDX[j]] = np.einsum('qi,qi->q', lam, zcol(s)[P])
        return LM @ (Ki @ Z).T
    Wz = weights(m['elec_nodes'][D.VIDX])
    chk = dict(W_zero_vs_W_rel_maxdiff=float(np.abs(Wz - W0).max() / np.abs(W0).max()))
    ns = len(SD_EXT)
    Wd = np.zeros((ns, NREP, 8, len(pts)), np.float32)
    nodes = np.zeros((ns, NREP, 6), np.int64); xyz = np.zeros((ns, NREP, 6, 3))
    geo = np.zeros((ns, NREP, 6)); eu = np.zeros((ns, NREP, 6)); seeds = np.zeros((ns, NREP), np.int64)
    Xe = m['Xe'][sel]; Lw = m['Lw'][sel]; en = m['euclid_node'][sel]
    for k, (sdi, sd) in enumerate(SD_EXT):
        for rep in range(NREP):
            r = (Rp[:, 1].astype(int) == sdi) & (Rp[:, 2].astype(int) == rep)
            assert r.sum() == 6
            o = np.argsort(Rp[r, 3])
            vn = enode[r][o]
            Wd[k, rep] = weights(vn)
            nodes[k, rep] = vn; xyz[k, rep] = Xe[r][o]; geo[k, rep] = Lw[r][o]; eu[k, rep] = en[r][o]
            seeds[k, rep] = D.seed_of(pid, sdi, rep)
    os.makedirs(OUT_EXT, exist_ok=True)
    fn = os.path.join(OUT_EXT, f'patient{pid:03d}_W_displaced_ext.h5')
    assert not os.path.exists(fn), f'refusing to overwrite {fn}'
    rel = np.array([[np.linalg.norm(Wd[a, b] - W0) / np.linalg.norm(W0) for b in range(NREP)] for a in range(ns)])
    with h5py.File(fn, 'w') as f:
        f.create_dataset('W_disp', data=Wd, compression='gzip', compression_opts=4)
        f['sd_mm'] = np.array([s for _, s in SD_EXT]); f['sd_index'] = np.array([i for i, _ in SD_EXT])
        f['seed'] = seeds; f['elec_node'] = nodes; f['elec_xyz_walk'] = xyz
        f['walk_mm'] = geo; f['euclid_from_original_node_mm'] = eu; f['W_rel_change'] = rel
        f.attrs.update(patient=pid, construction=os.path.relpath(D.ref_h5(pid), A.ROOT),
                       dims='W_disp[k (sd_index 3, 4 = 30, 40 mm), rep, lead (I, II, V1-V6), node of the construction]',
                       seed_rule='20261028 + 1000*patient + 100*sd_index + rep, sd_index 3 (30 mm), 4 (40 mm); '
                                 'default_rng(seed).standard_normal((6,2)) for V1..V6',
                       displaced='V1-V6 (limb RA, LA, LL and reference RL fixed)',
                       solver='pyamg SA max_coarse 500, numpy seed 20261028 before setup, CG rel res 1e-10, pinned RL',
                       torso='Malik 2026 heart 09 full mesh, homogeneous 0.2 S/m, electrode rule v2',
                       code='code/full/mesh/displace_ext.py (owner noise-ext) over displace.py (owner mesh2)', **chk)
    print(f'patient{pid}: {json.dumps(chk)} rel |W_disp - W| median by SD '
          f'{[round(float(np.median(rel[a])), 4) for a in range(ns)]} '
          f'euclid median {[round(float(np.median(eu[a])), 1) for a in range(ns)]} {time.time()-t0:.0f}s', flush=True)
    return fn


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2], [int(x) for x in sys.argv[3].split(',')])
    elif cmd == 'solve':
        D.solve(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == 'assemble':
        assemble(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
