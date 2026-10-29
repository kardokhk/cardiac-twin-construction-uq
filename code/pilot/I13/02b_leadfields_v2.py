"""Recompute rule-v2 electrodes and own-torso lead fields for one heart; overwrite Z/elec in heartNN.h5
(rule-v1 values kept as Z_rule1/elec_xyz_rule1).  Usage: python -I 02b_leadfields_v2.py <heart.h5> <malik.npz>"""
import sys, os, time, numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I11'))
import eplib
hf, npz = sys.argv[1:3]
t0 = time.time()
d = np.load(npz); pts = d['pts'].astype(np.float64); tets = d['tets'].astype(np.int64)
with h5py.File(hf, 'a') as H:
    c = H.attrs['heart_centroid']
    bf = eplib.boundary_faces(tets)
    spine = pts[np.unique(tets[d['tags'] == 4])]
    xyz, en = eplib.place_electrodes_v2(pts, bf, c, spine)
    print(os.path.basename(hf), 'electrodes v2 relative to heart (mm):', dict(zip(eplib.ELECS, np.round(xyz - c).astype(int).tolist())), flush=True)
    K = eplib.stiffness(pts, tets, sigma=eplib.SIG_BULK)
    Z = np.column_stack([eplib.solve_leadfields(K, en[:-1], en[-1], verbose=False), np.zeros(len(pts))])
    if 'Z_rule1' not in H:
        H.move('Z', 'Z_rule1'); H.move('elec_xyz', 'elec_xyz_rule1'); H.move('elec_nodes', 'elec_nodes_rule1')
    else:
        del H['Z'], H['elec_xyz'], H['elec_nodes']
    H.create_dataset('Z', data=Z.astype(np.float32), compression='gzip', compression_opts=1)
    H['elec_xyz'] = xyz; H['elec_nodes'] = en
    H.attrs['electrode_rule'] = 'v2'
print(f'done {time.time()-t0:.0f}s')
