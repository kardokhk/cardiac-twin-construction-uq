"""Per-heart preparation (pilot I13; reused by I11).
Input  : malikNN.npz (from 01_convert.py)
Output : <out>/heartNN.h5  (myocardial submesh, endocardial fast layer, root nodes, heart frame, rule-placed
         electrodes, own-torso lead fields Z on all torso nodes, own-torso ECG weights W_own)
         <meshdir>/heartNN.{pts,elem,lon} and heartNN_stim.vtx for openCARP eikonal.
Tag convention (verified per heart below): 22 RV myocardium, 23 LV myocardium (fibred), 20 RV blood, 21 LV blood.
Usage: python -I 02_prep_heart.py <npz> <out_dir> <mesh_dir>
"""
import sys, os, time, json
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I11'))
import eplib, carpio
from scipy.spatial import cKDTree

D_ENDO = 1.0     # mm, fast endocardial layer: myocardial elements with any node within D_ENDO of the blood interface
R_ROOT = 2.5     # mm, radius of each root-node stimulus patch

npz, out_dir, mesh_dir = sys.argv[1:4]
hid = os.path.basename(npz).replace('malik', '').replace('.npz', '')
t0 = time.time()
d = np.load(npz)
pts = d['pts'].astype(np.float64); tets = d['tets'].astype(np.int64); tags = d['tags']
fib_all = np.zeros((len(tets), 3), np.float32); fib_all[d['fibred_idx']] = d['fib'][:, :3]

myo_m = np.isin(tags, [22, 23])
fibred = np.zeros(len(tets), bool); fibred[d['fibred_idx']] = True
assert fibred[myo_m].mean() > 0.99 and fibred[~myo_m].mean() < 0.01, 'tag convention differs'
mt = tets[myo_m]; mtag = tags[myo_m]; mfib = fib_all[myo_m].astype(np.float64)
gnodes, inv = np.unique(mt, return_inverse=True)
mt_loc = inv.reshape(mt.shape)
P = pts[gnodes]
print(f'heart {hid}: myo {len(mt)} tets, {len(gnodes)} nodes', flush=True)

# blood interfaces
lvb = np.unique(tets[tags == 21]); rvb = np.unique(tets[tags == 20])
lv_endo_g = np.intersect1d(np.unique(tets[tags == 23]), lvb)
rv_endo_g = np.intersect1d(np.unique(tets[np.isin(tags, [22, 23])]), rvb)
assert len(lv_endo_g) > 1000 and len(rv_endo_g) > 1000, (len(lv_endo_g), len(rv_endo_g))
lv_endo = np.searchsorted(gnodes, lv_endo_g); rv_endo = np.searchsorted(gnodes, rv_endo_g)
endo_nodes = np.union1d(lv_endo, rv_endo)
dist = cKDTree(P[endo_nodes]).query(P)[0]
near = dist <= D_ENDO
endo_el = near[mt_loc].any(1)
etag = np.where(endo_el, 2, 1)
print(f'  endo layer: {endo_el.mean()*100:.1f}% of myo elements', flush=True)

# heart frame: LV long axis (PCA of LV myo nodes), apex toward (+x,-y,-z) in LPS
lvn = np.searchsorted(gnodes, np.unique(tets[tags == 23]))
c_lv = P[lvn].mean(0)
w, V = np.linalg.eigh(np.cov((P[lvn] - c_lv).T))
ax = V[:, -1]; ax = ax if ax @ np.array([1, -1, -1]) > 0 else -ax       # base -> apex
s = (P - c_lv) @ ax
s_ap, s_bs = s[lvn].max(), s[lvn].min()
h = (s_ap - s) / (s_ap - s_bs)                                            # 0 apex, 1 base (LV range)
c_lvb = pts[lvb].mean(0); c_rvb = pts[rvb].mean(0)
sep = (c_rvb - c_lvb); sep -= (sep @ ax) * ax; sep /= np.linalg.norm(sep)
ant = np.cross(ax, sep); ant = ant if ant[1] < 0 else -ant                 # anterior = -y

def pick(cands, hval, direction):
    m = cands[np.abs(h[cands] - hval) < 0.05]
    v = P[m] - c_lv; v -= np.outer(v @ ax, ax)
    return m[np.argmax(v @ direction)]

roots = {
    'LV_septal_mid': pick(lv_endo, 0.50, sep),
    'LV_ant_paraseptal': pick(lv_endo, 0.65, (sep + ant) / np.sqrt(2)),
    'LV_post_paraseptal': pick(lv_endo, 0.33, (sep - ant) / np.sqrt(2)),
    'RV_septal': pick(rv_endo, 0.40, -sep),
}
tree = cKDTree(P)
stim = np.unique(np.concatenate([tree.query_ball_point(P[r], R_ROOT) for r in roots.values()]))
print('  roots', {k: P[v].round(1).tolist() for k, v in roots.items()}, 'stim nodes', len(stim), flush=True)

mt_loc = carpio.fix_orientation(P, mt_loc)
if mesh_dir != '-':      # optional: write the native CT-derived myocardial mesh for openCARP (used only for timing)
    os.makedirs(mesh_dir, exist_ok=True)
    base = os.path.join(mesh_dir, f'heart{hid}')
    carpio.write_mesh(base, P * 1000.0, mt_loc, etag, mfib)
    carpio.write_vtx(base + '_stim.vtx', stim)
    print(f'  mesh written {time.time()-t0:.0f}s', flush=True)

# own-torso lead fields (homogeneous isotropic conductor)
bf = eplib.boundary_faces(tets)
xyz, enodes = eplib.place_electrodes(pts, bf, P.mean(0))
print('  electrodes', dict(zip(eplib.ELECS, np.round(xyz, 1).tolist())), flush=True)
K = eplib.stiffness(pts, tets, sigma=eplib.SIG_BULK)
ref = enodes[-1]
Z = eplib.solve_leadfields(K, enodes[:-1], ref)
Z = np.column_stack([Z, np.zeros(len(pts))])
del K
W_own = np.zeros((8, 1))   # ECG weights are computed on the voxel constructions (06_constructions.py)
print(f'  lead fields done {time.time()-t0:.0f}s', flush=True)

with h5py.File(os.path.join(out_dir, f'heart{hid}.h5'), 'w') as f:
    f['pts'] = P.astype(np.float64); f['tets'] = mt_loc.astype(np.int32); f['etag'] = etag.astype(np.int8)
    f['fib'] = mfib.astype(np.float32); f['gnodes'] = gnodes.astype(np.int32)
    f['lv_endo'] = lv_endo.astype(np.int32); f['rv_endo'] = rv_endo.astype(np.int32)
    f['stim'] = stim.astype(np.int32); f['roots'] = np.array(list(roots.values()), np.int32)
    f['frame'] = np.vstack([c_lv, ax, sep, ant]); f['h'] = h.astype(np.float32)
    f['elec_xyz'] = xyz; f['elec_nodes'] = enodes
    f.create_dataset('Z', data=Z.astype(np.float32), compression='gzip', compression_opts=1)
    f['W_own'] = W_own
    f.attrs['torso_volume_mL'] = float(np.abs(carpio.tet_volumes(pts, tets)).sum() / 1000)
    f.attrs['myo_volume_mL'] = float(eplib.tet_grads(P, mt_loc)[0].sum() / 1000)
    f.attrs['heart_centroid'] = P.mean(0)
print(f'heart {hid} done {time.time()-t0:.0f}s', flush=True)
