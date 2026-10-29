"""I11 pilot SUBSTITUTE constructions (documented substitute for ACDC segmentation constructions).
For one Malik 2026 heart: voxelise the deposited ventricular labels at 1 mm in the heart frame (long axis =
first grid axis) -> REFERENCE construction. Then K alternative constructions that mimic a short-axis cine
segmentation pipeline (ACDC-like): labels sampled on short-axis slices (8 mm spacing, random slice offset,
1.5 mm in-plane), observer contour shifts (LV endo, LV epi, RV endo; SD 0.75 mm each), a smooth random
boundary field (SD 0.5 mm, 15 mm correlation length), random omission of the most basal slice (p = 0.5),
shape-based (signed-distance) interpolation back to 1 mm, and an assumed 3 mm RV free wall (ACDC has no RV
wall label). Each construction is meshed (voxel -> 6 tets), given fibres from the nearest deposited element,
a fast endocardial layer (nodes within 1 mm of the blood interface), and root nodes mapped from the deposited
heart's root nodes. ECG weights use the heart's own-torso lead field interpolated at construction nodes.
Mask-readable lengths: L_scale = myocardial volume^(1/3); L_wall = 4 x mean distance-to-boundary in LV
myocardium (equals thickness for a flat wall); L_base = LV myocardial extent along the long axis.
Usage: python -I 06_constructions.py <heart.h5> <malik.npz> <outdir> <meshdir> <K> <seed>
"""
import sys, os, json, time
import numpy as np, h5py
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I13'))
import carpio, eplib

hf, npz, outdir, meshdir, K, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
VOX = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5    # voxel size (mm)
hid = os.path.basename(hf).replace('heart', '').replace('.h5', '')
rng = np.random.default_rng(seed)
t0 = time.time()
H = h5py.File(hf, 'r')
P0 = H['pts'][:]; T0 = H['tets'][:].astype(np.int64); fib0 = H['fib'][:]
frame = H['frame'][:]; c = frame[0]; Rf = np.vstack(frame[1:4])          # rows ax, sep, ant
roots_xyz = P0[H['roots'][:]]
Z_torso = H['Z'][:].astype(np.float64)
d = np.load(npz)
tp = d['pts'].astype(np.float64); tt = d['tets'].astype(np.int64); tg = d['tags']
sel = np.isin(tg, [20, 21, 22, 23])
ht, htag = tt[sel], tg[sel]
LAB = {23: 1, 22: 2, 21: 3, 20: 4}           # 1 LV myo, 2 RV myo, 3 LV blood, 4 RV blood
# --- voxelise at VOX mm in heart frame
U = (tp[np.unique(ht)] - c) @ Rf.T
lo = np.floor(U.min(0) / VOX) - 4; hi = np.ceil(U.max(0) / VOX) + 4
shape = tuple((hi - lo).astype(int) + 1)
g = (np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), -1).reshape(-1, 3) + lo) * VOX
X = g @ Rf + c
cent = tp[ht].mean(1); tree = cKDTree(cent)
lab = np.zeros(len(X), np.int8)
bb = tree.query(X, k=1)[0] < 3.0
idx = np.nonzero(bb)[0]
_, cand = tree.query(X[idx], k=16)
best = np.full(len(idx), -np.inf); bl = np.zeros(len(idx), np.int8)
for k in range(16):
    Tk = ht[cand[:, k]]; Xk = tp[Tk]
    D = np.stack([Xk[:, 1] - Xk[:, 0], Xk[:, 2] - Xk[:, 0], Xk[:, 3] - Xk[:, 0]], axis=2)
    l = np.linalg.solve(D, (X[idx] - Xk[:, 0])[:, :, None])[:, :, 0]
    s = np.minimum(1 - l.sum(1), l.min(1))
    up = s > best; best[up] = s[up]
    bl[up] = np.vectorize(LAB.get)(htag[cand[up, k]])
lab[idx] = np.where(best > -1e-6, bl, 0)
lab = lab.reshape(shape)
print(f'heart {hid}: voxel grid {shape}, counts', {k: int((lab == v).sum()) for k, v in LAB.items()}, f'{time.time()-t0:.0f}s', flush=True)


def sdf(mask, spacing):
    """signed distance (negative inside), per short-axis slice (2D, in-plane spacing), slices along axis 0."""
    out = np.empty(mask.shape, np.float32)
    for i in range(mask.shape[0]):
        m = mask[i]
        if m.any() and (~m).any():
            out[i] = ndi.distance_transform_edt(~m, sampling=spacing) - ndi.distance_transform_edt(m, sampling=spacing)
        else:
            out[i] = 50.0 if not m.any() else -50.0
    return out


def smooth_field(shape, sd, corr, rng):
    f = ndi.gaussian_filter(rng.standard_normal(shape), corr)
    return (f / f.std() * sd).astype(np.float32)


def sax_construction(lab, rng):
    p = dict(spacing=8.0, offset=float(rng.uniform(0, 8)), inplane=1.5, d_endo=float(rng.normal(0, 0.75)),
             d_epi=float(rng.normal(0, 0.75)), d_rv=float(rng.normal(0, 0.75)), drop_basal=int(rng.random() < 0.5),
             field_sd=0.5, rv_wall=3.0)
    n0 = lab.shape[0]
    zs = np.arange(p['offset'] / VOX, n0 - 1, p['spacing'] / VOX)
    # slices containing LV myocardium (axis 0 runs base -> apex because ax points base->apex)
    has = [z for z in zs if (lab[int(round(z))] == 1).any()]
    if p['drop_basal'] and len(has) > 3:
        has = has[1:]
    zs = np.array(has)
    step = max(1.0, p['inplane'] / VOX)          # in-plane sampling in voxels (1 at VOX = 1.5 mm)
    yi = np.arange(0, lab.shape[1] - 1, step); xi = np.arange(0, lab.shape[2] - 1, step)
    S = lab[np.round(zs).astype(int)][:, np.round(yi).astype(int)][:, :, np.round(xi).astype(int)]
    lvcav = S == 3; lvepi = (S == 1) | (S == 3); rvcav = S == 4
    F = {k: sdf(m, (step * VOX, step * VOX)) for k, m in dict(lvcav=lvcav, lvepi=lvepi, rvcav=rvcav).items()}
    # upsample to 1 mm grid (linear in z between slices, linear in-plane); outside slice range: open (no tissue)
    gz, gy, gx = np.meshgrid(np.arange(lab.shape[0]), np.arange(lab.shape[1]), np.arange(lab.shape[2]), indexing='ij')
    cz = np.interp(gz, zs, np.arange(len(zs)), left=-1, right=len(zs))
    coords = np.stack([np.clip(cz, 0, len(zs) - 1), gy / step, gx / step])
    outside = (cz < 0) | (cz > len(zs) - 1)
    fld = smooth_field(lab.shape, p['field_sd'], 15.0 / VOX, rng)
    up = {k: ndi.map_coordinates(v, coords, order=1, mode='nearest') for k, v in F.items()}
    lvcav_m = (up['lvcav'] - p['d_endo'] + fld < 0) & ~outside
    lvepi_m = (up['lvepi'] - p['d_epi'] + fld < 0) & ~outside
    rvcav_m = (up['rvcav'] - p['d_rv'] + fld < 0) & ~outside & ~lvepi_m
    # apical cap: myocardium closes within half a slice beyond the most apical slice (cone of the last epi contour)
    dz = (gz - zs[-1]) * VOX
    cap = (dz > 0) & (dz <= p['spacing'] / 2) & (up['lvepi'] + 1.5 * dz - p['d_epi'] + fld < 0)
    lvepi_m |= cap
    # apex closure: the most apical slice is capped by LV myocardium half a slice beyond (shape interpolation)
    new = np.zeros(lab.shape, np.int8)
    new[lvepi_m] = 1; new[lvcav_m] = 3; new[rvcav_m] = 4
    # assumed RV free wall: dilate RV cavity by rv_wall mm outside LV
    dist_rv = ndi.distance_transform_edt(~rvcav_m, sampling=VOX)
    rvw = (dist_rv <= p['rv_wall']) & ~rvcav_m & ~lvepi_m
    zr = np.nonzero(rvcav_m.any((1, 2)))[0]
    new[rvw & (gz >= zr.min()) & (gz <= zr.max())] = 2
    return new, p


def mesh_from_labels(L):
    """myocardial voxels (1,2) -> tets (6 per voxel), world coordinates; returns pts, tets, endo-node flags."""
    myo = (L == 1) | (L == 2)
    cc_, ncc = ndi.label(myo)
    if ncc > 1:
        sizes = ndi.sum(myo, cc_, range(1, ncc + 1)); myo = cc_ == (1 + int(np.argmax(sizes)))
    vi = np.argwhere(myo)
    shp = np.array(L.shape) + 1
    corner = lambda a, b, cc: ((vi[:, 0] + a) * shp[1] + (vi[:, 1] + b)) * shp[2] + (vi[:, 2] + cc)
    cn = [corner(a, b, cc) for a in (0, 1) for b in (0, 1) for cc in (0, 1)]
    paths = [(4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3)]
    tets = np.concatenate([np.column_stack([cn[0], cn[p], cn[q], cn[7]]) for p, q in paths])
    used, inv = np.unique(tets, return_inverse=True)
    tets = inv.reshape(tets.shape)
    ijk = np.column_stack(np.unravel_index(used, tuple(shp))).astype(np.float64) - 0.5   # voxel corners
    pts = ((ijk + lo) * VOX) @ Rf + c
    tets = carpio.fix_orientation(pts, tets)
    # blood-interface nodes: corners of blood voxels (3,4)
    bv = np.argwhere((L == 3) | (L == 4))
    bc = np.unique(np.concatenate([((bv[:, 0] + a) * shp[1] + (bv[:, 1] + b)) * shp[2] + (bv[:, 2] + cc)
                                   for a in (0, 1) for b in (0, 1) for cc in (0, 1)]))
    iface = np.isin(used, bc)
    lvb = np.argwhere(L == 3)
    lc = np.unique(np.concatenate([((lvb[:, 0] + a) * shp[1] + (lvb[:, 1] + b)) * shp[2] + (lvb[:, 2] + cc)
                                   for a in (0, 1) for b in (0, 1) for cc in (0, 1)]))
    lv_if = np.isin(used, lc)
    return pts, tets, iface, lv_if


def lengths(L):
    lvmyo = L == 1
    vol = float(((L == 1) | (L == 2)).sum()) * VOX ** 3
    edt = ndi.distance_transform_edt(lvmyo, sampling=VOX)
    zz = np.nonzero(lvmyo.any((1, 2)))[0]
    return dict(L_scale=vol ** (1 / 3), L_wall=float(4 * edt[lvmyo].mean()), L_base=float(zz.max() - zz.min() + 1) * VOX,
                myo_mL=vol / 1000.0, lv_myo_mL=float(lvmyo.sum()) * VOX ** 3 / 1000.0,
                lv_cav_mL=float((L == 3).sum()) * VOX ** 3 / 1000.0)


Ptree = cKDTree(P0[T0].mean(1))
_cen = tp[tt].mean(1); _lo, _hi = P0.min(0) - 25, P0.max(0) + 25
tt_near = tt[np.all((_cen > _lo) & (_cen < _hi), axis=1)]; del _cen
os.makedirs(meshdir, exist_ok=True)
summary = []
cons = [('c0ref', lab, dict(reference=True))]
for k in range(1, K + 1):
    newlab, p = sax_construction(lab, rng)
    cons.append((f'c{k}', newlab, p))
with h5py.File(os.path.join(outdir, f'cons{hid}.h5'), 'w') as fo:
    for name, L, p in cons:
        pts, tets, iface, lv_if = mesh_from_labels(L)
        cen = pts[tets].mean(1)
        fib = fib0[Ptree.query(cen)[1]]
        dist = cKDTree(pts[iface]).query(pts)[0]
        etag = np.where((dist <= 1e-6)[tets].any(1), 2, 1)          # one voxel layer touching the blood
        lvn = np.nonzero(lv_if)[0]; rvn = np.nonzero(iface & ~lv_if)[0]
        rnodes = [lvn[cKDTree(pts[lvn]).query(r)[1]] for r in roots_xyz[:3]] + [rvn[cKDTree(pts[rvn]).query(roots_xyz[3])[1]]]
        stim = np.unique(np.concatenate([cKDTree(pts).query_ball_point(pts[r], 2.5) for r in rnodes]))
        base = os.path.join(meshdir, f'h{hid}_{name}')
        carpio.write_mesh(base, pts * 1000, tets, etag, fib.astype(np.float64))
        carpio.write_vtx(base + '_stim.vtx', stim)
        Zs, sc = eplib.locate_and_interp(tp, tt_near, Z_torso, pts)
        Ki = eplib.stiffness(pts, tets, fib=fib.astype(np.float64), sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
        W = eplib.ecg_weights(Ki, Zs)
        Ls = lengths(L)
        root_shift = float(np.mean([np.linalg.norm(pts[r] - x) for r, x in zip(rnodes, roots_xyz)]))
        gp = fo.create_group(name)
        gp['W'] = W; gp['lab'] = L; gp['pts'] = pts; gp['tets'] = tets.astype(np.int32); gp['fib'] = fib.astype(np.float32)
        gp['etag'] = etag.astype(np.int8); gp['stim'] = stim.astype(np.int32); gp['roots'] = np.array(rnodes, np.int32)
        gp.attrs['vox'] = VOX; gp.attrs.update({**Ls, **{k: v for k, v in p.items()}, 'n_nodes': len(pts),
                                                      'n_tets': len(tets), 'endo_frac': float((etag == 2).mean()),
                                                      'root_shift_mm': root_shift, 'outside_nodes': int((sc < -1e-6).sum())})
        summary.append(dict(name=name, **Ls, n_tets=len(tets), root_shift_mm=root_shift, **{k: v for k, v in p.items()}))
        print(f'  {name}: {len(tets)} tets, endo {np.mean(etag==2)*100:.0f}%, lengths', {k: round(v, 2) for k, v in Ls.items()},
              f'root shift {root_shift:.1f} mm, {time.time()-t0:.0f}s', flush=True)
json.dump(summary, open(os.path.join(outdir, f'cons{hid}.json'), 'w'), indent=1)
