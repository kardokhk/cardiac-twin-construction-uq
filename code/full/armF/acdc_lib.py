"""ACDC constructions for the I11 study (owner mesh, I11 gates, 2026-10-28; copied to code/full/mesh by owner mesh2 for the
full arm, 2026-10-28. Changes: prepare_slices takes rv_wall; materialize/forward take etag_key. The fast endocardial
layer of the full arm (Amendment A2) is in endo_layer.py; the one-voxel layer described below is kept as a sensitivity.)


One construction = (heart, segmentation source, voxel size). Pipeline (fixed for all hearts; see
results/2026-10-28_G2_meshing/G2_criteria.md):
  labels -> per-class largest 26-connected component -> LV slices -> apex end by area rule -> sax_route
  reconstruction (pilot algebra) -> voxel tets -> Bayer 2012 fibres (I09 implementation via ka's bayer_i09.py)
  -> one-voxel fast endocardial layer (tag 2) -> root nodes (rule on the reference construction, mapped)
  -> placement in Malik heart 09 torso (frame alignment, transform fixed by the heart's expert construction)
  -> ECG weights W = K_i Z (homogeneous lead field of heart 09, electrode rule v2) -> smoke eikonal run.

Orientation (ACDC NIfTI affines carry no spacing and no orientation; the stored affine is diag(-1,-1,1)):
  * Image-physical coordinates are x = i*sx, y = j*sy, z = k*sz from the header zooms (a proper, unmirrored map,
    the same handedness as the stored affine). Mirroring of the original acquisition cannot be detected from the
    labels; it is an assumption.
  * Long axis = slice normal (SAX planes are planned perpendicular to the LV long axis), pointing base -> apex.
    The apex end is the stack end whose two outermost LV slices have the smaller mean LV epicardial area.
  * Septal direction = LV cavity centroid -> RV cavity centroid, projected onto the short-axis plane.
  * Anterior = cross(septal, long axis). This is the anatomical chirality found in 16 of 17 Malik heart frames
    (the 17th, heart 07, has a pilot sign artefact), including heart 09.
  * Rotation R maps (axis, septal, anterior) of the heart onto heart 09's frame; translation maps the LV
    myocardial centroid onto heart 09's LV myocardial centroid. det(R) = +1 is asserted.
  * ICP refinement onto heart 09's 1.5 mm surface is computed as a diagnostic, and its ECG weights are stored as
    W_icp; W (frame alignment) is the primary.
Units: mm, ms, m/s (= mm/ms).
"""
import json
import os
import sys
import time
import shutil

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'code', 'pilot', 'I13'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'gates', 'ka'))
import carpio            # noqa: E402  (code/gates/mesh/carpio.py)
import sax_route         # noqa: E402
import eplib             # noqa: E402  (code/pilot/I13/eplib.py, unchanged)

V_FIB = 0.6
THETA_SMOKE = (1.55, 0.29)          # geometric centre of the pilot theta* range (1.0-2.4, 0.2-0.42)
R_ROOT = 2.5                        # mm, stimulus patch radius around each root node (pilot)
RV_WALL = 3.0
BAYER_SEED = 20261028               # numpy global seed set before each Bayer call (pyamg spectral-radius estimate)
TORSO = os.path.join(ROOT, 'data', 'processed', 'acdc_mesh', 'torso09_near.npz')
SOURCES = ['expert', 'nnunet_task027_acdc', 'nnunet_task114_mnms', 'nnunet_ukb_sax', 'cinema_acdc_sax',
           'ukbb_cardiac_fcn_sa']


class GateFail(Exception):
    def __init__(self, code, msg):
        super().__init__(f'{code}: {msg}')
        self.code = code


# ----------------------------------------------------------------------------------------------- inputs
def info_cfg(pid):
    d = {}
    for line in open(os.path.join(ROOT, 'data/raw/acdc/extracted/testing', f'patient{pid:03d}', 'Info.cfg')):
        if ':' in line:
            k, v = line.split(':', 1); d[k.strip()] = v.strip()
    return d


def label_path(pid, source):
    if source == 'expert':
        ed = int(info_cfg(pid)['ED'])
        return os.path.join(ROOT, 'data/raw/acdc/extracted/testing', f'patient{pid:03d}', f'patient{pid:03d}_frame{ed:02d}_gt.nii.gz')
    return os.path.join(ROOT, 'data/processed/acdc_seg', source, f'patient{pid:03d}_ED.nii.gz')


def load_labels(path):
    import nibabel as nib
    im = nib.load(path)
    lab = np.rint(np.asarray(im.dataobj)).astype(np.int16)
    zooms = tuple(float(z) for z in im.header.get_zooms()[:3])
    return lab, zooms


# ----------------------------------------------------------------------------------------------- slices
def prepare_slices(lab, zooms, vox, direction=None, crop_mm=15.0, grid_mm=10.0, rv_wall=RV_WALL):
    """ACDC labels (i, j, k; 1 RV cavity, 2 LV myo, 3 LV cavity) -> sax_route ACDC input + grid geometry."""
    sx, sy, sz = zooms
    P = np.zeros(lab.shape, np.int8)
    P[lab == 2] = 1; P[lab == 3] = 3; P[lab == 1] = 4
    raw = dict(lv_myo_label_mL=float((lab == 2).sum()) * sx * sy * sz / 1000,
               lv_cav_label_mL=float((lab == 3).sum()) * sx * sy * sz / 1000,
               rv_cav_label_mL=float((lab == 1).sum()) * sx * sy * sz / 1000)
    st = np.ones((3, 3, 3), bool)
    removed = {}
    for v in (1, 3, 4):
        m = P == v
        cc, n = ndi.label(m, structure=st)
        if n > 1:
            sizes = ndi.sum(m, cc, range(1, n + 1))
            keep = cc == (1 + int(np.argmax(sizes)))
            removed[v] = int(m.sum() - keep.sum())
            P[m & ~keep] = 0
    lv_sl = [k for k in range(P.shape[2]) if (P[:, :, k] == 1).any()]
    if len(lv_sl) < 2:
        raise GateFail('F2', f'{len(lv_sl)} slices with LV myocardium')
    if not (P[:, :, lv_sl] == 4).any():
        raise GateFail('F2', 'no RV cavity in any LV slice')
    area = {k: float(((P[:, :, k] == 1) | (P[:, :, k] == 3)).sum()) * sx * sy for k in lv_sl}
    a_lo = np.mean([area[k] for k in lv_sl[:2]]); a_hi = np.mean([area[k] for k in lv_sl[-2:]])
    auto_dir = 1 if a_hi < a_lo else -1          # +1: apex at high k
    d = auto_dir if direction is None else int(direction)
    order = lv_sl if d == 1 else lv_sl[::-1]
    k_base = order[0]
    sub = P[:, :, lv_sl]
    ii, jj = np.nonzero((sub > 0).any(2))
    m1, m2 = int(np.ceil(crop_mm / sx)), int(np.ceil(crop_mm / sy))
    i0, i1 = max(ii.min() - m1, 0), min(ii.max() + m1, P.shape[0] - 1)
    j0, j1 = max(jj.min() - m2, 0), min(jj.max() + m2, P.shape[1] - 1)
    S = np.stack([P[i0:i1 + 1, j0:j1 + 1, k] for k in order]).astype(np.int8)
    zpos = np.array([d * (k - k_base) * sz for k in order])            # mm, base -> apex, zpos[0] = 0
    z0 = -2.0 * vox
    n0 = int(np.ceil((zpos[-1] + sz / 2 + 2 * vox - z0) / vox)) + 1
    x0 = ii.min() * sx - grid_mm; x1 = ii.max() * sx + grid_mm
    y0 = jj.min() * sy - grid_mm; y1 = jj.max() * sy + grid_mm
    n1 = int(np.ceil((x1 - x0) / vox)) + 1; n2 = int(np.ceil((y1 - y0) / vox)) + 1
    p1 = (x0 + np.arange(n1) * vox) / sx - i0
    p2 = (y0 + np.arange(n2) * vox) / sy - j0
    acdc = dict(S=S, zs=(zpos - z0) / vox, dyx=(sx, sy), n0=n0, p1=p1, p2=p2, spacing=sz, vox=vox, rv_wall=float(rv_wall))
    lo = np.array([z0 / vox, x0 / vox, y0 / vox])
    Rg = np.array([[0, 0, d], [1, 0, 0], [0, 1, 0]], float)             # grid axes (0,1,2) -> image x, y, z
    cimg = np.array([0.0, 0.0, k_base * sz])
    info = dict(n_lv_slices=len(lv_sl), slice_mm=sz, inplane_mm=[sx, sy], direction=d, auto_direction=auto_dir,
                k_base=int(k_base), k_apex=int(order[-1]), removed_vox=removed, **raw)
    return acdc, lo, Rg, cimg, info


def voxel_centres(L, value, vox, lo, R, c):
    idx = np.argwhere(L == value).astype(float)
    return ((idx + lo) * vox) @ R + c


# ----------------------------------------------------------------------------------------------- torso
def load_torso():
    d = np.load(TORSO)
    return {k: d[k] for k in d.files}


def frame_transform(L, vox, lo, Rg, cimg, direction, T):
    """Heart frame in image-physical coordinates and the rigid map onto heart 09's frame."""
    ax = np.array([0, 0, float(direction)])
    c_lvcav = voxel_centres(L, 3, vox, lo, Rg, cimg).mean(0)
    c_rvcav = voxel_centres(L, 4, vox, lo, Rg, cimg).mean(0)
    c_lv = voxel_centres(L, 1, vox, lo, Rg, cimg).mean(0)
    sep = c_rvcav - c_lvcav; sep -= (sep @ ax) * ax; sep /= np.linalg.norm(sep)
    ant = np.cross(sep, ax)
    A = np.vstack([ax, sep, ant]); F9 = T['frame09']; B = np.vstack(F9[1:4])
    R = B.T @ A
    assert abs(np.linalg.det(R) - 1) < 1e-6, np.linalg.det(R)
    t = F9[0] - R @ c_lv
    return dict(ax_img=ax, sep_img=sep, ant_img=ant, c_lv_img=c_lv, R=R, t=t)


# ----------------------------------------------------------------------------------------------- features
def boundary_nodes(tets):
    return np.unique(eplib.boundary_faces(tets))


def nonmanifold_edges(tets):
    f = eplib.boundary_faces(tets)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum()), int(len(cnt))


def root_rule(pts, tet_lab, tets, lv_if, iface, T):
    """Pilot rule (code/pilot/I13/02_prep_heart.py) on the reference construction, in torso coordinates."""
    c09, ax, sep, ant = T['frame09']
    lvn = np.unique(tets[tet_lab == 1])
    s = (pts - c09) @ ax
    s_ap, s_bs = s[lvn].max(), s[lvn].min()
    h = (s_ap - s) / (s_ap - s_bs)
    lv_endo = np.nonzero(lv_if)[0]; rv_endo = np.nonzero(iface & ~lv_if)[0]
    c_lv = pts[lvn].mean(0)

    def pick(cands, hval, direction):
        for tol in (0.05, 0.1, 0.2):
            m = cands[np.abs(h[cands] - hval) < tol]
            if len(m):
                v = pts[m] - c_lv; v -= np.outer(v @ ax, ax)
                return m[np.argmax(v @ direction)]
        raise GateFail('F1', f'no root candidate at h={hval}')
    r = [pick(lv_endo, 0.50, sep), pick(lv_endo, 0.65, (sep + ant) / np.sqrt(2)),
         pick(lv_endo, 0.33, (sep - ant) / np.sqrt(2)), pick(rv_endo, 0.40, -sep)]
    return pts[np.array(r)]


def map_roots(pts, lv_if, iface, roots_xyz):
    lvn = np.nonzero(lv_if)[0]; rvn = np.nonzero(iface & ~lv_if)[0]
    if len(lvn) == 0 or len(rvn) == 0:
        raise GateFail('F1', 'no LV or RV endocardial nodes for roots')
    r = [lvn[cKDTree(pts[lvn]).query(x)[1]] for x in roots_xyz[:3]] + [rvn[cKDTree(pts[rvn]).query(roots_xyz[3])[1]]]
    r = np.array(r)
    stim = np.unique(np.concatenate([cKDTree(pts).query_ball_point(pts[i], R_ROOT) for i in r]))
    shift = np.linalg.norm(pts[r] - roots_xyz, axis=1)
    return r, stim, shift


def bayer(pts, tets, lv_if, iface, vox, T):
    import bayer_i09
    c09, ax = T['frame09'][0], T['frame09'][1]
    s = (pts - c09) @ ax
    bn = boundary_nodes(tets)
    lv = np.nonzero(lv_if)[0]; rv = np.nonzero(iface & ~lv_if)[0]
    apex = np.nonzero(s >= s.max() - 0.75 * vox)[0]
    base = bn[s[bn] <= s.min() + 0.75 * vox]
    # pyamg's smoothed-aggregation setup estimates spectral radii from a random start vector drawn from numpy's
    # global (legacy) RNG; unseeded, the Laplace solutions differ at the 1e-10 level between calls and the bislerp
    # step of the rule flips fibres in about 0.6% of elements. Seeding makes every construction reproducible.
    np.random.seed(BAYER_SEED)
    out, meta = bayer_i09.bayer_fibres(pts, tets, lv, rv, bn, apex, base)
    meta['numpy_global_seed'] = BAYER_SEED
    return out['fib'], out['nonfinite'], meta, dict(n_apex=len(apex), n_base=len(base))


def ecg_weights(pts, tets, fib, T, R_extra=None, t_extra=None):
    q = pts if R_extra is None else pts @ R_extra.T + t_extra
    Zs, score = eplib.locate_and_interp(T['tp'], T['tt'], T['Z'].astype(np.float64), q)
    Ki = eplib.stiffness(pts, tets, fib=fib.astype(np.float64), sig_l=eplib.SIG_IL, sig_t=eplib.SIG_IT)
    return eplib.ecg_weights(Ki, Zs), int((score < -1e-6).sum()), float(score.min())


# ----------------------------------------------------------------------------------------------- forward model
def materialize(h5path, outdir, etag_key='etag'):
    """Write the openCARP text mesh (pts in um, tags 1 myocardium / 2 fast endocardium, per-element fibres) and the
    root stimulus vtx of one construction file into outdir. Returns (meshbase, vtx)."""
    import h5py
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(h5path, 'r') as f:
        base = os.path.join(outdir, os.path.basename(h5path).replace('.h5', ''))
        carpio.write_mesh(base, f['pts'][:] * 1000.0, f['tets'][:].astype(np.int64), f[etag_key][:].astype(np.int64),
                          f['fib'][:].astype(np.float64))
        carpio.write_vtx(base + '_stim.vtx', f['stim'][:])
    return base, base + '_stim.vtx'


def eikonal(meshbase, vtx, theta, simdir, tend=400):
    """theta = (v_endo, v_t) m/s; fibre velocity fixed at V_FIB; renumbering 0."""
    return carpio.run_eikonal0(meshbase, vtx, [(1, V_FIB, theta[1], theta[1]), (2, theta[0], theta[0], theta[0])],
                               simdir, threads=1, tend=tend)


def forward(h5path, theta, workdir, Wkey='W', etag_key='etag'):
    """One forward model: activation times and the 8-lead activation-template ECG (eplib.ECGModel, tau 1 ms,
    dt 0.5 ms, tmax = max activation + 20 ms). Returns (act, ecg (8, nt), wall seconds)."""
    import h5py
    mb, vtx = materialize(h5path, workdir, etag_key)
    act, wall = eikonal(mb, vtx, theta, os.path.join(workdir, 'sim'))
    with h5py.File(h5path, 'r') as f:
        W = f[Wkey][:]
    ecg = eplib.ECGModel(W, tmax=float(act.max()) + 20)(act)
    return act, ecg, wall
