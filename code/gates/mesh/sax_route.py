"""Short-axis (SAX) reconstruction and voxel meshing route for ACDC constructions (I11 gates, owner mesh, 2026-10-28).

Derived from the pilot route code/pilot/I11/06_constructions.py (SHA-256 9a76f506...a4eac). The algebra of the
pilot is kept: per-slice in-plane signed distance (sdf), linear interpolation between slice centres, no tissue
outside the slice range except an apical cap (cone of the most apical epicardial contour, slope 1.5, within half a
slice spacing), assumed RV free wall by dilating the RV cavity by `rv_wall` mm outside the LV, restricted to the
RV cavity's axial range; voxel -> 6 tets, largest 6-connected myocardial component.

Changes from the pilot, and why:
  1. sax_construction(lab, rng, acdc=None) has a second input path: with `acdc` given it takes real slice labels
     (ACDC) instead of sampling slices from a fine 3D grid. Without `acdc` it reproduces the pilot path exactly
     (same rng draws, same parameter dict), so the ka owner's ellipsoid test (code/gates/ka/ellipsoid_sax.py,
     which extracts sdf, smooth_field, sax_construction, mesh_from_labels and lengths with `ast`) runs on this
     file unchanged. Slice positions may be any real grid-index values and in-plane pixel size need not be a multiple of the
     voxel size (in-plane grid positions are given in slice-pixel units).
  2. The RV-wall axial restriction is skipped when there is no RV cavity (the pilot crashed on an LV-only shell).
  3. mesh_from_labels and lengths take optional explicit geometry (vox, lo, Rf, c) instead of module globals;
     with no arguments they read the module globals, as in the pilot. mesh_from_labels can fill a QC dict.
Label convention of reconstructions (pilot): 0 background, 1 LV myocardium, 2 RV free wall (assumed), 3 LV blood,
4 RV blood. Grid axis 0 runs base -> apex.
"""
import numpy as np
from scipy import ndimage as ndi
import carpio

VOX = 1.5                  # module globals read by the pilot-style call path (the ellipsoid test sets them)
lo = np.zeros(3)
Rf = np.eye(3)
c = np.zeros(3)


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


def sax_construction(lab, rng, acdc=None):
    """Pilot path (acdc is None): sample `lab` (fine grid, axis 0 base->apex, voxel VOX) on short-axis slices with
    random observer perturbations, then reconstruct on the same grid. Returns (new labels, parameter dict).
    ACDC path: acdc = dict(S=(ns, n1s, n2s) slice labels in the pilot convention ordered base->apex,
    zs=(ns,) slice-centre positions in grid-index units of axis 0 (increasing), dyx=(d1, d2) in-plane pixel size
    in mm, n0=grid size along axis 0, p1=(n1,), p2=(n2,) grid positions in slice-pixel units, spacing=slice
    spacing in mm, vox=grid voxel size in mm, rv_wall=mm). No perturbations. Returns (new labels, parameter dict)."""
    if acdc is None:
        V = VOX
        p = dict(spacing=8.0, offset=float(rng.uniform(0, 8)), inplane=1.5, d_endo=float(rng.normal(0, 0.75)),
                 d_epi=float(rng.normal(0, 0.75)), d_rv=float(rng.normal(0, 0.75)), drop_basal=int(rng.random() < 0.5),
                 field_sd=0.5, rv_wall=3.0)
        n0 = lab.shape[0]
        zs = np.arange(p['offset'] / V, n0 - 1, p['spacing'] / V)
        has = [z for z in zs if (lab[int(round(z))] == 1).any()]
        if p['drop_basal'] and len(has) > 3:
            has = has[1:]
        zs = np.array(has)
        step = max(1.0, p['inplane'] / V)
        yi = np.arange(0, lab.shape[1] - 1, step); xi = np.arange(0, lab.shape[2] - 1, step)
        S = lab[np.round(zs).astype(int)][:, np.round(yi).astype(int)][:, :, np.round(xi).astype(int)]
        gz = np.arange(lab.shape[0])
        p1 = np.arange(lab.shape[1]) / step; p2 = np.arange(lab.shape[2]) / step
        dyx = (step * V, step * V)
        fld = smooth_field(lab.shape, p['field_sd'], 15.0 / V, rng)
        spacing = p['spacing']
    else:
        V = float(acdc['vox'])
        p = dict(spacing=float(acdc['spacing']), offset=0.0, inplane=float(min(acdc['dyx'])), d_endo=0.0, d_epi=0.0,
                 d_rv=0.0, drop_basal=0, field_sd=0.0, rv_wall=float(acdc.get('rv_wall', 3.0)))
        S = acdc['S']; zs = np.asarray(acdc['zs'], float); gz = np.arange(int(acdc['n0']))
        p1 = np.asarray(acdc['p1'], float); p2 = np.asarray(acdc['p2'], float); dyx = tuple(acdc['dyx'])
        fld = np.float32(0.0)
        spacing = p['spacing']
    ns = len(zs)
    lvcav = S == 3; lvepi = (S == 1) | (S == 3); rvcav = S == 4
    F = {k: sdf(m, dyx) for k, m in dict(lvcav=lvcav, lvepi=lvepi, rvcav=rvcav).items()}
    GZ, P1, P2 = np.meshgrid(gz, p1, p2, indexing='ij')
    cz = np.interp(GZ, zs, np.arange(ns), left=-1, right=ns)
    coords = np.stack([np.clip(cz, 0, ns - 1), P1, P2])
    outside = (cz < 0) | (cz > ns - 1)
    up = {k: ndi.map_coordinates(v, coords, order=1, mode='nearest') for k, v in F.items()}
    del coords, P1, P2
    lvcav_m = (up['lvcav'] - p['d_endo'] + fld < 0) & ~outside
    lvepi_m = (up['lvepi'] - p['d_epi'] + fld < 0) & ~outside
    rvcav_m = (up['rvcav'] - p['d_rv'] + fld < 0) & ~outside & ~lvepi_m
    dz = (GZ - zs[-1]) * V
    cap = (dz > 0) & (dz <= spacing / 2) & (up['lvepi'] + 1.5 * dz - p['d_epi'] + fld < 0)
    lvepi_m |= cap
    new = np.zeros(GZ.shape, np.int8)
    new[lvepi_m] = 1; new[lvcav_m] = 3; new[rvcav_m] = 4
    dist_rv = ndi.distance_transform_edt(~rvcav_m, sampling=V)
    rvw = (dist_rv <= p['rv_wall']) & ~rvcav_m & ~lvepi_m
    zr = np.nonzero(rvcav_m.any((1, 2)))[0]
    if len(zr):
        new[rvw & (GZ >= zr.min()) & (GZ <= zr.max())] = 2
    return new, p


def mesh_from_labels(L, vox=None, lo_=None, Rf_=None, c_=None, qc=None):
    """myocardial voxels (1,2) -> tets (6 per voxel), world coordinates; returns pts, tets, endo-node flags
    (blood interface, LV blood interface). World point of voxel corner index ijk: ((ijk - 0.5 + lo) * vox) @ Rf + c.
    If `qc` is a dict it receives: n_myo_vox, kept_frac, n_components, tet_lab (label 1 or 2 per tet)."""
    V = VOX if vox is None else vox
    LO = lo if lo_ is None else np.asarray(lo_, float)
    R_ = Rf if Rf_ is None else np.asarray(Rf_, float)
    C_ = c if c_ is None else np.asarray(c_, float)
    myo = (L == 1) | (L == 2)
    n_myo = int(myo.sum())
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
    pts = ((ijk + LO) * V) @ R_ + C_
    tets = carpio.fix_orientation(pts, tets)
    bv = np.argwhere((L == 3) | (L == 4))
    bc = np.unique(np.concatenate([((bv[:, 0] + a) * shp[1] + (bv[:, 1] + b)) * shp[2] + (bv[:, 2] + cc)
                                   for a in (0, 1) for b in (0, 1) for cc in (0, 1)]))
    iface = np.isin(used, bc)
    lvb = np.argwhere(L == 3)
    lc = np.unique(np.concatenate([((lvb[:, 0] + a) * shp[1] + (lvb[:, 1] + b)) * shp[2] + (lvb[:, 2] + cc)
                                   for a in (0, 1) for b in (0, 1) for cc in (0, 1)]))
    lv_if = np.isin(used, lc)
    if qc is not None:
        qc.update(n_myo_vox=n_myo, kept_frac=float(len(vi)) / max(n_myo, 1), n_components=int(ncc),
                  tet_lab=np.tile(L[vi[:, 0], vi[:, 1], vi[:, 2]].astype(np.int8), 6))
    return pts, tets, iface, lv_if


def lengths(L, vox=None):
    V = VOX if vox is None else vox
    lvmyo = L == 1
    vol = float(((L == 1) | (L == 2)).sum()) * V ** 3
    edt = ndi.distance_transform_edt(lvmyo, sampling=V)
    zz = np.nonzero(lvmyo.any((1, 2)))[0]
    return dict(L_scale=vol ** (1 / 3), L_wall=float(4 * edt[lvmyo].mean()), L_base=float(zz.max() - zz.min() + 1) * V,
                myo_mL=vol / 1000.0, lv_myo_mL=float(lvmyo.sum()) * V ** 3 / 1000.0,
                lv_cav_mL=float((L == 3).sum()) * V ** 3 / 1000.0)
