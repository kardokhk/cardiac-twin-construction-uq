"""Amendment A2 fast endocardial layer (owner mesh2, I11 full arm, 2026-10-28).

Definition (final/AMENDMENT_2026-10-28.md, A2): all myocardial elements whose centroid lies within `thr_mm`
(1.5 mm) of the LV or RV endocardial surface, at both mesh resolutions. The endocardial surface is the set of
voxel faces shared by a kept myocardial voxel (labels 1 or 2, largest 6-connected component, exactly as in
sax_route.mesh_from_labels) and a blood voxel (label 3 LV blood or 4 RV blood). Faces between myocardium and
background (epicardium, including the RV free-wall epicardium, and the basal cut) are not endocardial, so the
layer lies on the cavity side only.

Distances are exact Euclidean point-to-square distances in grid-index space, scaled by the voxel size (the grid map
((q + lo) * vox) @ R + c is a rigid map times vox, so distances are preserved). No random numbers are drawn.

The gates-run definition (one voxel: tets with any node on a blood-interface voxel corner) is kept as
`onevoxel_layer` for the A2 sensitivity.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

THR_MM = 1.5
HALF_DIAG = np.sqrt(0.5)          # index units: centre-to-corner distance of a unit square face


def kept_myocardium(L):
    """Largest 6-connected component of labels 1 or 2 (identical to sax_route.mesh_from_labels)."""
    myo = (L == 1) | (L == 2)
    cc_, ncc = ndi.label(myo)
    if ncc > 1:
        sizes = ndi.sum(myo, cc_, range(1, ncc + 1)); myo = cc_ == (1 + int(np.argmax(sizes)))
    return myo


def endo_faces(L, myo=None):
    """Endocardial voxel faces: centres (F, 3) in grid-index units and normal axis (F,).
    Also returns which cavity each face bounds (3 LV, 4 RV)."""
    if myo is None:
        myo = kept_myocardium(L)
    cen, axs, cav = [], [], []
    for ax in range(3):
        for s in (-1, 1):
            nb = np.zeros(L.shape, np.int8)
            src = [slice(None)] * 3; dst = [slice(None)] * 3
            if s == 1:
                dst[ax] = slice(0, -1); src[ax] = slice(1, None)
            else:
                dst[ax] = slice(1, None); src[ax] = slice(0, -1)
            nb[tuple(dst)] = L[tuple(src)]
            m = myo & ((nb == 3) | (nb == 4))
            v = np.argwhere(m).astype(np.float64)
            cv = nb[m]
            v[:, ax] += 0.5 * s
            cen.append(v); axs.append(np.full(len(v), ax, np.int8)); cav.append(cv.astype(np.int8))
    return np.concatenate(cen), np.concatenate(axs), np.concatenate(cav)


def _face_dist(p, fc, fa):
    """Exact distance from points p (m, 3) to axis-aligned unit squares (centre fc (m, 3), normal axis fa (m,))."""
    d = np.abs(p - fc)
    out = np.maximum(d - 0.5, 0.0)
    n = np.arange(len(p))
    out[n, fa] = d[n, fa]
    return np.sqrt((out ** 2).sum(1))


def dist_to_faces(q, fcen, fax, max_idx, k=32):
    """Distance (index units) from query points q to the nearest face, exact when the distance is <= max_idx;
    np.inf where no face lies within max_idx. k nearest face centres are examined; points whose k-th candidate is
    still within the search radius are recomputed with a ball query, so the result is exact."""
    tree = cKDTree(fcen)
    rad = max_idx + HALF_DIAG
    out = np.full(len(q), np.inf)
    dd, ii = tree.query(q, k=k, distance_upper_bound=rad)
    for j in range(k):
        ok = np.isfinite(dd[:, j])
        if not ok.any():
            continue
        dj = np.full(len(q), np.inf)
        dj[ok] = _face_dist(q[ok], fcen[ii[ok, j]], fax[ii[ok, j]])
        out = np.minimum(out, dj)
    full = np.isfinite(dd[:, -1])            # all k candidates inside the radius: could be more
    if full.any():
        for i in np.nonzero(full)[0]:
            cand = np.array(tree.query_ball_point(q[i], rad))
            out[i] = _face_dist(np.repeat(q[i][None], len(cand), 0), fcen[cand], fax[cand]).min()
    out[out > max_idx] = np.inf
    return out


def tet_centroids_index(pts, tets, vox, lo, R, c):
    """Tet centroids in grid-index units (inverse of ((q + lo) * vox) @ R + c, R orthonormal)."""
    cen = pts[tets].mean(1)
    return ((cen - c) @ np.asarray(R).T) / vox - np.asarray(lo)


def a2_layer(L, pts, tets, vox, lo, R, c, thr_mm=THR_MM):
    """Returns (in_layer bool (M,), distance in mm (M,) with inf beyond 2*thr_mm, n_faces)."""
    R = np.asarray(R, float)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), 'grid map must be rigid'
    fcen, fax, _ = endo_faces(L)
    if len(fcen) == 0:
        return np.zeros(len(tets), bool), np.full(len(tets), np.inf), 0
    q = tet_centroids_index(pts, tets, vox, lo, R, c)
    d = dist_to_faces(q, fcen, fax, 2.0 * thr_mm / vox) * vox
    return d <= thr_mm + 1e-9, d, len(fcen)      # ties (d == thr, axis-aligned faces at 1.0 mm) are inside


def onevoxel_layer(tets, iface):
    """Gates-run definition: any node of the tet on a blood-interface voxel corner."""
    return iface[tets].any(1)


# ================================================================================================================
# Amendment 2, B1: smooth-surface fast layer (owner layer-sdf, 2026-10-28). Declared sensitivity; the A2 functions
# above are unchanged. Definition (final/AMENDMENT_2_2026-10-28.md, B1): all myocardial elements whose centroid lies
# within 1.5 mm of the smooth LV or RV endocardial surface; the surface is the zero level set of a signed distance
# field of the blood pool, built from the label mask upsampled to 0.25 mm with Gaussian smoothing (sigma = one native
# voxel) and marching cubes; cavity side only, as in A2.
# Choices where B1 is silent (assumptions, listed in results/2026-10-28_layer_sdf/README.md):
#   * "native voxel" = the construction's label-grid voxel `vox` (1.5 or 1.0 mm), the grid the mesh is built on and
#     the grid of the A2 unit test; sigma = vox mm = vox / 0.25 upsampled voxels.
#   * Each cavity (label 3 LV blood, label 4 RV blood) is processed separately: nearest-neighbour upsampling of its
#     indicator by f = vox / 0.25 (6 or 4) per axis, Gaussian filter (scipy.ndimage, truncate 4, zero outside), surface
#     = marching cubes (skimage 0.25.2, lewiner) at level 0.5 of the smoothed indicator. The signed distance field
#     whose zero level set this is: distance to that triangulated surface, negative where the smoothed indicator
#     exceeds 0.5. Smoothing the two cavities separately keeps the septum from closing.
#   * Cavity side only: a surface triangle is kept when the nearest non-blood native voxel to its centroid is kept
#     myocardium (kept_myocardium), so the smooth analogue of the blood-background faces (basal cut) is dropped, as A2
#     drops blood-background voxel faces. The RV free-wall epicardium is never part of a blood-pool surface.
#   * Distance = exact Euclidean point-to-triangle distance from the tet centroid (unsigned); inside if <= 1.5 mm.
# No random numbers are drawn.
SDF_UP_MM = 0.25
SKIMAGE_EXTRA = '/project/home/p201509/envs/cvri-sim-b1-extra'      # scikit-image 0.25.2 + lazy_loader 0.4 (--no-deps)


def _marching_cubes():
    import sys
    if SKIMAGE_EXTRA not in sys.path:
        sys.path.append(SKIMAGE_EXTRA)
    from skimage.measure import marching_cubes
    return marching_cubes


def point_triangle_dist(p, a, b, c):
    """Exact distance from points p (m, 3) to triangles (a, b, c) (m, 3) each (Ericson, Real-Time Collision Detection,
    5.1.5), vectorised. Degenerate triangles fall back to the nearest vertex or edge."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = p - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    q = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def put(m, val):
        m = m & ~done
        q[m] = val[m]; done[m] = True
    with np.errstate(divide='ignore', invalid='ignore'):
        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        v = d1 / (d1 - d3); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        w = d2 / (d2 - d6); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        den = 1.0 / (va + vb + vc)
        v = vb * den; w = vc * den
        put(np.ones(len(p), bool), a + v[:, None] * ab + w[:, None] * ac)
    out = np.sqrt(((p - q) ** 2).sum(1))
    bad = ~np.isfinite(out)
    if bad.any():                                    # degenerate triangle: nearest vertex (upper bound, exact enough)
        out[bad] = np.min([np.sqrt(((p[bad] - x[bad]) ** 2).sum(1)) for x in (a, b, c)], 0)
    return out


def smooth_endo_surface(L, vox, up_mm=SDF_UP_MM, sigma_mm=None, myo=None):
    """Smooth endocardial surface of a label grid. Returns (verts (V, 3) in native grid-index units, faces (F, 3),
    cavity label per face (F,), info dict). Only cavity-side triangles (see block comment) are returned."""
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    marching_cubes = _marching_cubes()
    if myo is None:
        myo = kept_myocardium(L)
    sigma_mm = float(vox) if sigma_mm is None else float(sigma_mm)
    f = int(round(vox / up_mm))
    assert abs(f * up_mm - vox) < 1e-9, (vox, up_mm)
    s_up = sigma_mm / up_mm
    margin = int(np.ceil((4.0 * sigma_mm + vox) / vox)) + 1
    V, F, CV = [], [], []
    info = dict(upsample_factor=f, sigma_mm=sigma_mm, sigma_up_vox=s_up, up_mm=up_mm, n_tri_all=0, n_tri_kept=0)
    nv = 0
    for cav in (3, 4):
        blood = L == cav
        if not blood.any():
            continue
        idx = np.argwhere(blood)
        i0 = idx.min(0) - margin; i1 = idx.max(0) + margin + 1
        pad_lo = np.maximum(-i0, 0); pad_hi = np.maximum(i1 - np.array(L.shape), 0)
        c0 = np.maximum(i0, 0); c1 = np.minimum(i1, L.shape)
        crop = blood[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
        crop = np.pad(crop, [(int(a), int(b)) for a, b in zip(pad_lo, pad_hi)])
        org = i0                                           # native index of crop voxel 0 (may be negative)
        U = crop.astype(np.float32).repeat(f, 0).repeat(f, 1).repeat(f, 2)
        G = gaussian_filter(U, sigma=s_up, mode='constant', cval=0.0, truncate=4.0)
        del U
        if not (G.max() > 0.5):
            continue
        vv, ff, _, _ = marching_cubes(G, level=0.5, method='lewiner')
        del G
        vv = org + (vv + 0.5) / f - 0.5                     # upsampled index -> native index units
        # cavity side: nearest non-blood native voxel of each triangle centroid must be kept myocardium
        cen = vv[ff].mean(1)
        _, ind = distance_transform_edt(blood, return_indices=True)
        r = np.clip(np.rint(cen).astype(int), 0, np.array(L.shape) - 1)
        nn = ind[:, r[:, 0], r[:, 1], r[:, 2]].T             # nearest non-blood voxel (itself if not blood)
        keep = myo[nn[:, 0], nn[:, 1], nn[:, 2]]
        info['n_tri_all'] += int(len(ff)); info['n_tri_kept'] += int(keep.sum())
        info[f'n_tri_cav{cav}'] = int(keep.sum())
        V.append(vv); F.append(ff[keep] + nv); CV.append(np.full(int(keep.sum()), cav, np.int8)); nv += len(vv)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros(0, np.int8), info
    return np.concatenate(V), np.concatenate(F), np.concatenate(CV), info


def dist_to_triangles(q, verts, faces, max_d, k=16):
    """Exact distance from points q (m, 3) to a triangle soup (same units), np.inf beyond max_d. The k triangles with
    the nearest centroids are checked; a point is certified when its k-th candidate centroid lies farther than
    (best distance + largest centroid-to-vertex radius) or no further centroid lies within max_d + radius;
    uncertified points are redone with 4k candidates, repeatedly, and finally with a ball query."""
    tri = verts[faces]
    cen = tri.mean(1)
    rmax = float(np.sqrt(((tri - cen[:, None]) ** 2).sum(-1)).max())
    tree = cKDTree(cen)
    out = np.full(len(q), np.inf)
    todo = np.arange(len(q))
    n_ball = 0
    while len(todo):
        kk = min(k, len(cen))
        dd, ii = tree.query(q[todo], k=kk, distance_upper_bound=max_d + rmax)
        dd = dd.reshape(len(todo), -1); ii = ii.reshape(len(todo), -1)
        best = np.full(len(todo), np.inf)
        for j in range(kk):
            ok = np.isfinite(dd[:, j])
            if not ok.any():
                continue
            t = tri[ii[ok, j]]
            best[ok] = np.minimum(best[ok], point_triangle_dist(q[todo][ok], t[:, 0], t[:, 1], t[:, 2]))
        out[todo] = best
        unsure = np.isfinite(dd[:, -1]) & (dd[:, -1] <= np.minimum(best, max_d) + rmax) & (kk < len(cen))
        todo = todo[unsure]
        if k >= 1024 and len(todo):
            for i in todo:
                cand = np.array(tree.query_ball_point(q[i], min(out[i], max_d) + rmax))
                t = tri[cand]
                out[i] = point_triangle_dist(np.repeat(q[i][None], len(cand), 0), t[:, 0], t[:, 1], t[:, 2]).min()
            n_ball = int(len(todo))
            break
        k *= 4
    out[out > max_d] = np.inf
    return out, n_ball


def sdf_layer(L, pts, tets, vox, lo, R, c, thr_mm=THR_MM, up_mm=SDF_UP_MM, sigma_mm=None):
    """B1 layer. Returns (in_layer bool (M,), distance in mm (M,) with inf beyond 2*thr_mm, info dict)."""
    R = np.asarray(R, float)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), 'grid map must be rigid'
    verts, faces, _, info = smooth_endo_surface(L, vox, up_mm, sigma_mm)
    if len(faces) == 0:
        return np.zeros(len(tets), bool), np.full(len(tets), np.inf), info
    q = tet_centroids_index(pts, tets, vox, lo, R, c) * vox          # mm in the grid-index frame (rigid map)
    d, nb = dist_to_triangles(q, verts * vox, faces, 2.0 * thr_mm)
    info['n_ball_queries'] = nb
    return d <= thr_mm + 1e-9, d, info
