"""I09 pilot step 1: template (mean-shape) preprocessing shared by every KIT instance.

All KIT SSM instances share the mean mesh topology (heart_vol.vtu; Zenodo 4506463, CC BY 4.0), so labels,
cavity caps, stimulus node sets and the fast endocardial layer are defined once on the template and carried to
every instance by node or element index (point correspondence of the SSM).

Inputs  : heart_vol.vtu, heart_sur.vtp (class: 1 base, 2 epi, 3 LV endo, 4 RV endo, 5 apex), pc_vol.mat
Outputs : <out>/template.npz, <out>/pc_vol_f32.npy, <out>/tmpl.belem (openCARP binary elements, tags 1 and 2)
Usage   : python -I 01_template.py <kit_dir> <sur_vtp> <out_dir>
"""
import os
import subprocess
import sys

import numpy as np
import pyvista as pv
import scipy.io as sio
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carpio  # noqa: E402

kit, sur_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)

vol = pv.read(os.path.join(kit, "heart_vol.vtu"))
X0 = np.asarray(vol.points, float)
T = vol.cells_dict[10].astype(np.int64)  # VTK_TETRA
sur = pv.read(sur_path)
Xs = np.asarray(sur.points, float)
cls_s = np.asarray(sur.point_data["class"]).astype(int)
print("vol nodes", X0.shape[0], "tets", T.shape[0], "| sur nodes", Xs.shape[0])

# orient tets positively
a, b, c, d = (X0[T[:, k]] for k in range(4))
sv = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
neg = sv < 0
T[neg] = T[neg][:, [0, 2, 1, 3]]
print("tets reoriented", int(neg.sum()))

# boundary faces, outward oriented (for positive tet (0,1,2,3) outward faces are (1,2,3),(0,3,2),(0,1,3),(0,2,1))
F = np.concatenate([T[:, [1, 2, 3]], T[:, [0, 3, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 1]]])
key = np.sort(F, axis=1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
bF = F[cnt[inv.ravel()] == 1]
bnodes = np.unique(bF)
print("boundary faces", bF.shape[0], "boundary nodes", bnodes.size)

# map surface classes onto volume boundary nodes
tree = cKDTree(Xs)
dist, j = tree.query(X0[bnodes])
print("surface->volume node match: max distance %.3g mm" % dist.max())
ncls = np.zeros(X0.shape[0], int)
ncls[bnodes] = cls_s[j]
print("boundary node classes", {k: int((ncls == k).sum()) for k in range(1, 6)})

# face class: majority of node classes, ignoring apex (5) which is mapped to epi (2)
fc_nodes = ncls[bF].copy()
fc_nodes[fc_nodes == 5] = 2
fcls = np.zeros(bF.shape[0], int)
for k in range(1, 5):
    m = (fc_nodes == k).sum(1)
    fcls[m >= 2] = k
und = fcls == 0
fcls[und] = fc_nodes[und].max(1)  # rare 1-1-1 mixes
print("face classes", {k: int((fcls == k).sum()) for k in range(1, 5)})


def loops_of(faces):
    """Boundary edges (u->v in face orientation) of an open surface, grouped into closed loops."""
    E = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    k = np.sort(E, 1)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    be = E[cnt[inv.ravel()] == 1]
    nxt = {int(u): int(v) for u, v in be}
    loops, seen = [], set()
    for u0 in nxt:
        if u0 in seen:
            continue
        lp, u = [], u0
        while u not in seen:
            seen.add(u); lp.append(u); u = nxt[u]
        loops.append(np.array(lp))
    return be, loops


faces_lv = bF[fcls == 3]
faces_rv = bF[fcls == 4]
be_lv, loops_lv = loops_of(faces_lv)
be_rv, loops_rv = loops_of(faces_rv)
print("LV endo loops", [len(l) for l in loops_lv], "RV endo loops", [len(l) for l in loops_rv])


def loop_id(be, loops):
    lid = {}
    for i, lp in enumerate(loops):
        for u in lp:
            lid[int(u)] = i
    return np.array([lid[int(u)] for u in be[:, 0]])


# node sets
lv_nodes = np.unique(faces_lv); rv_nodes = np.unique(faces_rv)
epi_nodes = np.unique(bF[fcls == 2]); base_nodes = np.unique(bF[fcls == 1])
apex_nodes = np.where(ncls == 5)[0]
print("apex nodes", apex_nodes.size)

# fast endocardial layer: elements with any node on LV or RV endocardium -> tag 2
endo_mask = np.zeros(X0.shape[0], bool); endo_mask[lv_nodes] = True; endo_mask[rv_nodes] = True
tag = np.where(endo_mask[T].any(1), 2, 1)
print("fast-layer elements", int((tag == 2).sum()), "fraction %.3f" % (tag == 2).mean())

# stimulus sites (template geometry), Durrer-inspired: three LV endocardial sites plus one RV free-wall site
apex = X0[apex_nodes].mean(0)
base_c = np.mean([X0[l].mean(0) for l in loops_lv], 0)
ax = base_c - apex; Lax = np.linalg.norm(ax); ax /= Lax
rvc = X0[rv_nodes].mean(0) - apex
esep = rvc - (rvc @ ax) * ax; esep /= np.linalg.norm(esep)
eper = np.cross(ax, esep)


def cyl(x):
    r = x - apex
    z = (r @ ax) / Lax
    th = np.degrees(np.arctan2(r @ eper, r @ esep))
    return z, th


def pick_lv(z0, th0):
    z, th = cyl(X0[lv_nodes])
    dth = (th - th0 + 180) % 360 - 180
    score = ((z - z0) * Lax) ** 2 + (np.radians(dth) * 20.0) ** 2
    return int(lv_nodes[np.argmin(score)])


sites = {
    "LV_midseptal_z0.50": pick_lv(0.50, 0.0),
    "LV_ant_paraseptal_z0.65": pick_lv(0.65, 60.0),
    "LV_post_paraseptal_z0.33": pick_lv(0.33, -60.0),
}
z_rv, _ = cyl(X0[rv_nodes])
band = np.abs(z_rv - 0.40) < 0.03
proj = (X0[rv_nodes] - apex) @ esep
sites["RV_freewall_z0.40"] = int(rv_nodes[band][np.argmax(proj[band])])
R_SITE = 3.0  # mm, radius of each stimulus patch on the template
stim_sets = {}
for k, n in sites.items():
    pool = lv_nodes if k.startswith("LV") else rv_nodes
    dd = np.linalg.norm(X0[pool] - X0[n], axis=1)
    stim_sets[k] = pool[dd <= R_SITE]
    print("site", k, "node", n, "patch nodes", stim_sets[k].size, "z/theta", np.round(cyl(X0[n][None])[0], 2),
          np.round(cyl(X0[n][None])[1], 1))

np.savez_compressed(
    os.path.join(out, "template.npz"), X0=X0, T=T.astype(np.int32), bF=bF.astype(np.int32), fcls=fcls,
    ncls=ncls, faces_lv=faces_lv, faces_rv=faces_rv, be_lv=be_lv, be_rv=be_rv,
    beloop_lv=loop_id(be_lv, loops_lv), beloop_rv=loop_id(be_rv, loops_rv),
    nloop_lv=len(loops_lv), nloop_rv=len(loops_rv),
    lv_nodes=lv_nodes, rv_nodes=rv_nodes, epi_nodes=epi_nodes, base_nodes=base_nodes, apex_nodes=apex_nodes,
    tag=tag, **{"stim_" + k: v for k, v in stim_sets.items()}, stim_names=np.array(list(stim_sets)),
    flipped=neg)

# PCs as float32 npy for memory mapping (node-interleaved xyz, verified by the I10 red team)
pc = sio.loadmat(os.path.join(kit, "pc_vol.mat"))["pc_vol"].astype(np.float32)
np.save(os.path.join(out, "pc_vol_f32.npy"), pc)
print("pc", pc.shape)

# binary element file through meshtool (text -> carp_bin)
tb = os.path.join(out, "tmpl_txt")
carpio.write_mesh(tb, X0, T, tag=tag)
r = subprocess.run([os.path.join(carpio.CARP_BIN, "meshtool"), "convert", f"-imsh={tb}", f"-omsh={out}/tmpl",
                    "-ofmt=carp_bin"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
for ext in (".pts", ".elem", ".lon"):
    os.remove(tb + ext)
print("done")
