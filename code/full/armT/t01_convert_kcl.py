"""Convert one KCL whole-torso model (Zenodo 7253863, openCARP text in a zip) to the Malik-style NPZ of the I13 pilot
(code/pilot/I13/01_convert.py): pts in mm, tets, tags, fibres of fibred elements.

Tag remapping (KCL info PDF): every KCL tag t -> 100 + t, then ventricles to the Malik convention used by the pilot code:
35 LV wall -> 23, 34 RV wall -> 22, 25 LV blood pool -> 21, 24 RV blood pool -> 20.
Orientation: the pilot code assumes LPS (x patient left, y posterior, z superior). The KCL axes are not documented, so
they are found from anatomy and snapped to signed world axes: superior = lungs (108) minus liver (105) centroid; left =
ventricular centroid minus liver centroid, superior part removed; posterior = superior x left (proper rotation,
det +1, no mirroring). Checked afterwards: ventricles anterior to the torso centroid, LV blood left of RV blood.
Spine proxy for electrode rule v2 (Malik tag 4 does not exist here): bone (103) nodes within 35 mm of the torso's
mid-sagittal x and posterior to the ventricular centroid. Stored as spine_pts.
Usage: python -I t01_convert_kcl.py <zip> <out.npz> <diag.json>      (reads the archive only; executes nothing in it)
"""
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

zp, out, diag = sys.argv[1:4]
t0 = time.time()
z = zipfile.ZipFile(zp)
stem = os.path.basename(zp).replace('.zip', '')


def rd(ext, **kw):
    with z.open(f'{stem}/{stem}.{ext}') as f:
        return pd.read_csv(f, sep=r'\s+', header=None, skiprows=1, engine='c', **kw)


pts = rd('pts').values.astype(np.float64) / 1000.0              # um -> mm
el = rd('elem', usecols=[1, 2, 3, 4, 5]).values
tets = el[:, :4].astype(np.int32); ktag = el[:, 4].astype(np.int16)
lon = rd('lon').values.astype(np.float32)
assert len(lon) == len(tets) and len(np.unique(tets)) == len(pts), (len(lon), len(tets))
fibred = np.linalg.norm(lon[:, :3], axis=1) > 0.5


def cen(tags):
    m = np.isin(ktag, tags)
    return pts[np.unique(tets[m])].mean(0), int(m.sum())


C = {name: cen(t) for name, t in dict(lv_wall=[35], rv_wall=[34], lv_blood=[25], rv_blood=[24], liver=[5], lungs=[8],
                                      bones=[3], heart=[34, 35], body=list(np.unique(ktag))).items()}
c = {k: v[0] for k, v in C.items()}


def snap(v, exclude=()):
    v = np.array(v, float)
    for e in exclude:
        v[e] = 0
    i = int(np.argmax(np.abs(v)))
    e = np.zeros(3); e[i] = np.sign(v[i])
    return e, i, float(abs(v[i]) / np.linalg.norm(v))


sup, i_s, f_s = snap(c['lungs'] - c['liver'])
left, i_l, f_l = snap(c['heart'] - c['liver'], exclude=(i_s,))
post = np.cross(sup, left)
M = np.vstack([left, post, sup])                                  # world -> LPS
assert abs(np.linalg.det(M) - 1) < 1e-9
P = pts @ M.T
cL = {k: M @ v for k, v in c.items()}
checks = dict(heart_anterior_of_body_mm=float(cL['body'][1] - cL['heart'][1]),
              lvblood_left_of_rvblood_mm=float(cL['lv_blood'][0] - cL['rv_blood'][0]),
              heart_left_of_body_mm=float(cL['heart'][0] - cL['body'][0]),
              lungs_above_liver_mm=float(cL['lungs'][2] - cL['liver'][2]))
ok = checks['heart_anterior_of_body_mm'] > 0 and checks['lvblood_left_of_rvblood_mm'] > 0 and checks['lungs_above_liver_mm'] > 0

tags = (ktag + 100).astype(np.int16)
for k, v in {135: 23, 134: 22, 125: 21, 124: 20}.items():
    tags[tags == k] = v
# spine proxy
bn = np.unique(tets[ktag == 3]); B = P[bn]
xmid = 0.5 * (P[:, 0].min() + P[:, 0].max())
spine = B[(np.abs(B[:, 0] - xmid) < 35) & (B[:, 1] > cL['heart'][1] + 20)]
np.savez(out, pts=P.astype(np.float32), tets=tets, tags=tags, fib=(lon[fibred][:, :3] @ M.T).astype(np.float32),
         fibred_idx=np.nonzero(fibred)[0].astype(np.int32), spine_pts=spine.astype(np.float32), M_world_to_lps=M)
d = dict(zip=os.path.basename(zp), n_nodes=int(len(pts)), n_tets=int(len(tets)), n_fibred=int(fibred.sum()),
         fibred_tags=sorted(int(t) for t in np.unique(ktag[fibred])), tag_counts={int(t): int((ktag == t).sum()) for t in np.unique(ktag)},
         centroids_world_mm={k: np.round(v[0], 1).tolist() for k, v in C.items()},
         axis_purity=dict(superior=f_s, left=f_l), M_world_to_lps=M.tolist(), checks_mm=checks, orientation_ok=bool(ok),
         n_spine_proxy_nodes=int(len(spine)), xmid_mm=float(xmid), seconds=round(time.time() - t0, 1))
json.dump(d, open(diag, 'w'), indent=1)
print(json.dumps(d), flush=True)
if not ok:
    sys.exit(f'orientation check failed for {stem}')
