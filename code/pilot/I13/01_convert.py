"""Convert one Malik 2026 heart-in-torso archive (CARP txt) to a compact NPZ and print a tag table.
Usage: python -I 01_convert.py <zip> <outdir>.  Never executes anything from the archive."""
import sys, os, zipfile, numpy as np, pandas as pd, io, time
zp, outdir = sys.argv[1], sys.argv[2]
hid = os.path.basename(zp).split('.')[0]
t0 = time.time()
z = zipfile.ZipFile(zp)
def rd(name, **kw):
    with z.open(f'{hid}/{name}') as f:
        return pd.read_csv(f, sep=r'\s+', header=None, skiprows=1, engine='c', **kw)
pts = rd('torso_tet.pts').values.astype(np.float64) / 1000.0   # um -> mm
el = rd('torso_tet.elem', usecols=[1, 2, 3, 4, 5]).values
tets = el[:, :4].astype(np.int32); tags = el[:, 4].astype(np.int16)
lon = rd('torso_tet.lon').values.astype(np.float32)
fibred = np.linalg.norm(lon[:, :3], axis=1) > 0.5
np.savez(os.path.join(outdir, f'malik{hid}.npz'), pts=pts.astype(np.float32), tets=tets, tags=tags,
         fib=lon[fibred], fibred_idx=np.nonzero(fibred)[0].astype(np.int32))
print(f'heart {hid}: {len(pts)} nodes, {len(tets)} tets, lon cols {lon.shape[1]}, fibred {fibred.sum()}, {time.time()-t0:.0f}s')
for t in np.unique(tags):
    m = tags == t
    c = pts[np.unique(tets[m])]
    print(f'  tag {t:3d}: {m.sum():9d} el, fibred {fibred[m].mean():.2f}, centroid {c.mean(0).round(1)}, extent {np.ptp(c,0).round(0)}')
