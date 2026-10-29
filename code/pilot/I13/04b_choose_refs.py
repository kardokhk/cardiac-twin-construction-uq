"""Choose reference torsos among the Malik hearts by torso volume: R1 = median, R2 = 75th percentile.
Usage: python -I 04b_choose_refs.py <h5dir>   -> prints 'R1 R2' heart ids and writes refs.json"""
import sys, os, glob, json, h5py, numpy as np
d = sys.argv[1]
v = {os.path.basename(f)[5:7]: h5py.File(f, 'r').attrs['torso_volume_mL'] for f in sorted(glob.glob(os.path.join(d, 'heart??.h5')))}
ids = sorted(v, key=v.get); n = len(ids)
r1 = ids[n // 2]; r2 = ids[int(round(0.75 * (n - 1)))]
json.dump(dict(torso_volume_mL=v, R1=r1, R2=r2), open(os.path.join(d, 'refs.json'), 'w'), indent=1)
print(r1, r2)
