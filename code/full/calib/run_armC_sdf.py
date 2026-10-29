"""Arm C rerun with the Amendment 2 B1 smooth-surface fast layer (owner layer-sdf, 2026-10-28).
Usage: python -I -B run_armC_sdf.py --layer-h5 <layer_sdf.h5> --out <dir> --hearts 101 ... --workers 126

Runs run_armC.main() unchanged (imported, not copied: same theta* (seed 20261028 + patient), 9 x 9 log grid,
cubic-spline emulator with kriged refinement, refinement rule, 25 x 25 grid then Nelder-Mead, both losses, rounds 0
to 2, bound rule) with exactly two differences:
  1. The openCARP mesh of every construction (including the reference construction that generates the targets) is
     written with element tags from <layer_sdf.h5>:/cons/<id>/etag_sdf instead of the construction file's `etag`
     (A2). Points, tets, fibres and stimulus nodes are read from the construction file as before.
  2. Constructions: include_primary rows only (RV wall 3 mm, both resolutions), and arm N is skipped (--skip-noise);
     B1 asks for arm C only.
After main() the per-heart HDF5 attributes that run_armC hard-codes as 'etag (A2)' are relabelled 'etag_sdf (B1)'.
"""
import hashlib
import os
import sys
import tempfile

import numpy as np
import h5py
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_armC as RC      # noqa: E402

C = RC.C
LAYER_H5 = None
LABEL = 'etag_sdf (B1)'


def _mat_sdf(name, tmp):
    """As run_armC._mat / acdc_lib.materialize, with the B1 element tag."""
    outdir = os.path.join(tmp, 'm_' + name)
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(LAYER_H5, 'r') as G:
        tag = G[f'cons/{name}/etag_sdf'][:].astype(np.int64)
    with h5py.File(C.cons_path(name), 'r') as f:
        tets = f['tets'][:].astype(np.int64)
        assert len(tag) == len(tets), (name, len(tag), len(tets))
        base = os.path.join(outdir, os.path.basename(C.cons_path(name)).replace('.h5', ''))
        C.A.carpio.write_mesh(base, f['pts'][:] * 1000.0, tets, tag, f['fib'][:].astype(np.float64))
        C.A.carpio.write_vtx(base + '_stim.vtx', f['stim'][:])
    return name, (base, base + '_stim.vtx')


def main():
    global LAYER_H5
    argv = sys.argv[1:]
    i = argv.index('--layer-h5'); LAYER_H5 = os.path.abspath(argv[i + 1]); del argv[i:i + 2]
    M = pd.read_csv(os.path.join(C.MESH_DIR, 'MANIFEST.tsv'), sep='\t')
    M = M[M.include_primary.astype(bool)]
    tmpd = os.environ.get('TMPDIR', tempfile.gettempdir())
    man = os.path.join(tmpd, 'MANIFEST_primary.tsv')
    M.to_csv(man, sep='\t', index=False)
    RC._mat = _mat_sdf                                   # the only change to the arm C logic
    sys.argv = [sys.argv[0]] + argv + ['--manifest', man, '--skip-noise']
    RC.main()
    sha = hashlib.sha256(open(LAYER_H5, 'rb').read()).hexdigest()
    out = argv[argv.index('--out') + 1]
    hearts = []
    for x in argv[argv.index('--hearts') + 1:]:
        if x.startswith('--'):
            break
        hearts.append(int(x))
    for p in hearts:
        with h5py.File(os.path.join(out, 'hearts', f'heart_patient{p:03d}.h5'), 'a') as H:
            H.attrs.update(etag_key=LABEL, layer='sdf_B1', layer_h5=LAYER_H5, layer_h5_sha256=sha,
                           driver='code/full/calib/run_armC_sdf.py')
            for n in H['cons']:
                H['cons'][n].attrs['etag'] = LABEL
    os.remove(man)


if __name__ == '__main__':
    main()
