"""Minimal CARP text mesh / IGB / vtx I/O and structured box mesher (pilot I11/I13, 2026-10-22)."""
import numpy as np, os

def write_mesh(base, pts_um, tets, tags, fib=None):
    """pts in micrometres, tets (M,4) int, tags (M,), fib (M,3) or (M,6)."""
    with open(base + '.pts', 'w') as f:
        f.write(f'{len(pts_um)}\n'); np.savetxt(f, pts_um, fmt='%.3f')
    with open(base + '.elem', 'w') as f:
        f.write(f'{len(tets)}\n')
        a = np.column_stack([tets, tags]).astype(np.int64)
        np.savetxt(f, a, fmt='Tt %d %d %d %d %d')
    if fib is None:
        fib = np.tile([1.0, 0, 0], (len(tets), 1))
    nax = 1 if fib.shape[1] == 3 else 2
    with open(base + '.lon', 'w') as f:
        f.write(f'{nax}\n'); np.savetxt(f, fib, fmt='%.6f')

def read_pts(fn):
    return np.loadtxt(fn, skiprows=1, dtype=np.float64) if os.path.getsize(fn) < 5e7 else _fast(fn, 3)

def _fast(fn, ncol, usecols=None, dtype=np.float64):
    import pandas as pd
    df = pd.read_csv(fn, sep=r'\s+', header=None, skiprows=1, usecols=usecols, engine='c')
    return df.values.astype(dtype)

def read_elem(fn):
    import pandas as pd
    df = pd.read_csv(fn, sep=r'\s+', header=None, skiprows=1, engine='c')
    assert (df[0] == 'Tt').all(), 'only tets supported'
    return df[[1, 2, 3, 4]].values.astype(np.int64), df[5].values.astype(np.int64)

def read_lon(fn):
    import pandas as pd
    with open(fn) as f: nax = int(f.readline().split()[0])
    df = pd.read_csv(fn, sep=r'\s+', header=None, skiprows=1, engine='c')
    return df.values.astype(np.float64), nax

def write_vtx(fn, idx, intra=False):
    with open(fn, 'w') as f:
        f.write(f'{len(idx)}\n{"intra" if intra else "extra"}\n')
        np.savetxt(f, np.asarray(idx, dtype=np.int64), fmt='%d')

def read_igb(fn):
    with open(fn, 'rb') as f:
        hdr = f.read(1024).decode('latin1')
        kv = {}
        for tok in hdr.replace('\r', ' ').replace('\n', ' ').split():
            if ':' in tok:
                k, v = tok.split(':', 1); kv[k] = v
        f.seek(1024)
        typ = kv.get('type', 'float')
        dt = {'float': '<f4', 'double': '<f8', 'int': '<i4'}[typ]
        if kv.get('systeme', 'little_endian') == 'big_endian': dt = dt.replace('<', '>')
        data = np.fromfile(f, dtype=dt)
    x = int(kv['x']); t = int(kv.get('t', 1))
    return data[: x * t].reshape(t, x) if len(data) >= x * t else data.reshape(-1, x)

def box_mesh(L, h, tag=1):
    """Structured tet mesh of box [0,Lx]x[0,Ly]x[0,Lz] (mm) at spacing h (mm); 6 tets per hex (Kuhn).
    Returns pts in mm, tets, tags."""
    n = [int(round(l / h)) for l in L]
    xs = [np.linspace(0, L[i], n[i] + 1) for i in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    def nid(i, j, k): return (i * (n[1] + 1) + j) * (n[2] + 1) + k
    I, J, K = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing='ij')
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    c = [nid(I + a, J + b, K + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]  # c[abd] index = 4a+2b+d
    v0, v7 = c[0], c[7]
    paths = [(4, 6), (4, 5), (2, 6), (2, 3), (1, 5), (1, 3)]
    tets = np.concatenate([np.column_stack([v0, c[p], c[q], v7]) for p, q in paths])
    tets = fix_orientation(pts, tets)
    return pts, tets, np.full(len(tets), tag)

def fix_orientation(pts, tets):
    v = tet_volumes(pts, tets); tets = tets.copy()
    neg = v < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    return tets

def tet_volumes(pts, tets):
    a = pts[tets[:, 1]] - pts[tets[:, 0]]; b = pts[tets[:, 2]] - pts[tets[:, 0]]; c = pts[tets[:, 3]] - pts[tets[:, 0]]
    return np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0

OPENCARP = '/project/home/p201509/opt/opencarp-v20.0/openCARP-v20.0-x86_64_AppImage/squashfs-root/usr/bin/openCARP'

def run_eikonal(meshbase, vtx, regions, simdir, threads=8, tend=400, extra=(), keep=False):
    """Pure eikonal (dream.solve=0) with openCARP. regions: list of (tag, vel_l, vel_t, vel_n) in m/s.
    Returns (activation times (N,), wall seconds)."""
    import subprocess, time, shutil
    os.makedirs(os.path.dirname(os.path.abspath(simdir)), exist_ok=True)
    tags = sorted({r[0] for r in regions})
    a = [OPENCARP, '-meshname', meshbase, '-simID', simdir, '-tend', str(tend), '-dt', '1000',
         '-spacedt', str(tend), '-timedt', str(tend), '-dream.solve', '0', '-dream.fim.max_iter', '500',
         '-num_imp_regions', '1', '-imp_region[0].im', 'tenTusscherPanfilov',
         '-num_phys_regions', '1', '-phys_region[0].ptype', '2', '-phys_region[0].num_IDs', str(len(tags))]
    for i, t in enumerate(tags):
        a += [f'-phys_region[0].ID[{i}]', str(t)]
    a += ['-num_gregions', str(len(regions))]
    for i, (t, vl, vt, vn) in enumerate(regions):
        a += [f'-gregion[{i}].num_IDs', '1', f'-gregion[{i}].ID[0]', str(t),
              f'-gregion[{i}].dream.vel_l', f'{float(vl)*1000:.6g}', f'-gregion[{i}].dream.vel_t', f'{float(vt)*1000:.6g}',
              f'-gregion[{i}].dream.vel_n', f'{float(vn)*1000:.6g}']
    a += ['-num_stim', '1', '-stim[0].crct.type', '0', '-stim[0].elec.vtx_file', vtx,
          '-stim[0].ptcl.start', '0', '-stim[0].ptcl.duration', '1', '-stim[0].pulse.strength', '100']
    a += list(extra)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.time()
    r = subprocess.run(a, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    fn = os.path.join(simdir, 'act.igb')
    if r.returncode != 0 or not os.path.exists(fn):
        raise RuntimeError(f'openCARP failed ({r.returncode}): {r.stdout[-2000:]} {r.stderr[-2000:]}')
    act = read_igb(fn)[-1].astype(np.float64)
    if not keep:
        shutil.rmtree(simdir, ignore_errors=True)
    return act, wall
