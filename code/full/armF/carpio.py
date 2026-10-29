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


# ---------------------------------------------------------------------------------------------------------------
# Added by owner mesh, I11 gates, 2026-10-28. Above this line: verbatim copy of code/pilot/I11/carpio.py
# (SHA-256 39edda03...8225). run_eikonal is unchanged; run_eikonal0 passes '-renumbering 0' explicitly (bug #553).

def run_eikonal0(meshbase, vtx, regions, simdir, threads=1, tend=400, keep=False):
    """Pilot run_eikonal with renumbering forced to 0 (openCARP v20.0 bug #553)."""
    return run_eikonal(meshbase, vtx, regions, simdir, threads=threads, tend=tend, extra=['-renumbering', '0'],
                       keep=keep)


def run_reaction_eikonal(meshbase, vtx, regions, simdir, mode=1, np_mpi=1, tend=200.0, dt_us=25.0, spacedt=0.5,
                         im='tenTusscherPanfilov', cond=(0.174, 0.625, 0.019, 0.236, 0.019, 0.236), keep=True,
                         timeout=36000, layout='default0'):
    """openCARP v20.0 reaction-eikonal (dream.solve = 1 RE+ with diffusion, or 2 RE-), monodomain, full ionic model.
    regions: list of (tag, vel_l, vel_t, vel_n) in m/s; each region gets the same conductivities `cond`
    (g_il, g_el, g_it, g_et, g_in, g_en; S/m; openCARP defaults rounded). Writes vm.igb every `spacedt` ms.
    Returns (vm (T, N) float32, act (N,), wall seconds, log text).
    layout (added 2026-10-28, follow-up round): 'legacy' writes regions as gregion[0..n-1] (openCARP v20.0 then warns
    'check_unassigned_tags: ... carry tags {first tag} not assigned to any gregion_i region; these elements default
    to region 0'); 'default0' writes an ID-less default gregion[0] with the parameters of regions[0] and the regions
    as gregion[1..n]; 'default0dup' is 'default0' with regions[0]'s tag also listed in gregion[0]. Which layout is
    warning-free and bit-identical to 'legacy' is recorded in results/2026-10-28_G2_meshing/README.md (Correction
    2026-10-28): on patient101 expert 1.5 mm, 1 thread, 20 ms, 'default0' and 'default0dup' gave a Vm byte-identical
    to 'legacy' (SHA-256 f919b380...) without the warning; 'default0' is the default from 2026-10-28."""
    import subprocess, time, shutil
    assert layout in ('legacy', 'default0', 'default0dup'), layout
    os.makedirs(simdir, exist_ok=True)
    tags = sorted({r[0] for r in regions})
    a = ['-meshname', meshbase, '-simID', simdir, '-tend', str(tend), '-dt', str(dt_us),
         '-spacedt', str(spacedt), '-timedt', str(max(spacedt, 10.0)), '-bidomain', '0', '-renumbering', '0',
         '-dream.solve', str(mode), '-dream.fim.max_iter', '500',
         '-num_imp_regions', '1', '-imp_region[0].im', im, '-imp_region[0].num_IDs', str(len(tags)),
         '-num_phys_regions', '1', '-phys_region[0].ptype', '2', '-phys_region[0].num_IDs', str(len(tags))]
    for i, t in enumerate(tags):
        a += [f'-imp_region[0].ID[{i}]', str(t), f'-phys_region[0].ID[{i}]', str(t)]
    gil, gel, git, get, gin, gen = cond
    regs = list(regions)
    if layout != 'legacy':
        regs = [regions[0]] + regs
    a += ['-num_gregions', str(len(regs))]
    for i, (t, vl, vt, vn) in enumerate(regs):
        if layout == 'default0' and i == 0:
            a += ['-gregion[0].num_IDs', '0']
        else:
            a += [f'-gregion[{i}].num_IDs', '1', f'-gregion[{i}].ID[0]', str(t)]
        a += [f'-gregion[{i}].g_il', str(gil), f'-gregion[{i}].g_el', str(gel), f'-gregion[{i}].g_it', str(git),
              f'-gregion[{i}].g_et', str(get), f'-gregion[{i}].g_in', str(gin), f'-gregion[{i}].g_en', str(gen),
              f'-gregion[{i}].dream.vel_l', f'{float(vl)*1000:.6g}', f'-gregion[{i}].dream.vel_t', f'{float(vt)*1000:.6g}',
              f'-gregion[{i}].dream.vel_n', f'{float(vn)*1000:.6g}']
    a += ['-num_stim', '1', '-stim[0].crct.type', '0', '-stim[0].elec.vtx_file', vtx,
          '-stim[0].ptcl.start', '0', '-stim[0].ptcl.duration', '1', '-stim[0].pulse.strength', '100']
    # openCARP v20.0: "DREAM/Eikonal physics currently do not support MPI parallelization. Use openMP instead."
    # so np_mpi is used as the OpenMP thread count of a single rank.
    cmd = [OPENCARP] + a
    env = dict(os.environ, OMP_NUM_THREADS=str(np_mpi))
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    full = (r.stdout or '') + (r.stderr or '')
    warn = ''.join('WARNING-LINE ' + l + '\n' for l in full.splitlines() if 'unassigned' in l or 'IDs provided' in l)
    log = warn + (r.stdout or '')[-4000:] + (r.stderr or '')[-4000:]
    fv = os.path.join(simdir, 'vm.igb'); fa = os.path.join(simdir, 'act.igb')
    if r.returncode != 0 or not os.path.exists(fv):
        raise RuntimeError(f'openCARP RE failed ({r.returncode}): {log}')
    vm = read_igb(fv).astype(np.float32)
    act = read_igb(fa)[-1].astype(np.float64) if os.path.exists(fa) else None
    if not keep:
        shutil.rmtree(simdir, ignore_errors=True)
    return vm, act, wall, log
