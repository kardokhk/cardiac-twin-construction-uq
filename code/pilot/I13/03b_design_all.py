"""Run the eikonal design (03_design_runs.design) for many constructions through one global pool of single-thread
openCARP processes (openCARP v20.0 eikonal showed no OpenMP speed-up: 528 s with 8 and 527 s with 32 threads).
Input: a text file with lines '<meshbase> <stim.vtx> <out.h5> <seed>'.  Each construction's runs are written
to its own HDF5 as soon as they are complete.
Usage: python -I 03b_design_all.py <list.txt> --workers 120 [--tmp DIR]
"""
import sys, os, time, argparse, threading, importlib.util
import numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'I11'))
import carpio
spec = importlib.util.spec_from_file_location('dr', os.path.join(here, '03_design_runs.py'))
dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)

ap = argparse.ArgumentParser()
ap.add_argument('lst'); ap.add_argument('--workers', type=int, default=120)
ap.add_argument('--tmp', default=os.environ.get('TMPDIR', '/tmp')); ap.add_argument('--tend', type=float, default=400)
a = ap.parse_args()
jobs = []
cons = []
for line in open(a.lst):
    if not line.strip():
        continue
    mb, vtx, out, seed = line.split()
    if os.path.exists(out):
        continue
    grid, tstar, held = dr.design(int(seed))
    th = np.vstack([grid, tstar, held]); kind = np.array([0] * len(grid) + [1] * 3 + [2] * len(held))
    ci = len(cons)
    cons.append(dict(mb=mb, vtx=vtx, out=out, seed=int(seed), th=th, kind=kind, T=[None] * len(th), wall=[0.0] * len(th),
                     left=len(th)))
    jobs += [(ci, i) for i in range(len(th))]
# interleave constructions so that the slowest (low-velocity) runs are spread out
lock = threading.Lock()
t0 = time.time()
print(f'{len(cons)} constructions, {len(jobs)} runs, {a.workers} workers', flush=True)


def one(job):
    ci, i = job
    c = cons[ci]; th = c['th'][i]
    tag = os.path.basename(c['out']).replace('.h5', '')
    try:
        act, wall = carpio.run_eikonal(c['mb'], c['vtx'], [(1, dr.V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                       os.path.join(a.tmp, f'{tag}_{i}'), threads=1, tend=a.tend)
    except Exception as e:
        print('FAILED', tag, i, str(e)[:300], flush=True)
        act, wall = np.full(1, -1.0), -1.0
    with lock:
        c['T'][i] = act.astype(np.float32); c['wall'][i] = wall; c['left'] -= 1
        if c['left'] == 0:
            n = max(len(x) for x in c['T'])
            T = np.vstack([x if len(x) == n else np.full(n, -1, np.float32) for x in c['T']])
            unact = int((T > 1e6).sum() + (T < 0).sum())
            with h5py.File(c['out'], 'w') as f:
                f['theta'] = c['th']; f['kind'] = c['kind']; f['wall_s'] = np.array(c['wall'])
                f.create_dataset('T', data=T, compression='gzip', compression_opts=1)
                f['G1'] = dr.G1; f['G2'] = dr.G2
                f.attrs.update(seed=c['seed'], v_fib=dr.V_FIB, unactivated=unact, threads=1)
            w = np.array(c['wall'])
            print(f'{tag}: done, median wall {np.median(w):.1f}s max {w.max():.1f}s, max act {T.max():.1f} ms, '
                  f'unactivated {unact}, elapsed {time.time()-t0:.0f}s', flush=True)
            c['T'] = None


order = sorted(jobs, key=lambda j: (j[1], j[0]))
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(one, order))
print(f'all done {time.time()-t0:.0f}s', flush=True)
