"""Packed eikonal timing on ACDC constructions with the pilot calibration design (owner mesh, I11 gates, 2026-10-28).
Design identical to code/pilot/I13/03_design_runs.py: 9 x 9 log grid over v_endo 0.8-3.0 and v_t 0.15-0.5 m/s,
3 theta* draws (1.0-2.4, 0.2-0.42) and 6 held-out points from default_rng(seed), seed 13013 (pilot). (The study plan
widens the v_t prior for the main run; this script times the pilot design, which is the cost unit of arm C.)
All runs of all listed constructions go into one pool with `workers` concurrent single-thread openCARP processes.
Usage: python -I design_runs.py <out_dir> <workers> <cons.h5> [<cons.h5> ...]
Writes <out_dir>/runs__<construction>.h5 with theta, kind (0 grid, 1 theta*, 2 held-out), T (activation, float32),
wall_s per run.
"""
import os, sys, time
import numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import acdc_lib as A

B_LO, B_HI = np.array([0.8, 0.15]), np.array([3.0, 0.5])
G1 = np.exp(np.linspace(np.log(B_LO[0]), np.log(B_HI[0]), 9))
G2 = np.exp(np.linspace(np.log(B_LO[1]), np.log(B_HI[1]), 9))
T_LO, T_HI = np.array([1.0, 0.2]), np.array([2.4, 0.42])


def design(seed):
    rng = np.random.default_rng(seed)
    grid = np.array([[a, b] for a in G1 for b in G2])
    tstar = np.exp(np.log(T_LO) + rng.random((3, 2)) * (np.log(T_HI) - np.log(T_LO)))
    held = np.exp(np.log(B_LO * 1.05) + rng.random((6, 2)) * (np.log(B_HI / 1.05) - np.log(B_LO * 1.05)))
    return grid, tstar, held


if __name__ == '__main__':
    out, workers, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    tmp = os.environ.get('TMPDIR', '/tmp')
    grid, tstar, held = design(13013)
    thetas = np.vstack([grid, tstar, held]); kind = np.array([0] * len(grid) + [1] * 3 + [2] * len(held))
    mesh = {f: A.materialize(f, os.path.join(tmp, 'm_' + os.path.basename(f)[:-3])) for f in files}
    jobs = [(f, i) for f in files for i in range(len(thetas))]

    def one(j):
        f, i = j
        mb, vtx = mesh[f]
        act, wall = A.eikonal(mb, vtx, thetas[i], os.path.join(tmp, f'r_{os.path.basename(f)[:-3]}_{i}'))
        return f, i, act.astype(np.float32), wall
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, jobs))
    for f in files:
        r = sorted([x for x in res if x[0] == f], key=lambda x: x[1])
        T = np.vstack([x[2] for x in r]); walls = np.array([x[3] for x in r])
        with h5py.File(os.path.join(out, 'runs__' + os.path.basename(f)), 'w') as h:
            h['theta'] = thetas; h['kind'] = kind; h['wall_s'] = walls; h['G1'] = G1; h['G2'] = G2
            h.create_dataset('T', data=T, compression='gzip', compression_opts=1)
            h.attrs.update(seed=13013, workers=workers, threads=1, v_fib=A.V_FIB, n_nodes=T.shape[1],
                           unactivated=int(((T > 1e6) | (T < 0)).sum()), pool_wall_s=time.time() - t0,
                           construction=f)
        print(f'{os.path.basename(f)}: {T.shape[1]} nodes, {len(walls)} runs, wall median {np.median(walls):.1f}s '
              f'[{np.percentile(walls, 2.5):.1f}, {np.percentile(walls, 97.5):.1f}] max {walls.max():.1f}, '
              f'max act {T[T < 1e6].max():.1f} ms, unactivated {int(((T > 1e6) | (T < 0)).sum())}', flush=True)
    print(f'pool of {len(jobs)} runs on {workers} workers: {time.time()-t0:.0f}s', flush=True)
