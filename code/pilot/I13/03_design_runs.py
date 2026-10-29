"""Eikonal design runs for one construction (pilot I13; reused by I11).
Parameters: theta = (v_endo, v_t) in m/s. Region tag 1 = myocardium (v_l = V_FIB fixed, v_t = v_n = theta[1]),
tag 2 = fast endocardial layer (isotropic, theta[0]). Root nodes (stim vtx) fixed.
Runs: 9 x 9 log grid (emulator), 3 theta* targets, 6 held-out points (emulator error).
Usage: python -I 03_design_runs.py <meshbase> <stim.vtx> <out.h5> <seed> [--workers W --threads T --limit N --tmp DIR]
"""
import sys, os, time, argparse
import numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'I11'))
import carpio

V_FIB = 0.6
B_LO, B_HI = np.array([0.8, 0.15]), np.array([3.0, 0.5])
G1 = np.exp(np.linspace(np.log(B_LO[0]), np.log(B_HI[0]), 9))
G2 = np.exp(np.linspace(np.log(B_LO[1]), np.log(B_HI[1]), 9))
T_LO, T_HI = np.array([1.0, 0.2]), np.array([2.4, 0.42])     # range for theta* draws


def design(seed):
    rng = np.random.default_rng(seed)
    grid = np.array([[a, b] for a in G1 for b in G2])
    tstar = np.exp(np.log(T_LO) + rng.random((3, 2)) * (np.log(T_HI) - np.log(T_LO)))
    held = np.exp(np.log(B_LO * 1.05) + rng.random((6, 2)) * (np.log(B_HI / 1.05) - np.log(B_LO * 1.05)))
    return grid, tstar, held


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('meshbase'); ap.add_argument('vtx'); ap.add_argument('out'); ap.add_argument('seed', type=int)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--tmp', default=os.environ.get('TMPDIR', '/tmp'))
    ap.add_argument('--tend', type=float, default=400)
    a = ap.parse_args()
    grid, tstar, held = design(a.seed)
    thetas = np.vstack([grid, tstar, held]); kind = np.array([0] * len(grid) + [1] * 3 + [2] * len(held))
    if a.limit:
        sel = np.linspace(0, len(thetas) - 1, a.limit).astype(int); thetas, kind = thetas[sel], kind[sel]
    tag = os.path.basename(a.out).replace('.h5', '')

    def one(i):
        th = thetas[i]
        act, wall = carpio.run_eikonal(a.meshbase, a.vtx, [(1, V_FIB, th[1], th[1]), (2, th[0], th[0], th[0])],
                                       os.path.join(a.tmp, f'{tag}_{i}'), threads=a.threads, tend=a.tend)
        print(f'   run {i} theta {np.round(th,3)} wall {wall:.1f}s max act {act.max():.1f}', flush=True)
        return i, act.astype(np.float32), wall
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one, range(len(thetas))))
    T = np.vstack([o[1] for o in outs]); walls = np.array([o[2] for o in outs])
    unact = int((T > 1e6).sum() + (T < 0).sum())
    with h5py.File(a.out, 'w') as f:
        f['theta'] = thetas; f['kind'] = kind; f['wall_s'] = walls
        f.create_dataset('T', data=T, compression='gzip', compression_opts=1)
        f['G1'] = G1; f['G2'] = G2
        f.attrs.update(seed=a.seed, workers=a.workers, threads=a.threads, v_fib=V_FIB, unactivated=unact,
                       total_wall_s=time.time() - t0)
    print(f'{tag}: {len(thetas)} runs, median wall {np.median(walls):.1f}s (max {walls.max():.1f}), '
          f'total {time.time()-t0:.0f}s, max act {T[T<1e6].max():.1f} ms, unactivated {unact}', flush=True)
