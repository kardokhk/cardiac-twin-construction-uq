"""Niederer et al. 2011 N-version benchmark (doi:10.1098/rsta.2011.0139) with openCARP v20.0 monodomain.
Slab 20 x 7 x 3 mm, fibres along x, ten Tusscher-Panfilov 2006, sigma_il 0.17 sigma_el 0.62 sigma_it 0.019
sigma_et 0.24 S/m (monodomain harmonic mean), chi 1400 /cm, Cm 1 uF/cm2, stimulus 1.5 mm cube at the origin
corner, 2 ms, 35.71 uA/cm2 (= 50,000 uA/cm3 / chi).  Reports activation time (0 mV crossing) at the far corner
(20,7,3) mm and the maximum, per resolution h and time step dt, plus wall time.
Usage: python -I stageA_niederer.py <workdir> <out.json> <cases "h:dt_us:np,...">
"""
import sys, os, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carpio
work, out, cases = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(work, exist_ok=True)
mpirun = os.path.join(os.path.dirname(carpio.OPENCARP), 'mpirun')
cases = [tuple(float(x) for x in c.split(':')) for c in cases.split(',')]
meshes = {}
for h, _, _ in cases:
    if h not in meshes:
        pts, tets, tags = carpio.box_mesh((20, 7, 3), h)
        base = os.path.join(work, f'nied_h{h}')
        carpio.write_mesh(base, pts * 1000, tets, tags)
        far = int(np.argmin(np.linalg.norm(pts - np.array([20, 7, 3]), axis=1)))
        meshes[h] = (base, far, len(pts))


def run(c):
    h, dt, npr = c
    base, far, n = meshes[h]
    sim = os.path.join(work, f'run_h{h}_dt{dt}')
    tend = 160 if h >= 0.5 else 90
    sb = os.path.join(work, 'stimblock.par')
    open(sb, 'w').write('stim[0].elec.p0 = -1 -1 -1\nstim[0].elec.p1 = 1500.5 1500.5 1500.5\n')
    par = ['+F', sb, '-meshname', base, '-simID', sim, '-tend', str(tend), '-dt', str(dt), '-spacedt', '1', '-timedt', '10',
           '-bidomain', '0', '-num_imp_regions', '1', '-imp_region[0].im', 'tenTusscherPanfilov',
           '-num_gregions', '1', '-gregion[0].num_IDs', '1', '-gregion[0].ID[0]', '1',
           '-gregion[0].g_il', '0.17', '-gregion[0].g_el', '0.62', '-gregion[0].g_it', '0.019', '-gregion[0].g_et', '0.24',
           '-gregion[0].g_in', '0.019', '-gregion[0].g_en', '0.24',
           '-num_stim', '1', '-stim[0].crct.type', '0', '-stim[0].elec.geom_type', '2',
           '-stim[0].ptcl.start', '0', '-stim[0].ptcl.duration', '2', '-stim[0].pulse.strength', '35.71',
           '-num_LATs', '1', '-lats[0].measurand', '0', '-lats[0].threshold', '0', '-lats[0].method', '1', '-lats[0].all', '0']
    t0 = time.time()
    with open(sim + '.log', 'w') as lf:
        r = subprocess.run([mpirun, '-launcher', 'fork', '-np', str(int(npr)), carpio.OPENCARP] + par, stdout=lf, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS='1'))
    wall = time.time() - t0
    try:
        lat = np.loadtxt(os.path.join(sim, 'init_acts_vm_act-thresh.dat'))
        d = dict(h=h, dt_us=dt, ranks=int(npr), nodes=n, far_corner_ms=float(lat[far]), max_ms=float(lat.max()),
                 unactivated=int((lat < 0).sum()), wall_s=wall, rc=r.returncode)
    except Exception as e:
        d = dict(h=h, dt_us=dt, error=str(e), rc=r.returncode, tail=open(sim + '.log').read()[-1500:])
    print(d, flush=True)
    return d


with ThreadPoolExecutor(len(cases)) as ex:
    res = list(ex.map(run, cases))
json.dump(res, open(out, 'w'), indent=1)
