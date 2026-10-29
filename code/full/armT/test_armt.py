"""Known-answer tests for armt_lib (owner armT, 2026-10-28). Usage: python -I test_armt.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L

rng = np.random.default_rng(1)
# 1. closest point on triangle vs dense barycentric sampling
a, b, c = rng.normal(size=(3, 500, 3)); p = rng.normal(size=(500, 3)) * 2
cp = L.closest_on_triangles(p, a, b, c)
u = np.linspace(0, 1, 201); U, V = np.meshgrid(u, u); m = U + V <= 1; U, V = U[m], V[m]
err = []
for i in range(len(p)):
    S = a[i] + np.outer(U, b[i] - a[i]) + np.outer(V, c[i] - a[i])
    dmin = np.linalg.norm(S - p[i], axis=1).min()
    err.append(np.linalg.norm(cp[i] - p[i]) - dmin)
err = np.array(err)
assert err.max() < 1e-9 and err.min() > -0.02, (err.min(), err.max())
# 2. distance to a plane made of two triangles
pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0.]]); tris = np.array([[0, 1, 2], [1, 3, 2]])
q = np.array([[2, 3, 1.5], [8, 8, -0.7], [12, 5, 0.]])
d = L.dist_to_surface(q, pts, tris, k=2)
assert np.allclose(d, [1.5, 0.7, 2.0]), d
# 3. theta* inside the A4 box, one per stratum, reproducible
for h in ['01', 'K3']:
    t = L.theta_star(h)
    assert (t >= L.T_LO).all() and (t <= L.T_HI).all()
    st = np.floor((np.log(t) - np.log(L.T_LO)) / (np.log(L.T_HI) - np.log(L.T_LO)) * 3)
    assert sorted(st[:, 0]) == [0, 1, 2] and sorted(st[:, 1]) == [0, 1, 2]
    assert np.array_equal(t, L.theta_star(h))
assert L.pid_of('K3') == 203 and L.pid_of('09') == 9
# 4. bound hits
assert L.bound_hit([0.5, 0.3]) and L.bound_hit([1.0, 0.8]) and not L.bound_hit([1.0, 0.3])
print('test_armt: all pass; closest-point max excess', float(err.max()), 'theta* 01', L.theta_star('01').round(4).tolist())
