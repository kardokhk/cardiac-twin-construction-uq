"""I09 pilot: openCARP v20.0 eikonal I/O helpers (written 2026-10-22).

Mesh units: openCARP .pts are in micrometres. Velocities in the parameter file are in mm/s (openCARP.prm,
DREAM_CV). Activation times are read back from the IGB file `act.igb` (last cycle).
"""
import os
import subprocess
import numpy as np

CARP_BIN = ("/project/home/p201509/opt/opencarp-v20.0/openCARP-v20.0-x86_64_AppImage/"
            "squashfs-root/usr/bin")
OPENCARP = os.path.join(CARP_BIN, "openCARP")
MPIRUN = os.path.join(CARP_BIN, "mpirun")


def write_mesh(base, X_mm, T, fib=None, she=None, tag=1, binary=False):
    """Write openCARP text mesh. X_mm in mm (converted to um). T (ne,4) zero-based. fib/she (ne,3)."""
    X = np.asarray(X_mm, float) * 1000.0
    with open(base + ".pts", "w") as f:
        f.write(f"{X.shape[0]}\n")
        np.savetxt(f, X, fmt="%.3f")
    write_elem(base + ".elem", T, tag)
    write_lon(base + ".lon", T.shape[0], fib, she)


def write_elem(path, T, tag=1):
    T = np.asarray(T, np.int64)
    tags = np.broadcast_to(np.asarray(tag, np.int64), (T.shape[0],))
    with open(path, "w") as f:
        f.write(f"{T.shape[0]}\n")
        arr = np.column_stack([T, tags])
        np.savetxt(f, arr, fmt="Tt %d %d %d %d %d")


def write_lon(path, ne, fib=None, she=None):
    if fib is None:
        fib = np.tile([1.0, 0.0, 0.0], (ne, 1))
    with open(path, "w") as f:
        if she is None:
            f.write("1\n")
            np.savetxt(f, fib, fmt="%.6f")
        else:
            f.write("2\n")
            np.savetxt(f, np.column_stack([fib, she]), fmt="%.6f")


def write_vtx(path, nodes):
    nodes = np.atleast_1d(np.asarray(nodes, np.int64))
    with open(path, "w") as f:
        f.write(f"{nodes.size}\nextra\n")
        np.savetxt(f, nodes, fmt="%d")


def write_par(path, meshname, simid, stim_vtx, vel=(600.0, 400.0, 200.0), tag=1, tend=1.0,
              extra=""):
    """Parameter file for a pure eikonal solve (dream.solve = 0). All stimulus files fire at t = 0."""
    lines = [
        f"meshname = {meshname}",
        f"simID = {simid}",
        "experiment = 0",
        f"tend = {tend}",
        "dt = 1.0",
        "spacedt = 1.0",
        "timedt = 1.0",
        "dream.solve = 0",
        "dream.fim.tol = 1e-4",
        "dream.fim.max_iter = 5000",
        "num_phys_regions = 1",
        "phys_region[0].ptype = 2",
        "phys_region[0].num_IDs = 1",
        f"phys_region[0].ID[0] = {tag}",
        "num_imp_regions = 1",
        "imp_region[0].im = MitchellSchaeffer",
        "imp_region[0].num_IDs = 1",
        f"imp_region[0].ID[0] = {tag}",
        "num_gregions = 1",
        "gregion[0].num_IDs = 1",
        f"gregion[0].ID[0] = {tag}",
        f"gregion[0].dream.vel_l = {vel[0]}",
        f"gregion[0].dream.vel_t = {vel[1]}",
        f"gregion[0].dream.vel_n = {vel[2]}",
        f"num_stim = {len(stim_vtx)}",
    ]
    for i, v in enumerate(stim_vtx):
        lines += [f"stim[{i}].crct.type = 0", f"stim[{i}].elec.vtx_file = {v}",
                  f"stim[{i}].ptcl.start = 0.0", f"stim[{i}].ptcl.npls = 1",
                  f"stim[{i}].ptcl.duration = 1.0", f"stim[{i}].pulse.strength = 100.0"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n" + extra)


def run(parfile, cwd, np_mpi=1, omp=1, log="carp.log", timeout=3600):
    env = dict(os.environ, OMP_NUM_THREADS=str(omp))
    cmd = [OPENCARP, "+F", parfile]
    if np_mpi > 1:
        cmd = [MPIRUN, "-n", str(np_mpi)] + cmd
    with open(os.path.join(cwd, log), "w") as fh:
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return r.returncode


def read_igb(path):
    """Return array (nt, nx) from an openCARP IGB file (1024-byte ASCII header)."""
    with open(path, "rb") as f:
        hdr = f.read(1024).decode("latin1")
        data = f.read()
    kv = dict(tok.split(":", 1) for tok in hdr.replace("\r", " ").replace("\n", " ").split() if ":" in tok)
    nx = int(kv["x"])
    typ = kv.get("type", "float")
    dt = {"float": "<f4", "double": "<f8"}[typ]
    if kv.get("systeme", "little_endian").startswith("big"):
        dt = dt.replace("<", ">")
    a = np.frombuffer(data, dt)
    nt = a.size // nx
    return a[: nt * nx].reshape(nt, nx)
