"""I09 pilot step 3: weight table. Rows 0-199: N(0, I_100) draws (numpy default_rng, seed 20261022);
rows 200-299: deposited KIT instance weights (instances_weights.csv.gz, scrambled Halton U(-3, 3)).
Writes <out>/weights.csv (no header) and <out>/cases.txt (case_id arm row).
Usage: python -I 03_make_cases.py <instances_weights.csv.gz> <out_dir>"""
import gzip, os, sys
import numpy as np
src, out = sys.argv[1:3]
rng = np.random.default_rng(20261022)
Wg = rng.standard_normal((200, 100))
Wu = np.loadtxt(gzip.open(src), delimiter=",")
assert Wu.shape == (100, 100)
W = np.vstack([Wg, Wu])
np.savetxt(os.path.join(out, "weights.csv"), W, delimiter=",", fmt="%.10f")
with open(os.path.join(out, "cases.txt"), "w") as f:
    for i in range(200):
        f.write(f"G{i:03d} gauss {i}\n")
    for i in range(100):
        f.write(f"U{i+1:03d} uniform {200+i}\n")
print(W.shape, "Gaussian |w|^2 range", (Wg**2).sum(1).min().round(1), (Wg**2).sum(1).max().round(1),
      "uniform", (Wu**2).sum(1).min().round(1), (Wu**2).sum(1).max().round(1))
