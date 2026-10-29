"""Fig. 2 (v03): per-heart construction spread (A, B) and heart x network v_t recovery ratio (C).

Inputs (read only):
  results/2026-10-28_analysis/tables/per_heart_cv_primary.csv     (voxel-face layer, corr loss)
  results/2026-10-28_analysis/tables/B1_per_heart_cv_sdf_corr.csv (smooth-surface layer, corr loss)
  results/2026-10-28_analysis/results.json  E1_primary, B1_smooth_layer_SENSITIVITY.E1.corr
  results/2026-10-29_v03_analyses/heatmap_ratio_v_t_heart_x_network.csv  (item 2, independently reproduced)
No new statistics: medians, CIs and per heart-network ratios are copied from archived results.
Run: FIG_QA_DIR=<scratch> /project/home/p201509/envs/cvri-sim/bin/python -I code/figures/fig2_v03.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script directory
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figcommon as fc  # noqa: E402

fam = fc.setup(scripts_floor=True)  # CMPB: sub- and superscripts >= 6 pt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

V03 = fc.RES / "2026-10-29_v03_analyses"
T = fc.ANALYSIS / "tables"
a2 = pd.read_csv(T / "per_heart_cv_primary.csv")
b1 = pd.read_csv(T / "B1_per_heart_cv_sdf_corr.csv")
m = a2.merge(b1, on=["patient", "group"], suffixes=("_A2", "_B1"))
assert len(m) == 50
R = fc.results_json()
summ = {"A2": R["E1_primary"], "B1": R["B1_smooth_layer_SENSITIVITY"]["E1"]["corr"]}
for lay, df in (("A2", a2), ("B1", b1)):
    for p in ("v_endo", "v_t"):
        assert abs(np.median(df[f"cv_{p}"]) - summ[lay][p]["cv_median"]) < 1e-12
LAYER_TICK = {"A2": "Voxel-face layer\n(primary)", "B1": "Smooth-surface layer\n(sensitivity)"}

hm = pd.read_csv(V03 / "heatmap_ratio_v_t_heart_x_network.csv")
assert len(hm) == 50
COLS = [  # column, row label
    ("nnunet_task027_acdc", "nnU-Net Task027"),
    ("nnunet_task114_mnms", "nnU-Net Task114"),
    ("nnunet_ukb_sax", "nnU-Net UKB SAX"),
    ("cinema_acdc_sax", "CineMA"),
    ("expert_1.0", "Expert labels, 1.0 mm"),
    ("ukbb_cardiac_fcn_sa", "ukbb_cardiac (sensitivity)"),
]
hm["gorder"] = hm.group.map({g: i for i, g in enumerate(fc.GROUPS)})
hm = hm.sort_values(["gorder", "patient"]).reset_index(drop=True)
M = hm[[c for c, _ in COLS]].to_numpy(float).T  # rows = sources, columns = hearts

rng = np.random.default_rng(fc.SEED)
jit = rng.uniform(-0.16, 0.16, size=len(m))
X = {"A2": 0.0, "B1": 1.0}

fig = plt.figure(figsize=(fc.figstyle.mm(fc.WIDTH_CMPB_MM), fc.figstyle.mm(fc.cmpb_height(132))), layout="constrained")
sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[72, 60])
axes = sf_top.subplots(1, 2)
for ax, p in zip(axes, ("v_endo", "v_t")):
    for i, r in m.iterrows():
        ax.plot([X["A2"] + jit[i], X["B1"] + jit[i]], [100 * r[f"cv_{p}_A2"], 100 * r[f"cv_{p}_B1"]],
                color="0.82", lw=0.5, zorder=1)
    for lay in ("A2", "B1"):
        for g in fc.GROUPS:
            s = m[m.group == g]
            col, mk = fc.GROUP_STYLE[g]
            ax.scatter(X[lay] + jit[s.index], 100 * s[f"cv_{p}_{lay}"], s=9, marker=mk,
                       facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)  # >= 0.5 pt (V3)
        e = summ[lay][p]
        med, lo, hi = 100 * e["cv_median"], 100 * e["ci"][0], 100 * e["ci"][1]
        xm = X[lay] + 0.32
        ax.errorbar(xm, med, yerr=[[med - lo], [hi - med]], fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=2.0, capthick=0.8, zorder=4)
        ax.plot([xm - 0.07, xm + 0.07], [med, med], color="black", lw=1.5, zorder=5)
        ax.text(xm + 0.09, med, f"{med:.1f}%", va="center", ha="left")
    ax.set_xticks([0, 1], [LAYER_TICK["A2"], LAYER_TICK["B1"]])
    ax.set_xlim(-0.35, 1.7)
    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Fast endocardial layer (lead-correlation fit)")
    ax.set_ylabel(f"Spread of {fc.PARAM_LABEL[p]} across constructions, CV (%)")
handles = [Line2D([], [], ls="none", marker=fc.GROUP_STYLE[g][1], color=fc.GROUP_STYLE[g][0],
                  markeredgecolor="white", markeredgewidth=0.5, markersize=4, label=fc.GROUP_LABEL[g])
           for g in fc.GROUPS]
handles.append(Line2D([], [], color="black", lw=1.0, marker="_", markersize=6,
                      label="Median over hearts, 95% CI"))
sf_top.legend(handles=handles, loc="outside upper center", ncols=6, frameon=False)

# --- panel C: heatmap -------------------------------------------------------
axc = sf_bot.subplots(1, 1)
norm = colors.LogNorm(vmin=0.5, vmax=2.0)  # geometric centre at 1: 0.5 and 2 are equal fold errors
cmap = plt.get_cmap("RdBu_r").copy()  # ColorBrewer RdBu, colourblind-safe diverging
cmap.set_bad("white")
Mm = np.ma.masked_invalid(M)
im = axc.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), Mm, cmap=cmap, norm=norm,
                    edgecolors="white", linewidth=0.5, rasterized=False)  # >= 0.5 pt (V3)
# missing cells: grey cross so they are not read as ratio = 1
for i, j in zip(*np.where(~np.isfinite(M))):
    axc.plot([j + 0.2, j + 0.8], [i + 0.2, i + 0.8], color="0.55", lw=0.5)
    axc.plot([j + 0.2, j + 0.8], [i + 0.8, i + 0.2], color="0.55", lw=0.5)
# small dot in every cell below 1, so direction survives greyscale (both arms of the map print dark)
for i, j in zip(*np.where(np.isfinite(M) & (M < 1.0))):
    rgb = cmap(norm(M[i, j]))[:3]
    light = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    axc.plot([j + 0.5], [i + 0.5], ls="none", marker="o", ms=1.3, mew=0,
             color="white" if light < 0.5 else "black")
axc.set_ylim(M.shape[0], 0)
axc.set_yticks(np.arange(M.shape[0]) + 0.5, [lab for _, lab in COLS])
axc.axhline(4, color="black", lw=0.8)
bounds = [0]
for g in fc.GROUPS:
    bounds.append(bounds[-1] + int((hm.group == g).sum()))
for b in bounds[1:-1]:
    axc.axvline(b, color="black", lw=0.8)
axc.set_xticks([(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])], [fc.GROUP_LABEL[g] for g in fc.GROUPS])
axc.tick_params(axis="x", length=0)
axc.set_xlabel("Heart (10 per ACDC group, ordered by patient number within group)")
for s in axc.spines.values():
    s.set_visible(False)
cb = sf_bot.colorbar(im, ax=axc, extend="both", fraction=0.03, pad=0.01, aspect=15)
cb.set_ticks([0.5, 0.67, 1.0, 1.5, 2.0], labels=["0.5", "0.67", "1", "1.5", "2"])
cb.minorticks_off()
cb.set_label(r"Recovered/true $v_\mathrm{t}$")

fc.panel_letters(list(axes), ["A", "B"], dx=-0.12, dy=1.0)
axc.text(-0.155, 1.04, "C", transform=axc.transAxes, fontsize=fc.PANEL_PT, fontweight="bold",
         va="bottom", ha="left")

# raw data beside the figure
long = []
for lay, name in (("A2", "voxel-face"), ("B1", "smooth-surface")):
    for p in ("v_endo", "v_t"):
        for _, r in m.iterrows():
            long.append(dict(panel="A" if p == "v_endo" else "B", patient=r.patient, group=r.group, layer=name,
                             parameter=p, cv=r[f"cv_{p}_{lay}"]))
pd.DataFrame(long).to_csv(fc.OUT / "fig2_v03_points.csv", index=False)
hm.drop(columns="gorder").to_csv(fc.OUT / "fig2_v03_heatmap.csv", index=False)

probs = fc.finish(fig, "fig2_v03", fc.WIDTH_CMPB_MM, __file__)
print("font", fam)
print("\n".join(probs))
print("ratio range", np.nanmin(M), np.nanmax(M), "n>2", np.sum(M > 2), "n<0.5", np.sum(M < 0.5))
