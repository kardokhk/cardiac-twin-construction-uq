"""F2: per-heart between-construction CV of theta_hat/theta*, A2 and B1 layers.

Inputs (read only):
  results/2026-10-28_analysis/tables/per_heart_cv_primary.csv     (A2, corr loss)
  results/2026-10-28_analysis/tables/B1_per_heart_cv_sdf_corr.csv (B1, corr loss)
  results/2026-10-28_analysis/results.json  E1_primary, B1_smooth_layer_SENSITIVITY.E1.corr
No new statistics: medians and 95% CIs are copied from results.json.
"""
import numpy as np
import pandas as pd

import figcommon as fc

fam = fc.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

T = fc.ANALYSIS / "tables"
a2 = pd.read_csv(T / "per_heart_cv_primary.csv")
b1 = pd.read_csv(T / "B1_per_heart_cv_sdf_corr.csv")
m = a2.merge(b1, on=["patient", "group"], suffixes=("_A2", "_B1"))
assert len(m) == 50
R = fc.results_json()
summ = {
    "A2": R["E1_primary"],
    "B1": R["B1_smooth_layer_SENSITIVITY"]["E1"]["corr"],
}
# consistency check between the tables and results.json
for lay, df in (("A2", a2), ("B1", b1)):
    for p in ("v_endo", "v_t"):
        assert abs(np.median(df[f"cv_{p}"]) - summ[lay][p]["cv_median"]) < 1e-12

rng = np.random.default_rng(fc.SEED)
jit = rng.uniform(-0.16, 0.16, size=len(m))
X = {"A2": 0.0, "B1": 1.0}

fig, axes = fc.figstyle.figure_mm(fc.WIDTH_DOUBLE_MM, 72, ncols=2)
rows = []
for ax, p in zip(axes, ("v_endo", "v_t")):
    for i, r in m.iterrows():
        ya, yb = 100 * r[f"cv_{p}_A2"], 100 * r[f"cv_{p}_B1"]
        ax.plot([X["A2"] + jit[i], X["B1"] + jit[i]], [ya, yb], color="0.82", lw=0.5, zorder=1)
    for lay in ("A2", "B1"):
        for g in fc.GROUPS:
            s = m[m.group == g]
            col, mk = fc.GROUP_STYLE[g]
            ax.scatter(X[lay] + jit[s.index], 100 * s[f"cv_{p}_{lay}"], s=9, marker=mk,
                       facecolor=col, edgecolor="white", linewidth=0.3, zorder=3)
        e = summ[lay][p]
        med, lo, hi = 100 * e["cv_median"], 100 * e["ci"][0], 100 * e["ci"][1]
        xm = X[lay] + 0.32
        ax.errorbar(xm, med, yerr=[[med - lo], [hi - med]], fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=2.0, capthick=0.8, zorder=4)
        ax.plot([xm - 0.07, xm + 0.07], [med, med], color="black", lw=1.5, zorder=5)
        ax.text(xm + 0.09, med, f"{med:.1f}%", va="center", ha="left")
        rows.append(dict(parameter=p, layer=lay, median_cv=e["cv_median"],
                         ci_low=e["ci"][0], ci_high=e["ci"][1], n_hearts=e["per_heart"]["n"]))
    ax.set_xticks([0, 1], ["A2 voxel layer\n(primary)", "B1 smooth layer\n(sensitivity)"])
    ax.set_xlim(-0.35, 1.7)
    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("Fast endocardial layer definition (correlation loss)")
    ax.set_ylabel(f"CV of {fc.PARAM_LABEL[p]} across constructions (%)")

handles = [Line2D([], [], ls="none", marker=fc.GROUP_STYLE[g][1], color=fc.GROUP_STYLE[g][0],
                  markeredgecolor="white", markeredgewidth=0.3, markersize=4, label=fc.GROUP_LABEL[g])
           for g in fc.GROUPS]
handles.append(Line2D([], [], color="black", lw=1.0, marker="_", markersize=6,
                      label="Median over hearts, 95% CI"))
fig.legend(handles=handles, loc="outside upper center", ncols=6, frameon=False)
fc.panel_letters(axes, ["A", "B"], dx=-0.12, dy=1.0)

# raw data beside the figure
long = []
for lay in ("A2", "B1"):
    for p in ("v_endo", "v_t"):
        for _, r in m.iterrows():
            long.append(dict(patient=r.patient, group=r.group, layer=lay, parameter=p,
                             cv=r[f"cv_{p}_{lay}"], n_cons=r[f"n_cons_{lay}"]))
pd.DataFrame(long).to_csv(fc.OUT / "fig2_construction_cv_points.csv", index=False)
pd.DataFrame(rows).to_csv(fc.OUT / "fig2_construction_cv_summary.csv", index=False)

probs = fc.finish(fig, "fig2_construction_cv", fc.WIDTH_DOUBLE_MM, __file__)
print("font", fam)
print("\n".join(probs))
