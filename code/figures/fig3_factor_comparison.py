"""F3: construction CV (A2, B1) beside torso bias (arm T) and model-form bias (arm F).

Correlation loss (primary) throughout.
Inputs (read only):
  results/2026-10-28_analysis/results.json
      E1_primary, B1_smooth_layer_SENSITIVITY.E1.corr  (construction CV, cluster bootstrap CI)
      E5_armT  (round 2, set pooled_HCM_heavy, mean absolute bias over hearts, cluster bootstrap CI)
      E5_armF  (set re, 1.5 mm, corr, qrs window: geometric mean ratio, t-based 95% CI)
  results/2026-10-28_armT/estimates_v2.csv (round 2) for the DESCRIPTIVE median over hearts of
      absolute bias (I11-full-results section 5 item 5: show mean and median). The median and its
      percentile cluster-bootstrap CI (2,000 resamples, seed 20261028) are computed here.
"""
import numpy as np
import pandas as pd

import figcommon as fc

fam = fc.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

R = fc.results_json()
PARAMS = ("v_endo", "v_t")

# --- arm T medians (descriptive) -------------------------------------------
est = pd.read_csv(fc.RES / "2026-10-28_armT" / "estimates_v2.csv", dtype={"heart": str})
est = est[(est["round"] == 2) & (est.loss == "corr") & (est.ref != "own")]


def armT_median(ref, reg, p):
    d = est[(est.ref == ref) & (est.reg == reg)]
    per = d.assign(a=d[f"rel_{p}"].abs()).groupby("heart")["a"].mean()
    mean_check = per.mean()
    rng = np.random.default_rng(fc.SEED)
    v = per.to_numpy()
    boots = np.array([np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(fc.N_BOOT)])
    return np.median(v), np.percentile(boots, [2.5, 97.5]), mean_check, len(v)


rows = []
for p in PARAMS:
    for lay, e in (("A2", R["E1_primary"]), ("B1", R["B1_smooth_layer_SENSITIVITY"]["E1"]["corr"])):
        q = e[p]
        rows.append(dict(parameter=p, factor="construction", label=f"Construction, {lay} layer",
                         stat="median over hearts of between-construction CV", value=q["cv_median"],
                         lo=q["ci"][0], hi=q["ci"][1], n=50, ci_method="cluster bootstrap over hearts, percentile"))
    for reg, regl in (("trans", "translation"), ("icp", "ICP")):
        for ref in ("09", "17"):
            t = [x for x in R["E5_armT"] if x["loss"] == "corr" and x["ref"] == ref and x["reg"] == reg
                 and x["set"] == "pooled_HCM_heavy"]
            assert len(t) == 1
            t = t[0]
            val, ci = t[f"{p}_abs"]
            med, mci, mean_check, n = armT_median(ref, reg, p)
            assert abs(mean_check - val) < 1e-9, (ref, reg, p, mean_check, val)
            lab = f"Torso, ref {ref}, {regl}"
            rows.append(dict(parameter=p, factor="torso", label=lab, stat="mean absolute relative bias",
                             value=val, lo=ci[0], hi=ci[1], n=n, ci_method="cluster bootstrap over hearts"))
            rows.append(dict(parameter=p, factor="torso_median", label=lab,
                             stat="median absolute relative bias (descriptive)", value=med, lo=mci[0], hi=mci[1],
                             n=n, ci_method="cluster bootstrap over hearts, percentile, computed in fig3 script"))
    f = [x for x in R["E5_armF"] if x["set"] == "re" and x["vox"] == 1.5 and x["loss"] == "corr"
         and x["window"] == "qrs"]
    assert len(f) == 1
    f = f[0][p]
    gm, (tlo, thi) = f["gm"], f["t_ci"]
    rows.append(dict(parameter=p, factor="model_form", label="Model form (reaction-eikonal targets)",
                     stat="|geometric mean ratio - 1| (signed value is negative)", value=abs(gm - 1),
                     lo=abs(thi - 1), hi=abs(tlo - 1), n=10, ci_method="t interval on log ratio over hearts"))
df = pd.DataFrame(rows)
df.to_csv(fc.OUT / "fig3_factor_comparison.csv", index=False)

# --- plot -------------------------------------------------------------------
order = ["Construction, A2 layer", "Construction, B1 layer",
         "Torso, ref 09, translation", "Torso, ref 17, translation",
         "Torso, ref 09, ICP", "Torso, ref 17, ICP",
         "Model form (reaction-eikonal targets)"]
ypos = {lab: i for i, lab in enumerate(order[::-1])}
col = {"construction": fc.OI["blue"], "torso": fc.OI["orange"], "torso_median": fc.OI["orange"],
       "model_form": fc.OI["green"]}
mk = {"construction": "o", "torso": "s", "torso_median": "s", "model_form": "D"}

fig, axes = fc.figstyle.figure_mm(fc.WIDTH_DOUBLE_MM, 68, ncols=2, sharey=True)
for ax, p in zip(axes, PARAMS):
    d = df[df.parameter == p]
    for _, r in d.iterrows():
        off = {"torso": 0.13, "torso_median": -0.13}.get(r.factor, 0.0)
        y = ypos[r.label] + off
        v, lo, hi = 100 * r.value, 100 * r.lo, 100 * r.hi
        filled = r.factor != "torso_median"
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=mk[r.factor], color=col[r.factor],
                    mfc=col[r.factor] if filled else "white", mec=col[r.factor], ms=3.5,
                    elinewidth=0.8, capsize=1.5, capthick=0.6)
    for yy in (4.5, 0.5):
        ax.axhline(yy, color="0.8", lw=0.5)
    ax.set_xlim(0, None)
    ax.set_xlabel(f"Magnitude for {fc.PARAM_LABEL[p]} (%)")
axes[0].set_yticks(list(ypos.values()), list(ypos.keys()))
axes[0].set_ylabel("Factor (correlation loss)")
axes[1].set_ylabel("")
axes[1].tick_params(labelleft=False)
handles = [
    Line2D([], [], ls="none", marker="o", color=col["construction"], ms=3.5,
           label="Construction: median CV over 50 hearts"),
    Line2D([], [], ls="none", marker="s", color=col["torso"], ms=3.5,
           label="Torso: mean |bias| over 21 hearts"),
    Line2D([], [], ls="none", marker="s", color=col["torso"], mfc="white", ms=3.5,
           label="Torso: median |bias| (descriptive)"),
    Line2D([], [], ls="none", marker="D", color=col["model_form"], ms=3.5,
           label="Model form: |GM ratio - 1|, 10 hearts"),
]
fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)
# A sits at the figure's left edge because panel A carries the row labels
from matplotlib.transforms import blended_transform_factory  # noqa: E402
axes[0].text(0.005, 1.01, "A", transform=blended_transform_factory(fig.transFigure, axes[0].transAxes),
             fontsize=fc.PANEL_PT, fontweight="bold", va="bottom", ha="left")
fc.panel_letters([axes[1]], ["B"], dx=-0.04, dy=1.01)

probs = fc.finish(fig, "fig3_factor_comparison", fc.WIDTH_DOUBLE_MM, __file__)
print("font", fam)
print("\n".join(probs))
print(df[["parameter", "label", "factor", "value", "lo", "hi"]].to_string())
