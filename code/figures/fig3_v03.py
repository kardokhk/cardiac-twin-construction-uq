"""Fig. 3 (v03) error budget, and Fig. S (torso registration).

fig3_v03: construction spread (voxel-face and smooth-surface layers), error from a generic torso
(translation-only registration) and error from simplified physics, lead-correlation fit.
figS_torso_registration: torso error under translation-only against iterative closest point registration.
Inputs (read only), as fig3_factor_comparison.py:
  results/2026-10-28_analysis/results.json  E1_primary, B1_smooth_layer_SENSITIVITY.E1.corr, E5_armT, E5_armF
  results/2026-10-28_armT/estimates_v2.csv  round 2, for the descriptive median of torso error
Medians of torso error and their percentile cluster-bootstrap CIs (2,000 resamples over hearts, fresh
numpy default_rng(20261028) per call) are computed here, identically to fig3_factor_comparison.py.
Run: FIG_QA_DIR=<scratch> /project/home/p201509/envs/cvri-sim/bin/python -I code/figures/fig3_v03.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figcommon as fc  # noqa: E402

fam = fc.setup(scripts_floor=True)  # CMPB: sub- and superscripts >= 6 pt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

R = fc.results_json()
PARAMS = ("v_endo", "v_t")
est = pd.read_csv(fc.RES / "2026-10-28_armT" / "estimates_v2.csv", dtype={"heart": str})
est = est[(est["round"] == 2) & (est.loss == "corr") & (est.ref != "own")]


def torso_median(ref, reg, p):
    d = est[(est.ref == ref) & (est.reg == reg)]
    per = d.assign(a=d[f"rel_{p}"].abs()).groupby("heart")["a"].mean()
    rng = np.random.default_rng(fc.SEED)
    v = per.to_numpy()
    boots = np.array([np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(fc.N_BOOT)])
    return np.median(v), np.percentile(boots, [2.5, 97.5]), per.mean(), len(v)


def torso_rows(p, reg, regl):
    out = []
    for ref in ("09", "17"):
        t = [x for x in R["E5_armT"] if x["loss"] == "corr" and x["ref"] == ref and x["reg"] == reg
             and x["set"] == "pooled_HCM_heavy"]
        assert len(t) == 1
        val, ci = t[0][f"{p}_abs"]
        med, mci, mean_check, n = torso_median(ref, reg, p)
        assert abs(mean_check - val) < 1e-9
        lab = f"reference torso {ref}, {regl}"
        out.append(dict(parameter=p, factor="torso", label=lab, stat="mean over hearts of |ratio - 1|",
                        value=val, lo=ci[0], hi=ci[1], n=n, ci_method="cluster bootstrap over hearts (archived)"))
        out.append(dict(parameter=p, factor="torso_median", label=lab,
                        stat="median over hearts of |ratio - 1| (descriptive)", value=med, lo=mci[0], hi=mci[1],
                        n=n, ci_method="percentile cluster bootstrap over hearts, computed in this script"))
    return out


COL = {"construction": fc.OI["blue"], "torso": fc.OI["orange"], "torso_median": fc.OI["orange"],
       "model_form": fc.OI["green"]}
MK = {"construction": "o", "torso": "s", "torso_median": "s", "model_form": "D"}


def draw(df, order, stem, height, handles, group_lines):
    ypos = {lab: i for i, lab in enumerate(order[::-1])}
    fig, axes = fc.figstyle.figure_mm(fc.WIDTH_CMPB_MM, fc.cmpb_height(height), ncols=2, sharey=True)
    for ax, p in zip(axes, PARAMS):
        for _, r in df[df.parameter == p].iterrows():
            off = {"torso": 0.13, "torso_median": -0.13}.get(r.factor, 0.0)
            v, lo, hi = 100 * r.value, 100 * r.lo, 100 * r.hi
            filled = r.factor != "torso_median"
            ax.errorbar(v, ypos[r.label] + off, xerr=[[v - lo], [hi - v]], fmt=MK[r.factor], color=COL[r.factor],
                        mfc=COL[r.factor] if filled else "white", mec=COL[r.factor], ms=3.5,
                        elinewidth=0.8, capsize=1.5, capthick=0.6)
        for yy in group_lines:
            ax.axhline(yy, color="0.8", lw=0.5)
        ax.set_xlim(0, 45)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel(f"Size for {fc.PARAM_LABEL[p]} (%)")
    axes[0].set_yticks(list(ypos.values()), list(ypos.keys()))
    axes[1].tick_params(labelleft=False)
    fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)
    axes[0].text(0.005, 1.01, "A", transform=blended_transform_factory(fig.transFigure, axes[0].transAxes),
                 fontsize=fc.PANEL_PT, fontweight="bold", va="bottom", ha="left")
    fc.panel_letters([axes[1]], ["B"], dx=-0.04, dy=1.01)
    probs = fc.finish(fig, stem, fc.WIDTH_CMPB_MM, __file__)
    print(stem, "font", fam)
    print("\n".join(probs))


# ---- main figure -------------------------------------------------------------
rows = []
for p in PARAMS:
    for lay, lname, e in (("A2", "voxel-face layer", R["E1_primary"]),
                          ("B1", "smooth-surface layer", R["B1_smooth_layer_SENSITIVITY"]["E1"]["corr"])):
        q = e[p]
        rows.append(dict(parameter=p, factor="construction", label=f"Construction spread (CV), {lname}",
                         stat="median over hearts of between-construction CV", value=q["cv_median"],
                         lo=q["ci"][0], hi=q["ci"][1], n=50, ci_method="percentile cluster bootstrap over hearts"))
    for r in torso_rows(p, "trans", "translation"):
        r["label"] = "Error from a generic torso,\n" + r["label"]
        rows.append(r)
    f = [x for x in R["E5_armF"] if x["set"] == "re" and x["vox"] == 1.5 and x["loss"] == "corr"
         and x["window"] == "qrs"]
    assert len(f) == 1
    gm, (tlo, thi) = f[0][p]["gm"], f[0][p]["t_ci"]
    assert gm < 1 and thi < 1
    rows.append(dict(parameter=p, factor="model_form", label="Error from simplified physics",
                     stat="|geometric mean ratio - 1| (signed value negative)", value=abs(gm - 1),
                     lo=abs(thi - 1), hi=abs(tlo - 1), n=10, ci_method="t interval on log ratio over hearts"))
df = pd.DataFrame(rows)
df.to_csv(fc.OUT / "fig3_v03.csv", index=False)
order = ["Construction spread (CV), voxel-face layer", "Construction spread (CV), smooth-surface layer",
         "Error from a generic torso,\nreference torso 09, translation",
         "Error from a generic torso,\nreference torso 17, translation",
         "Error from simplified physics"]
handles = [
    Line2D([], [], ls="none", marker="o", color=COL["construction"], ms=3.5,
           label="Spread: median CV over 50 hearts, 95% CI"),
    Line2D([], [], ls="none", marker="s", color=COL["torso"], ms=3.5,
           label="Bias: mean |recovered/true − 1| over 21 hearts, 95% CI"),
    Line2D([], [], ls="none", marker="s", color=COL["torso"], mfc="white", ms=3.5,
           label="Bias: median |recovered/true − 1| over 21 hearts, 95% CI"),
    Line2D([], [], ls="none", marker="D", color=COL["model_form"], ms=3.5,
           label="Bias: |geometric mean recovered/true − 1|, 10 hearts, 95% CI"),
]
draw(df, order, "fig3_v03", 62, handles, [2.5, 0.5])

# ---- supplementary: registration ------------------------------------------
rows = []
for p in PARAMS:
    for reg, regl in (("trans", "translation only"), ("icp", "iterative closest point")):
        rows += torso_rows(p, reg, regl)
dfs = pd.DataFrame(rows)
dfs.to_csv(fc.OUT / "figS_torso_registration.csv", index=False)
order = ["reference torso 09, translation only", "reference torso 09, iterative closest point",
         "reference torso 17, translation only", "reference torso 17, iterative closest point"]
draw(dfs, order, "figS_torso_registration", 52, handles[1:3], [1.5])
print(df[["parameter", "label", "value", "lo", "hi"]].to_string())
print(dfs[["parameter", "label", "factor", "value", "lo", "hi"]].to_string())
