"""F5: variance shares of log(theta_hat/theta*), pre-registered E2 against secondary E2d.

Correlation loss (primary). Inputs (read only): results/2026-10-28_analysis/results.json
  E2_shares.primary          E2: random heart intercept; all primary rows incl. reference (1,482 estimates)
  E2d_secondary_heart_source.sets.primary_no_ref
                             E2d: + (1 | heart:source); non-reference primary rows (1,332 estimates)
Shares and percentile 95% CIs from 2,000 cluster resamples over hearts (seed 20261028), copied as stored.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script directory
import pandas as pd  # noqa: E402

import figcommon as fc  # noqa: E402

fam = fc.setup(scripts_floor=True)  # CMPB: sub- and superscripts >= 6 pt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

R = fc.results_json()
E2 = R["E2_shares"]["primary"]
E2d = R["E2d_secondary_heart_source"]["sets"]["primary_no_ref"]

TERMS = [  # key, label
    ("segmentation_S", "Segmentation source S"),
    ("mesh_M", "Mesh resolution M"),
    ("SxM", "S x M"),
    ("theta_star", r"Truth draw $\theta^*$"),
    ("heart", "Heart (random intercept)"),
    ("heart_source", "Heart x source (random)"),
    ("residual", "Residual"),
    ("within_heart_construction", "Within-heart construction share"),
]
MODELS = {
    "E2": dict(src=E2, color=fc.OI["blue"], marker="o", off=0.17,
               label="E2, pre-registered: (1 | heart), all rows"),
    "E2d": dict(src=E2d, color=fc.OI["vermillion"], marker="s", off=-0.17,
                label="E2d, secondary: + (1 | heart:source), non-reference rows"),
}
rows = []
fig, axes = fc.figstyle.figure_mm(fc.WIDTH_CMPB_MM, fc.cmpb_height(70), ncols=2, sharey=True)
ypos = {k: len(TERMS) - 1 - i for i, (k, _) in enumerate(TERMS)}
for ax, p in zip(axes, ("v_endo", "v_t")):
    for mname, mm_ in MODELS.items():
        sh = mm_["src"][p]["shares"]
        for k, lab in TERMS:
            if k not in sh:
                rows.append(dict(parameter=p, model=mname, term=k, share=None, ci_low=None, ci_high=None))
                continue
            v, (lo, hi) = 100 * sh[k]["share"], (100 * sh[k]["ci"][0], 100 * sh[k]["ci"][1])
            y = ypos[k] + mm_["off"]
            ax.plot([lo, hi], [y, y], color=mm_["color"], lw=0.9)
            ax.plot([v], [y], ls="none", marker=mm_["marker"], color=mm_["color"], ms=3.5)
            rows.append(dict(parameter=p, model=mname, term=k, share=sh[k]["share"],
                             ci_low=sh[k]["ci"][0], ci_high=sh[k]["ci"][1]))
    ax.axhline(0.5, color="0.8", lw=0.5)
    ax.set_xlim(-2, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(f"Share of variance of log ratio, {fc.PARAM_LABEL[p]} (%)")
axes[0].set_yticks(list(ypos.values()), [lab for _, lab in TERMS])
axes[0].set_ylabel("Variance term (correlation loss)")
axes[1].tick_params(labelleft=False)
ax0 = axes[0]
for k in ("heart_source",):
    ax0.text(1.5, ypos[k] + MODELS["E2"]["off"], "not in E2", va="center", ha="left",
             color=MODELS["E2"]["color"])
    axes[1].text(1.5, ypos[k] + MODELS["E2"]["off"], "not in E2", va="center", ha="left",
                 color=MODELS["E2"]["color"])
handles = [Line2D([], [], color=m["color"], marker=m["marker"], ms=3.5, lw=0.9, label=m["label"])
           for m in MODELS.values()]
fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False)
from matplotlib.transforms import blended_transform_factory  # noqa: E402
axes[0].text(0.005, 1.01, "A", transform=blended_transform_factory(fig.transFigure, axes[0].transAxes),
             fontsize=fc.PANEL_PT, fontweight="bold", va="bottom", ha="left")
fc.panel_letters([axes[1]], ["B"], dx=-0.04, dy=1.01)
pd.DataFrame(rows).to_csv(fc.OUT / "fig5_variance_shares.csv", index=False)

probs = fc.finish(fig, "fig5_variance_shares", fc.WIDTH_CMPB_MM, __file__)
print("font", fam)
print("\n".join(probs))
