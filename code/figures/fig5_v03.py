"""Fig. 5 (v03): per-network v_t recovery ratio beside myocardial Dice, and the ensemble-size curve.

Replaces the variance-shares figure in the main text (playbook C7), because the per-network and
ensemble-size analyses were independently reproduced (results/2026-10-29_v03_analyses/README.md,
"Independent recomputation 2026-10-29", 356 of 356 REPRODUCED).
Inputs (read only), results/2026-10-29_v03_analyses/:
  per_heart_network_long.csv   r_hn (median over 3 truth draws x 2 resolutions) and myocardial Dice per heart
  table1_per_network.csv       median and IQR of r_hn; median myocardial Dice over the hearts used
  ensemble_cv_v_t_k{2,3,4}_complete_hearts.csv  per-heart CV for each k-subset of the four networks
  results.json item4_ensemble_size.complete_hearts_v_t  plotted value and 95% CI per k
No new statistics: every plotted summary is asserted equal to the archived value.
Run: FIG_QA_DIR=<scratch> /project/home/p201509/envs/cvri-sim/bin/python -I code/figures/fig5_v03.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figcommon as fc  # noqa: E402
from pictorial_common import SRC_STYLE  # noqa: E402

fam = fc.setup(scripts_floor=True)  # CMPB: sub- and superscripts >= 6 pt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

V03 = fc.RES / "2026-10-29_v03_analyses"
long = pd.read_csv(V03 / "per_heart_network_long.csv")
tab = pd.read_csv(V03 / "table1_per_network.csv").set_index("source")
R = json.load(open(V03 / "results.json"))
ENS = R["item4_ensemble_size"]["complete_hearts_v_t"]

ROWS = [  # source, label, colour, marker
    ("nnunet_task027_acdc", "nnU-Net Task027") + SRC_STYLE["nnunet_task027_acdc"][::2],
    ("nnunet_task114_mnms", "nnU-Net Task114") + SRC_STYLE["nnunet_task114_mnms"][::2],
    ("nnunet_ukb_sax", "nnU-Net UKB SAX") + SRC_STYLE["nnunet_ukb_sax"][::2],
    ("cinema_acdc_sax", "CineMA") + SRC_STYLE["cinema_acdc_sax"][::2],
    ("expert_1.0", "Expert labels, 1.0 mm", "#000000", "o"),
    ("ukbb_cardiac_fcn_sa", "ukbb_cardiac (sensitivity)", "0.45", "X"),
]
ypos = {s: len(ROWS) - 1 - i for i, (s, *_) in enumerate(ROWS)}

fig = plt.figure(figsize=(fc.figstyle.mm(fc.WIDTH_CMPB_MM), fc.figstyle.mm(fc.cmpb_height(70))), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 1.25])
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharey=axA)
axC = fig.add_subplot(gs[2])
rng = np.random.default_rng(fc.SEED)
out = []
for s, lab, col, mk in ROWS:
    d = long[long.source == s]
    assert len(d) == tab.loc[s, "n_hearts"]
    y = ypos[s]
    jy = rng.uniform(-0.18, 0.18, len(d))
    edge = "black" if s == "cinema_acdc_sax" else col
    axA.scatter(d.ratio_v_t, y + jy, s=4, marker=mk, facecolor=col, edgecolor="none", alpha=0.65, zorder=2)
    med, q1, q3 = np.median(d.ratio_v_t), *np.percentile(d.ratio_v_t, [25, 75])
    assert np.allclose([med, q1, q3], tab.loc[s, ["ratio_v_t_median", "ratio_v_t_q25", "ratio_v_t_q75"]])
    axA.plot([q1, q3], [y + 0.32] * 2, color="black", lw=1.0, zorder=3)
    axA.plot([med], [y + 0.32], ls="none", marker=mk, ms=4, mfc=col, mec=edge if s != "expert_1.0" else "black",
             mew=0.6, zorder=4)
    axA.text(5.6, y, f"{med:.2f}", ha="right", va="center")
    row = dict(source=s, n_hearts=len(d), ratio_v_t_median=med, ratio_v_t_q25=q1, ratio_v_t_q75=q3)
    if d.myo_dice.notna().any():
        dm = d.myo_dice.dropna()
        assert abs(np.median(dm) - tab.loc[s, "myo_dice_median_used_hearts"]) < 1e-9
        axB.scatter(dm, y + jy[: len(d)][d.myo_dice.notna().to_numpy()], s=4, marker=mk, facecolor=col,
                    edgecolor="none", alpha=0.65, zorder=2)
        axB.plot([np.median(dm)], [y + 0.32], ls="none", marker=mk, ms=4, mfc=col, mec=edge, mew=0.6, zorder=4)
        axB.plot(np.percentile(dm, [25, 75]), [y + 0.32] * 2, color="black", lw=1.0, zorder=3)
        axB.text(1.075, y, f"{np.median(dm):.3f}", ha="right", va="center")
        row.update(myo_dice_median=np.median(dm), myo_dice_q25=np.percentile(dm, 25),
                   myo_dice_q75=np.percentile(dm, 75), n_dice=len(dm))
    else:
        axB.text(0.72, y, "reference labels", ha="left", va="center", color="0.3")
    out.append(row)
axA.text(5.6, len(ROWS) - 0.45, "Median", ha="right", va="bottom", color="0.3")
axB.text(1.075, len(ROWS) - 0.45, "Median", ha="right", va="bottom", color="0.3")
axA.axvline(1.0, color="0.3", lw=0.6, ls="--", zorder=1)
axA.axhline(1.5, color="0.8", lw=0.5)
axB.axhline(1.5, color="0.8", lw=0.5)
axA.set_xscale("log")
axA.set_xlim(0.3, 5.8)
axA.set_xticks([0.33, 0.5, 0.67, 1, 1.5, 2, 3], ["0.33", "0.5", "0.67", "1", "1.5", "2", "3"])
axA.minorticks_off()
axA.set_yticks(list(ypos.values()), [r[1] for r in ROWS])
axA.set_xlabel(r"Recovered/true $v_\mathrm{t}$ per heart")
axB.set_xlim(0.7, 1.08)
axB.set_xticks([0.7, 0.8, 0.9, 1.0])
axB.tick_params(labelleft=False)
axB.set_xlabel("Myocardial Dice per heart")

# --- C: ensemble size ---------------------------------------------------------
crow = []
for k in (2, 3, 4):
    e = pd.read_csv(V03 / f"ensemble_cv_v_t_k{k}_complete_hearts.csv").set_index("patient")
    assert len(e) == ENS["n_hearts"] == 47 and e.shape[1] == ENS["by_k"][str(k)]["n_subsets"]
    per_subset = 100 * e.median(axis=0)  # median over hearts, one value per subset
    val = ENS["by_k"][str(k)]["value"]
    assert abs(per_subset.mean() / 100 - val) < 1e-9  # archived statistic: mean over subsets (CSV holds 10 digits)
    lo, hi = ENS["by_k"][str(k)]["ci"]
    xs = k + np.linspace(-0.12, 0.12, len(per_subset)) if len(per_subset) > 1 else np.array([k])
    axC.plot(xs - 0.25, per_subset, ls="none", marker="o", ms=2.5, mfc="white", mec="0.35", mew=0.6)
    axC.errorbar(k, 100 * val, yerr=[[100 * (val - lo)], [100 * (hi - val)]], fmt="o", ms=4, color=fc.OI["blue"],
                 elinewidth=1.0, capsize=2, capthick=0.8, zorder=4)
    axC.text(k + 0.1, 100 * hi + 0.3, f"{100 * val:.1f}%", va="bottom", ha="left")
    for c, v in per_subset.items():
        crow.append(dict(k=k, subset=c, median_cv_over_hearts=v / 100, n_hearts=len(e)))
    crow.append(dict(k=k, subset="mean over subsets (plotted)", median_cv_over_hearts=val, ci_low=lo,
                     ci_high=hi, n_hearts=len(e)))
axC.plot([2, 3, 4], [100 * ENS["by_k"][str(k)]["value"] for k in (2, 3, 4)], color=fc.OI["blue"], lw=0.8, zorder=3)
axC.set_xlim(1.55, 4.6)
axC.set_xticks([2, 3, 4])
axC.set_ylim(0, 22)
axC.set_yticks([0, 5, 10, 15, 20])
axC.set_xlabel("Networks per heart, k")
axC.set_ylabel(r"Median spread of $v_\mathrm{t}$, CV (%)")
handles = [Line2D([], [], ls="none", marker="o", mfc="white", mec="0.35", ms=2.5, label="One subset of k"),
           Line2D([], [], ls="-", marker="o", color=fc.OI["blue"], ms=4, lw=0.8,
                  label="Mean over subsets, 95% CI")]
axC.legend(handles=handles, loc="lower right", frameon=False)
fc.panel_letters([axA], ["A"], dx=-0.62, dy=1.01)
fc.panel_letters([axB], ["B"], dx=-0.12, dy=1.01)
fc.panel_letters([axC], ["C"], dx=-0.36, dy=1.01)

pd.DataFrame(out).to_csv(fc.OUT / "fig5_v03_per_network.csv", index=False)
pd.DataFrame(crow).to_csv(fc.OUT / "fig5_v03_ensemble.csv", index=False)
long[["patient", "group", "source", "n_est", "ratio_v_t", "myo_dice"]].to_csv(fc.OUT / "fig5_v03_points.csv",
                                                                              index=False)
probs = fc.finish(fig, "fig5_v03", fc.WIDTH_CMPB_MM, __file__)
print("font", fam)
print("\n".join(probs))
print(pd.DataFrame(out).to_string())
