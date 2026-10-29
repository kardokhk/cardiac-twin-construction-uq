"""F4: noise-induced SD of theta_hat/theta* against noise level, per comparator, both losses.

Inputs (read only): results/2026-10-28_analysis/results.json
  B4_E3_extended_grid.{primary, amp_loss}   median over hearts of noise SD per level, break-even, CI
  E1_primary.{v_endo, v_t}.sd_median/sd_ci  construction SD, correlation loss
  E7_sensitivity_cv[set == amp_loss]        construction SD, amplitude loss
  E3_noise_censored + B4_E3_extended_grid.noise_censored_ext   censored calibrations per level
No new statistics. Levels with more than 10% censored calibrations (of 3,000 = 50 hearts x 3 theta* x
20 replicates) are drawn open: their SD is a lower bound.
"""
import numpy as np
import pandas as pd

import figcommon as fc

fam = fc.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

R = fc.results_json()
B4 = R["B4_E3_extended_grid"]
N_PER_LEVEL = 50 * 3 * 20
CENS_FRAC = 0.10
amp_cons = [x for x in R["E7_sensitivity_cv"] if x.get("set") == "amp_loss"]
assert len(amp_cons) == 1
CONS = {
    "corr": {p: (R["E1_primary"][p]["sd_median"], R["E1_primary"][p]["sd_ci"]) for p in ("v_endo", "v_t")},
    "amp": {p: (amp_cons[0][p]["sd_median"], amp_cons[0][p]["sd_ci"]) for p in ("v_endo", "v_t")},
}
cens = {}
for c in R["E3_noise_censored"] + B4["noise_censored_ext"]:
    cens[(c["loss"], c["comparator"], float(c["level"]))] = c["censored"]

COMP = [
    ("white", "White noise SNR (dB)", "lin"),
    ("lp40", "40 Hz band-limited noise SNR (dB)", "lin"),
    ("wander", "Baseline wander (% of lead QRS peak-to-peak)", "log"),
    ("disp", "Electrode displacement SD (mm)", "log"),
]
LOSSKEY = {"corr": "primary", "amp": "amp_loss"}

fig, axes = fc.figstyle.figure_mm(fc.WIDTH_DOUBLE_MM, 165, nrows=4, ncols=2)
rows = []
for i, (comp, xlab, xs) in enumerate(COMP):
    for j, p in enumerate(("v_endo", "v_t")):
        ax = axes[i, j]
        for loss in ("corr", "amp"):
            st = fc.LOSS_STYLE[loss]
            e = B4[LOSSKEY[loss]][f"{comp}|{p}"]
            assert abs(e["sd_cons_median"] - CONS[loss][p][0]) < 1e-12
            lv = np.array(e["levels"], float)
            sd = np.array([e["sd_noise_median"][str(int(l)) if float(l).is_integer() else str(l)] for l in e["levels"]])
            cf = np.array([cens.get((loss, comp, l), 0) / N_PER_LEVEL for l in lv])
            sc, (clo, chi) = CONS[loss][p]
            ax.axhspan(clo, chi, color=st["color"], alpha=0.15, lw=0)
            ax.axhline(sc, color=st["color"], lw=0.6, ls=st["ls"])
            ax.plot(lv, sd, color=st["color"], ls=st["ls"], lw=0.9)
            ok = cf <= CENS_FRAC
            ax.plot(lv[ok], sd[ok], ls="none", marker=st["marker"], color=st["color"], ms=3)
            ax.plot(lv[~ok], sd[~ok], ls="none", marker=st["marker"], mfc="white", mec=st["color"], ms=3)
            be, ci = e["breakeven"], e["ci"]
            lo = ci[0] if not isinstance(ci[0], str) else None
            hi = ci[1] if not isinstance(ci[1], str) else None
            gmin, gmax = lv.min(), lv.max()
            # break-even marker on the construction line; CI as a horizontal bar, open end if beyond grid
            if comp in ("white", "lp40"):
                lo_x = lo if lo is not None else gmin
                hi_x = hi if hi is not None else gmax
            else:
                lo_x = lo if lo is not None else gmax
                hi_x = hi if hi is not None else gmax
            xa, xb = sorted([lo_x, hi_x])
            # offset vertically (log scale) so the two losses' CIs do not overprint the bands
            yb = sc * (0.7 if loss == "corr" else 1.45)
            ax.plot([xa, xb], [yb, yb], color=st["color"], lw=0.9)
            open_lo, open_hi = (lo is None), (hi is None)
            for xe, is_open in ((xa, open_lo if comp in ("white", "lp40") else False),
                                (xb, open_hi if comp not in ("white", "lp40") else False)):
                if is_open:
                    ax.plot([xe], [yb], ls="none", marker=">", ms=3, color=st["color"])
                else:
                    ax.plot([xe, xe], [yb / 1.12, yb * 1.12], color=st["color"], lw=0.8)
            ax.plot([be], [yb], ls="none", marker="D", ms=3.2, mfc="black", mec="black")
            for k, l in enumerate(lv):
                rows.append(dict(comparator=comp, parameter=p, loss=loss, level=l, sd_noise_median=sd[k],
                                 censored_fraction=cf[k], sd_cons_median=sc, sd_cons_ci_low=clo,
                                 sd_cons_ci_high=chi, breakeven=be, breakeven_ci_low=ci[0],
                                 breakeven_ci_high=ci[1]))
        ax.set_yscale("log")
        if xs == "log":
            ax.set_xscale("log", base=2)
            ax.set_xticks(lv, [f"{l:g}" for l in lv])
            ax.minorticks_off()
        else:
            ax.set_xlim(32, -17)  # severity increases to the right
            ax.set_xticks([30, 20, 10, 0, -10])
        ax.set_ylim(1e-4 if p == "v_endo" else 5e-4, 2.5)
        yt = [1e-4, 1e-3, 1e-2, 1e-1, 1] if p == "v_endo" else [1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(yt, [f"{v:g}" for v in yt])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(xlab)
        sub = "endo" if p == "v_endo" else "t"
        ax.set_ylabel(rf"SD of $\hat{{v}}_\mathrm{{{sub}}}/v^*_\mathrm{{{sub}}}$")
pd.DataFrame(rows).to_csv(fc.OUT / "fig4_noise_breakeven.csv", index=False)

handles = [
    Line2D([], [], color=fc.LOSS_STYLE["corr"]["color"], ls="-", marker="o", ms=3, lw=0.9,
           label="Noise SD, correlation loss (primary)"),
    Line2D([], [], color=fc.LOSS_STYLE["amp"]["color"], ls="--", marker="s", ms=3, lw=0.9,
           label="Noise SD, amplitude loss"),
    Line2D([], [], color="0.4", ls="none", marker="o", mfc="white", ms=3,
           label="Level with > 10% censored calibrations"),
    Patch(facecolor="0.6", alpha=0.35, label="Construction SD: line median, band 95% CI"),
    Line2D([], [], color="black", ls="-", lw=0.8, marker="D", ms=3.2,
           label="Break-even level with 95% CI (> beyond grid)"),
]
fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False)
fc.panel_letters(list(axes.flat), list("ABCDEFGH"), dx=-0.13, dy=1.0)

probs = fc.finish(fig, "fig4_noise_breakeven", fc.WIDTH_DOUBLE_MM, __file__)
print("font", fam)
print("\n".join(probs))
