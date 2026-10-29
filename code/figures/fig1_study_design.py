"""F1: study design schematic (boxes and arrows only; no data).

Content taken from final/STUDY_PLAN.md section 3 as governed by final/AMENDMENT_2026-10-28.md (A1 to A4)
and final/AMENDMENT_2_2026-10-28.md (B1). No result values are drawn.
"""
import pandas as pd

import figcommon as fc

fam = fc.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

W, H = fc.WIDTH_DOUBLE_MM, 80.0
fig, ax = fc.figstyle.figure_mm(W, H)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

C_MAIN = fc.OI["blue"]
C_COMP = fc.OI["orange"]
C_OUT = fc.OI["green"]
boxes = []


def box(x, y, w, h, title, body, edge):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                       fc="white", ec=edge, lw=0.9)
    ax.add_patch(p)
    ax.text(x + 1.8, y + h - 1.8, title, ha="left", va="top", fontweight="bold")
    ax.text(x + 1.8, y + h - 5.6, body, ha="left", va="top", linespacing=1.25)
    boxes.append(dict(title=title, body=body.replace("\n", " "), x_mm=x, y_mm=y, w_mm=w, h_mm=h))
    return (x, y, w, h)


def arrow(p0, p1, color="black", rad=0.0, style=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                        shrinkA=0, shrinkB=0, connectionstyle=style or f"arc3,rad={rad}")
    ax.add_patch(a)


# top row: arm C, per heart
y1, h1 = 50, 28
b1 = box(1, y1, 33, h1, "Heart (n = 50)",
         "ACDC test set,\nend-diastolic cine CMR,\n10 per group: NOR, DCM,\nHCM, MINF, ARV", C_MAIN)
b2 = box(39, y1, 40, h1, "Constructions S x M",
         "S: expert labels (reference)\n   + 4 segmentation networks\nM: 1.5 mm and 1.0 mm\n"
         "Up to 10 per heart (494 in all)\nFast layer: A2 (primary), B1", C_MAIN)
b3 = box(84, y1, 41, h1, "Forward model",
         "Eikonal activation, fixed rules\nfor fibres and root nodes\n"
         "8-lead QRS in reference torso\n(Malik 09), one rigid transform\nper heart", C_MAIN)
b4 = box(130, y1, 43, h1, "Calibration",
         "Spline emulator, 9 x 9 log grid,\nrefinement runs; "
         r"Fit $\theta$ = ($v_\mathrm{endo}$, $v_\mathrm{t}$)" "\nLoss: correlation (primary),\namplitude (secondary)\n"
         r"Output: $\hat{\theta}/\theta^*$ and its CV" "\nacross constructions",
         C_MAIN)

# targets
bt = box(39, 26.5, 40, 17, r"Truth draws $\theta^*$",
         "3 per heart (Latin hypercube)\nTargets simulated on the\nreference construction", C_MAIN)

# comparators
yc, hc = 2, 16
bN = box(1, yc, 52, hc, "Arm N: target noise",
         "White and 40 Hz band-limited noise, baseline\nwander, electrode displacement; reference\n"
         "emulator; break-even level against arm C", C_COMP)
bF = box(57, yc, 46, hc, "Arm F: model form (10 hearts)",
         "Targets from reaction-eikonal ECGs,\ncalibrated with the eikonal emulator", C_COMP)
bT = box(107, yc, 66, hc, "Arm T: torso (22 heart-in-torso models)",
         "Targets in own torso, calibration in reference\ntorso 09 or 17; registration translation only\n"
         "(primary) or ICP", C_COMP)

def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b, f=0.5):
    x, y, w, h = b
    return (x + f * w, y + h)


def bottom(b, f=0.5):
    x, y, w, h = b
    return (x + f * w, y)


arrow(right(b1), left(b2))
arrow(right(b2), left(b3))
arrow(right(b3), left(b4))
arrow(right(bt), bottom(b4, 0.12), rad=0.0, style="angle,angleA=0,angleB=90,rad=0")
arrow(top(bN, 0.6), (bt[0], bt[1] + 4), color=C_COMP)
arrow(top(bF, 0.3), (bF[0] + 0.3 * bF[2], bt[1]), color=C_COMP)
arrow(top(bT, 0.75), (bT[0] + 0.75 * bT[2], b4[1]), color=C_COMP)

pd.DataFrame(boxes).to_csv(fc.OUT / "fig1_study_design_boxes.csv", index=False)
probs = fc.finish(fig, "fig1_study_design", W, __file__)
print("font", fam)
print("\n".join(probs))
