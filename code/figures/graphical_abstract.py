"""Graphical abstract (Elsevier, CMPB).

Specification read 2026-10-29 at https://www.elsevier.com/researcher/author/tools-and-resources/graphical-abstract:
at least 531 x 1328 px (h x w), or proportionally more at the same 2.5:1 ratio; shown in a 200 x 500 px window;
at least 300 dpi; TIFF, EPS, PDF or MS Office; fonts Times, Arial, Courier or Symbol, legible at the reduced size;
left-to-right reading; no heading or outline. The CMPB Guide for Authors page returned HTTP 403 the same day, so no
journal-specific override was checked.

Built at 1328 x 531 px at 300 dpi (112.4 x 45.0 mm); the PNG and TIFF are written at 600 dpi (2656 x 1062 px, same
ratio). Font: Liberation Sans, the metric-compatible Arial clone (Arial itself is not installed here).

Panels: left, the end-diastolic cine slice of heart 141 (the Fig. 1 example) with expert and four network
contours; centre, recovered/true v_t for its ten constructions (mean over the three truth draws; the individual
draws are in the CSV); right, the error budget for v_t from results.json, construction spread against model-form
and torso bias. Values are asserted against results.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # needed under python -I
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figcommon as fc  # noqa: E402
from pictorial_common import SOURCES, SRC_LABEL, SRC_STYLE  # noqa: E402

fam = fc.figstyle.use_print_style("Liberation Sans")
assert fam == "Liberation Sans", fam
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

PX_W, PX_H, DPI_SPEC = 1328, 531, 300
# CineMA as in Fig. 1: darker yellow, so the graphical abstract matches the figure and reads in greyscale
CINEMA_DARK = "#B8A800"
W_IN, H_IN = PX_W / DPI_SPEC, PX_H / DPI_SPEC
W_MM, H_MM = W_IN * 25.4, H_IN * 25.4
TXT, BIG = 8.0, 9.0          # pt; at the 200 px thumbnail 8 pt is about 12 px
fc.floor_mathtext_scripts(base_pt=TXT)  # Elsevier: subscripts >= 6 pt (8 pt base gave 5.6 pt); layout only
plt.rcParams.update({"font.size": TXT, "axes.labelsize": TXT, "xtick.labelsize": TXT, "ytick.labelsize": TXT,
                     "mathtext.cal": fam, "mathtext.sf": fam, "mathtext.tt": fam})
PID = 141  # Fig. 1 example heart (selection rule in fig1_pipeline.py)

# ------------------------------------------------------------- values, asserted
R = fc.results_json()
cv_vt = R["E1_primary"]["v_t"]["cv_median"]
ci_vt = R["E1_primary"]["v_t"]["ci"]
cmp_ = {(r["factor"], r["parameter"], r["metric"]): r for r in R["E5_comparison"]}
mf = cmp_[("model form (arm F, reaction-eikonal targets, QRS window)", "v_t",
           "geometric-mean theta_hat/theta* minus 1")]
t17 = cmp_[("torso (arm T, reference 17, translation only)", "v_t", "mean absolute relative bias")]
t09 = cmp_[("torso (arm T, reference 09, translation only)", "v_t", "mean absolute relative bias")]
pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
assert pct(cv_vt) == "16.1" and pct(abs(mf["value"])) == "7.1"
assert pct(t17["value"]) == "9.0" and pct(t09["value"]) == "16.1"
assert mf["value"] < 0

# ------------------------------------------------------------- data
est = pd.read_csv(fc.RES / "2026-10-28_armC/estimates.csv")
h = est[(est["round"] == 2) & est.include_primary & (est.loss == "corr") & (est.patient == PID)]
assert len(h) == 30
pdir = fc.ROOT / f"data/raw/acdc/extracted/testing/patient{PID}"
cfg = dict(l.split(": ") for l in (pdir / "Info.cfg").read_text().strip().splitlines())
ed = int(cfg["ED"])
img = np.asarray(nib.load(pdir / f"patient{PID}_frame{ed:02d}.nii.gz").dataobj, dtype=float)
labs = {"expert": np.asarray(nib.load(pdir / f"patient{PID}_frame{ed:02d}_gt.nii.gz").dataobj)}
for s in SOURCES[1:]:
    labs[s] = np.asarray(nib.load(fc.ROOT / f"data/processed/acdc_seg/{s}/patient{PID}_ED.nii.gz").dataobj)
lv_slices = [k for k in range(img.shape[2]) if (labs["expert"][:, :, k] == 3).any()]
KSL = lv_slices[len(lv_slices) // 2]
union = np.zeros(img.shape[:2], bool)
for s in SOURCES:
    union |= labs[s][:, :, KSL] > 0
ii, jj = np.nonzero(union)
side = max(np.ptp(ii), np.ptp(jj)) + 8
ic, jc = (ii.min() + ii.max()) // 2, (jj.min() + jj.max()) // 2
i0, j0 = ic - side // 2, jc - side // 2
crop = img[i0:i0 + side, j0:j0 + side, KSL]
lo, hi = np.percentile(crop, [1, 99.5])

# ------------------------------------------------------------- figure
fig = plt.figure(figsize=(W_IN, H_IN))


def ax_mm(x, y, w, hh):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, hh / H_MM])


def head(x_mm, text):
    fig.text(x_mm / W_MM, (H_MM - 1.2) / H_MM, text, ha="left", va="top", fontsize=BIG, fontweight="bold")


# left: cine with five label sets
axA = ax_mm(1.0, 1.5, 30.0, 30.0)
axA.imshow(np.clip((crop - lo) / (hi - lo), 0, 1), cmap="gray", interpolation="bilinear")
for s in SOURCES:
    col, ls, _ = SRC_STYLE[s]
    col = CINEMA_DARK if s == "cinema_acdc_sax" else col
    L = labs[s][i0:i0 + side, j0:j0 + side, KSL]
    for v in (1, 2):
        axA.contour((L == v).astype(float), levels=[0.5], colors=[col], linewidths=0.8, linestyles=[ls])
axA.set_axis_off()
head(1.0, "5 segmentations")
fig.text(1.0 / W_MM, (H_MM - 5.6) / H_MM, "1 heart, 2 mesh sizes", ha="left", va="top")

# centre: recovered / true v_t per construction
axB = ax_mm(43.0, 12.0, 21.0, 21.5)
xpos = {s: i for i, s in enumerate(SOURCES)}
rows = []
for s in SOURCES:
    col, _, mk = SRC_STYLE[s]
    col = CINEMA_DARK if s == "cinema_acdc_sax" else col
    for r, dx in ((1.5, -0.17), (1.0, 0.17)):
        sub = h[(h.source == s) & (h.resolution_mm == r)]
        assert len(sub) == 3
        m = sub.ratio_v_t.mean()
        edge = "black" if s in ("expert", "cinema_acdc_sax") else col
        axB.plot(xpos[s] + dx, m, ls="none", marker=mk, markersize=5.0, markeredgewidth=0.8,
                 markerfacecolor=col if r == 1.5 else "white", markeredgecolor=edge)
        for _, e in sub.iterrows():
            rows.append(dict(panel="centre", patient=PID, source=s, resolution_mm=r, istar=int(e.istar),
                             ratio_v_t=e.ratio_v_t, plotted_mean_over_istar=m))
axB.axhline(1.0, color="black", lw=0.7, ls="--", zorder=0)
axB.set_ylim(0.2, 1.6)
axB.set_yticks([0.5, 1.0, 1.5])
axB.set_xlim(-0.6, 4.6)
axB.set_xticks(range(5), ["Expert", "T027", "T114", "UKB", "CineMA"], rotation=90, ha="center")
axB.tick_params(axis="x", length=0, pad=2)
axB.set_ylabel(r"Recovered / true $v_\mathrm{t}$", labelpad=1)
head(33.5, "10 constructions,")
fig.text(33.5 / W_MM, (H_MM - 5.6) / H_MM, "10 answers", ha="left", va="top",
         fontsize=BIG, fontweight="bold")
axB.text(-0.5, 0.23, "filled 1.5 mm\nopen 1.0 mm", ha="left", va="bottom", fontsize=TXT, linespacing=1.0)

# right: error budget, horizontal bars from zero
axC = ax_mm(90.0, 8.5, 21.0, 24.5)
C_CONS, C_TORSO, C_MF = fc.OI["blue"], fc.OI["orange"], fc.OI["green"]
y_cons, y_mf, y_t = 2.3, 1.0, 0.0
axC.barh(y_cons, 100 * cv_vt, height=0.62, color=C_CONS, lw=0)
axC.barh(y_mf, 100 * abs(mf["value"]), height=0.62, color=C_MF, lw=0, hatch="////", edgecolor="white")
axC.barh(y_t, 100 * t17["value"], height=0.62, color=C_TORSO, lw=0, hatch="....", edgecolor="white")
axC.barh(y_t, 100 * (t09["value"] - t17["value"]), left=100 * t17["value"], height=0.62, color="white",
         edgecolor=C_TORSO, lw=0.8, hatch="....")
axC.text(100 * cv_vt + 0.6, y_cons, f"{pct(cv_vt)}%", va="center", ha="left")
axC.text(100 * abs(mf["value"]) + 0.6, y_mf, f"{pct(abs(mf['value']))}%", va="center", ha="left")
axC.text(100 * t09["value"] + 0.6, y_t, f"{pct(t17['value'])} to\n{pct(t09['value'])}%", va="center", ha="left",
         linespacing=1.0)
axC.set_yticks([y_cons, y_mf, y_t], ["Construction\n(spread)", "Simplified\nphysics (bias)", "Generic\ntorso (bias)"])
axC.tick_params(axis="y", length=0, pad=2)
axC.set_ylim(-0.55, 3.0)
axC.set_xlim(0, 30)
axC.set_xticks([0, 10, 20])  # axis still runs to 30; a '30' label would touch the figure edge
axC.set_xlabel(r"Error in $v_\mathrm{t}$ (%)", labelpad=1)
axC.spines["left"].set_visible(False)
# spread vs bias brackets
head(69.5, "Spread across")
fig.text(69.5 / W_MM, (H_MM - 5.6) / H_MM, "constructions vs bias", ha="left", va="top", fontsize=BIG,
         fontweight="bold")

for x0, x1 in ((31.3, 34.3), (65.0, 68.0)):
    fig.patches.append(FancyArrowPatch((x0 / W_MM, 20.0 / H_MM), (x1 / W_MM, 20.0 / H_MM),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                       color="black"))

rows += [dict(panel="right", quantity="construction spread, median per-heart CV", value=cv_vt,
              ci_lo=ci_vt[0], ci_hi=ci_vt[1], key="E1_primary.v_t.cv_median"),
         dict(panel="right", quantity="model-form bias, |geometric-mean ratio - 1|", value=abs(mf["value"]),
              signed=mf["value"], ci_lo=mf["ci"][0], ci_hi=mf["ci"][1], key="E5_comparison model form v_t"),
         dict(panel="right", quantity="torso bias, reference 17, translation only", value=t17["value"],
              ci_lo=t17["ci"][0], ci_hi=t17["ci"][1], key="E5_comparison torso 17 v_t abs"),
         dict(panel="right", quantity="torso bias, reference 09, translation only", value=t09["value"],
              ci_lo=t09["ci"][0], ci_hi=t09["ci"][1], key="E5_comparison torso 09 v_t abs")]
stem = "graphical_abstract"
pd.DataFrame(rows).to_csv(fc.OUT / f"{stem}_data.csv", index=False)

out_pdf, out_png, out_tif = (fc.OUT / f"{stem}.{e}" for e in ("pdf", "png", "tif"))
fc.fontguard.render_strict(fig, str(out_pdf))
fc.fontguard.render_strict(fig, str(out_png), dpi=2 * DPI_SPEC)
from PIL import Image  # noqa: E402
im = Image.open(out_png)
flat = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA")).convert("RGB")
flat.save(out_png, dpi=(2 * DPI_SPEC, 2 * DPI_SPEC))
flat.save(out_tif, compression="tiff_lzw", dpi=(2 * DPI_SPEC, 2 * DPI_SPEC))
assert flat.size == (2 * PX_W, 2 * PX_H), flat.size
assert os.path.getmtime(out_png) > os.path.getmtime(__file__)
# thumbnail as ScienceDirect shows it, plus greyscale, for inspection only
qa = fc.QA_DIR
qa.mkdir(parents=True, exist_ok=True)
flat.resize((500, 200), Image.LANCZOS).save(qa / f"{stem}_thumb500x200.png")
flat.convert("L").save(qa / f"{stem}_grey.png")
flat.resize((PX_W, PX_H), Image.LANCZOS).save(qa / f"{stem}_1328x531.png")
probs = fc.figqa.report(fig)
print("font", fam, "size px", flat.size)
print("\n".join(probs))
