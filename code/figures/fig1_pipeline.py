"""Fig. 1 (replacement): pictorial pipeline for one example heart.

Panels, left to right: (A) end-diastolic cine short-axis slice with expert and four network label contours;
(B) mid-ventricular cut of the expert construction at 1.5 mm and 1.0 mm (zoomed wall window, element edges
shown); (C) target activation time on the expert 1.5 mm construction (reference), same cut; (D) the stored
8-lead target ECG for the same truth draw; (E) recovered/true v_t for the heart's ten constructions and three
truth draws. Bottom strip: the three benchmarks. Nothing is simulated here: every field is read from archived
outputs. Fitted ECGs are not stored (the calibration archive keeps activation times only), so (D) shows the
target leads only.

Example heart rule (scripted below): among hearts with all 10 primary constructions and no censored
estimate, the one whose per-heart v_t CV is closest to the upper quartile over the 50 hearts.

Sources (read only):
  data/raw/acdc/extracted/testing/patientNNN/patientNNN_frame01.nii.gz (+ _gt) and Info.cfg (ED frame)
  data/processed/acdc_seg/<network>/patientNNN_ED.nii.gz
  data/processed/acdc_mesh_v2/cons/patientNNN__expert__{1.5,1.0}__rv3.0.h5
  results/2026-10-28_armC/hearts/heart_patientNNN.h5 (targets/T, targets/ecg, targets/theta_star)
  results/2026-10-28_armC/estimates.csv (round 2, include_primary, loss corr)
  results/2026-10-28_analysis/tables/per_heart_cv_primary.csv
  results/2026-10-28_G1_segmentation/metrics_test50.tsv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # needed under python -I
import h5py
import nibabel as nib
import numpy as np
import pandas as pd

import figcommon as fc
from pictorial_common import SOURCES, SRC_LABEL, SRC_STYLE

fam = fc.setup(scripts_floor=True)  # CMPB: sub- and superscripts >= 6 pt
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = fc.ROOT
# Layout is written in design millimetres on the original 174 x 84 mm canvas (DW x DH); the figure is
# rendered at 190 mm width with every position and size scaled by the same factor, so the layout and aspect
# ratio are unchanged and only the physical size grows (text stays 7 pt, lines keep their pt widths).
DW, DH = fc.WIDTH_DOUBLE_MM, 84.0
W_MM, H_MM = fc.WIDTH_CMPB_MM, fc.cmpb_height(DH)
# CineMA in Fig. 1: darker yellow for contours, legend key and filled markers, so the key matches the
# contours and filled (1.5 mm) markers read as grey, not white, in greyscale (figure-check defects 2 and 3)
CINEMA_DARK = "#B8A800"
ISTAR = 0          # truth draw shown in C and D (all three are shown in E)
DT_MS = 0.5        # target ECG sampling interval (code/full/calib/calib_lib.py, DT)
ZOOM_MM = 15.0     # side of the wall window in B


# ---------------------------------------------------------------- choose the heart
est = pd.read_csv(fc.RES / "2026-10-28_armC/estimates.csv")
prim = est[(est["round"] == 2) & est.include_primary & (est.loss == "corr")]
cvt = pd.read_csv(fc.ANALYSIS / "tables/per_heart_cv_primary.csv")
q75 = cvt.cv_v_t.quantile(0.75)
cand = cvt[(cvt.n_cons == 10) & (cvt.n_censored == 0)].copy()
cand["d"] = (cand.cv_v_t - q75).abs()
PID = int(cand.sort_values("d").patient.iloc[0])
row = cvt[cvt.patient == PID].iloc[0]
GROUP = row.group
h = prim[prim.patient == PID]
assert len(h) == 30 and h.censored.sum() == 0
cv_check = h.groupby("istar").ratio_v_t.agg(lambda x: x.std(ddof=1) / x.mean()).median()
assert abs(cv_check - row.cv_v_t) < 1e-12, (cv_check, row.cv_v_t)
rank = int((cvt.cv_v_t < row.cv_v_t).sum()) + 1
print(f"heart {PID} ({GROUP}): v_t CV {row.cv_v_t:.4f}, upper quartile {q75:.4f}, rank {rank} of {len(cvt)}")

# ---------------------------------------------------------------- A: cine slice and contours
pdir = ROOT / f"data/raw/acdc/extracted/testing/patient{PID}"
cfg = dict(l.split(": ") for l in (pdir / "Info.cfg").read_text().strip().splitlines())
ed = int(cfg["ED"])
img_nii = nib.load(pdir / f"patient{PID}_frame{ed:02d}.nii.gz")
img = np.asarray(img_nii.dataobj, dtype=float)
px = float(img_nii.header.get_zooms()[0])
labs = {"expert": np.asarray(nib.load(pdir / f"patient{PID}_frame{ed:02d}_gt.nii.gz").dataobj)}
for s in SOURCES[1:]:
    labs[s] = np.asarray(nib.load(ROOT / f"data/processed/acdc_seg/{s}/patient{PID}_ED.nii.gz").dataobj)
    assert labs[s].shape == img.shape
lv_slices = [k for k in range(img.shape[2]) if (labs["expert"][:, :, k] == 3).any()]
KSL = lv_slices[len(lv_slices) // 2]
union = np.zeros(img.shape[:2], bool)
for s in SOURCES:
    union |= labs[s][:, :, KSL] > 0
ii, jj = np.nonzero(union)
pad = 10
i0, i1 = max(ii.min() - pad, 0), min(ii.max() + pad, img.shape[0])
j0, j1 = max(jj.min() - pad, 0), min(jj.max() + pad, img.shape[1])
# square crop so the panel is not distorted
side = max(i1 - i0, j1 - j0)
ic, jc = (i0 + i1) // 2, (j0 + j1) // 2
i0, i1 = ic - side // 2, ic - side // 2 + side
j0, j1 = jc - side // 2, jc - side // 2 + side
crop = img[i0:i1, j0:j1, KSL]
lo, hi = np.percentile(crop, [1, 99.5])

dice = pd.read_csv(fc.RES / "2026-10-28_G1_segmentation/metrics_test50.tsv", sep="\t")
dice = dice[dice.patient == f"patient{PID}"].set_index("network")

# ---------------------------------------------------------------- B, C: mesh cuts


def load_mesh(vox):
    with h5py.File(ROOT / f"data/processed/acdc_mesh_v2/cons/patient{PID}__expert__{vox}__rv3.0.h5", "r") as f:
        pts, tets = f["pts"][:], f["tets"][:]
        R, c = f["grid_R"][:], f["grid_c"][:]
        etag, lv = f["etag"][:], f["lv_endo"][:]
        rv = f["rv_endo"][:]
    loc = (pts - c) @ R.T   # local frame: axis 0 = slice (long) axis, axes 1 and 2 in plane
    return loc, tets, etag, lv, rv


EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def cut(loc, tets, z, node_val=None):
    """Intersect tets with the plane loc[:,0] == z. Returns polygons (in-plane coords), tet ids, mean value."""
    s = loc[:, 0] - z
    st = s[tets]
    hit = (st.min(1) < 0) & (st.max(1) > 0)
    tid = np.nonzero(hit)[0]
    T = tets[tid]
    a, b = T[:, EDGES[:, 0]], T[:, EDGES[:, 1]]
    sa, sb = s[a], s[b]
    cross = (sa * sb) < 0
    w = np.where(cross, sa / np.where(cross, sa - sb, 1.0), 0.0)
    P = loc[a][..., 1:] + w[..., None] * (loc[b][..., 1:] - loc[a][..., 1:])
    vals = None
    if node_val is not None:
        vals = node_val[a] + w * (node_val[b] - node_val[a])
    polys, out_val = [], []
    for k in range(len(tid)):
        q = P[k][cross[k]]
        cen = q.mean(0)
        order = np.argsort(np.arctan2(q[:, 1] - cen[1], q[:, 0] - cen[0]))
        polys.append(q[order])
        if vals is not None:
            out_val.append(vals[k][cross[k]].mean())
    return polys, tid, (np.array(out_val) if vals is not None else None)


loc15, tets15, etag15, lv15, rv15 = load_mesh("1.5")
loc10, tets10, etag10, _, _ = load_mesh("1.0")
# mid-ventricular plane, same frame for both meshes; offset 0.37 mm so the plane never lies on a node layer
# of the 1.0 mm or 1.5 mm voxel-derived meshes (a plane through nodes drops the tets that touch it)
Z = 0.5 * (loc15[lv15, 0].min() + loc15[lv15, 0].max()) + 0.37
with h5py.File(fc.RES / f"2026-10-28_armC/hearts/heart_patient{PID}.h5", "r") as f:
    Tt = f["targets/T"][ISTAR].astype(float)
    ecg = f["targets/ecg"][ISTAR]
    theta_star = f["targets/theta_star"][ISTAR]
    assert f.attrs["reference"] == f"patient{PID}__expert__1.5__rv3.0"
assert Tt.shape[0] == loc15.shape[0]
polyC, tidC, valC = cut(loc15, tets15, Z, Tt)
poly15, tid15, _ = cut(loc15, tets15, Z)
poly10, tid10, _ = cut(loc10, tets10, Z)
# zoom window centre: LV wall on the side away from the RV
near = np.abs(loc15[:, 0] - Z) < 3
lv_c = loc15[lv15][near[lv15]][:, 1:].mean(0)
rv_c = loc15[rv15][near[rv15]][:, 1:].mean(0)
u = (lv_c - rv_c) / np.linalg.norm(lv_c - rv_c)
lvn = loc15[lv15][near[lv15]][:, 1:]
r_endo = np.median(np.linalg.norm(lvn - lv_c, axis=1))
ZC = lv_c + u * (r_endo + 4.0)
# rotate the in-plane mesh coordinates so the LV-to-RV direction matches panel A (rotation only; the mirror
# sense between the image and mesh frames was not verified)
L3 = np.argwhere(labs['expert'][i0:i1, j0:j1, KSL] == 3).mean(0)
L1 = np.argwhere(labs['expert'][i0:i1, j0:j1, KSL] == 1).mean(0)
dA = np.array([L1[1] - L3[1], -(L1[0] - L3[0])])
dC = rv_c - lv_c
ang = np.arctan2(dA[1], dA[0]) - np.arctan2(dC[1], dC[0])
ROT = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
rot = lambda polys: [q @ ROT.T for q in polys]  # noqa: E731
polyC, poly15, poly10 = rot(polyC), rot(poly15), rot(poly10)
ZC = ZC @ ROT.T

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
assert H_MM <= fc.MAX_HEIGHT_CMPB_MM


def ax_mm(x, y, w, hgt):
    return fig.add_axes([x / DW, y / DH, w / DW, hgt / DH])


TITLE_Y = 76.0  # baseline of panel titles (mm from bottom)
axes_for_letters = []


def title(ax, text, x_mm=None):
    x = ax.get_position().x0 if x_mm is None else x_mm / DW
    fig.text(x, TITLE_Y / DH, text, ha="left", va="bottom")


# A
axA = ax_mm(5.0, 36.0, 34.0, 34.0)
axA.imshow(np.clip((crop - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
           origin="upper")
for s in SOURCES:
    col, ls, _ = SRC_STYLE[s]
    col = CINEMA_DARK if s == "cinema_acdc_sax" else col
    L = labs[s][i0:i1, j0:j1, KSL]
    for lab_val in (1, 2):   # RV cavity, LV myocardium (ACDC convention)
        axA.contour((L == lab_val).astype(float), levels=[0.5], colors=[col], linewidths=0.7,
                    linestyles=[ls])
axA.add_patch(Rectangle((3, side - 4.5), 10 / px, 1.2, color="white", lw=0))
axA.text(3, side - 5.5, "10 mm", color="white", ha="left", va="bottom")
axA.set_axis_off()
title(axA, f"Cine ED slice, heart {PID}")
handles = []
for s in SOURCES:
    col, ls, _ = SRC_STYLE[s]
    lab = SRC_LABEL[s] if s == "expert" else f"{SRC_LABEL[s]} ({dice.loc[s, 'MYO_dice']:.2f})"
    handles.append(Line2D([], [], color=CINEMA_DARK if s == "cinema_acdc_sax" else col, ls=ls, lw=1.0,
                          label=lab))
axA.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.02), frameon=False,
           handlelength=2.4, borderaxespad=0.0, labelspacing=0.25,
           title="Labels (myocardial Dice)", title_fontsize=fc.BODY_PT, alignment="left")
axes_for_letters.append(axA)

# B: two zoomed cuts
LAYER = {1: "#D9D9D9", 2: fc.OI["blue"]}
for k, (polys, tid, etag, lab, y0) in enumerate([(poly15, tid15, etag15, "1.5 mm", 54.0),
                                                 (poly10, tid10, etag10, "1.0 mm", 31.0)]):
    ax = ax_mm(46.0, y0, 20.0, 20.0)
    pc = PolyCollection(polys, facecolors=[LAYER[int(t)] for t in etag[tid]], edgecolors="#4D4D4D",
                        linewidths=0.5)
    ax.add_collection(pc)
    ax.set_xlim(ZC[0] - ZOOM_MM / 2, ZC[0] + ZOOM_MM / 2)
    ax.set_ylim(ZC[1] - ZOOM_MM / 2, ZC[1] + ZOOM_MM / 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.text(0.0, -0.03, lab, transform=ax.transAxes, ha="left", va="top")
    if k == 0:
        title(ax, "Expert mesh")
        axB = ax
axL = ax_mm(46.0, 17.0, 20.0, 8.0)
axL.set_axis_off()
axL.legend(handles=[Rectangle((0, 0), 1, 1, fc=LAYER[2], ec="black", lw=0.5, label="Fast endocardial layer"),
                    Rectangle((0, 0), 1, 1, fc=LAYER[1], ec="black", lw=0.5, label="Myocardium")],
           loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.0, labelspacing=0.25)
axes_for_letters.append(axB)

# C: activation map
axC = ax_mm(71.0, 36.0, 32.0, 34.0)
pcC = PolyCollection(polyC, array=valC, cmap="viridis", edgecolors="face", linewidths=0.0)
pcC.set_rasterized(True)
axC.add_collection(pcC)
allp = np.vstack(polyC)
mid = 0.5 * (allp.min(0) + allp.max(0))
half = 0.5 * (allp.max(0) - allp.min(0)).max() + 3
axC.set_xlim(mid[0] - half, mid[0] + half)
axC.set_ylim(mid[1] - half, mid[1] + half)
axC.set_aspect("equal")
axC.set_axis_off()
axC.add_patch(Rectangle(ZC - ZOOM_MM / 2, ZOOM_MM, ZOOM_MM, fill=False, ec="black", lw=0.6, ls="--"))
axC.add_patch(Rectangle((mid[0] + half - 21, mid[1] - half + 1), 20, 1.2, color="black", lw=0))
axC.text(mid[0] + half - 11, mid[1] - half + 2.8, "20 mm", ha="center", va="bottom")
title(axC, "Activation, expert 1.5 mm")
cax = ax_mm(73.0, 30.0, 28.0, 2.0)
cb = fig.colorbar(pcC, cax=cax, orientation="horizontal")
cb.set_label("Activation time (ms)", labelpad=1.5)
cb.outline.set_linewidth(0.5)
axes_for_letters.append(axC)

# D: target ECG, 8 leads, each scaled to its own maximum absolute value
LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
t = np.arange(ecg.shape[1]) * DT_MS
TMAX = 100.0
axD = ax_mm(110.0, 22.0, 24.0, 48.0)
OFF = 2.3
for k, ld in enumerate(LEADS):
    y = ecg[k] / np.abs(ecg[k]).max()
    base = (len(LEADS) - 1 - k) * OFF
    axD.plot(t[t <= TMAX], base + y[t <= TMAX], color="black", lw=0.6)
    axD.text(-3, base, ld, ha="right", va="center")
axD.set_xlim(0, TMAX)
axD.set_ylim(-1.3, (len(LEADS) - 1) * OFF + 1.3)
axD.set_yticks([])
axD.spines["left"].set_visible(False)
axD.set_xticks([0, 50, 100])
axD.set_xlabel("Time (ms)")
title(axD, "Target ECG")
axes_for_letters.append(axD)

# E: recovered / true v_t per construction and truth draw
order = [(s, r) for s in SOURCES for r in (1.5, 1.0)]
axE = ax_mm(158.0, 22.0, 14.0, 48.0)
rows_out = []
for k, (s, r) in enumerate(order):
    col, _, mk = SRC_STYLE[s]
    col = CINEMA_DARK if s == "cinema_acdc_sax" else col
    sub = h[(h.source == s) & (h.resolution_mm == r)].sort_values("istar")
    assert len(sub) == 3
    ypos = len(order) - 1 - k
    for j, (_, e) in enumerate(sub.iterrows()):
        filled = r == 1.5
        axE.plot(e.ratio_v_t, ypos + (j - 1) * 0.22, ls="none", marker=mk, markersize=3.2,
                 markerfacecolor=col if filled else "white", markeredgecolor="black" if s == "cinema_acdc_sax" or s == "expert" else col,
                 markeredgewidth=0.6)
        rows_out.append(dict(patient=PID, source=s, resolution_mm=r, istar=int(e.istar),
                             v_t_star=e.v_t_star, v_t_hat=e.v_t_hat, ratio_v_t=e.ratio_v_t,
                             is_reference=bool(e.is_reference)))
    lbl = f"{SRC_LABEL[s].replace('nnU-Net ', '')}, {r:.1f}"
    if s == "expert" and r == 1.5:
        lbl += " (ref.)"
    axE.text(-0.04, ypos, lbl, transform=axE.get_yaxis_transform(), ha="right", va="center")
axE.axvline(1.0, color="black", lw=0.6, ls="--", zorder=0)
axE.set_ylim(-0.7, len(order) - 0.3)
axE.set_yticks([])
axE.set_xlim(0.5, 1.5)
axE.set_xticks([0.5, 1.0, 1.5])
axE.set_xlabel("Ratio")
axE.spines["left"].set_visible(False)
title(axE, r"Recovered/true $v_\mathrm{t}$", x_mm=138.0)
axes_for_letters.append(axE)

# arrows between panels (figure mm)


def farrow(x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0 / DW, y / DH), (x1 / DW, y / DH), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))


for x0, x1 in [(40.0, 44.5), (67.0, 70.0), (104.0, 107.0), (135.0, 138.0)]:
    farrow(x0, x1, 53.0)

# letters at the panel titles
for ax, L in zip(axes_for_letters, "ABCDE"):
    bb = ax.get_position()
    x = bb.x0 if L != "E" else 138.0 / DW
    fig.text(x - 3.5 / DW, TITLE_Y / DH, L, fontsize=fc.PANEL_PT, fontweight="bold",
             ha="left", va="bottom")

# bottom benchmark strip
axS = ax_mm(2.0, 1.0, DW - 4.0, 11.5)
axS.set_xlim(0, DW - 4.0); axS.set_ylim(0, 11.5)
axS.set_axis_off()
axS.add_patch(Rectangle((0.2, 0.2), DW - 4.4, 11.1, fill=False, ec="black", lw=0.5))
axS.text(2.0, 5.75, "Benchmarks\nfor scale", ha="left", va="center", fontweight="bold")
items = [
    (24.0, "ECG noise", "white, band-limited, baseline wander,\nelectrode displacement; 50 hearts"),
    (75.0, "Simplified physics", "eikonal fit to reaction-eikonal\ntargets; 10 hearts"),
    (124.0, "Generic torso", "heart-in-torso models fitted in\n2 reference torsos; 21 hearts"),
]
rng = np.random.default_rng(fc.SEED)
for x, head, body in items:
    # icon, 8 mm wide
    xi = np.linspace(0, 8, 120)
    if head == "ECG noise":
        y = 5.75 + 2.2 * np.exp(-((xi - 4) / 0.6) ** 2) - 0.8 * np.exp(-((xi - 5) / 0.5) ** 2)
        axS.plot(x + xi, y + rng.normal(0, 0.25, xi.size), color="black", lw=0.5)
    elif head == "Simplified physics":
        axS.plot(x + xi, np.where(xi < 4, 3.5, 8.0), color="black", lw=0.6)
        axS.plot(x + xi, 3.5 + 4.5 / (1 + np.exp(-(xi - 4) / 0.35)), color=fc.OI["green"], lw=0.8, ls="--")
    else:
        th = np.linspace(0, 2 * np.pi, 100)
        axS.plot(x + 4 + 3.8 * np.cos(th), 5.75 + 3.6 * np.sin(th), color="black", lw=0.6)
        axS.plot(x + 4.8 + 1.4 * np.cos(th), 6.1 + 1.1 * np.sin(th), color=fc.OI["orange"], lw=0.8)
    axS.text(x + 10.0, 8.9, head, ha="left", va="top", fontweight="bold")
    axS.text(x + 10.0, 6.2, body, ha="left", va="top", linespacing=1.15)

# ---------------------------------------------------------------- data beside the figure
stem = "fig1_pipeline"
pd.DataFrame(rows_out).to_csv(fc.OUT / f"{stem}_ratios.csv", index=False)
ecg_long = pd.DataFrame({"t_ms": np.repeat(t[None], 8, 0).ravel(), "lead": np.repeat(LEADS, len(t)),
                         "ecg_au": ecg.ravel()})
ecg_long.to_csv(fc.OUT / f"{stem}_target_ecg.csv", index=False)
meta = dict(patient=PID, group=GROUP, selection_rule="10 primary constructions, 0 censored; v_t CV closest to "
            "upper quartile over 50 hearts", cv_v_t=row.cv_v_t, cv_v_t_q75=q75, rank_of_50=rank,
            cv_v_endo=row.cv_v_endo, ed_frame=ed, cine_slice_index=KSL, n_lv_slices=len(lv_slices),
            pixel_mm=px, mesh_cut_axis0_mm=Z, istar_shown=ISTAR, v_endo_star=theta_star[0],
            v_t_star=theta_star[1], act_max_ms=float(Tt.max()), n_nodes_1p5=len(loc15), n_nodes_1p0=len(loc10),
            fitted_ecg_stored=False,
            **{f"myo_dice_{s}": dice.loc[s, "MYO_dice"] for s in SOURCES[1:]})
pd.DataFrame([meta]).to_csv(fc.OUT / f"{stem}_meta.csv", index=False)

probs = fc.finish(fig, stem, W_MM, __file__)
print("font", fam)
print("\n".join(probs))
