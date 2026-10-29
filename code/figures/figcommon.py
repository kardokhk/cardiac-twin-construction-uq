"""Shared settings for the I11 manuscript figures (IJNMBE target; CMPB figures use WIDTH_CMPB_MM).

Journal constants are kept here so a venue change is a one-line edit.
IJNMBE (Wiley) figure widths could not be confirmed on 2026-10-29 (author
guidelines page returned HTTP 403), so the fallback widths are
used: 174 mm double column, 84 mm single column.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import figstyle  # noqa: E402  (sets the Agg backend on import)
import figqa  # noqa: E402
import fontguard  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "manuscript"
QA_DIR = Path(os.environ.get("FIG_QA_DIR", "/tmp")) # derived QA images, not deliverables
RES = ROOT / "results"
ANALYSIS = RES / "2026-10-28_analysis"

WIDTH_DOUBLE_MM = 174.0   # v02 figures (IJNMBE fallback); kept so they rebuild unchanged
WIDTH_SINGLE_MM = 84.0
# CMPB (Elsevier) full-page width, artwork-sizing page (90 / 140 / 190 mm), rebuild 2026-10-29.
# The v03, pictorial and supplementary CMPB figures were designed at 174 mm; heights are scaled by the
# same factor so each figure keeps its aspect ratio (cmpb_height).
WIDTH_CMPB_MM = 190.0
CMPB_SCALE = WIDTH_CMPB_MM / WIDTH_DOUBLE_MM
MAX_HEIGHT_CMPB_MM = 240.0
MIN_TEXT_PT = 6.0          # Elsevier: no text, sub- or superscripts included, below 6 pt
MAX_HEIGHT_MM = 170.0
BODY_PT = 7.0
PANEL_PT = 8.0
SEED = 20261028
N_BOOT = 2000

# Okabe-Ito; one colour per condition across all figures.
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73", purple="#CC79A7",
          orange="#E69F00", sky="#56B4E9", yellow="#F0E442", black="#000000")
GROUPS = ["NOR", "DCM", "HCM", "MINF", "RV"]
GROUP_STYLE = {  # colour + marker, so groups survive greyscale
    "NOR": (OI["black"], "o"),
    "DCM": (OI["blue"], "s"),
    "HCM": (OI["vermillion"], "^"),
    "MINF": (OI["green"], "D"),
    "RV": (OI["purple"], "v"),
}
GROUP_LABEL = {"NOR": "NOR", "DCM": "DCM", "HCM": "HCM", "MINF": "MINF", "RV": "ARV (RV)"}
LOSS_STYLE = {  # correlation loss primary, amplitude secondary
    "corr": dict(color=OI["blue"], marker="o", ls="-", label="Correlation loss"),
    "amp": dict(color=OI["vermillion"], marker="s", ls="--", label="Amplitude loss"),
}
LAYER_STYLE = {
    "A2": dict(color=OI["blue"], label="A2 voxel layer (primary)"),
    "B1": dict(color=OI["orange"], label="B1 smooth layer (sensitivity)"),
}
PARAM_LABEL = {"v_endo": r"$v_\mathrm{endo}$", "v_t": r"$v_\mathrm{t}$"}


def cmpb_height(design_height_mm: float) -> float:
    """Height at 190 mm width for a figure designed at 174 mm, same aspect ratio."""
    h = design_height_mm * CMPB_SCALE
    assert h <= MAX_HEIGHT_CMPB_MM
    return h


def floor_mathtext_scripts(base_pt: float = BODY_PT, min_pt: float = MIN_TEXT_PT) -> float:
    """Raise matplotlib's mathtext sub/superscript scale so first-level scripts are >= min_pt.

    matplotlib shrinks scripts by the module constant _mathtext.SHRINK_FACTOR (0.7), so a 7 pt v_t has a
    4.9 pt subscript. The constant is read at layout time, so setting it before any text is drawn applies
    to the whole figure. Layout only; no plotted value changes.
    """
    from matplotlib import _mathtext
    f = max(_mathtext.SHRINK_FACTOR, min_pt / base_pt + 1e-3)
    _mathtext.SHRINK_FACTOR = f
    return f


def setup(scripts_floor: bool = False) -> str:
    fam = figstyle.use_print_style()
    if scripts_floor:
        floor_mathtext_scripts()
    plt.rcParams.update({
        "font.size": BODY_PT, "axes.labelsize": BODY_PT, "axes.titlesize": BODY_PT,
        "xtick.labelsize": BODY_PT, "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "mathtext.cal": fam, "mathtext.sf": fam, "mathtext.tt": fam,
    })
    return fam


def results_json() -> dict:
    with open(ANALYSIS / "results.json") as fh:
        return json.load(fh)


def panel_letters(axes, letters=None, dx=-0.14, dy=1.02):
    figstyle.panel_labels(axes, letters, dx=dx, dy=dy, size=PANEL_PT)


def finish(fig, stem: str, width_mm: float, script: str) -> list[str]:
    """Render strictly, save PDF + 600 dpi PNG, run programmatic QA, write QA images."""
    OUT.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fontguard.render_strict(fig, str(pdf))
    fontguard.render_strict(fig, str(png), dpi=600)
    from PIL import Image
    im = Image.open(png)  # flatten RGBA onto white: journals reject alpha channels
    Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA")).convert("RGB").save(
        png, dpi=(600, 600))
    assert os.path.getmtime(png) > os.path.getmtime(script), "stale PNG"
    problems = figqa.report(fig)
    qa_png = QA_DIR / f"{stem}.png"
    import shutil
    shutil.copy(png, qa_png)
    figqa.greyscale_and_downscale(str(qa_png), width_mm)
    with open(QA_DIR / f"{stem}_qa.txt", "w") as fh:
        fh.write("\n".join(problems) + "\n")
    return problems
