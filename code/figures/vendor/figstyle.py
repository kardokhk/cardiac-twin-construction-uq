"""Print-ready matplotlib style for journal figures. No LaTeX required."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM_PER_IN = 25.4
COL = {"single": 89.0, "onehalf": 136.0, "double": 183.0}

# Nature's page depth is the binding constraint at 170 mm. Some journals allow
# more (NAR 230 mm); pass max_height_mm to figure_mm() to raise it deliberately
# rather than editing this default.
MAX_HEIGHT_MM = 170.0

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

BASE_PT = 7.0

# Same chain as fontguard.HELVETICA_CHAIN, and deliberately WITHOUT Helvetica
# Neue: Windows does not alias it to Arial and may substitute something unrelated.
# DejaVu Sans is last only as a tripwire; its metrics are wider, so reaching it
# changes every label's width.
SANS_CHAIN = ["Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arimo",
              "Liberation Sans", "Arial", "DejaVu Sans"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": SANS_CHAIN,
    "text.usetex": False,
    # mathtext.rm/it/bf are rewritten by use_print_style() to whichever family
    # actually resolved, so maths does not silently come from DejaVu.
    "mathtext.fontset": "custom",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "mathtext.default": "regular",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT - 1,
    "ytick.labelsize": BASE_PT - 1,
    "legend.fontsize": BASE_PT - 1,
    "figure.titlesize": BASE_PT + 1,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,          # Science: nothing below 0.5 pt at final size
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "patch.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlelocation": "left",
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.3,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.transparent": False,
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.01,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "pdf.compression": 6,
    "agg.path.chunksize": 10000,
}


def _resolves(name: str, weight: str = "normal"):
    from matplotlib import font_manager as fm

    try:
        return fm.findfont(fm.FontProperties(family=name, weight=weight),
                           fallback_to_default=False)
    except ValueError:
        return None


def resolve_family(chain: list[str] = None, *, require_bold: bool = True) -> str:
    """First family in the chain that genuinely exists and has a distinct bold.

    Raises if only DejaVu Sans resolves, because reaching the tripwire means the
    figure will not use the requested typeface and every label will be wider
    than intended. Install a metric-compatible clone instead of accepting it.

    require_bold matters on two platforms for opposite reasons. Linux ships each
    weight as its own file, so a partial install can leave bold missing. macOS
    ships Helvetica as a .ttc collection whose bold resolves to the same face,
    so bold panel letters render at regular weight and nothing warns. Either way
    a family whose bold is not distinct is unusable for panel labels.
    """
    chain = chain or SANS_CHAIN
    skipped = []
    for name in chain:
        if name == "DejaVu Sans":
            break
        regular = _resolves(name)
        if regular is None:
            continue
        if require_bold:
            bold = _resolves(name, "bold")
            if bold is None or bold == regular:
                skipped.append(name)
                continue
        return name
    if skipped:
        raise RuntimeError(
            "Found " + ", ".join(skipped) + " but bold resolves to the same "
            "file as regular, so bold panel letters would render at regular "
            "weight. Install a clone that ships separate weight files "
            "(Liberation Sans, Arimo, Nimbus Sans), then clear the matplotlib "
            "font cache. Pass require_bold=False only if the figure has no "
            "bold text at all. See references/figures.md.")
    raise RuntimeError(
        "No Helvetica-metric sans font found; matplotlib would fall back to "
        "DejaVu Sans and silently change every label width. Install one of "
        "Nimbus Sans, TeX Gyre Heros, Liberation Sans or Arimo, then clear the "
        "matplotlib font cache. See references/figures.md.")


def use_print_style(family: str = None) -> str:
    """Apply the print style and point mathtext at the family that resolved."""
    plt.rcParams.update(RC)
    family = family or resolve_family()
    plt.rcParams["font.sans-serif"] = [family] + [
        n for n in SANS_CHAIN if n != family]
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = f"{family}:italic"
    plt.rcParams["mathtext.bf"] = f"{family}:bold"
    return family


def mm(*vals: float) -> tuple[float, ...] | float:
    out = tuple(v / MM_PER_IN for v in vals)
    return out[0] if len(out) == 1 else out


def figure_mm(width_mm: float, height_mm: float,
              *, max_height_mm: float = MAX_HEIGHT_MM, **kw):
    assert height_mm <= max_height_mm, (
        f"height {height_mm} mm exceeds {max_height_mm} mm; pass "
        "max_height_mm= explicitly if the target journal allows more")
    return plt.subplots(figsize=(width_mm / MM_PER_IN, height_mm / MM_PER_IN), **kw)


def panel_labels(axes, letters=None, *, dx=-0.16, dy=1.04, size=8.0):
    letters = letters or [chr(65 + i) for i in range(len(axes))]
    for ax, letter in zip(axes, letters):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
                fontweight="bold", va="bottom", ha="left")


def save_all(fig, stem: str, *, dpi_raster: int = 600) -> list[str]:
    from PIL import Image

    paths = [f"{stem}.pdf", f"{stem}.png"]
    fig.savefig(paths[0])                      # vector master, fonttype 42
    fig.savefig(paths[1], dpi=dpi_raster)      # the preview image
    tif = f"{stem}.tif"
    # matplotlib's own TIFF writer emits RGBA; PLOS and ASM reject alpha
    # channels and layered files, so flatten onto white and re-save with LZW.
    im = Image.open(paths[1])
    Image.alpha_composite(Image.new("RGBA", im.size, "white"),
                          im.convert("RGBA")).convert("RGB").save(
        tif, compression="tiff_lzw", dpi=(dpi_raster, dpi_raster))
    paths.append(tif)
    return paths
