"""Detect matplotlib's SILENT font substitution. Linux/headless safe.

matplotlib reports a missing font family through the `matplotlib.font_manager`
LOGGER, not through `warnings`, so `warnings.catch_warnings()` never sees it.
A caller must either capture that logger or use strict findfont.
"""
from __future__ import annotations

import logging

import matplotlib
from matplotlib import font_manager as fm

HELVETICA_CHAIN = [
    "Helvetica",        # genuine Linotype Helvetica, rarely present on Linux
    "Nimbus Sans",      # URW clone, metric-compatible  (apt: fonts-urw-base35)
    "TeX Gyre Heros",   # GUST clone, metric-compatible  (apt: fonts-texgyre)
    "Arimo",            # Croscore Arial-metric clone   (apt: fonts-croscore)
    "Liberation Sans",  # Arial-metric clone            (apt: fonts-liberation)
    "Arial",
    "DejaVu Sans",      # matplotlib's bundled last resort - WIDER metrics
]


def available(names: list[str]) -> list[str]:
    have = set(fm.get_font_names())          # matplotlib >= 3.6
    return [n for n in names if n in have]


INSTALL_HINT = (
    "Install one, e.g.\n"
    "  sudo apt-get install -y fonts-urw-base35 fonts-texgyre fonts-liberation "
    "fonts-croscore\n"
    "  sudo dnf install -y urw-base35-fonts liberation-sans-fonts\n"
    "  # no root: drop .ttf/.otf into ~/.local/share/fonts\n"
    "then: fc-cache -fv && rm -f \"$(python3 -c 'import matplotlib;"
    "print(matplotlib.get_cachedir())')\"/fontlist-*.json")


def _resolves(name: str, weight: str = "normal"):
    try:
        return fm.findfont(fm.FontProperties(family=name, weight=weight),
                           fallback_to_default=False)
    except ValueError:
        return None


def resolve_or_fail(chain: list[str] = None, *, require_bold: bool = True) -> str:
    """Return the first family that genuinely exists; raise if only DejaVu does.

    Kept behaviourally identical to figstyle.resolve_family() so the two cannot
    disagree about which family is acceptable. require_bold rejects a family
    whose bold resolves to the same file as its regular face, which is the macOS
    .ttc collection case and the partial-install case on Linux; either way bold
    panel letters would render at regular weight with no warning.
    """
    chain = chain or HELVETICA_CHAIN
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
            "Found " + ", ".join(skipped) + " but bold resolves to the same file "
            "as regular, so bold panel letters would render at regular weight.\n"
            + INSTALL_HINT)
    raise RuntimeError("No Helvetica-metric sans font found. " + INSTALL_HINT)


class FontSubstitutionError(RuntimeError):
    pass


class _Trap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.hits: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "findfont" in msg or "not found" in msg:
            self.hits.append(msg)


def render_strict(fig, path: str, *, dpi: int = 600, strict: bool = True) -> list[str]:
    """Draw and save, treating any font substitution as a DEFECT, not noise."""
    trap = _Trap()
    for name in ("matplotlib.font_manager", "matplotlib"):
        logging.getLogger(name).addHandler(trap)
    prev = matplotlib.set_loglevel("warning")
    try:
        fig.canvas.draw()
        fig.savefig(path, dpi=dpi)
    finally:
        for name in ("matplotlib.font_manager", "matplotlib"):
            logging.getLogger(name).removeHandler(trap)
    uniq = sorted(set(trap.hits))
    if uniq and strict:
        raise FontSubstitutionError(
            "matplotlib silently substituted a font; the rendered figure does "
            "NOT use the requested family:\n  " + "\n  ".join(uniq))
    return uniq
