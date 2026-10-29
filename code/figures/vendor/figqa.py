"""Programmatic pre-checks that complement visual inspection of a matplotlib figure."""
from __future__ import annotations

import itertools

from PIL import Image


UNIT_HINT = ("(", "[", "/", "%", "^{-", "$^{", "log", "z-score", "a.u.",
             "count", "fraction", "proportion", "index", "no.")


def _visible_ticklabels(ax):
    """Tick labels for ticks inside the view limits only.

    matplotlib keeps Text artists for ticks outside the view; those sit at
    nonsense display coordinates and produce phantom clipping reports.
    """
    out = []
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = sorted(lim)
        for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
            if lo - 1e-9 <= loc <= hi + 1e-9:
                out.append(lab)
    return out


def _texts(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += [ax.title, ax.xaxis.label, ax.yaxis.label]
        out += list(ax.texts)
        out += _visible_ticklabels(ax)
        if ax.get_legend() is not None:
            out += ax.get_legend().get_texts()
    return [t for t in out if t.get_text().strip() and t.get_visible()]


def report(fig, *, min_pt=6.0, pad_px=0.5):
    """min_pt defaults to 6.0: the working floor in the checklist. 5 pt is the
    absolute journal minimum (Nature, NAR), not a target. Lower it only for a
    journal that genuinely permits 5 pt, and never to make a figure pass."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    problems = []

    boxes = []
    for t in _texts(fig):
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        boxes.append((t, bb))

    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        if ba.overlaps(bb):
            ix = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            iy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if ix > pad_px and iy > pad_px:
                problems.append(
                    f"TEXT OVERLAP: {a.get_text()!r} x {b.get_text()!r} "
                    f"({ix:.1f}x{iy:.1f} px)")

    fb = fig.bbox
    for t, bb in boxes:
        if bb.x0 < fb.x0 - pad_px or bb.y0 < fb.y0 - pad_px \
           or bb.x1 > fb.x1 + pad_px or bb.y1 > fb.y1 + pad_px:
            problems.append(f"CLIPPED AT FIGURE EDGE: {t.get_text()!r}")

    sizes = {round(t.get_fontsize(), 2) for t, _ in boxes}
    for s in sorted(sizes):
        if s < min_pt:
            problems.append(f"FONT TOO SMALL: {s} pt (< {min_pt} pt)")
    if len(sizes) > 4:
        problems.append(f"FONT SIZE SPRAWL: {len(sizes)} distinct sizes {sorted(sizes)}")

    for i, ax in enumerate(fig.axes):
        off = ax.yaxis.get_offset_text().get_text() or ax.xaxis.get_offset_text().get_text()
        if off:
            problems.append(f"ax[{i}]: scientific-notation offset text {off!r} "
                            "present - fold the factor into the axis label instead")
        if not ax.get_xlabel():
            problems.append(f"ax[{i}]: missing x label")
        if not ax.get_ylabel():
            problems.append(f"ax[{i}]: missing y label")
        for lab in ("x", "y"):
            txt = ax.get_xlabel() if lab == "x" else ax.get_ylabel()
            if txt and not any(h in txt.lower() for h in UNIT_HINT):
                problems.append(f"ax[{i}]: {lab} label {txt!r} may state no unit "
                                "- confirm the quantity is dimensionless")
        leg = ax.get_legend()
        if leg is not None:
            labs = [t.get_text() for t in leg.get_texts()]
            if len(labs) != len(set(labs)):
                problems.append(f"ax[{i}]: duplicate legend entries {labs}")
        for coll in ax.collections + ax.lines:
            try:
                bb = coll.get_window_extent(r)
            except Exception:
                continue
            ab = ax.bbox
            if bb.width and (bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1
                             or bb.y0 < ab.y0 - 1 or bb.y1 > ab.y1 + 1):
                problems.append(f"ax[{i}]: data artist {type(coll).__name__} "
                                "extends past the axes box - data may be clipped")

    w_in, h_in = fig.get_size_inches()
    problems.append(f"INFO: {w_in*25.4:.1f} x {h_in*25.4:.1f} mm, "
                    f"base {min(sizes) if sizes else 0}-{max(sizes) if sizes else 0} pt, "
                    f"{dpi:.0f} dpi")
    return problems


def greyscale_and_downscale(png_path, target_mm, dpi_screen=110):
    im = Image.open(png_path)
    grey = png_path.replace(".png", "_grey.png")
    im.convert("L").save(grey)
    px = max(1, int(round(target_mm / 25.4 * dpi_screen)))
    actual = png_path.replace(".png", "_actualsize.png")
    im.convert("RGB").resize((px, max(1, int(im.height * px / im.width))),
                             Image.LANCZOS).save(actual)
    return grey, actual
