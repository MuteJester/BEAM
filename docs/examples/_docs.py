"""Helpers the documentation examples use to save figures and data.

The example runner (docs/tools/run_examples.py) tells each example where to
write through $DOCS_FIGURES and $DOCS_DATA. Run by hand, an example writes
into the current directory instead.

    from _docs import figure, save_data      # hide

    with figure("saturation") as (fig, ax):  # writes saturation-light.svg and
        ax.plot(depths, richness)            # saturation-dark.svg
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# One palette for both themes: mid-tone colours that read on white and on
# the site's dark background: BEAM's teal and coral first.
PALETTE = ["#438f91", "#ef785f", "#6c7bd9", "#c4568a", "#8a9a2b", "#7d7d7d"]
INK = {"light": "#1f2328", "dark": "#e6edf3"}
GRID = {"light": "#d0d7de", "dark": "#3d444d"}


def _out(var: str) -> Path:
    path = Path(os.environ.get(var, "."))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _style(fig, theme: str) -> None:
    ink, grid = INK[theme], GRID[theme]
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.tick_params(colors=ink, which="both")
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(ink)
        ax.grid(True, color=grid, linewidth=0.6, alpha=0.8)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_alpha(0.0)
            for text in legend.get_texts():
                text.set_color(ink)
    for text in fig.texts:
        text.set_color(ink)


def _plotted(fig) -> list:
    """The numbers a figure shows: every line's points and every bar."""
    axes = []
    for ax in fig.axes:
        lines = [line.get_xydata().tolist() for line in ax.get_lines()]
        bars = [[float(v) for v in (p.get_x(), p.get_width(), p.get_y(), p.get_height())]
                for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
        shapes = [[float(v) for v in p.get_path().get_extents().bounds]
                  for p in ax.patches if isinstance(p, matplotlib.patches.PathPatch)]
        images = [np.asarray(im.get_array(), dtype=float).ravel().tolist() for im in ax.images]
        axes.append({"lines": lines, "bars": bars, "shapes": shapes, "images": images})
    return axes


@contextlib.contextmanager
def figure(name: str, *, size=(6.4, 3.8), ncols: int = 1, nrows: int = 1):
    """A matplotlib figure saved as `<name>-light.svg` and `<name>-dark.svg`.

    The numbers it plots are saved beside it as `<name>.data.json`, so the
    runner's --check can tell when a figure's content changed. The SVGs carry
    no date and fixed element ids, so an unchanged figure is byte-identical.
    """
    with plt.rc_context({"axes.prop_cycle": matplotlib.cycler(color=PALETTE),
                         "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "beam-docs",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(nrows, ncols, figsize=size, layout="constrained")
        yield fig, ax
        out = _out("DOCS_FIGURES")
        (out / f"{name}.data.json").write_text(
            json.dumps(_plotted(fig), separators=(",", ":")))
        for theme in ("light", "dark"):
            _style(fig, theme)
            fig.savefig(out / f"{name}-{theme}.svg", transparent=True,
                        metadata={"Date": None})
        plt.close(fig)


def _plain(obj):
    """JSON-ready copies of NumPy scalars and arrays."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def save_data(name: str, obj) -> None:
    """Write data for an interactive figure as `<name>.json`."""
    (_out("DOCS_DATA") / f"{name}.json").write_text(
        json.dumps(_plain(obj), separators=(",", ":")))


# Residue colours for logos, by chemistry: hydrophobic, aromatic, polar,
# positive, negative, and the special cases G and P.
RESIDUE_COLOURS = {
    **dict.fromkeys("AVLIM", "#8a9a2b"), **dict.fromkeys("FWY", "#6c7bd9"),
    **dict.fromkeys("STNQC", "#14a394"), **dict.fromkeys("KRH", "#c4568a"),
    **dict.fromkeys("DE", "#ef785f"), "G": "#7d7d7d", "P": "#b08a2e",
}


def draw_logo(ax, heights, residues: str) -> None:
    """A sequence logo: at each position, letters stacked to their heights (bits)."""
    font = FontProperties(family="DejaVu Sans", weight="bold")
    heights = np.asarray(heights, dtype=float)
    for i, column in enumerate(heights):
        y = 0.0
        for a in np.argsort(column):
            h = column[a]
            if h < 0.02:
                continue
            path = TextPath((0, 0), residues[a], size=1, prop=font)
            box = path.get_extents()
            place = (matplotlib.transforms.Affine2D()
                     .translate(-box.x0, -box.y0)
                     .scale(0.86 / box.width, h / box.height)
                     .translate(i - 0.43, y))
            ax.add_patch(matplotlib.patches.PathPatch(
                place.transform_path(path), color=RESIDUE_COLOURS.get(residues[a], "#7d7d7d"), lw=0))
            y += h
    ax.set_xlim(-0.6, len(heights) - 0.4)
    ax.set_ylim(0, max(1.0, float(heights.sum(axis=1).max()) * 1.05))
    ax.set_xticks(range(len(heights)), [str(i + 1) for i in range(len(heights))])
    ax.set_ylabel("bits")

