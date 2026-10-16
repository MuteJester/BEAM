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
import matplotlib.pyplot as plt  # noqa: E402

# One palette for both themes: mid-tone colours that read on white and on
# the site's dark background. The first is the site's teal accent.
PALETTE = ["#14a394", "#e07a3f", "#6c7bd9", "#c4568a", "#8a9a2b", "#7d7d7d"]
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


@contextlib.contextmanager
def figure(name: str, *, size=(6.4, 3.8), ncols: int = 1):
    """A matplotlib figure saved as `<name>-light.svg` and `<name>-dark.svg`."""
    with plt.rc_context({"axes.prop_cycle": matplotlib.cycler(color=PALETTE),
                         "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(1, ncols, figsize=size, layout="constrained")
        yield fig, ax
        out = _out("DOCS_FIGURES")
        for theme in ("light", "dark"):
            _style(fig, theme)
            fig.savefig(out / f"{name}-{theme}.svg", transparent=True)
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
