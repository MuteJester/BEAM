"""Surprisal spectra: how a model's probability mass is spread over `-ln P`.

One spectrum at a length or pooled, the counting and generated spectra on a
shared grid, and every length's spectrum as one matrix.

These are the implementations behind `BeamModel.spectrum`,
`BeamModel.spectrum_pair` and `BeamModel.spectra_by_length`. They take the
model as their first argument rather than `self`; the methods remain on
`BeamModel`, carry the public documentation, and delegate here. Bin and
length validation stays on the model -- `_check_bins` and `_check_length`
have callers outside the math domain -- and is reached back through it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import _cbeam as _c

if TYPE_CHECKING:
    from ..model.facade import BeamModel


def spectrum(model: BeamModel, *, measure: str = "generated",
             length: int | None = None, bins: int = 1024,
             spacing: float | None = None) -> dict:
    """Behind :meth:`BeamModel.spectrum`."""
    if measure not in ("counting", "generated"):
        raise ValueError(
            f"measure must be 'counting' or 'generated', got {measure!r}")
    bins = model._check_bins(bins, spacing)
    if spacing is not None and not (float(spacing) > 0.0):
        raise ValueError(
            f"spacing must be positive, got {spacing!r}; pass None to have "
            f"it chosen from accumulator_max()")
    w, ds = _c.spectrum(
        model._g, measure == "counting",
        model._check_length(length, allow_none=True),
        bins, 0.0 if spacing is None else float(spacing))
    w = np.asarray(w, dtype=np.float64)
    return {
        "surprisal": np.arange(bins) * ds,
        "weights": w,
        "spacing": ds,
        "total": float(w.sum()),
        "measure": measure,
    }


def spectrum_pair(model: BeamModel, *, length: int | None = None,
                  bins: int = 1024, spacing: float | None = None) -> dict:
    """Behind :meth:`BeamModel.spectrum_pair`."""
    bins = model._check_bins(bins, spacing)
    if spacing is not None and not (float(spacing) > 0.0):
        raise ValueError(
            f"spacing must be positive, got {spacing!r}; pass None to have "
            f"it chosen from accumulator_max()")
    cb, gb, ds = _c.spectrum_pair(
        model._g, model._check_length(length, allow_none=True), bins,
        0.0 if spacing is None else float(spacing))
    return {
        "surprisal": np.arange(bins) * ds,
        "counting": np.frombuffer(cb, dtype=np.float64).copy(),
        "generated": np.frombuffer(gb, dtype=np.float64).copy(),
        "spacing": ds,
    }


def spectra_by_length(model: BeamModel, *, measure: str = "generated",
                      max_length: int | None = None, bins: int = 1024,
                      spacing: float | None = None) -> dict:
    """Behind :meth:`BeamModel.spectra_by_length`."""
    if measure not in ("counting", "generated"):
        raise ValueError(
            f"measure must be 'counting' or 'generated', got {measure!r}")
    bins = model._check_bins(bins, spacing)
    if spacing is not None and not (float(spacing) > 0.0):
        raise ValueError(
            f"spacing must be positive, got {spacing!r}; pass None to have "
            f"it chosen from accumulator_max()")
    hi = int(model.config["max_length"])
    if max_length is None:
        want = 0                      # 0 means "the model's own maximum"
    else:
        max_length = int(max_length)
        if not 0 <= max_length <= hi:
            raise ValueError(
                f"max_length must be between 0 and {hi}, got {max_length}")
        want = max_length
    b, rows, cols, ds = _c.spectra_by_length(
        model._g, measure == "counting", want, bins,
        0.0 if spacing is None else float(spacing))
    return {
        "surprisal": np.arange(cols) * ds,
        "weights": np.frombuffer(b, dtype=np.float64).reshape(rows,
                                                              cols).copy(),
        "spacing": ds,
        "measure": measure,
    }
