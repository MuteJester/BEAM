"""The distribution of `-ln P(s)` under a model: its moments and its shape.

Moments and cumulants straight from the trellis, summary statistics derived
from them, the exact reachable range, the per-depth decomposition, and the
length-mixture of normals that gives a density and a distribution function.

These are the implementations behind `BeamModel.surprisal_moments` and its
seven relatives. They take the model as their first argument rather than
`self`; the methods remain on `BeamModel`, carry the public documentation,
and delegate here. Nothing in this module imports `BeamModel` at runtime.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .. import _cbeam as _c
from ..constants import BEAM_MAX_MOMENT
from ..model.validation import _check_int, _is_scalar_arg
from .saddlepoint import _normal_cdf

if TYPE_CHECKING:
    from typing import Any

    from ..model.facade import BeamModel


def surprisal_moments(model: BeamModel, *, q: float = 1.0,
                      order: int = 4) -> np.ndarray:
    """Behind :meth:`BeamModel.surprisal_moments`."""
    order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
    return np.asarray(
        _c.surprisal_moments(model._g, float(q), order),
        dtype=np.float64)


def surprisal_cumulants(model: BeamModel, *, q: float = 1.0,
                        order: int = 4) -> np.ndarray:
    """Behind :meth:`BeamModel.surprisal_cumulants`."""
    order = _check_int("order", order, 1, BEAM_MAX_MOMENT)
    return np.asarray(
        _c.surprisal_cumulants(model._g, float(q), order),
        dtype=np.float64)


def surprisal_stats(model: BeamModel, *, q: float = 1.0) -> dict[str, float]:
    """Behind :meth:`BeamModel.surprisal_stats`."""
    k = model.surprisal_cumulants(q=q, order=4)
    var = float(k[1])
    sd = math.sqrt(var) if var > 0 else 0.0
    return {
        "mean": float(k[0]),
        "variance": var,
        "std": sd,
        "skewness": float(k[2]) / sd ** 3 if sd > 0 else 0.0,
        "excess_kurtosis": float(k[3]) / var ** 2 if var > 0 else 0.0,
    }


def surprisal_range(model: BeamModel) -> tuple[float, float]:
    """Behind :meth:`BeamModel.surprisal_range`."""
    return tuple(_c.surprisal_range(model._g))


def surprisal_by_depth(model: BeamModel, *, q: float = 1.0) -> dict[str, Any]:
    """Behind :meth:`BeamModel.surprisal_by_depth`."""
    depths, konst = _c.surprisal_by_depth(model._g, float(q))
    depths = np.asarray(depths, dtype=np.float64)
    return {
        "depth": np.arange(len(depths)),
        "by_depth": depths,
        "length_constant": float(konst),
        "total": float(depths.sum() + konst),
    }


def surprisal_distribution(model: BeamModel) -> dict[str, np.ndarray]:
    """Behind :meth:`BeamModel.surprisal_distribution`."""
    lm = model.length_marginals()
    keep = lm["generated"] > 0
    return {
        "length": lm["length"][keep],
        "weight": lm["generated"][keep],
        "mean": lm["mean_surprisal"][keep],
        "variance": lm["variance"][keep],
        "std": np.sqrt(lm["variance"][keep]),
    }


def surprisal_pdf(model: BeamModel, x) -> np.ndarray:
    """Behind :meth:`BeamModel.surprisal_pdf`."""
    d = model.surprisal_distribution()
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(xs)
    for w, mu, sd in zip(d["weight"], d["mean"], d["std"]):
        if sd <= 0:
            continue
        z = (xs - mu) / sd
        out += w * np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))
    return float(out[0]) if _is_scalar_arg(x) else out


def surprisal_cdf(model: BeamModel, x) -> np.ndarray:
    """Behind :meth:`BeamModel.surprisal_cdf`."""
    d = model.surprisal_distribution()
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(xs)
    for w, mu, sd in zip(d["weight"], d["mean"], d["std"]):
        if sd <= 0:
            out += w * (xs >= mu)
            continue
        out += w * np.array([_normal_cdf((v - mu) / sd) for v in xs])
    return float(out[0]) if _is_scalar_arg(x) else out
