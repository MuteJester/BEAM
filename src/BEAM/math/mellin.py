"""The Mellin transform of a model's sequence distribution.

`ln sum_s P(s)**q`, at a real order, at many real orders in one pass, at
complex orders, and along a vertical line in the complex plane. Every
diversity quantity in the library is built from the first of these.

These are the implementations behind `BeamModel.log_mellin` and its three
relatives. They take the model as their first argument rather than `self`;
the methods remain on `BeamModel`, carry the public documentation, and
delegate here. Nothing in this module imports `BeamModel` at runtime: it
uses only the model's capsule and its public methods, so the dependency runs
one way, from the facade to here.
"""
from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

import numpy as np

from .. import _cbeam as _c
from ..errors import BeamTypeError, BeamValueError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..model.facade import BeamModel


def _orders_vector(orders: Iterable[float]) -> np.ndarray:
    """``orders`` as a fresh one-dimensional float64 array of finite values.

    Fresh on purpose, not merely contiguous: the C side reads it with the GIL
    released, and an array the caller still holds could be written by another
    thread while it does.
    """
    if isinstance(orders, (str, bytes)):
        raise BeamTypeError("orders must be a sequence of numbers, not a string")
    try:
        seq = orders if isinstance(orders, np.ndarray) else list(orders)
        q = np.array(seq, dtype=np.float64)
    except (TypeError, ValueError):
        raise BeamTypeError(
            f"orders must be a sequence of numbers, got "
            f"{type(orders).__name__}") from None
    if q.ndim != 1:
        raise BeamValueError(
            f"orders must be one-dimensional, got shape {q.shape}")
    bad = q[~np.isfinite(q)]
    if bad.size:
        raise BeamValueError(
            f"orders must be finite, got {bad[:3].tolist()}")
    return np.ascontiguousarray(q)


def log_mellin(model: BeamModel, q: float) -> float:
    """Behind :meth:`BeamModel.log_mellin`."""
    return float(_c.log_mellin(model._g, float(q)))


def log_mellin_many(model: BeamModel, orders: Iterable[float]) -> np.ndarray:
    """Behind :meth:`BeamModel.log_mellin_many`."""
    q = _orders_vector(orders)
    if q.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.frombuffer(_c.log_mellin_many(model._g, q),
                         dtype=np.float64).copy()


def log_mellin_complex(model: BeamModel, orders: Iterable[complex]) -> np.ndarray:
    """Behind :meth:`BeamModel.log_mellin_complex`."""
    if isinstance(orders, (str, bytes)):
        raise BeamTypeError("orders must be a sequence of numbers, not a string")
    try:
        seq = orders if isinstance(orders, np.ndarray) else list(orders)
        q = np.array(seq, dtype=np.complex128)
    except (TypeError, ValueError):
        raise BeamTypeError(
            f"orders must be a sequence of numbers, got "
            f"{type(orders).__name__}") from None
    if q.ndim != 1:
        raise BeamValueError(
            f"orders must be one-dimensional, got shape {q.shape}")
    bad = q[~np.isfinite(q)]
    if bad.size:
        raise BeamValueError(
            f"orders must be finite, got {bad[:3].tolist()}")
    if q.size == 0:
        return np.empty(0, dtype=np.complex128)
    # Fresh arrays: the C side reads them with the GIL released.
    re = np.array(q.real, dtype=np.float64)
    im = np.array(q.imag, dtype=np.float64)
    return np.frombuffer(_c.log_mellin_complex(model._g, re, im),
                         dtype=np.complex128).copy()


def log_mellin_line(model: BeamModel, re: float, im_start: float,
                    im_step: float, n: int) -> np.ndarray:
    """Behind :meth:`BeamModel.log_mellin_line`."""
    for name, v in (("re", re), ("im_start", im_start), ("im_step", im_step)):
        if not math.isfinite(float(v)):
            raise BeamValueError(f"{name} must be finite, got {v!r}")
    n = operator.index(n)
    if n < 0:
        raise BeamValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return np.empty(0, dtype=np.complex128)
    return np.frombuffer(
        _c.log_mellin_line(model._g, float(re), float(im_start),
                           float(im_step), n),
        dtype=np.complex128).copy()
