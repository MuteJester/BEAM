"""Argument checks and the scalar/array calling convention.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._simulation_result import SimulationResult


def _check_int(name: str, value: Any, lo: int, hi: int) -> int:
    """Range-check a configuration integer before it reaches the C layer.

    The C entry point takes these as ``uint8_t``, so an out-of-range value is
    silently truncated rather than rejected: ``max_length=283`` becomes 27 and
    ``order=260`` becomes 4, both of which then build a perfectly healthy
    model that is not the one that was asked for. Checking here is the only
    place that difference can still be seen.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _is_scalar_arg(x: Any) -> bool:
    """True when ``x`` is a single number rather than a container.

    One convention for the whole module: an argument that arrives as a scalar
    leaves as a ``float``, and anything with a shape leaves as an array of the
    same shape. A one-element list is a container and stays one.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
