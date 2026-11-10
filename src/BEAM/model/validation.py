"""Argument checks and the scalar/array calling convention.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._simulation_result import SimulationResult
from ..errors import BeamTypeError, BeamValueError


def _check_int(name: str, value: Any, lo: int, hi: int, *,
               context: str = "") -> int:
    """Convert a configuration integer and check it lies in ``lo..hi``.

    This is the early, friendly gate; the C layer validates again. Raises
    :class:`BeamTypeError` for something that is not an integer and
    :class:`BeamValueError` for an out-of-range one. ``context`` is appended
    to the range in the message, for a bound that depends on another argument.
    Both configuration bounds are per-alphabet -- 2..4 orders and 1..255
    residues for amino acids against 2..9 and 1..512 for nucleotides -- so
    without it a refusal quotes a number without the argument that chose it.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        # BeamTypeError, which IS a TypeError, so `except TypeError` around a
        # build keeps working and `except BEAM.BeamTypeError` starts working.
        # This is the one place in the library that raises it: the C layer has
        # no code for a type error, because by the time a value reaches C it
        # has already been converted.
        raise BeamTypeError(
            f"{name} must be an integer, got {value!r}") from None
    if not lo <= v <= hi:
        raise BeamValueError(f"{name} must be between {lo} and {hi}"
                             f"{f' {context}' if context else ''}, got {v}")
    return v


def _is_scalar_arg(x: Any) -> bool:
    """True when ``x`` is a single number rather than a container.

    One convention for the whole module: an argument that arrives as a scalar
    leaves as a ``float``, and anything with a shape leaves as an array of the
    same shape. A one-element list is a container and stays one.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
