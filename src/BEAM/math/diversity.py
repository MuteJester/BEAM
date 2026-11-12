"""Entropy and the Hill numbers: effective sequence counts at each order.

`D_0` is the support size, `D_1` the exponential of the entropy, `D_2` the
inverse collision probability, and the profile over `q` interpolates between
them.

These are the implementations behind `BeamModel.entropy`,
`BeamModel.hill_number` and their relatives. They take the model as their
first argument rather than `self`; the methods remain on `BeamModel`, carry
the public documentation, and delegate here. Nothing in this module imports
`BeamModel` at runtime.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .. import _cbeam as _c
from .mellin import _orders_vector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..model.facade import BeamModel


def _hill_from_mellin(q: float, log_mellin: float) -> float:
    """``D_q`` from ``ln sum P**q``, by the expression ``hill_number`` ends in.

    ``beam_model_hill_number`` returns 0.0 for a model with no live length,
    where ``log_mellin`` is ``-inf``; above ``q = 1`` the division would turn
    that into ``+inf`` instead, so it is special-cased the same way. C's
    ``exp`` saturates to ``inf`` where Python's raises, and a support too
    large for a double is an answer, not an error.
    """
    if log_mellin == -math.inf:
        return 0.0
    try:
        return math.exp(log_mellin / (1.0 - q))
    except OverflowError:
        return math.inf


def entropy(model: BeamModel, *, base: float = math.e) -> float:
    """Behind :meth:`BeamModel.entropy`."""
    if not (base > 0.0) or base == 1.0:
        raise ValueError(f"base must be positive and not 1, got {base!r}")
    h = float(_c.entropy(model._g))
    return h if base == math.e else h / math.log(base)


def hill_number(model: BeamModel, q: float) -> float:
    """Behind :meth:`BeamModel.hill_number`."""
    return float(_c.hill_number(model._g, float(q)))


def hill_numbers(model: BeamModel, orders: Iterable[float]) -> np.ndarray:
    """Behind :meth:`BeamModel.hill_numbers`."""
    return np.array([model.hill_number(float(a)) for a in orders])


def hill_profile(model: BeamModel, orders: Iterable[float]) -> np.ndarray:
    """Behind :meth:`BeamModel.hill_profile`."""
    q = _orders_vector(orders)
    out = np.empty(q.size, dtype=np.float64)
    one = np.abs(q - 1.0) < 1e-12        # `fabs(alpha - 1.0) < 1e-12` in C
    zero = q == 0.0
    rest = ~(one | zero)
    if rest.any():
        qr = q[rest]
        lm = model.log_mellin_many(qr)
        out[rest] = [_hill_from_mellin(float(a), float(v))
                     for a, v in zip(qr, lm)]
    if zero.any():
        out[zero] = model.path_count()
    if one.any():
        out[one] = math.exp(model.entropy())
    return out


def hill_curve(model: BeamModel,
               orders: Iterable[float] | None = None) -> dict[str, np.ndarray]:
    """Behind :meth:`BeamModel.hill_curve`."""
    if orders is None:
        orders = np.concatenate([np.linspace(0.0, 1.0, 11)[:-1],
                                 np.linspace(1.0, 4.0, 13)])
    orders = np.asarray(list(orders), dtype=np.float64)
    return {"orders": orders, "hill": model.hill_numbers(orders)}


def diversity_profile(model: BeamModel) -> dict[str, float]:
    """Behind :meth:`BeamModel.diversity_profile`."""
    return {
        "D0": model.hill_number(0.0),
        "D1": model.hill_number(1.0),
        "D2": model.hill_number(2.0),
        # Spelled as FlashBackGraph spells it. Cross-model summary code
        # reads this key and would otherwise have to special-case BeamModel.
        "entropy_nats": model.entropy(),
    }


def effective_diversity(model: BeamModel) -> float:
    """Behind :meth:`BeamModel.effective_diversity`."""
    return model.hill_number(1.0)
