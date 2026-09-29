"""Configuration assembly: what a caller may set and what it defaults to.
"""
from __future__ import annotations

from typing import Any


from ..constants import BEAM_MAX_LENGTH_CEILING, BEAM_MAX_MOMENT
from . import alphabets
from .validation import _check_int

from .._simulation_result import SimulationResult


#: The strengths a model reports before it has estimated any, in
#: pseudo-observations. The same numbers as BEAM_DEFAULT_ALPHA and
#: BEAM_DEFAULT_ALPHA_LENGTH in include/beam/model.h, which a test holds them
#: to. A model built without ``alpha=`` estimates its own from its counts.
DEFAULT_ALPHA = 10.0
DEFAULT_ALPHA_LENGTH = 10.0


def _is_auto(value) -> bool:
    """``None`` or ``"auto"``: leave the strength to the data."""
    return value is None or (isinstance(value, str) and value == "auto")


def _alpha_levels(alpha, order: int):
    """One positive, finite strength per level 0..order, or None for the
    strengths the model estimates from its counts."""
    import math
    if _is_auto(alpha):
        return None
    if isinstance(alpha, str):
        raise ValueError(f"alpha={alpha!r}: pass numbers, or 'auto' to "
                         f"estimate the strengths from the counts")
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        levels = (float(alpha),) * (order + 1)
    else:
        levels = tuple(float(a) for a in alpha)
        if len(levels) != order + 1:
            raise ValueError(
                f"alpha has {len(levels)} entries, but an order-{order} model "
                f"has {order + 1} levels (0..order): pass one number for every "
                f"level, or exactly one per level")
    for k, a in enumerate(levels):
        if not (math.isfinite(a) and a > 0.0):
            raise ValueError(
                f"alpha[{k}]={a!r}: every prior strength must be finite and "
                f"positive, since a zero would make an unobserved residue "
                f"impossible at that level")
    return levels


def _alpha_length(alpha_length):
    """A positive, finite strength, or None to estimate it."""
    import math
    if _is_auto(alpha_length):
        return None
    alpha_length = float(alpha_length)
    if not (math.isfinite(alpha_length) and alpha_length > 0.0):
        raise ValueError(f"alpha_length={alpha_length!r} must be finite and "
                         f"positive")
    return alpha_length


def _config_args(*, order, max_length, weights, alpha=None, alpha_length=None,
                 sparse=None, alphabet=alphabets.DEFAULT_ALPHABET) -> tuple:
    """Validate a build configuration and flatten it for the C entry points.

    Shared by :class:`BeamModel` and :class:`Builder` so a streamed build and a
    one-shot build cannot be validated differently, which would make the
    "byte-identical" claim on :meth:`Builder.build` a coincidence.

    `order` and `max_length` may be None, which means "whatever this
    alphabet's default is". Those defaults are read from the C descriptor, not
    kept here, so the two layers cannot drift apart.
    """
    info = alphabets.info(alphabet)
    if order is None:
        order = info["default_order"]
    if max_length is None:
        max_length = info["default_max_length"]

    # The order ceiling is the alphabet's own, because the real bound is on
    # radix**order and the two radices are far apart: 21**9 does not fit the
    # uint32 the context count lives in, while 6**9 fits comfortably.
    order = _check_int("order", order, 2, info["max_order"])
    max_length = _check_int("max_length", max_length, 2,
                            BEAM_MAX_LENGTH_CEILING)
    weights = tuple(float(w) for w in weights)
    alpha = _alpha_levels(alpha, order)
    alpha_length = _alpha_length(alpha_length)

    # The weights SHOULD sum to 1 and are not refused if they do not: (1, 0)
    # and (0, 1) are how each anchor's chain is measured on its own, which is
    # what separates a partition function below 1 into the two causes that can
    # put it there. Off 1, log W scales with the length, so within-length
    # ranking is untouched and cross-length comparison is not meaningful.
    # Sparse by default: a dense model materialises every row, which is
    # gigabytes at order 4, and represents exactly the same model.
    sparse = True if sparse is None else bool(sparse)

    # The alphabet is appended rather than placed first, where it belongs, so
    # every existing position in the tuple keeps its meaning.
    return (order, max_length, weights, alpha, alpha_length, sparse, alphabet)


def _config_args_from_dict(cfg: dict[str, Any]) -> tuple:
    """The same tuple, rebuilt from what :attr:`BeamModel.config` reports."""
    return _config_args(alphabet=cfg.get("alphabet",
                                        alphabets.DEFAULT_ALPHABET),
                        order=cfg["order"],
                        max_length=cfg["max_length"],
                        weights=cfg["weights"],
                        alpha=(None if cfg.get("alpha_auto", True)
                               else cfg.get("alpha")),
                        alpha_length=(None if cfg.get("alpha_length_auto", True)
                                      else cfg.get("alpha_length")),
                        sparse=cfg.get("sparse", True))
