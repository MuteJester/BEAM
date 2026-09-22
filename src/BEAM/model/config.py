"""Configuration assembly: what a caller may set and what it defaults to.
"""
from __future__ import annotations

from typing import Any


from ..constants import BEAM_MAX_LENGTH_CEILING, BEAM_MAX_MOMENT
from . import alphabets
from .validation import _check_int

from .._simulation_result import SimulationResult


def _config_args(*, order, max_length, weights, strict=True,
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
    strict = bool(strict)

    # The weights SHOULD sum to 1 and are not refused if they do not: (1, 0)
    # and (0, 1) are how each anchor's chain is measured on its own, which is
    # what separates a partition function below 1 into the two causes that can
    # put it there. Off 1, log W scales with the length, so within-length
    # ranking is untouched and cross-length comparison is not meaningful.
    if sparse is None:
        sparse = strict
    else:
        sparse = bool(sparse)
        if sparse and not strict:
            raise ValueError("sparse=True requires strict=True")

    # The alphabet is appended rather than placed first, where it belongs, so
    # every existing position in the tuple keeps its meaning.
    return (order, max_length, weights, strict, sparse, alphabet)


def _config_args_from_dict(cfg: dict[str, Any]) -> tuple:
    """The same tuple, rebuilt from what :attr:`BeamModel.config` reports."""
    return _config_args(alphabet=cfg.get("alphabet",
                                        alphabets.DEFAULT_ALPHABET),
                        order=cfg["order"],
                        max_length=cfg["max_length"],
                        weights=cfg["weights"],
                        strict=cfg.get("strict", True),
                        sparse=cfg.get("sparse", False))
