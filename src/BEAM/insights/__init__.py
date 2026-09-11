"""Derived, plottable views of a finalized BEAM model.

Nothing here computes anything new. Every quantity is built from the exact
machinery already in the library -- `positional_marginals` and the banded
surprisal sweep -- and what this package adds is the shaping: the arithmetic a
consumer would otherwise do by hand, and the refusals that stop a badly posed
question coming back as a confident number.

It never draws anything. It returns arrays, and you plot them.

    >>> s = model.insights.logos(15, pseq_below=1e-9)
    >>> s[0].as_dict()          # {residue: heights}, ready for logomaker
"""
from __future__ import annotations

from .logo import cost, logo, logos
from .results import Logo, LogoSet

__all__ = ["Logo", "LogoSet", "cost", "logo", "logos"]


class _Insights:
    """The `model.insights` accessor.

    A bound view rather than free functions on the model, so that this
    package can grow without adding a method to `BeamModel` for every
    quantity anyone ever wants derived from one.
    """

    __slots__ = ("_model",)

    def __init__(self, model) -> None:
        self._model = model

    def logos(self, length, **kw) -> LogoSet:
        return logos(self._model, length, **kw)

    def logo(self, length, **kw) -> Logo:
        return logo(self._model, length, **kw)

    def cost(self, length, **kw) -> dict:
        return cost(self._model, length, **kw)

    def __repr__(self) -> str:
        return "<BEAM insights: logos>"
