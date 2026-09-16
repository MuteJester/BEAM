"""What :func:`depth_matched_hill_profiles` returns, and the band drawn from it.

Both are frozen, hold read-only arrays, and define no ``__float__``. That is
the rule for every number in this library that needs a caveat: a bare float
gets pasted into a table and separated from the sentence that qualified it,
while a field has to be named, and naming ``resampling_band`` is the moment a
reader learns what the interval is and is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = ["DepthMatchedProfiles", "ResamplingBand"]


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class ResamplingBand:
    """Percentiles of each repertoire's Hill profile over its resamples.

    **This is not a confidence interval, and not for a new sample.** Every
    replicate is a resample of ONE repertoire: the same sequences, drawn again
    at the common depth in proportion to their abundance. So the band measures
    how much the profile moves when what you already have is reweighted, not
    what a second sample of the same person would give. Measured at ``q = 1``
    the two differ by an order of magnitude: the resampling band was about 3%
    wide where held-out disjoint blocks of the same repertoire spread over
    about 50%. Read it as a floor on the uncertainty, never as the
    uncertainty.

    Attributes:
        orders: ``(n_orders,)`` Hill orders.
        lo, hi: The percentiles the band spans, from 0 to 100.
        lower, median, upper: ``(k, n_orders)``, per repertoire and order.
        n_bootstraps: Resamples behind each percentile.
        depth: The common template depth every resample was drawn at.
    """
    orders: np.ndarray
    lo: float
    hi: float
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    n_bootstraps: int
    depth: int

    def __repr__(self) -> str:
        k, n = self.median.shape
        return (f"ResamplingBand({self.lo:g}-{self.hi:g}th percentile of "
                f"{self.n_bootstraps} resamples at depth {self.depth}; "
                f"{k} repertoires x {n} orders; not a confidence interval)")


@dataclass(frozen=True, eq=False)
class DepthMatchedProfiles:
    """Hill profiles of ``k`` repertoires, each resampled to a common depth.

    ``profiles[i, b, j]`` is ``D_q`` at ``orders[j]`` of a model fitted to the
    ``b``-th resample of repertoire ``i``: ``depth`` templates drawn with
    replacement in proportion to abundance, collapsed to per-sequence counts,
    and fitted fresh.

    **Comparable only within one call.** ``depth`` is the smallest template
    total among the repertoires passed together, so adding or removing a
    repertoire can change it, and with it every other repertoire's profiles.
    Two results are comparable only if their ``depth`` is equal and they were
    produced from the same configuration.

    Attributes:
        orders: ``(n_orders,)`` Hill orders.
        depth: ``D``, the templates each resample draws: the smallest
            repertoire's template total, never its distinct count.
        weighting: ``"abundance"`` fits each resample with its drawn counts,
            ``"presence"`` with weight 1 per distinct sequence drawn. The draw
            itself is abundance-proportional either way.
        n_bootstraps: Resamples per repertoire.
        seed: The seed the draws came from. When the call passed ``None`` this
            is the entropy that was drawn for it, so every result can be
            reproduced exactly from its own record.
        profiles: ``(k, n_bootstraps, n_orders)``.
        n_distinct_drawn: ``(k, n_bootstraps)``, the distinct sequences each
            resample reached. An output rather than an input: a repertoire
            whose templates sit in few clones reaches fewer from the same
            draws, which is a property of that repertoire.
        n_templates: ``(k,)``, each repertoire's template total.
        n_distinct: ``(k,)``, each repertoire's distinct sequences.
        config: The model configuration every resample was fitted with.
    """
    orders: np.ndarray
    depth: int
    weighting: str
    n_bootstraps: int
    seed: int
    profiles: np.ndarray
    n_distinct_drawn: np.ndarray
    n_templates: np.ndarray
    n_distinct: np.ndarray
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Frozen means the fields cannot be rebound; it says nothing about
        # the arrays they hold. Read-only copies close that, and a read-only
        # mapping does the same for the configuration.
        for name in ("orders", "profiles", "n_distinct_drawn", "n_templates",
                     "n_distinct"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "config",
                           MappingProxyType(dict(self.config)))

    @property
    def n_repertoires(self) -> int:
        return int(self.profiles.shape[0])

    @property
    def n_orders(self) -> int:
        return int(self.orders.shape[0])

    def resampling_band(self, lo: float = 5.0,
                        hi: float = 95.0) -> ResamplingBand:
        """The ``lo``-th to ``hi``-th percentile of each profile over resamples.

        A RESAMPLING band, not a confidence interval for a new sample: every
        replicate reweights the same repertoire, so this measures reweighting
        what you have and not what re-sequencing would give. At ``q = 1`` it
        was about 3% wide where held-out disjoint blocks of one repertoire
        spread over about 50%. See :class:`ResamplingBand`.

        Percentiles are numpy's default, linear between order statistics, and
        with twenty resamples the 5th and 95th sit on the second-smallest and
        second-largest values' neighbourhood, not on anything finer.
        """
        lo, hi = float(lo), float(hi)
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(
                f"need 0 <= lo < hi <= 100, got lo={lo!r}, hi={hi!r}")
        p = np.percentile(self.profiles, [lo, 50.0, hi], axis=1)
        return ResamplingBand(orders=_frozen(self.orders), lo=lo, hi=hi,
                              lower=_frozen(p[0]), median=_frozen(p[1]),
                              upper=_frozen(p[2]),
                              n_bootstraps=self.n_bootstraps,
                              depth=self.depth)

    def __repr__(self) -> str:
        return (f"DepthMatchedProfiles({self.n_repertoires} repertoires x "
                f"{self.n_bootstraps} resamples x {self.n_orders} orders, "
                f"depth={self.depth}, weighting={self.weighting!r}, "
                f"seed={self.seed})")
