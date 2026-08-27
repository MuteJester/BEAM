"""Histogram: a surprisal histogram over a shared grid.
"""
from __future__ import annotations

import math

import numpy as np

from ..model.validation import _is_scalar_arg

from ..constants import BEAM_MAX_MOMENT

from .._simulation_result import SimulationResult


class Histogram:
    """One reconstructed surprisal spectrum on a deterministic grid.

    Carries the attribute names :class:`FlashBackGraph`'s ``PseqHistogram``
    carries, so spectrum code written against that reads across unchanged.

    ``weights`` are raw masses: without a length restriction they sum to 1 for
    ``measure='generated'`` and to the support size for ``'counting'``.
    """

    __slots__ = ("surprisal", "weights", "measure", "q", "grid_spacing",
                 "max_rounding_error", "true_max_surprisal", "length")

    def __init__(self, *, surprisal, weights, measure, q, grid_spacing,
                 max_rounding_error, true_max_surprisal, length=None):
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.measure = measure
        self.q = float(q)
        self.grid_spacing = float(grid_spacing)
        self.max_rounding_error = float(max_rounding_error)
        self.true_max_surprisal = float(true_max_surprisal)
        self.length = length

    def __repr__(self) -> str:
        at = "" if self.length is None else f", length={self.length}"
        return (f"Histogram(measure={self.measure!r}, "
                f"bins={self.weights.size}{at}, "
                f"total_mass={self.total_mass:.6g})")

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.weights, dtype=np.longdouble))

    @property
    def normalized_weights(self) -> np.ndarray:
        total = self.total_mass
        return self.weights / total if total > 0 else np.zeros_like(self.weights)

    @property
    def mean(self) -> float:
        """Mean surprisal on this grid.

        Exact to machine precision: linear transport conserves the first
        moment. The VARIANCE read off the same grid is not — it is inflated by
        up to ``(L + 1) * spacing**2 / 6`` — so take that from
        :meth:`BeamModel.surprisal_moments` when it matters.
        """
        return float(np.dot(self.normalized_weights, self.surprisal))

    @property
    def variance(self) -> float:
        w = self.normalized_weights
        d = self.surprisal - self.mean
        return float(np.dot(w, d * d))

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def cdf(self, x):
        """Normalised CDF in surprisal space."""
        values = np.atleast_1d(np.asarray(x, dtype=np.float64))
        total = self.total_mass
        if total <= 0:
            out = np.zeros_like(values)
        else:
            cum = np.cumsum(self.weights, dtype=np.float64) / total
            out = np.interp(values, self.surprisal, cum, left=0.0, right=1.0)
        return float(out[0]) if _is_scalar_arg(x) else out

    def pdf(self, x):
        """Piecewise-linear normalised density of the reconstruction."""
        values = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self.grid_spacing <= 0 or self.total_mass <= 0:
            out = np.zeros_like(values)
        else:
            density = self.normalized_weights / self.grid_spacing
            out = np.interp(values, self.surprisal, density,
                            left=0.0, right=0.0)
        return float(out[0]) if _is_scalar_arg(x) else out

    def quantile(self, probability):
        """Approximate surprisal quantile of the reconstructed measure."""
        p = np.atleast_1d(np.asarray(probability, dtype=np.float64))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be between 0 and 1")
        cum = np.cumsum(self.normalized_weights)
        out = np.interp(p, cum, self.surprisal)
        return float(out[0]) if _is_scalar_arg(probability) else out
