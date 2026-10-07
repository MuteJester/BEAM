"""Atoms: the (probability, multiplicity) pair a spectrum reduces to.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from ..constants import BEAM_MAX_MOMENT

from .._simulation_result import SimulationResult


class Atoms:
    """A discrete ``(probability, multiplicity)`` description of the model.

    One entry per occupied grid bin: how probable a sequence in that bin is,
    and how many sequences are there. Every rarefaction-style quantity —
    expected richness, novelty, coverage, the Poisson-binomial occupancy a
    sharing prediction needs — is a sum over these pairs.

    **The probability is ``generated / counting``, not ``exp(-surprisal)``.**
    Both spectra come out of ONE call (:meth:`BeamModel.spectrum_pair`),
    which puts them on one grid and splits every edge of both by the same
    fractions, so they are the same weights carrying different mass and their
    ratio is the mass-weighted mean probability of the sequences in the bin.
    That the two share splitting fractions is what makes the ratio meaningful,
    and taking them from one call makes it structural rather than a
    coincidence of two calls doing the same arithmetic. Reading ``exp(-surprisal)`` off the bin's left edge instead biases
    every atom by up to ``exp(spacing)`` and shows up immediately as a total
    mass that is not 1. With this definition
    ``sum(multiplicities * probabilities)`` is the generated mass exactly,
    which is the identity the tests check.
    """

    __slots__ = ("probabilities", "multiplicities", "surprisal",
                 "grid_spacing", "length")

    def __init__(self, *, probabilities, multiplicities, surprisal,
                 grid_spacing, length=None):
        self.probabilities = np.asarray(probabilities, dtype=np.longdouble)
        self.multiplicities = np.asarray(multiplicities, dtype=np.longdouble)
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.grid_spacing = float(grid_spacing)
        self.length = length

    def __repr__(self) -> str:
        return (f"Atoms(n={self.probabilities.size}, "
                f"support={self.support_size:.6g}, mass={self.mass:.9f})")

    def __len__(self) -> int:
        return int(self.probabilities.size)

    @property
    def support_size(self) -> float:
        """Total multiplicity: the counting measure's integral."""
        return float(np.sum(self.multiplicities, dtype=np.longdouble))

    @property
    def mass(self) -> float:
        """``sum multiplicity * probability``. 1 for an unrestricted spectrum."""
        return float(np.sum(self.multiplicities * self.probabilities,
                            dtype=np.longdouble))

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(probabilities, multiplicities)``, the pair every R(n)-style
        calculation and ``PublicnessModel.expected_counts`` consumes."""
        return self.probabilities, self.multiplicities

    def _survival(self, n: float) -> np.ndarray:
        """``(1 - p)**n``, computed so it survives ``p ~ 1e-80``.

        ``(1 - p)**n`` in float64 rounds ``1 - p`` to 1 and collapses to 1.0;
        ``exp(n * log1p(-p))`` in long double stays representable.
        """
        return np.exp(np.longdouble(n) * np.log1p(-self.probabilities))

    def expected_richness(self, n: float) -> float:
        """``R(n) = sum_s [1 - (1 - P(s))**n]``, distinct sequences after
        ``n`` independent draws.

        ``R(1)`` is exactly the generated mass, 1 for an unrestricted
        spectrum, and ``R(n)`` rises monotonically to the support size.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return float(np.sum(self.multiplicities * (1.0 - self._survival(n)),
                            dtype=np.longdouble))

    def novelty(self, n: float) -> float:
        """``U(n) = sum_s P(s) (1 - P(s))**(n-1)``, the chance draw ``n`` is
        one never drawn before.

        Exactly ``R(n) - R(n-1)``, which is the identity the tests check.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return float(np.sum(
            self.multiplicities * self.probabilities * self._survival(n - 1.0),
            dtype=np.longdouble))

    def coverage(self, n: float) -> float:
        """Probability mass carried by the sequences seen in ``n`` draws.

        Good-Turing's target, computed exactly rather than estimated from the
        singletons.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        total = self.mass
        if total <= 0:
            return 0.0
        seen = np.sum(self.multiplicities * self.probabilities
                      * (1.0 - self._survival(n)), dtype=np.longdouble)
        return float(seen / np.longdouble(total))

    def expected_frequency(self, n: int, r: int) -> float:
        """Expected number of distinct sequences drawn exactly ``r`` times in
        ``n`` draws. ``r=1`` and ``r=2`` are Chao's ``f1`` and ``f2``."""
        n, r = int(n), int(r)
        if n < 0 or r < 0 or r > n:
            raise ValueError("require n >= 0 and 0 <= r <= n")
        p = self.probabilities
        keep = (p > 0) & (p < 1)
        log_coef = (math.lgamma(n + 1) - math.lgamma(r + 1)
                    - math.lgamma(n - r + 1))
        terms = np.exp(np.longdouble(log_coef)
                       + np.longdouble(r) * np.log(p[keep])
                       + np.longdouble(n - r) * np.log1p(-p[keep]))
        return float(np.sum(self.multiplicities[keep] * terms,
                            dtype=np.longdouble))

    def discovery_curve(self, draw_counts: Iterable[float]) -> dict[str, np.ndarray]:
        """``R(n)``, ``U(n)`` and coverage over many sampling depths at once.

        One pass of numpy per depth over the atoms, which is what makes a
        400-point curve seconds rather than one spectrum rebuild per point.
        """
        ns = np.asarray(list(draw_counts), dtype=np.float64)
        if np.any(ns < 1):
            raise ValueError("every draw count must be at least 1")
        return {
            "n": ns,
            "expected_richness": np.array([self.expected_richness(n) for n in ns]),
            "novelty": np.array([self.novelty(n) for n in ns]),
            "coverage": np.array([self.coverage(n) for n in ns]),
        }
