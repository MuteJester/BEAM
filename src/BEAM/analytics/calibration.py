"""Calibration: the SCALE surprisal calibration.
"""
from __future__ import annotations

import math

import numpy as np

from ..model.validation import _is_scalar_arg

from ..math.saddlepoint import _normal_pdf, _normal_cdf

from .._simulation_result import SimulationResult


class Calibration:
    """An exact length-conditioned null for the surprisal.

    Built by :meth:`BeamModel.calibrate`. Holds, for each length the model can
    produce, the survival function of ``-ln P`` among sequences of exactly that
    length. Reconstructed by dynamic programming rather than simulated, so
    there is no sampling noise, no minimum-count rule, and nothing to tune.
    """

    __slots__ = ("surprisal", "spacing", "survival", "prior", "_pooled",
                 "_stats")

    def __init__(self, *, surprisal, spacing, survival, prior=None):
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.spacing = float(spacing)
        self.survival = {int(k): np.asarray(v, dtype=np.float64)
                         for k, v in survival.items()}
        # Uniform if no prior is supplied, which is the only honest default
        # when the length mixture is not known.
        self.prior = ({int(k): float(v) for k, v in prior.items()}
                      if prior is not None
                      else dict.fromkeys(self.survival, 1.0))
        self._pooled: np.ndarray | None = None
        self._stats: dict[int | None, tuple[float, float]] = {}

    def __repr__(self) -> str:
        return (f"Calibration(lengths={len(self.survival)}, "
                f"resolution_floor={self.resolution_floor:.2e})")

    @property
    def lengths(self) -> tuple[int, ...]:
        """Lengths this calibration covers."""
        return tuple(sorted(self.survival))

    @property
    def resolution_floor(self) -> float:
        """Smallest tail probability the grid can resolve.

        A p-value at or below this is at the limit of what the reconstruction
        can distinguish from zero, and should be reported as such rather than
        quoted precisely.
        """
        return min(float(v[v > 0].min()) if np.any(v > 0) else 1.0
                   for v in self.survival.values())

    @property
    def pooled_survival(self) -> np.ndarray:
        """The length mixture's survival function, weighted by ``pi(L)``.

        The fallback null for a length the calibration does not cover.
        """
        if self._pooled is None:
            total = sum(self.prior.values())
            acc = np.zeros_like(self.surprisal)
            for L, sf in self.survival.items():
                acc += (self.prior.get(L, 0.0) / total) * sf
            self._pooled = acc
        return self._pooled

    def quantile(self, length: int | None, probability):
        """Surprisal ``u`` with ``P(U <= u) = probability`` at this length.

        ``length=None`` uses the pooled mixture. Linearly interpolated on the
        reconstruction grid, so it is exact to one bin width.
        """
        p = np.atleast_1d(np.asarray(probability, dtype=np.float64))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be between 0 and 1")
        sf = (self.pooled_survival if length is None
              else self.survival.get(int(length)))
        if sf is None:
            sf = self.pooled_survival
        # MID-DISTRIBUTION cdf. `1 - sf[i]` is P(U < u_i), strictly below the
        # atom, and inverting that puts every quantile half an atom too high —
        # a systematic bias of half a bin width on the median and so on every
        # SCALE score. Splitting the atom's own mass across the step removes it.
        below = np.clip(1.0 - sf, 0.0, 1.0)
        atom = np.empty_like(sf)
        atom[:-1] = sf[:-1] - sf[1:]
        atom[-1] = sf[-1]
        cdf = np.clip(below + 0.5 * atom, 0.0, 1.0)
        # np.interp needs an increasing x; a survival function is monotone, so
        # the CDF is already sorted up to floating-point flat spots.
        out = np.interp(p, cdf, self.surprisal)
        return float(out[0]) if _is_scalar_arg(probability) else out

    def _median_iqr(self, length: int | None) -> tuple[float, float]:
        """Median and interquartile range of the surprisal at one length."""
        key = None if length is None or int(length) not in self.survival \
            else int(length)
        if key not in self._stats:
            q25, q50, q75 = self.quantile(key, [0.25, 0.5, 0.75])
            iqr = float(q75 - q25)
            # A length whose whole mass falls in one bin has a zero IQR; the
            # grid spacing is the smallest difference that grid can express,
            # and dividing by it is better than dividing by zero.
            self._stats[key] = (float(q50), iqr if iqr > 0 else self.spacing)
        return self._stats[key]

    def median(self, length: int | None = None) -> float:
        """Median surprisal at this length; ``None`` pools over lengths."""
        return self._median_iqr(length)[0]

    def iqr(self, length: int | None = None) -> float:
        """Interquartile range of the surprisal at this length."""
        return self._median_iqr(length)[1]

    def scale(self, length: int, surprisal):
        """``(surprisal - median_L) / IQR_L``, the SCALE anomaly score.

        Length-normalised because ``-ln P`` grows with length, so a raw
        surprisal is not comparable across lengths. An uncovered length falls
        back to the pooled mixture.
        """
        m, sd = self._median_iqr(length)
        u = np.atleast_1d(np.asarray(surprisal, dtype=np.float64))
        out = (u - m) / sd
        return float(out[0]) if _is_scalar_arg(surprisal) else out

    def pvalue(self, length: int, surprisal) -> float | np.ndarray:
        """``P(U >= surprisal)`` for a sequence of this length.

        Returns 1.0 for a length the calibration does not cover, which is the
        conservative answer: an uncovered length is not evidence of anomaly.
        """
        sf = self.survival.get(int(length))
        u = np.atleast_1d(np.asarray(surprisal, dtype=np.float64))
        if sf is None:
            out = np.ones_like(u)
        else:
            idx = np.clip(np.searchsorted(self.surprisal, u, side="left"),
                          0, len(sf) - 1)
            # The survival function is a normalised cumulative sum, so its
            # first entry can land a few ulps above 1. A p-value may not.
            out = np.clip(sf[idx], 0.0, 1.0)
        if _is_scalar_arg(surprisal):
            return float(out[0])
        return out


