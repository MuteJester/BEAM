"""PublicnessModel: expected sharing across a cohort of repertoires.

Split out of BEAM._publicness. The class keeps `__module__ = "BEAM._publicness"`
on purpose: instances pickle, and a pickle records the class by module path,
so moving the definition would silently invalidate every pickle already
written. The old module still re-exports the name, and a test asserts the
round trip.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import _cbeam as _c


from .constants import (TAIL_SIGMA, TAIL_FLOOR, SMALL_MU, _TINY_LEVELS,
                        MASS_TOL, _CHUNK_DOUBLES)
from .cohort import _next_power_of_two_above, _depth_groups

class PublicnessModel:
    """Poisson-binomial occupancy model for one fixed cohort of depths.

    ``depths`` is the per-repertoire sampling depth, in distinct sequences
    contributed. The model is independent of any graph: it turns a sequence
    probability into the distribution of how many repertoires contain that
    sequence, and :meth:`expected_counts` aggregates that over a whole
    p-sequence spectrum.

    Read the module docstring before changing ``tail_sigma`` or
    ``tail_floor``. They are what keeps DFT roundoff, amplified by spectrum
    multiplicities of up to 1e29, out of the predicted tail.
    """

    def __init__(
        self,
        depths: Any,
        *,
        depth_bins: int = 64,
        k_fft: int | None = None,
        tail_sigma: float = TAIL_SIGMA,
        tail_floor: float = TAIL_FLOOR,
        mass_tol: float = MASS_TOL,
    ) -> None:
        if depth_bins < 1:
            raise ValueError("depth_bins must be positive")
        if tail_sigma < 0 or tail_floor < 0:
            raise ValueError("tail_sigma and tail_floor must be non-negative")
        if mass_tol <= 0:
            raise ValueError("mass_tol must be positive")

        self._depths, self._multiplicity, self._spread = _depth_groups(
            depths, depth_bins
        )
        self._n_repertoires = int(round(float(np.sum(self._multiplicity))))

        if k_fft is None:
            k_fft = _next_power_of_two_above(self._n_repertoires)
        if k_fft <= self._n_repertoires:
            raise ValueError(
                f"k_fft={k_fft} must exceed the {self._n_repertoires} "
                "repertoires, or occupancy mass wraps around the transform"
            )
        if k_fft % 2:
            raise ValueError("k_fft must be even")

        self._k_fft = int(k_fft)
        self.tail_sigma = float(tail_sigma)
        self.tail_floor = float(tail_floor)
        self.mass_tol = float(mass_tol)

    # ── Cohort description ────────────────────────────────────

    @property
    def n_repertoires(self) -> int:
        """Number of repertoires in the cohort."""
        return self._n_repertoires

    @property
    def n_depth_groups(self) -> int:
        """Number of factors in the generating-function product."""
        return int(self._depths.size)

    @property
    def depth_spread(self) -> float:
        """Widest within-group ratio of depths. ``1.0`` means exact grouping."""
        return self._spread

    @property
    def k_fft(self) -> int:
        """Number of roots of unity used to invert the generating function."""
        return self._k_fft

    def depth_groups(self) -> tuple[np.ndarray, np.ndarray]:
        """Representative depths and their multiplicities."""
        return self._depths.copy(), self._multiplicity.copy()

    # ── Per-sequence quantities ───────────────────────────────

    def moments(self, p: Any) -> tuple[Any, Any]:
        """Closed-form mean and variance of the occupancy count.

        Neither touches the PMF, which is exactly why they can be trusted to
        say which part of the inverted PMF is signal.
        """
        values = np.asarray(p, dtype=np.float64)
        flat = np.ascontiguousarray(values.reshape(-1))
        mean = np.empty(flat.size, dtype=np.float64)
        variance = np.empty(flat.size, dtype=np.float64)
        _c.publicness_moments(
            flat, self._depths, self._multiplicity, mean, variance
        )
        if np.ndim(p) == 0:
            return float(mean[0]), float(variance[0])
        return mean.reshape(values.shape), variance.reshape(values.shape)

    def pmf(self, p: float) -> np.ndarray:
        """Occupancy PMF of one sequence probability, over levels 0..R.

        Truncated, clipped and mass-checked exactly as the aggregate is, so
        this is the single-atom view of what :meth:`expected_counts` sums.
        """
        edges = np.arange(self._n_repertoires + 2, dtype=np.float64)
        counts, _, _, _ = self._aggregate(
            np.asarray([float(p)]), np.ones(1), edges
        )
        return counts

    def bin_mass(self, p: float, edges: Any) -> np.ndarray:
        """Probability the occupancy count falls in each half-open bin."""
        resolved = self._resolve_edges(edges)
        counts, _, _, _ = self._aggregate(
            np.asarray([float(p)]), np.ones(1), resolved
        )
        return counts

    # ── Aggregation over a spectrum ───────────────────────────

    def expected_counts(
        self,
        probabilities: Any,
        multiplicities: Any,
        *,
        levels: Any = None,
    ) -> dict[str, Any]:
        """Predicted sequences per publicness bin, over a whole spectrum.

        ``probabilities`` and ``multiplicities`` are a p-sequence spectrum:
        one probability atom and the number of distinct sequences carrying it.
        Atoms of zero multiplicity are dropped before any work is done, which
        matters because a deterministic surprisal grid is mostly empty.

        ``levels`` selects the publicness binning:

        * ``None`` (default) gives one bin per occupancy level, 0..R.
        * an ``int`` gives that many equal-width bins spanning 0..R+1, merged
          where rounding would produce empty ones.
        * a sequence gives explicit non-decreasing integer bin edges, read as
          half-open ranges ``[edges[i], edges[i+1])``.
        """
        p = np.ascontiguousarray(
            np.asarray(probabilities, dtype=np.float64).reshape(-1)
        )
        w = np.ascontiguousarray(
            np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        )
        if p.size != w.size:
            raise ValueError(
                "probabilities and multiplicities must be the same length"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("every multiplicity must be finite and non-negative")

        keep = w > 0
        p, w = np.ascontiguousarray(p[keep]), np.ascontiguousarray(w[keep])
        if p.size == 0:
            raise ValueError("the spectrum carries no mass")

        edges = self._resolve_edges(levels)
        counts, mean, variance, retained = self._aggregate(p, w, edges)
        return {
            "edges": edges.astype(np.int64),
            "expected_counts": counts,
            "expected_total": float(np.sum(counts)),
            "atom_mean": mean,
            "atom_variance": variance,
            "min_retained_mass": float(np.min(retained)),
            "n_atoms": int(p.size),
            "n_repertoires": self._n_repertoires,
            "n_depth_groups": self.n_depth_groups,
            "depth_spread": self._spread,
            "k_fft": self._k_fft,
        }

    def expected_counts_by_group(
        self,
        probabilities: Any,
        multiplicities: Any,
        groups: Any,
        *,
        n_groups: int | None = None,
        levels: Any = None,
    ) -> dict[str, Any]:
        """Predict publicness counts for several atom groups together.

        Each probability atom contributes only to the output row named by its
        non-negative integer ``groups`` value. The probability generating
        functions and FFT inversion are batched across all rows, while the
        native accumulator applies the same moment-derived truncation, clip
        order, and retained-mass test as :meth:`expected_counts` before adding
        each atom to its row.

        This is mathematically equivalent to selecting each group and calling
        :meth:`expected_counts` separately. It is intended for joint surfaces
        such as P-sequence probability bin by repertoire publicness.

        Args:
            probabilities: Sequence probability for every spectrum atom.
            multiplicities: Number of sequences represented by every atom.
            groups: Non-negative integer output row for every atom.
            n_groups: Number of output rows. Defaults to one above the largest
                group index; larger values retain explicit empty rows.
            levels: Publicness binning accepted by :meth:`expected_counts`.

        Returns:
            Diagnostics and an ``expected_counts`` float64 matrix with shape
            ``(n_groups, number_of_publicness_bins)``.
        """
        p = np.ascontiguousarray(
            np.asarray(probabilities, dtype=np.float64).reshape(-1)
        )
        w = np.ascontiguousarray(
            np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        )
        raw_groups = np.asarray(groups).reshape(-1)
        if p.size != w.size or p.size != raw_groups.size:
            raise ValueError(
                "probabilities, multiplicities and groups must be the same length"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("every multiplicity must be finite and non-negative")
        if not np.issubdtype(raw_groups.dtype, np.integer):
            raise TypeError("groups must contain integers")
        if np.any(raw_groups < 0) or np.any(raw_groups > np.iinfo(np.uint32).max):
            raise ValueError("groups must lie within the uint32 range")
        group = np.ascontiguousarray(raw_groups, dtype=np.uint32)

        keep = w > 0
        p = np.ascontiguousarray(p[keep])
        w = np.ascontiguousarray(w[keep])
        group = np.ascontiguousarray(group[keep])
        if p.size == 0:
            raise ValueError("the spectrum carries no mass")

        required_groups = int(np.max(group)) + 1
        if n_groups is None:
            n_groups = required_groups
        elif not isinstance(n_groups, (int, np.integer)):
            raise TypeError("n_groups must be an integer or None")
        n_groups = int(n_groups)
        if n_groups < required_groups:
            raise ValueError(
                f"n_groups={n_groups} cannot hold group index {required_groups - 1}"
            )

        edges = self._resolve_edges(levels)
        counts, mean, variance, retained = self._aggregate_grouped(
            p, w, group, n_groups, edges
        )
        return {
            "edges": edges.astype(np.int64),
            "expected_counts": counts,
            "expected_total": float(np.sum(counts, dtype=np.longdouble)),
            "group_totals": np.asarray(
                np.sum(counts, axis=1, dtype=np.longdouble), dtype=np.float64
            ),
            "atom_mean": mean,
            "atom_variance": variance,
            "min_retained_mass": float(np.min(retained)),
            "n_atoms": int(p.size),
            "n_groups": n_groups,
            "n_repertoires": self._n_repertoires,
            "n_depth_groups": self.n_depth_groups,
            "depth_spread": self._spread,
            "k_fft": self._k_fft,
        }

    # ── Internals ─────────────────────────────────────────────

    def _resolve_edges(self, levels: Any) -> np.ndarray:
        if levels is None:
            return np.arange(self._n_repertoires + 2, dtype=np.float64)
        if np.ndim(levels) == 0:
            n_bins = int(levels)
            if n_bins < 1:
                raise ValueError("levels must be a positive bin count")
            grid = np.linspace(0.0, self._n_repertoires + 1.0, n_bins + 1)
            return np.unique(np.rint(grid))

        edges = np.asarray(levels, dtype=np.float64).reshape(-1)
        if edges.size < 2:
            raise ValueError("explicit bin edges need at least two entries")
        if np.any(edges != np.floor(edges)):
            raise ValueError("bin edges must be integers")
        if np.any(np.diff(edges) < 0):
            raise ValueError("bin edges must be non-decreasing")
        if edges[0] < 0 or edges[-1] > self._k_fft:
            raise ValueError(f"bin edges must lie within [0, {self._k_fft}]")
        return edges

    def _aggregate(
        self,
        p: np.ndarray,
        w: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        groups = np.zeros(p.size, dtype=np.uint32)
        counts, mean, variance, retained = self._aggregate_grouped(
            p, w, groups, 1, edges
        )
        return counts[0], mean, variance, retained

    def _aggregate_grouped(
        self,
        p: np.ndarray,
        w: np.ndarray,
        groups: np.ndarray,
        n_groups: int,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros((n_groups, edges.size - 1), dtype=np.float64)
        mean = np.empty(p.size, dtype=np.float64)
        variance = np.empty(p.size, dtype=np.float64)
        retained = np.empty(p.size, dtype=np.float64)
        _c.publicness_moments(p, self._depths, self._multiplicity,
                              mean, variance)

        # Atoms whose mean occupancy is far below machine precision cannot go
        # through the transform at all: their generating function is
        # 1 + mu(z-1) with mu ~ 1e-13, and adding that to 1.0 in double
        # precision destroys it. See SMALL_MU for what the transform returns
        # instead. They are inverted in closed form.
        tiny = mean < SMALL_MU
        big = np.flatnonzero(~tiny)

        step = max(1, _CHUNK_DOUBLES // self._k_fft)
        for start in range(0, big.size, step):
            idx = big[start:start + step]
            g = np.empty((idx.size, self._k_fft), dtype=np.complex128)
            _c.publicness_pgf(np.ascontiguousarray(p[idx]), self._depths,
                              self._multiplicity,
                              g.view(np.float64).reshape(-1))
            # Forward transform divided by K. np.fft.ifft would return the
            # same coefficients reflected about zero, which reads as a
            # plausible distribution and is wrong at every level but 0.
            pmf = np.ascontiguousarray(
                np.fft.fft(g, axis=-1).real / self._k_fft
            )
            chunk_retained = np.empty(idx.size, dtype=np.float64)
            _c.publicness_accumulate_grouped(
                pmf, np.ascontiguousarray(mean[idx]),
                np.ascontiguousarray(variance[idx]),
                np.ascontiguousarray(w[idx]),
                np.ascontiguousarray(groups[idx]),
                edges, counts, chunk_retained,
                self.tail_sigma, self.tail_floor, self.mass_tol,
            )
            retained[idx] = chunk_retained

        if tiny.any():
            self._accumulate_tiny_grouped(
                mean[tiny], w[tiny], groups[tiny], n_groups,
                edges, counts, retained, np.flatnonzero(tiny)
            )
        return counts, mean, variance, retained

    def _accumulate_tiny_grouped(
        self,
        mu: np.ndarray,
        w: np.ndarray,
        groups: np.ndarray,
        n_groups: int,
        edges: np.ndarray,
        counts: np.ndarray,
        retained: np.ndarray,
        where: np.ndarray,
    ) -> None:
        """Closed-form occupancy for atoms the transform cannot represent.

        Below ``SMALL_MU`` every per-repertoire detection probability is
        itself minute, so the Poisson-binomial count collapses to a Poisson
        with the same mean. The approximation error is bounded by Le Cam:
        total variation is at most ``sum_r pi_r**2 <= mu * max_r pi_r``, which
        at ``mu = 1e-6`` is below 1e-25 and shrinks quadratically from there.
        That is exact to double precision, and it is the regime where the
        transform returns nothing but rectified quantisation residue.

        Only the first few levels can carry mass: ``mu**k / k!`` at
        ``mu = SMALL_MU`` is already below 1e-30 by ``k = 5``, so levels above
        ``_TINY_LEVELS`` are exactly zero rather than nearly zero.
        """
        # Forward recurrence P(k) = P(k-1) * mu / k rather than
        # exp(-mu + k*log(mu) - log(k!)), which is undefined at mu = 0. A
        # probability-zero atom is a legitimate input and must give a delta at
        # occupancy zero, not a NaN.
        pmf = np.empty((mu.size, _TINY_LEVELS + 1), dtype=np.float64)
        pmf[:, 0] = np.exp(-mu)
        for k in range(1, _TINY_LEVELS + 1):
            pmf[:, k] = pmf[:, k - 1] * mu / k

        total = pmf.sum(axis=1)
        if not np.all(np.abs(total - 1.0) <= self.mass_tol):
            worst = int(np.argmax(np.abs(total - 1.0)))
            raise ValueError(
                f"closed-form occupancy for a mu={mu[worst]:.6g} atom retains "
                f"{total[worst]:.12f} of its mass over levels "
                f"0..{_TINY_LEVELS}; SMALL_MU and _TINY_LEVELS disagree"
            )
        retained[where] = total

        # Each level is a single integer, so it lands in exactly one bin.
        n_bins = edges.size - 1
        for k in range(_TINY_LEVELS + 1):
            b = int(np.searchsorted(edges, k, side="right")) - 1
            if 0 <= b < n_bins:
                counts[:, b] += np.bincount(
                    groups,
                    weights=w * pmf[:, k],
                    minlength=n_groups,
                )

    def __repr__(self) -> str:
        return (
            f"PublicnessModel(n_repertoires={self._n_repertoires}, "
            f"n_depth_groups={self.n_depth_groups}, "
            f"depth_spread={self._spread:.3f}, k_fft={self._k_fft})"
        )

# See the module docstring: the pickle path must not move.
PublicnessModel.__module__ = "BEAM._publicness"
