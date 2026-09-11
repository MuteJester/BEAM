"""Result container for :meth:`BeamModel.positional_banded`."""
from __future__ import annotations

import numpy as np


class BandedMarginals:
    """Per-position residue distributions, one set per band of surprisal.

    Returned rather than a bare array because the marginals alone do not say
    what they describe. Two things travel with them and are needed to read
    them honestly: how much of the model each band holds, and whether the
    grid was fine enough to separate the bands at all.

    Attributes:
        marginals: ``(n_bands, length, n_residues)``. Row ``[j, i]`` is
            ``P(x_i = a | length, edges[j] <= surprisal < edges[j+1])`` and
            sums to 1, or is all zeros where the band holds no mass.
        band_mass: ``(n_bands,)``. How much mass each band holds, in the
            chosen measure. A zero here means that band's marginals describe
            nothing.
        position_mass: ``(n_bands, length)``. The same total measured
            independently at every position. See :attr:`resolution_error`.
        edges: the band edges in nats, as given.
        spacing: the grid spacing actually used, in nats.
        counting: which measure was taken.
        length: the sequence length these describe.
    """

    __slots__ = ('marginals', 'band_mass', 'position_mass', 'edges',
                 'spacing', 'counting', 'length')

    def __init__(self, marginals, band_mass, position_mass, edges,
                 spacing: float, counting: bool, length: int) -> None:
        self.marginals = marginals
        self.band_mass = band_mass
        self.position_mass = position_mass
        self.edges = edges
        self.spacing = float(spacing)
        self.counting = bool(counting)
        self.length = int(length)

    @property
    def n_bands(self) -> int:
        return int(self.band_mass.shape[0])

    @property
    def fraction(self) -> np.ndarray:
        """Each band's share of the mass that landed in any band."""
        total = self.band_mass.sum()
        if not total > 0.0:
            return np.zeros_like(self.band_mass)
        return self.band_mass / total

    @property
    def resolution_error(self) -> float:
        """How far the grid is from resolving these bands, as a fraction.

        Every position of a band independently measures that band's mass, and
        in exact arithmetic they are the same number. They are not, because
        conditioning on a surprisal band bins a real-valued quantity, and the
        disagreement is the clearest available reading of how much that
        binning cost.

        Below about ``1e-9`` the bands are as sharp as the arithmetic allows.
        Anything larger means band edges are cutting through mass at a scale
        the grid cannot see, and the usual cause is bands only a few bins
        wide: raise ``bins``, or widen the bands.
        """
        worst = 0.0
        for j in range(self.n_bands):
            m = self.band_mass[j]
            if not m > 0.0:
                continue
            worst = max(worst, float(np.abs(self.position_mass[j] - m).max() / m))
        return worst

    def __len__(self) -> int:
        return self.n_bands

    def __repr__(self) -> str:
        measure = 'counting' if self.counting else 'generated'
        return (f'BandedMarginals(length={self.length}, '
                f'n_bands={self.n_bands}, measure={measure}, '
                f'spacing={self.spacing:.4g}, '
                f'resolution_error={self.resolution_error:.2g})')
