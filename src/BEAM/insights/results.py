"""Containers for what :mod:`BEAM.insights` returns."""
from __future__ import annotations

import numpy as np


class Logo:
    """One band's sequence logo.

    Everything a plotting package needs, and the three things it does not ask
    for but a reader of the plot does: how much of the model this band is,
    how far the grid was from resolving it, and how many residues each
    position can actually spell.

    Attributes:
        probability: ``(length, n_residues)``. Rows sum to 1. A residue the
            model cannot spell here is exactly 0 -- not floored, because in a
            strict model that zero is information.
        height_bits, height_nats: ``(length, n_residues)``, the stack heights
            a logo is drawn from. Bits is the axis every logo tool assumes.
            The two are the same numbers scaled by ``ln 2``.
        entropy_bits, entropy_nats: ``(length,)``.
        information_bits, information_nats: ``(length,)``,
            ``max_entropy - entropy``. This is a relative entropy against a
            uniform background; see :attr:`max_entropy_bits`.
        n_support: ``(length,)``, how many residues carry any probability.
            Where this is small the ``log R`` reference is generous and the
            information content is correspondingly flattering.
        consensus: ``(length,)`` residue indices; ``consensus_probability``
            their probabilities.
        residues: the alphabet, as a string, so a column can label itself.
        lo_nats, hi_nats: the band in surprisal, after any infinity was
            resolved against the grid.
        mass, log_mass: this band's mass in the chosen measure. ``log_mass``
            is primary; a far-tail band underflows a float.
        fraction: this band's share of the mass the grid held.
        resolution_error: how far the grid was from resolving the band, or
            ``None`` when no grid was involved.
        position_mass: ``(length,)`` or ``None``, the per-position estimates
            whose disagreement ``resolution_error`` summarises.
    """

    __slots__ = ('probability', 'height_bits', 'height_nats',
                 'entropy_bits', 'entropy_nats',
                 'information_bits', 'information_nats',
                 'n_support', 'consensus', 'consensus_probability',
                 'position_mass', 'residues', 'lo_nats', 'hi_nats',
                 'log_mass', 'mass', 'fraction', 'resolution_error',
                 'max_entropy_bits', 'max_entropy_nats')

    def __init__(self, **kw) -> None:
        for k in self.__slots__:
            setattr(self, k, kw[k])

    @property
    def length(self) -> int:
        return int(self.probability.shape[0])

    @property
    def n_residues(self) -> int:
        return int(self.probability.shape[1])

    @property
    def consensus_sequence(self) -> str:
        """The argmax residue at each position, as a string.

        The first thing anyone computes from a logo, and the one place a
        caller would otherwise have to know the alphabet's order.
        """
        return "".join(self.residues[int(a)] for a in self.consensus)

    def as_dict(self, unit: str = "bits") -> dict:
        """The arrays a plotting package wants, keyed by residue.

        ``{residue: array of length `length`}`` of stack heights, which is
        the shape logomaker and ggseqlogo both take.
        """
        if unit not in ("bits", "nats"):
            raise ValueError(f"unit must be 'bits' or 'nats', got {unit!r}")
        h = self.height_bits if unit == "bits" else self.height_nats
        return {self.residues[a]: h[:, a].copy() for a in range(self.n_residues)}

    def __repr__(self) -> str:
        band = f"[{self.lo_nats:.4g}, {self.hi_nats:.4g}) nats"
        res = ("exact" if self.resolution_error is None
               else f"resolution_error={self.resolution_error:.2g}")
        return (f"Logo(length={self.length}, band={band}, "
                f"fraction={self.fraction:.4f}, {res})")


class LogoSet:
    """The logos of one length, and what they were built from.

    Attributes:
        logos: one :class:`Logo` per band, in ascending surprisal -- which is
            DESCENDING probability, so ``logos[0]`` is the commonest band.
        exact: true when no grid error is possible. It describes the answer,
            not which function produced it: a single band covering the whole
            grid is exact because there is no interior edge for a sequence to
            be misclassified across.
        spacing: the grid, or ``None`` when none was used.
        on_grid_total: the mass the grid actually held. Compare with
            ``length_total`` to find a grid that is too SHORT; compare a
            band's ``resolution_error`` to find one that is too COARSE.
        length_total: the whole length's mass in the same measure.
        max_entropy_bits, max_entropy_nats: the ceiling information is
            measured against, ``log(reference_residues)``.
    """

    __slots__ = ('logos', 'length', 'residues', 'reference_residues',
                 'max_entropy_bits', 'max_entropy_nats', 'on_grid_total',
                 'length_total', 'spacing', 'exact', 'counting')

    def __init__(self, **kw) -> None:
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __len__(self) -> int:
        return len(self.logos)

    def __getitem__(self, i):
        return self.logos[i]

    def __iter__(self):
        return iter(self.logos)

    @property
    def grid_shortfall(self) -> float:
        """Fraction of the length's mass the grid failed to hold.

        Zero when the grid was long enough. A grid too SHORT loses mass off
        its top end; that is a different fault from a grid too COARSE, which
        misplaces mass across a band edge and shows up in
        ``resolution_error``. Neither implies the other.
        """
        if not self.length_total > 0.0:
            return 0.0
        return max(0.0, 1.0 - self.on_grid_total / self.length_total)

    def __repr__(self) -> str:
        measure = 'counting' if self.counting else 'generated'
        return (f"LogoSet(length={self.length}, bands={len(self.logos)}, "
                f"measure={measure}, exact={self.exact})")
