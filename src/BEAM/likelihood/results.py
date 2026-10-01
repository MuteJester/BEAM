"""What :func:`repertoire_likelihood` and :func:`pair_likelihood` return.

Frozen, holding read-only arrays, and defining no ``__float__``: every piece
but the totals carries a caveat, and a caveat has to travel with its number.
Each :class:`Piece` names its own ``method`` and says whether it is ``exact``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Piece", "RepertoireLikelihood", "PairLikelihood"]


def _frozen(a) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Piece:
    """One factor of the chain, conditioned on every factor before it.

    Attributes:
        name: ``richness``, ``sharing``, ``identity``, ``profile`` or
            ``assignment``.
        question: The conditional probability this piece is, in words.
        log_prob: Its natural log.
        exact: Whether ``log_prob`` is exact. Pieces resting on a
            saddlepoint are not, and the piece after each is an exact quantity
            minus that approximation, so it carries the same error with the
            opposite sign. The SUM of the pieces is exact. A repertoire that
            leaves nothing to approximate after the set (every count 1) has
            its profile and assignment labelled exact.
        method: How ``log_prob`` was obtained.
        error: Size of the approximation in ``log_prob``, in nats: the
            saddlepoint's ``sum_j 1/(12 v_j)`` over its tilted variances
            ``v_j`` (an estimate, not a ceiling), plus, for the single-
            repertoire ``richness`` and ``identity``, the spectrum binning
            error measured against half the bins. Above a nat, read only the
            piece's sum with its neighbour; the method says so. The pair
            pieces' binning is not in it.
        observed: The summary this piece is about, when it has one: the number
            of unique sequences, the number shared, the count of sequences in
            each abundance bin. Empty for ``identity`` and ``assignment``.
        expected: The model's exact expectation of ``observed``, same shape.
    """
    name: str
    question: str
    log_prob: float
    exact: bool
    method: str
    error: float
    observed: np.ndarray
    expected: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "observed", _frozen(self.observed))
        object.__setattr__(self, "expected", _frozen(self.expected))

    def __repr__(self) -> str:
        tag = "exact" if self.exact else "approximate"
        return f"Piece({self.name}: log {self.question} = {self.log_prob:.6g}, {tag})"


@dataclass(frozen=True, eq=False)
class RepertoireLikelihood:
    """``log P(repertoire | model)``, whole and in pieces.

    Attributes:
        log_likelihood: EXACT. ``log N! - sum log c! + sum c log P(s)``: the
            probability of these read counts when N reads are independent
            draws from the model. ``-inf`` if the model cannot produce one of
            the sequences.
        poissonized_log_likelihood: EXACT. The same counts when every sequence
            is an independent Poisson(N P(s)) and so the total is free. It is
            what the pieces multiply to, and it differs from
            ``log_likelihood`` by exactly ``log_prob_total_reads``.
        log_prob_total_reads: EXACT. ``log Poisson(N; N)``, the chance the free
            total lands on N.
        pieces: ``richness``, ``identity``, ``profile``, ``assignment``, in
            chain order. Empty when the repertoire is impossible.
        n_reads, n_unique: N and the number of distinct sequences, after
            repeated sequences were merged.
        n_unspellable: Sequences the model gives probability zero.
        model_mass: Total probability of the model's spectrum, 1 for a model
            over its whole support.
        bins: Spectrum bins behind the saddlepoint pieces; None when the
            caller passed the atoms in.
        profile_edges: Lower edges of the abundance bins.
    """
    log_likelihood: float
    poissonized_log_likelihood: float
    log_prob_total_reads: float
    pieces: tuple
    n_reads: int
    n_unique: int
    n_unspellable: int
    model_mass: float
    bins: int | None
    profile_edges: tuple

    def piece(self, name: str) -> Piece:
        for p in self.pieces:
            if p.name == name:
                return p
        raise KeyError(name)

    def __repr__(self) -> str:
        return (f"RepertoireLikelihood(log P = {self.log_likelihood:.6g} exact; "
                f"N={self.n_reads}, n={self.n_unique}; "
                f"{len(self.pieces)} pieces, see .pieces)")


@dataclass(frozen=True, eq=False)
class PairLikelihood:
    """``log P(R1, R2 | model)`` for two repertoires under ONE model.

    Given the model the two are independent, so the totals are the sums of the
    two single totals. What the pair adds is the chain through what they
    share: ``richness``, ``sharing``, ``identity``, ``profile``,
    ``assignment``.

    Attributes:
        log_likelihood, poissonized_log_likelihood, log_prob_total_reads:
            EXACT, as in :class:`RepertoireLikelihood`, summed over the two.
        pieces: In chain order; ``observed``/``expected`` of the per-repertoire
            pieces have one row per repertoire. Empty when impossible.
        n_reads, n_unique, n_unspellable: One entry per repertoire.
        n_shared: Distinct sequences present in both.
        shared_counts_log_prob, private_counts_log_prob: EXACT. The read
            counts given which sequences were seen, split by whether the
            sequence is in both repertoires. They sum to ``profile`` plus
            ``assignment``.
        model_mass, bins, profile_edges: As in :class:`RepertoireLikelihood`.
    """
    log_likelihood: float
    poissonized_log_likelihood: float
    log_prob_total_reads: float
    pieces: tuple
    n_reads: tuple
    n_unique: tuple
    n_unspellable: tuple
    n_shared: int
    shared_counts_log_prob: float
    private_counts_log_prob: float
    model_mass: float
    bins: int | None
    profile_edges: tuple

    def piece(self, name: str) -> Piece:
        for p in self.pieces:
            if p.name == name:
                return p
        raise KeyError(name)

    def __repr__(self) -> str:
        return (f"PairLikelihood(log P = {self.log_likelihood:.6g} exact; "
                f"N={self.n_reads}, n={self.n_unique}, shared={self.n_shared}; "
                f"{len(self.pieces)} pieces, see .pieces)")
