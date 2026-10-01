"""``P(repertoire | model)`` as a chain of conditional pieces.

THE MODEL OF A REPERTOIRE. An abundance-weighted BEAM gives ``P(s)``, the
probability of one read. A repertoire of N reads is N independent draws, so

    log P(R | M) = log N! - sum_s log c_s! + sum_s c_s log P(s)

which is exact and is ``log_likelihood``. Every piece below is a factor of
this same number, never an addition to it.

THE CHAIN. Let the total be free: each sequence independently Poisson with
mean ``lambda_s = N P(s)``. The Poisson field factorises exactly as

    P(counts) = P(n) * P(S | n) * P(F | S) * P(counts | S, F)

    n       how many distinct sequences were seen            richness
    S       which ones                                       identity
    F       how many were seen once, twice, ..., per bin     profile
    counts  which sequence carries which count               assignment

Each later quantity determines the earlier ones -- the set S fixes n, the
counts fix F -- so every conditional is a ratio, ``P(later) / P(earlier)``,
and conditioning costs nothing beyond each summary's own probability. The
product is ``P(Poisson field) = P(multinomial) * Poisson(N; N)``, exactly.

WHAT IS EXACT.
  * ``P(S)``: the observed sequences present, every other absent,
    ``prod_S (1 - e^-lambda_s) * exp(-N (mass - sum_S P(s)))``.
  * ``P(counts | S)``: each seen sequence a zero-truncated Poisson.
  * The expectations reported with each summary, up to the spectrum binning
    for the ones summed over the whole model.
WHAT IS NOT.
  * ``P(n)``, ``P(k shared | n1, n2)`` and ``P(F | S)`` are probabilities of
    category totals of independent units, by saddlepoint
    (:mod:`.lattice`; each piece reports the size of its error, and a
    category nearly fixed under the tilt is convolved exactly instead). ``P(n)`` and the
    sharing piece also sum over the model's spectrum, binned: relative error
    about ``spacing^2 / 8``, 4e-6 at 2048 bins.
  * ``identity`` and ``assignment`` are exact quantities divided by those,
    so they carry the same error with the opposite sign. The total does not.

TWO REPERTOIRES UNDER ONE MODEL are independent given it, so their totals
add. The pair chain threads the overlap through the middle:

    P(n1, n2) * P(k | n1, n2) * P(S1, S2 | n1, n2, k) * P(F1, F2 | S1, S2)
      * P(counts | S1, S2, F1, F2)

where ``(n1 - k, n2 - k, k)`` are the totals of a four-way categorical per
spectrum bin: in neither, only the first, only the second, both.

WHAT THIS DOES NOT MODEL. Reads of one clone are not independent draws, and a
sequence's counts in two samples of one person are not independent. Both are
properties of the data this model lacks, and they show up as the summaries
landing far from their expectations, not as a failure of the arithmetic.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..errors import BeamValueError
from .lattice import categorical_sum_log_pmf
from .results import PairLikelihood, Piece, RepertoireLikelihood

__all__ = ["DEFAULT_PROFILE_EDGES", "repertoire_likelihood", "pair_likelihood"]

#: Abundance bins 1, 2, 3, 4, 5, 6-10, 11-100 and over 100 reads.
DEFAULT_PROFILE_EDGES = (1, 2, 3, 4, 5, 6, 11, 101)

_SADDLE = "saddlepoint; see `error` for its size"


# ── Inputs ──────────────────────────────────────────────────────

def _check_edges(edges) -> tuple:
    try:
        e = tuple(int(x) for x in edges)
    except (TypeError, ValueError):
        raise BeamValueError(f"profile_edges must be integers, got {edges!r}") from None
    if not e or e[0] != 1 or any(b <= a for a, b in zip(e, e[1:])) \
            or any(x != y for x, y in zip(e, edges)):
        raise BeamValueError("profile_edges must be strictly increasing "
                             f"integers starting at 1, got {tuple(edges)!r}")
    return e


def _collapse(sequences: Sequence[str], counts: Iterable) -> tuple[list, np.ndarray]:
    """Merge repeated sequences, adding their reads. Counts are read counts,
    so they must be positive integers."""
    seqs = list(sequences)
    c = np.asarray(list(counts), dtype=float)
    if len(seqs) != c.size:
        raise BeamValueError(f"{len(seqs)} sequences but {c.size} counts")
    if not seqs:
        raise BeamValueError("an empty repertoire has no likelihood to compute")
    if not np.all(np.isfinite(c)) or np.any(c < 1) or np.any(c != np.floor(c)):
        raise BeamValueError("counts must be positive integers: they are read "
                             "counts, and the likelihood is of whole reads")
    merged: dict[str, int] = {}
    for s, v in zip(seqs, c.astype(np.int64)):
        merged[s] = merged.get(s, 0) + int(v)
    return list(merged), np.fromiter(merged.values(), dtype=np.int64,
                                     count=len(merged))


def _log_factorials(c: np.ndarray) -> np.ndarray:
    u, inv = np.unique(c, return_inverse=True)
    return np.array([math.lgamma(int(x) + 1) for x in u])[inv]


def _spectrum(model, bins, atoms):
    if atoms is None:
        atoms = model.pseq_analysis().atoms(bins)
    elif atoms.length is not None:
        raise BeamValueError("atoms must cover the whole model, not one length "
                             f"(got length={atoms.length})")
    else:
        bins = None
    p = np.asarray(atoms.probabilities, dtype=float)
    m = np.asarray(atoms.multiplicities, dtype=float)
    keep = (p > 0) & (m > 0)
    return p[keep], m[keep], float(atoms.mass), bins


# ── The per-repertoire parts ───────────────────────────────────

def _ztp_log_bins(lam: np.ndarray, edges: tuple) -> np.ndarray:
    """``(n, B)`` log zero-truncated-Poisson mass per abundance bin.

    The open top bin is a series from its edge up when ``lambda`` is below the
    edge, where ``1 - lower`` would cancel to nothing, and ``1 - lower`` above
    it, where the lower mass is at most about a half.
    """
    lam = np.asarray(lam, dtype=float)
    n, top = lam.size, edges[-1]
    log_lam = np.log(lam)
    log_z = np.log(-np.expm1(-lam))
    out = np.full((n, len(edges)), -np.inf)
    b = 0
    for r in range(1, top):
        while b + 1 < len(edges) and r >= edges[b + 1]:
            b += 1
        term = r * log_lam - lam - math.lgamma(r + 1) - log_z
        out[:, b] = np.logaddexp(out[:, b], term)
    lower = np.logaddexp.reduce(out[:, :-1], axis=1) if len(edges) > 1 \
        else np.full(n, -np.inf)
    tail = np.empty(n)
    big = lam >= top
    with np.errstate(divide="ignore"):
        tail[big] = np.log1p(-np.exp(lower[big]))
    small = ~big
    if np.any(small):
        ls = lam[small]
        first = top * np.log(ls) - ls - math.lgamma(top + 1) - log_z[small]
        total = np.ones(ls.size)
        term = np.ones(ls.size)
        k = 0
        while True:
            k += 1
            term = term * ls / (top + k)
            total += term
            if np.all(term <= 1e-17 * total):
                break
        tail[small] = first + np.log(total)
    out[:, -1] = tail
    return out


def _profile_counts(c: np.ndarray, edges: tuple) -> np.ndarray:
    hi = edges[1:] + (np.inf,)
    return np.array([((c >= lo) & (c < h)).sum() for lo, h in zip(edges, hi)],
                    dtype=float)


class _Rep:
    """Everything about one repertoire that does not involve the spectrum."""

    def __init__(self, model, sequences, counts, edges):
        self.seqs, self.c = _collapse(sequences, counts)
        self.N = int(self.c.sum())
        self.n = len(self.seqs)
        self.lp = model.log_pseq_batch(self.seqs)
        self.n_unspellable = int(np.sum(~np.isfinite(self.lp)))
        self.lfact = _log_factorials(self.c)
        N = self.N
        self.log_pois_N = N * math.log(N) - N - math.lgamma(N + 1)
        if self.n_unspellable:
            return
        self.multinomial = float(math.lgamma(N + 1) - self.lfact.sum()
                                 + np.sum(self.c * self.lp))
        lam = N * np.exp(self.lp)
        self.log_seen = np.log(-np.expm1(-lam))
        self.captured = float(np.sum(np.exp(self.lp)))
        # log P(c_s | seen), each seen sequence a zero-truncated Poisson
        self.per_seq_counts = (self.c * np.log(lam) - lam - self.lfact
                               - self.log_seen)
        self.log_counts = float(self.per_seq_counts.sum())
        log_q = _ztp_log_bins(lam, edges)
        self.F = _profile_counts(self.c, edges)
        self.F_expected = np.exp(log_q).sum(axis=0)
        self.log_profile, _, self.profile_error = categorical_sum_log_pmf(
            log_q, np.ones(self.n), self.F)

    def log_presence(self, mass: float) -> float:
        """``log P(exactly these sequences seen)``: every other one absent."""
        return float(self.log_seen.sum()) - self.N * (mass - self.captured)


def _log_richness(p, m, N, n):
    log_absent = -N * p
    log_present = np.log(-np.expm1(-N * p))
    lq = np.column_stack([log_absent, log_present])
    # The absent total is implied, never formed: m.sum() can be 4.5e24, where
    # m.sum() - n is m.sum() and n has vanished.
    val, _, err = categorical_sum_log_pmf(lq, m, np.array([0.0, n]), rest=0)
    return val, float(np.sum(m * np.exp(log_present))), err


# ── Public ──────────────────────────────────────────────────────

def repertoire_likelihood(model, sequences: Sequence[str], counts: Iterable, *,
                          bins: int = 2048,
                          profile_edges: Sequence[int] = DEFAULT_PROFILE_EDGES,
                          atoms=None) -> RepertoireLikelihood:
    """``log P(repertoire | model)``, exact, and the chain it factors into.

    Args:
        model: An abundance-weighted :class:`BeamModel`: ``P(s)`` is the
            probability of one read.
        sequences: The repertoire's sequences. Repeats are merged and their
            reads added, which is what two nucleotide clonotypes of one
            amino-acid sequence are to an amino-acid model.
        counts: Read (template) count of each, positive integers.
        bins: Spectrum resolution for the pieces summed over the whole model.
        profile_edges: Lower edges of the abundance bins, from 1 up. The cost
            of the profile piece grows with the last edge.
        atoms: ``model.pseq_analysis().atoms(bins)``, when scoring many
            repertoires against one model: the spectrum is the expensive step
            and depends only on the model.

    Returns:
        :class:`RepertoireLikelihood`. Its totals are exact; each piece says
        whether it is.
    """
    edges = _check_edges(profile_edges)
    r = _Rep(model, sequences, counts, edges)
    if r.n_unspellable:
        return RepertoireLikelihood(
            log_likelihood=-math.inf, poissonized_log_likelihood=-math.inf,
            log_prob_total_reads=r.log_pois_N, pieces=(), n_reads=r.N,
            n_unique=r.n, n_unspellable=r.n_unspellable, model_mass=math.nan,
            bins=bins if atoms is None else None, profile_edges=edges)
    p, m, mass, used_bins = _spectrum(model, bins, atoms)
    log_n, n_expected, err_n = _log_richness(p, m, r.N, r.n)
    log_S = r.log_presence(mass)
    pieces = (
        Piece("richness", "P(n unique)", log_n, False,
              f"{_SADDLE}, over the model spectrum", err_n, [r.n], [n_expected]),
        Piece("identity", "P(these sequences | n)", log_S - log_n, False,
              "exact P(sequences) divided by the richness saddlepoint", err_n,
              [], []),
        Piece("profile", "P(abundance pattern | sequences)", r.log_profile,
              False, _SADDLE, r.profile_error, r.F, r.F_expected),
        Piece("assignment", "P(which sequence has which count | pattern)",
              r.log_counts - r.log_profile, False,
              "exact P(counts | sequences) divided by the profile saddlepoint",
              r.profile_error, [], []),
    )
    return RepertoireLikelihood(
        log_likelihood=r.multinomial,
        poissonized_log_likelihood=log_S + r.log_counts,
        log_prob_total_reads=r.log_pois_N, pieces=pieces, n_reads=r.N,
        n_unique=r.n, n_unspellable=0, model_mass=mass, bins=used_bins,
        profile_edges=edges)


def pair_likelihood(model, sequences_1: Sequence[str], counts_1: Iterable,
                    sequences_2: Sequence[str], counts_2: Iterable, *,
                    bins: int = 2048,
                    profile_edges: Sequence[int] = DEFAULT_PROFILE_EDGES,
                    atoms=None) -> PairLikelihood:
    """``log P(R1, R2 | model)`` for two repertoires under ONE model, with the
    chain threaded through the sequences they share.

    The model is usually their union, which is the "same source" hypothesis;
    either donor's own model can also score the other's repertoire directly,
    since every sequence up to ``max_length`` is possible under the prior.
    Arguments are as for :func:`repertoire_likelihood`, once per repertoire.
    """
    edges = _check_edges(profile_edges)
    a = _Rep(model, sequences_1, counts_1, edges)
    b = _Rep(model, sequences_2, counts_2, edges)
    common = dict(n_reads=(a.N, b.N), n_unique=(a.n, b.n),
                  n_unspellable=(a.n_unspellable, b.n_unspellable),
                  log_prob_total_reads=a.log_pois_N + b.log_pois_N,
                  profile_edges=edges)
    shared = set(a.seqs) & set(b.seqs)
    if a.n_unspellable or b.n_unspellable:
        return PairLikelihood(
            log_likelihood=-math.inf, poissonized_log_likelihood=-math.inf,
            pieces=(), n_shared=len(shared), shared_counts_log_prob=math.nan,
            private_counts_log_prob=math.nan, model_mass=math.nan,
            bins=bins if atoms is None else None, **common)
    p, m, mass, used_bins = _spectrum(model, bins, atoms)
    k = len(shared)

    log_n1, e1, err1 = _log_richness(p, m, a.N, a.n)
    log_n2, e2, err2 = _log_richness(p, m, b.N, b.n)
    log_seen1 = np.log(-np.expm1(-a.N * p))
    log_seen2 = np.log(-np.expm1(-b.N * p))
    lq = np.column_stack([-(a.N + b.N) * p, log_seen1 - b.N * p,
                          -a.N * p + log_seen2, log_seen1 + log_seen2])
    x = np.array([0.0, a.n - k, b.n - k, k])      # "in neither" is implied
    log_joint, _, err_joint = categorical_sum_log_pmf(lq, m, x, rest=0)
    k_expected = float(np.sum(m * np.exp(log_seen1 + log_seen2)))

    log_S = a.log_presence(mass) + b.log_presence(mass)
    log_profile = a.log_profile + b.log_profile
    log_counts = a.log_counts + b.log_counts
    in_b = np.array([s in shared for s in a.seqs])
    in_a = np.array([s in shared for s in b.seqs])
    shared_counts = float(a.per_seq_counts[in_b].sum()
                          + b.per_seq_counts[in_a].sum())

    pieces = (
        Piece("richness", "P(n1, n2 unique)", log_n1 + log_n2, False,
              f"{_SADDLE}, over the model spectrum", err1 + err2,
              [a.n, b.n], [e1, e2]),
        Piece("sharing", "P(k shared | n1, n2)", log_joint - log_n1 - log_n2,
              False, "joint (n1, n2, k) saddlepoint over the model spectrum, "
              "divided by the richness saddlepoints", err_joint + err1 + err2,
              [k], [k_expected]),
        Piece("identity", "P(these sequences | n1, n2, k)", log_S - log_joint,
              False, "exact P(both sequence sets) divided by the joint "
              "saddlepoint", err_joint, [], []),
        Piece("profile", "P(both abundance patterns | sequences)",
              log_profile, False, _SADDLE, a.profile_error + b.profile_error,
              [a.F, b.F], [a.F_expected, b.F_expected]),
        Piece("assignment", "P(which sequence has which count | patterns)",
              log_counts - log_profile, False, "exact P(counts | sequences) "
              "divided by the profile saddlepoints",
              a.profile_error + b.profile_error, [], []),
    )
    return PairLikelihood(
        log_likelihood=a.multinomial + b.multinomial,
        poissonized_log_likelihood=log_S + log_counts, pieces=pieces,
        n_shared=k, shared_counts_log_prob=shared_counts,
        private_counts_log_prob=log_counts - shared_counts, model_mass=mass,
        bins=used_bins, **common)
