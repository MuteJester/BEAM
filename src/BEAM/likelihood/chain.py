"""``P(repertoire | model)`` as a chain of conditional pieces.

THE MODEL OF A REPERTOIRE. An abundance-weighted BEAM gives ``P(s)``, the
probability of one read. A repertoire of N reads is N independent draws, so

    log P(R | M) = log N! - sum_s log c_s! + sum_s c_s log P(s)

which is exact and is ``log_likelihood``. Every piece below is a factor of
this same number, never an addition to it.

THE CHAIN, GIVEN N READS. The total factorises exactly as

    P(counts | N) = P(n | N) * P(S | n, N) * P(F | S, N) * P(counts | S, F, N)

    n       how many distinct sequences were seen            richness
    S       which ones                                       identity
    F       how many were seen once, twice, ..., per bin     profile
    counts  which sequence carries which count               assignment

Each later quantity determines the earlier ones -- the set S fixes n, the
counts fix F -- so every conditional is a ratio, ``P(later) / P(earlier)``,
and nothing is counted twice. The four multiply to the multinomial itself.

Every piece is conditioned on N because that is the question: how surprising
are n distinct sequences GIVEN N reads. Left free, the read total would make
n look far more variable than it is where most reads are new, which is where
repertoires are (on a small model at N = 30: a spread of 4.8 instead of 1.5).
The Poisson field does the work: with each sequence independently Poisson
``lambda_s = N P(s)``, conditioning on the total is a ratio,
``P(x | N) = P(x, N) / P(N)``, ``P(N) = Poisson(N; N)``, and the joints are
sums of independent units (:mod:`.fixed_total`).

WHAT IS EXACT.
  * The total, and ``P(S)``: the observed sequences present, every other
    absent, ``prod_S (1 - e^-lambda_s) * exp(-N (mass - sum_S P(s)))``.
  * ``P(counts | S)``: each seen sequence a zero-truncated Poisson.
WHAT IS NOT.
  * ``P(n, N)`` over the whole model, and ``P(N | S)`` and ``P(F, N | S)``
    over the seen sequences, are sums of independent units, by saddlepoint;
    each piece reports the size of its error. ``P(n, N)`` also sums over the
    model's spectrum, binned; that error is measured against half the bins
    and added to the richness piece's.
  * A total at the edge of what the pattern allows (every count at its bin's
    floor or ceiling) fixes every count, so P(F, N | S) becomes the chance
    that every count sits there -- a categorical sum the lattice takes, its
    nearly fixed bins convolved exactly -- instead of the joint saddlepoint,
    which is singular there. Where a split's error still passes a nat, its
    method says so: read only profile + assignment, whose one approximation
    is then P(N | S).
  * Each joint appears in two neighbouring pieces with opposite signs, so the
    approximations move value between pieces and never change their sum.
  * The expected abundance pattern is to first order: each sequence's bin
    probabilities under the tilt that makes N the expected read total.

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

from .. import _cbeam as _c
from ..errors import BeamValueError
from .fixed_total import (counts_given_set_log_pmf, expected_bins_given_reads,
                          present_and_reads_log_pmf)
from .lattice import categorical_sum_log_pmf
from .results import (AbundanceSpectrum, PairLikelihood, Piece,
                      RepertoireLikelihood)

__all__ = ["DEFAULT_PROFILE_EDGES", "repertoire_likelihood", "pair_likelihood",
           "abundance_spectrum", "auto_profile_edges"]

#: Abundance bins 1, 2, 3, 4, 5, 6-10, 11-100 and over 100 reads. Fixed bins
#: are convenient to compare but can be degenerate: "over 100 reads" is
#: reached only by clones the model gives essentially no chance of, and a
#: coordinate with no variance is one no tilt can move. Pass ``"auto"`` for
#: bins chosen so the model expects enough sequences in each.
DEFAULT_PROFILE_EDGES = (1, 2, 3, 4, 5, 6, 11, 101)

#: The fewest sequences ``auto`` leaves the model expecting in a bin.
AUTO_MIN_EXPECTED = 5.0

#: Below this expected count a clone size is Poisson-like, not Gaussian, so
#: :class:`AbundanceSpectrum` reports no standardised departure for it.
GAUSSIAN_MIN_EXPECTED = 10.0

_SADDLE = "saddlepoint; see `error` for its size"


# ── Inputs ──────────────────────────────────────────────────────

def _count_spectrum(lam, ks):
    """Exact ``(E[F_k], Var[F_k])`` per count, in C (math/ztp_moments.c)."""
    ks = np.ascontiguousarray(np.asarray(ks, dtype=float))
    raw = _c.ztp_count_spectrum(np.ascontiguousarray(np.asarray(lam, dtype=float)), ks)
    v = np.frombuffer(raw, dtype=float)
    return v[:ks.size].copy(), v[ks.size:].copy()


def auto_profile_edges(lam, max_count: int,
                       min_expected: float = AUTO_MIN_EXPECTED) -> tuple:
    """Abundance bins the model expects at least ``min_expected`` sequences in.

    Counts are merged upward from 1 until each bin's exact expected count
    clears the floor, the last bin running open to infinity. That is what
    stops a bin being degenerate: a bin the model expects 1e-121 sequences in
    has no variance under any tilt, so the saddlepoint's Hessian is singular
    and its Newton cannot converge -- which is how fixed bins lost 9.3% of one
    cohort's donors. With a floor of five the tilt needed is a few nats.
    """
    ks = np.arange(1, int(max_count) + 1, dtype=float)
    e, _ = _count_spectrum(lam, ks)
    edges, run = [1], 0.0
    for k, ek in zip(ks, e):
        run += ek
        if run >= min_expected and int(k) + 1 <= int(max_count):
            edges.append(int(k) + 1)
            run = 0.0
    while len(edges) > 1 and run < min_expected:      # the open bin, too thin
        edges.pop()
        run += float(e[edges[-1] - 1:].sum()) if edges[-1] <= len(e) else 0.0
        break
    return tuple(edges)


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

    def __init__(self, model, sequences, counts, edges, given_total=False):
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
        log_lam = math.log(N) + np.asarray(self.lp, dtype=float)
        lam = np.exp(log_lam)
        # log(1 - e^-lam), which is log lam for a lam too small to hold
        tiny = lam < 1e-8
        self.log_seen = np.empty_like(lam)
        self.log_seen[tiny] = log_lam[tiny] + np.log1p(-0.5 * lam[tiny])
        self.log_seen[~tiny] = np.log(-np.expm1(-lam[~tiny]))
        self.captured = float(np.sum(np.exp(self.lp)))
        # log P(c_s | seen), each seen sequence a zero-truncated Poisson
        self.per_seq_counts = (self.c * log_lam - lam - self.lfact
                               - self.log_seen)
        self.log_counts = float(self.per_seq_counts.sum())
        self.F = _profile_counts(self.c, edges)
        if given_total:
            # P(N | S) and P(F, N | S): the read total held at N
            self.log_N_given_S, self.exact_N_given_S, self.err_N_given_S = \
                counts_given_set_log_pmf(lam, N)
            self.log_FN_given_S, self.exact_FN_given_S, self.err_FN_given_S = \
                counts_given_set_log_pmf(lam, N, edges, self.F)
            self.F_expected = expected_bins_given_reads(lam, N, edges)
            return
        log_q = _ztp_log_bins(lam, edges)
        self.F_expected = np.exp(log_q).sum(axis=0)
        self.log_profile, _, self.profile_error = categorical_sum_log_pmf(
            log_q, np.ones(self.n), self.F)

    def log_presence(self, mass: float) -> float:
        """``log P(exactly these sequences seen)``: every other one absent."""
        return float(self.log_seen.sum()) - self.N * (mass - self.captured)


def _log_richness_given_reads(p, m, N, n, mass):
    """``log P(n, N)``, ``log P(N)``, the expected n at exactly N reads
    (``sum 1 - (1 - p)^N``), whether ``P(n, N)`` is exact, and its
    saddlepoint error."""
    log_nN, exact, err = present_and_reads_log_pmf(p, m, N, n, N)
    log_pN = N * math.log(N * mass) - N * mass - math.lgamma(N + 1)
    expected = float(np.sum(m * -np.expm1(N * np.log1p(-p))))
    return log_nN, log_pN, expected, exact, err


def _binning_error(fine, coarse):
    """The binning error left in a value computed at ``bins`` from the same
    value at ``bins / 2``: the error falls with the square of the grid
    spacing, so the finer one keeps a third of their difference."""
    return abs(fine - coarse) / 3.0


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
                          bins: int = 8192,
                          profile_edges="auto",
                          atoms=None, atoms_coarse=None) -> RepertoireLikelihood:
    """``log P(repertoire | model)``, exact, and the chain it factors into.

    Args:
        model: An abundance-weighted :class:`BeamModel`: ``P(s)`` is the
            probability of one read.
        sequences: The repertoire's sequences. Repeats are merged and their
            reads added, which is what two nucleotide clonotypes of one
            amino-acid sequence are to an amino-acid model.
        counts: Read (template) count of each, positive integers.
        bins: Spectrum resolution for the richness piece, the one summed over
            the whole model. It is also computed at ``bins // 2``, and its
            reported error includes the binning error that comparison
            measures (on an Emerson donor at 2048 bins the expected richness
            was 85 sequences high and the piece 8 nats off).
        profile_edges: Lower edges of the abundance bins, from 1 up, or
            ``"auto"`` (the default) to choose them so the model expects at
            least `AUTO_MIN_EXPECTED` sequences in each. Fixed bins can be
            degenerate -- "over 100 reads" holds only clones the model gives
            essentially no chance of, and a bin with no variance is one no
            tilt can move, which lost 9.3% of one cohort's donors outright and
            left others with a split worth nothing. Automatic bins fixed every
            such case and cut the reported error fiftyfold on the rest. They
            depend on the model and the repertoire, so a `profile` piece is
            comparable between repertoires only at the same `profile_edges`;
            the bins used are reported back. For the clone-size comparison
            that needs no bins at all, see :func:`abundance_spectrum`.
        atoms: ``model.pseq_analysis().atoms(bins)``, when scoring many
            repertoires against one model: the spectrum is the expensive step
            and depends only on the model.
        atoms_coarse: the same at ``bins // 2``, for the binning error. When
            ``atoms`` is passed without it, that error is not estimated, and
            the richness piece says so.

    Returns:
        :class:`RepertoireLikelihood`. Its totals are exact; each piece says
        whether it is.
    """
    seqs0, c0 = _collapse(sequences, counts)
    if isinstance(profile_edges, str):
        if profile_edges != "auto":
            raise BeamValueError("profile_edges must be a sequence of integers "
                                 f"or \"auto\", got {profile_edges!r}")
        lam0 = int(c0.sum()) * np.exp(np.asarray(model.log_pseq_batch(seqs0), float))
        edges = _check_edges(auto_profile_edges(lam0, int(c0.max())))
    else:
        edges = _check_edges(profile_edges)
    r = _Rep(model, sequences, counts, edges, given_total=True)
    if r.n_unspellable:
        return RepertoireLikelihood(
            log_likelihood=-math.inf, poissonized_log_likelihood=-math.inf,
            log_prob_total_reads=r.log_pois_N, pieces=(), n_reads=r.N,
            n_unique=r.n, n_unspellable=r.n_unspellable, model_mass=math.nan,
            bins=bins if atoms is None else None, profile_edges=edges)
    if atoms is None:
        a = model.pseq_analysis()
        atoms, atoms_coarse = a.atoms(bins), a.atoms(max(2, bins // 2))
    p, m, mass, used_bins = _spectrum(model, bins, atoms)
    log_nN, log_pN, n_expected, _, err_n = _log_richness_given_reads(
        p, m, r.N, r.n, mass)
    how = "P(n, N) by saddlepoint over the model spectrum, over the exact P(N)"
    if atoms_coarse is not None:
        pc, mc, massc, _ = _spectrum(model, None, atoms_coarse)
        coarse = _log_richness_given_reads(pc, mc, r.N, r.n, massc)
        err_bin = _binning_error(log_nN - log_pN, coarse[0] - coarse[1])
        how += "; error includes the binning error against half the bins"
    else:
        err_bin = 0.0
        how += "; binning error NOT estimated (pass atoms_coarse)"
    err_n = err_n + err_bin
    log_S = r.log_presence(mass)
    split = ("P(pattern, N | sequences) over P(N | sequences), by saddlepoint"
             if r.err_FN_given_S <= 1.0 else
             "P(pattern, N | sequences) over P(N | sequences), by saddlepoint; "
             f"UNRELIABLE split (error {r.err_FN_given_S:.3g} nats): read only "
             "profile + assignment, whose one approximation is P(N | sequences)")
    pieces = (
        # richness and identity rest on the spectrum saddlepoint: never exact
        Piece("richness", "P(n unique | N reads)", log_nN - log_pN, False,
              how, err_n, [r.n], [n_expected]),
        Piece("identity", "P(these sequences | n, N)",
              log_S + r.log_N_given_S - log_nN, False,
              "exact P(sequences) times P(N | sequences), over P(n, N); "
              "both by saddlepoint", err_n + r.err_N_given_S, [], []),
        Piece("profile", "P(abundance pattern | sequences, N)",
              r.log_FN_given_S - r.log_N_given_S,
              r.exact_FN_given_S and r.exact_N_given_S,
              split, r.err_FN_given_S + r.err_N_given_S,
              r.F, r.F_expected),
        Piece("assignment", "P(which sequence has which count | pattern, N)",
              r.log_counts - r.log_FN_given_S, r.exact_FN_given_S,
              "exact P(counts | sequences) over the P(pattern, N | sequences) "
              "saddlepoint" + split[split.find(";"):] * (r.err_FN_given_S > 1.0),
              r.err_FN_given_S, [], []),
    )
    return RepertoireLikelihood(
        log_likelihood=r.multinomial,
        poissonized_log_likelihood=log_S + r.log_counts,
        log_prob_total_reads=log_pN, pieces=pieces, n_reads=r.N,
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


def abundance_spectrum(model, sequences: Sequence[str], counts: Iterable, *,
                       counts_wanted=None, max_count: int | None = None
                       ) -> AbundanceSpectrum:
    """The clone-size spectrum: how many sequences the model expects at each
    count, exactly, against how many the repertoire has.

    Given the set of sequences seen, each count is an independent
    zero-truncated Poisson with mean ``N P(s)``, so ``E[F_k]`` and
    ``Var[F_k]`` are sums over sequences -- exact, with no tilt, no
    saddlepoint and no binning. See :class:`AbundanceSpectrum`.

    Args:
        model: an abundance-weighted :class:`BeamModel`.
        sequences, counts: the repertoire, as for :func:`repertoire_likelihood`.
        counts_wanted: the clone sizes to report. The default is every size
            the repertoire actually has.
        max_count: report every size from 1 to this instead.

    Returns:
        :class:`AbundanceSpectrum`.
    """
    seqs, c = _collapse(sequences, counts)
    N = int(c.sum())
    lp = np.asarray(model.log_pseq_batch(seqs), dtype=float)
    if not np.all(np.isfinite(lp)):
        raise BeamValueError("the model cannot represent every sequence, so it "
                             "gives the repertoire no clone-size spectrum")
    lam = N * np.exp(lp)
    if counts_wanted is not None:
        ks = np.asarray(sorted({int(k) for k in counts_wanted}), dtype=np.int64)
    elif max_count is not None:
        ks = np.arange(1, int(max_count) + 1, dtype=np.int64)
    else:
        ks = np.unique(c)
    e, v = _count_spectrum(lam, ks.astype(float))
    obs = np.array([int(np.count_nonzero(c == k)) for k in ks], dtype=np.int64)
    sd = np.sqrt(v)
    # A standardised departure means nothing where the model expects almost
    # no sequences at this count: the count is Poisson-like, not Gaussian, and
    # dividing by a standard deviation of 1e-65 reports 1e136 for having seen
    # one. Those counts get NaN, and are read straight off `observed` against
    # `expected` instead.
    gaussian = e >= GAUSSIAN_MIN_EXPECTED
    z = np.full(e.shape, np.nan)
    np.divide(obs - e, sd, out=z, where=gaussian & (sd > 0))
    return AbundanceSpectrum(
        counts=ks, observed=obs, expected=e, sd=sd, z=z, n_sequences=len(seqs),
        n_reads=N, expected_reads=float(np.sum(lam / -np.expm1(-lam))))
