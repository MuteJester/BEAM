"""The independent reference for the likelihood group.

Every quantity here is computed by direct summation over an enumerated
support or by exact convolution, and shares no code with
``BEAM.likelihood``. A test that compares the two is comparing the library
against arithmetic, which is the only comparison that can find an error in it.

The model is abundance-weighted, as the repertoire models are: ``P(s)`` is the
probability of one READ, so a repertoire's counts are draws from it. It is a
NUCLEOTIDE model of short length because under the hierarchical prior the
support is every string over the alphabet up to ``max_length``: five residues
to length five is 3,905 strings, which can be enumerated, where twenty would
not.
"""
from __future__ import annotations

import itertools
import math
import random

from BEAM import BeamModel

LETTERS = "ACGTN"          # the whole nucleotide alphabet: the support
TRAIN = "ACGT"
MAX_LENGTH = 5

#: Headroom over the error a saddlepoint answer reports about itself,
#: sum_j 1/(12 v_j): a leading-order estimate, not a ceiling. The worst case
#: measured against exact convolution was 0.97 of it.
SLACK = 1.25


def tiny_weighted_model(seed: int, *, n: int = 120, lo: int = 3,
                        hi: int = 5) -> BeamModel:
    """An abundance-weighted model small enough to enumerate."""
    rng = random.Random(seed)
    seqs = ["".join(rng.choice(TRAIN) for _ in range(rng.randint(lo, hi)))
            for _ in range(n)]
    abund = [rng.randint(1, 30) for _ in seqs]
    return BeamModel(seqs, abundances=abund, alphabet="nt", order=2,
                     max_length=MAX_LENGTH)


def support(model: BeamModel) -> dict[str, float]:
    """``{sequence: P(sequence)}`` over every string up to ``max_length``,
    which under the hierarchical prior is the whole support."""
    out: dict[str, float] = {}
    for L in range(1, MAX_LENGTH + 1):
        for tup in itertools.product(LETTERS, repeat=L):
            s = "".join(tup)
            out[s] = math.exp(model.log_pseq(s))
    return out


def repertoire(sup: dict[str, float], n_unique: int, seed: int,
               *, big: int | None = None) -> tuple[list[str], list[int]]:
    """``n_unique`` distinct supported sequences with skewed read counts.

    Mostly ones and twos, a few larger, and optionally one clone of ``big``
    reads, so every profile bin the tests use is occupied by something.
    """
    rng = random.Random(seed)
    seqs = rng.sample(sorted(sup), n_unique)
    counts = [1 if rng.random() < 0.6 else rng.randint(2, 12) for _ in seqs]
    if big is not None:
        counts[0] = big
    return seqs, counts


def log_multinomial(sup, seqs, counts) -> float:
    """``log P(counts | N)``: N reads, each an independent draw."""
    N = sum(counts)
    return (math.lgamma(N + 1) - sum(math.lgamma(c + 1) for c in counts)
            + sum(c * math.log(sup[s]) for s, c in zip(seqs, counts)))


def log_poisson_field(sup, seqs, counts) -> float:
    """Every supported sequence an independent Poisson(N p), unseen ones at 0."""
    N = sum(counts)
    obs = dict(zip(seqs, counts))
    tot = 0.0
    for s, p in sup.items():
        lam = N * p
        c = obs.get(s, 0)
        tot += c * math.log(lam) - lam - math.lgamma(c + 1)
    return tot


def log_presence(sup, seqs, N) -> float:
    """``log P(exactly these sequences seen)`` at Poisson depth N."""
    S = set(seqs)
    tot = 0.0
    for s, p in sup.items():
        tot += math.log(-math.expm1(-N * p)) if s in S else -N * p
    return tot


def expected_richness(sup, N) -> float:
    return sum(-math.expm1(-N * p) for p in sup.values())


def expected_richness_at_reads(sup, N) -> float:
    """E[distinct | exactly N reads] = sum_s 1 - (1 - p)^N."""
    return sum(-math.expm1(N * math.log1p(-p)) for p in sup.values())


def expected_shared(sup, N1, N2) -> float:
    return sum((-math.expm1(-N1 * p)) * (-math.expm1(-N2 * p))
               for p in sup.values())


def exact_categorical_sum(unit_probs) -> dict[tuple[int, ...], float]:
    """Exact joint pmf of the category counts of independent units.

    ``unit_probs`` is one probability tuple per unit, all of length C. Returns
    ``{(count_0, ..., count_{C-1}): probability}`` by plain convolution.
    """
    C = len(unit_probs[0])
    dist = {tuple([0] * C): 1.0}
    for probs in unit_probs:
        nxt: dict[tuple[int, ...], float] = {}
        for key, w in dist.items():
            for j, q in enumerate(probs):
                if q == 0.0:
                    continue
                k = list(key)
                k[j] += 1
                k = tuple(k)
                nxt[k] = nxt.get(k, 0.0) + w * q
        dist = nxt
    return dist


def ztp_bin_probs(lam: float, edges) -> list[float]:
    """Zero-truncated Poisson(lam) mass in each bin ``[e_i, e_{i+1})``, the
    last bin open-ended. Direct summation, far past where the tail matters."""
    z = -math.expm1(-lam)
    pmf = lambda r: math.exp(r * math.log(lam) - lam - math.lgamma(r + 1)) / z
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else lo + 4000
        out.append(sum(pmf(r) for r in range(lo, hi)))
    return out


def exact_present_and_reads(probs, depth, n_max, total_max):
    """Joint pmf of (distinct sequences seen, total reads) in the Poisson
    field at ``depth``, ``D[n, N]``, by convolution over every sequence:
    each is Poisson(depth * p). Entries past the maxima are dropped, which
    leaves every entry inside them exact."""
    import numpy as np
    D = np.zeros((n_max + 1, total_max + 1))
    D[0, 0] = 1.0
    r = np.arange(total_max + 1)
    lgam = np.array([math.lgamma(k + 1) for k in r])
    for p in probs:
        lam = depth * p
        pr = np.exp(r * math.log(lam) - lam - lgam)
        new = D * pr[0]
        for k in range(1, total_max + 1):
            new[1:, k:] += D[:-1, :total_max + 1 - k] * pr[k]
        D = new
    return D


def exact_profile_and_reads(lams, edges, total):
    """``{(bin counts, total reads): probability}`` for seen sequences, each
    count zero-truncated Poisson(lam), binned by ``edges`` (the last bin
    open), keeping only totals up to ``total``. Plain convolution."""
    def bin_of(r):
        b = 0
        while b + 1 < len(edges) and r >= edges[b + 1]:
            b += 1
        return b
    dist = {((0,) * len(edges), 0): 1.0}
    for lam in lams:
        z = -math.expm1(-lam)
        nxt: dict = {}
        for (F, T), w in dist.items():
            for r in range(1, total - T + 1):
                q = math.exp(r * math.log(lam) - lam - math.lgamma(r + 1)) / z
                F2 = list(F)
                F2[bin_of(r)] += 1
                key = (tuple(F2), T + r)
                nxt[key] = nxt.get(key, 0.0) + w * q
        dist = nxt
    return dist
