"""Models small enough to enumerate, and the brute-force sums over them.

The point of this file is that every mixture quantity has a second,
independent implementation here that owes nothing to the trellis: a Python
dict of ``{sequence: probability}`` built by scoring candidates one at a time,
and plain sums over it. A test that compares the two is comparing a dynamic
program against arithmetic, which is the only comparison that can actually
find an error in the dynamic program.

The enumeration is itself validated before anything is built on it --
:func:`support` must return exactly ``path_count()`` entries and its
probabilities must sum to 1 -- because an enumeration that quietly missed a
corner of the support would make every downstream assertion agree with the
same mistake twice.
"""
from __future__ import annotations

import itertools
import math
import random

from BEAM import BeamModel

# Two disjoint residue pools. Models built over different pools cannot share a
# single sequence, which is what the beta = M tests need; models built over
# one pool overlap heavily, which is what the brute-force tests want.
POOL_A = "ACDG"
POOL_B = "FHKL"

ORDER = 2
MAX_LENGTH = 6


def tiny_model(seed: int, *, letters: str = POOL_A, n: int = 120,
               lo: int = 3, hi: int = 5, **config) -> BeamModel:
    """A closed-world model whose entire support can be enumerated."""
    rng = random.Random(seed)
    seqs = ["".join(rng.choice(letters) for _ in range(rng.randint(lo, hi)))
            for _ in range(n)]
    cfg = {"order": ORDER, "max_length": MAX_LENGTH}
    cfg.update(config)
    return BeamModel(seqs, **cfg)


def support(model: BeamModel, letters: str = POOL_A + POOL_B) -> dict[str, float]:
    """``{sequence: P(sequence)}`` over the model's whole support, by trial.

    Enumerates every string over ``letters`` at every length the model
    produces and keeps the ones that score above zero. Sound only because the
    model is closed-world: a residue the corpus never spelled has a zero count,
    hence a ``-inf`` weight, so nothing outside ``letters`` can be in the
    support. The caller checks that claim by comparing the size against
    ``path_count``.
    """
    out: dict[str, float] = {}
    for L in model.observed_lengths:
        for tup in itertools.product(letters, repeat=L):
            s = "".join(tup)
            lp = model.log_pseq(s)
            if lp != -math.inf:
                out[s] = math.exp(lp)
    return out


def mixture_pmf(supports, weights) -> dict[str, float]:
    """``pbar(s)`` as a dict, by direct summation. No trellis involved."""
    out: dict[str, float] = {}
    for w, sup in zip(weights, supports):
        for s, p in sup.items():
            out[s] = out.get(s, 0.0) + w * p
    return out


def brute_hill(pmf: dict[str, float], q: float) -> float:
    """``D_q`` of an explicit pmf, straight from the definition."""
    if q == 0.0:
        return float(len(pmf))
    if q == 1.0:
        return math.exp(-sum(p * math.log(p) for p in pmf.values()))
    return sum(p ** q for p in pmf.values()) ** (1.0 / (1.0 - q))


def brute_entropy(pmf: dict[str, float]) -> float:
    """``H(pbar)`` in nats, by direct summation."""
    return -sum(p * math.log(p) for p in pmf.values())
