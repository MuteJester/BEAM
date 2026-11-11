"""Models small enough to enumerate, and the brute-force sums over them.

The point of this file is that every mixture quantity has a second,
independent implementation here that owes nothing to the trellis: a Python
dict of ``{sequence: probability}`` built by scoring candidates one at a time,
and plain sums over it. A test that compares the two is comparing a dynamic
program against arithmetic, which is the only comparison that can actually
find an error in the dynamic program.

Every model carries the hierarchical prior, so its support is every string
over the WHOLE alphabet at every length up to ``max_length`` -- not the letters
its corpus happened to use. The fixtures are therefore nucleotide models with
``max_length`` 6: ``sum_L 5**L`` is 19,530 strings, small enough to walk in
Python, where the amino-acid space at the same length is 67 million.

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

#: The nucleotide alphabet as the model reads it. N is a residue, not a gap.
NT = "ACGTN"

#: What the ordinary fixtures' corpora are drawn from. N is left out on
#: purpose, so that one residue's whole mass comes from the prior alone.
CORPUS = "ACGT"

# Two disjoint residue pools. Models built over different pools still share
# every string -- the prior sees to that -- but with a weak prior the mass they
# share is some twenty orders of magnitude below either model's own, which is
# far below double precision. That is what the beta = M tests need. POOL_B is
# the larger pool so the two far models differ in diversity.
POOL_A = "AC"
POOL_B = "GTN"

#: The prior strength of the far fixtures. Each level multiplies an unseen
#: residue's share by about ``FAR_ALPHA / n``, so at order 2 an off-pool
#: residue costs ~1e-16 and the two far models overlap in ~4e-20 of mass.
FAR_ALPHA = 1e-3

ORDER = 2
MAX_LENGTH = 6


def tiny_model(seed: int, *, letters: str = CORPUS, n: int = 120,
               lo: int = 3, hi: int = 5, **config) -> BeamModel:
    """A nucleotide model whose entire support can be enumerated."""
    rng = random.Random(seed)
    seqs = ["".join(rng.choice(letters) for _ in range(rng.randint(lo, hi)))
            for _ in range(n)]
    cfg = {"alphabet": "nt", "order": ORDER, "max_length": MAX_LENGTH}
    cfg.update(config)
    return BeamModel(seqs, **cfg)


def strings(max_length: int = MAX_LENGTH, letters: str = NT):
    """Every string over ``letters`` of length 1 to ``max_length``."""
    for L in range(1, max_length + 1):
        for tup in itertools.product(letters, repeat=L):
            yield "".join(tup)


def support(model: BeamModel, letters: str = NT) -> dict[str, float]:
    """``{sequence: P(sequence)}`` over the model's whole support, by trial.

    Enumerates every string over the full alphabet at every length up to
    ``max_length`` and keeps the ones that score above zero. Under the prior
    that should be all of them; the filter stays so that a string wrongly
    scored ``-inf`` shows up as a count that disagrees with ``path_count``
    rather than as a crash, and the caller checks exactly that.
    """
    out: dict[str, float] = {}
    for s in strings(int(model.config["max_length"]), letters):
        lp = model.log_pseq(s)
        if lp != -math.inf:
            out[s] = math.exp(lp)
    return out


def whole_space(max_length: int = MAX_LENGTH, letters: str = NT) -> float:
    """``sum_L R**L``: how many strings the prior makes possible."""
    return float(sum(len(letters) ** L for L in range(1, max_length + 1)))


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
