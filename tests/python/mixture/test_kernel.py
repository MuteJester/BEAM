"""The product sweep, checked against machinery that already existed.

`<P_i, P_i> = sum_s P_i(s)^2 = 1 / D2(i)`, so running the new kernel on one
model against itself must reproduce `hill_number(2.0)`, which reaches the same
number down a completely separate path -- the Mellin transform of the tilted
partition function in math/diversity.c.

That is worth more than any brute-force check on a toy alphabet, because it
runs at full scale on a real model. A recursion that is subtly wrong about
rescaling, about which states carry mass, or about the sink terms will agree
with a four-residue enumeration and disagree here.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from BEAM import BeamModel, Mixture
from _tiny import support, tiny_model

FOUNDATION = Path(".dev/runtime_profile/cache/foundation.beam")


def _foundation_or_skip():
    """The cached foundation model, or a skip that says which of the two
    reasons applies.

    Presence is no longer sufficient. The cache is a file this build may
    decline to read: every model written before the single-expert change is a
    format this one refuses, on the grounds that its depth index was clamped
    and the exact depths cannot be recovered. That is a stale cache, not a
    kernel defect, and saying so is the difference between a skip somebody
    acts on and one they ignore.
    """
    if not FOUNDATION.exists():
        pytest.skip("the cached foundation model is not present")
    try:
        return BeamModel.from_file(str(FOUNDATION))
    except Exception as exc:                       # noqa: BLE001
        if "retrain" not in str(exc).lower():
            raise
        pytest.skip(f"the cached foundation model predates this build and "
                    f"must be rebuilt: {exc}")


def _self_inner_product_matches_d2(model, rel=1e-12):
    mix = Mixture([model, model])
    got = 1.0 / mix.inner_product(0, 0)
    want = model.hill_number(2.0)
    assert got == pytest.approx(want, rel=rel), (
        f"1/<P,P> = {got!r} but hill_number(2.0) = {want!r}; the product sweep "
        f"and the Mellin path disagree, and one of them is wrong")
    return abs(got - want) / want


def test_self_inner_product_is_one_over_d2_tiny(tiny_a):
    _self_inner_product_matches_d2(tiny_a)


def test_self_inner_product_is_one_over_d2_on_the_shared_fixture(small):
    """The order-2 model the rest of the suite uses, not a purpose-built toy."""
    _self_inner_product_matches_d2(small)


def test_self_inner_product_is_one_over_d2_at_order_three(model):
    """Order 3 exercises a wider state space and a different backoff shape."""
    _self_inner_product_matches_d2(model)


def test_self_inner_product_is_one_over_d2_open_world(open_small):
    """The open world: every cell positive, so nothing drops out of the sum.

    A different regime for the sweep -- no state is dead, the intersection of
    the reached sets is everything, and the rescaling has to hold a trellis
    with no zeros in it.
    """
    _self_inner_product_matches_d2(open_small)


def test_self_inner_product_is_one_over_d2_on_the_foundation():
    """The same identity on the 927M-sequence, order-4, 194k-state model.

    Slow (about ten seconds) and gated on a file that is not in the repository,
    but it is the only place the kernel meets a full-scale trellis, and it is
    where a rescaling defect would actually show. Measured agreement when this
    was written: exact, to the last bit.
    """
    m = _foundation_or_skip()
    err = _self_inner_product_matches_d2(m, rel=1e-12)
    assert err < 1e-12


def test_the_total_is_the_log_sum_of_the_per_length_terms(tiny_a, tiny_b):
    """The by-length vector and the scalar must be the same quantity."""
    from BEAM import _cbeam as _c
    total, by = _c.mixture_inner_product([tiny_a._g, tiny_b._g])
    v = np.frombuffer(by, dtype=np.float64)
    live = v[np.isfinite(v)]
    assert live.size
    mx = live.max()
    want = mx + math.log(np.exp(live - mx).sum())
    assert total == pytest.approx(want, rel=1e-13)


def test_one_member_sums_to_one(tiny_a):
    """`sum_s P(s) = 1` is the k=1 case, and it must be exactly 0 in logs."""
    mix = Mixture([tiny_a, tiny_a])
    assert mix.log_inner_product(0) == pytest.approx(0.0, abs=1e-12)


# A pair whose supports meet in exactly ONE sequence, at full CDR3 length.
# Two corpora over disjoint residue pools, each given a single copy of the same
# rare 27-mer: nothing else can be spelled by both, so the inner product is
# knowable in closed form -- the product of that one sequence's probability
# under each model -- and it is around 1e-25.
_RARE = "CADEFGHIKLMNPQRSTVWYACDEFGH"


def _one_shared_sequence(seed_a=1, seed_b=2, n=3000, length=27):
    import random

    def corpus(letters, seed):
        g = random.Random(seed)
        return ["".join(g.choice(letters) for _ in range(length))
                for _ in range(n)]

    a = BeamModel(corpus("ACDG", seed_a) + [_RARE],
                  abundances=[2000] * n + [1], order=2, max_length=27)
    b = BeamModel(corpus("FHKL", seed_b) + [_RARE],
                  abundances=[2000] * n + [1], order=2, max_length=27)
    return a, b


def test_a_tiny_overlap_survives_the_rescaling():
    """An inner product around 1e-25 must come back finite, nonzero and right.

    A product of per-edge probabilities over 27 layers is far below what a
    double holds, which is why the accumulator is rescaled every layer with the
    log of the scale carried alongside. Without that the sweep returns 0.0,
    which looks exactly like disjoint supports -- a wrong answer indistinguish-
    able from a right one, and the reason this is asserted rather than assumed.

    The expected value is known in closed form here: the two supports meet in
    exactly one sequence, so `<P_a, P_b>` is that sequence's probability under
    each, multiplied.
    """
    a, b = _one_shared_sequence()
    mix = Mixture([a, b])
    want = a.pseq(_RARE) * b.pseq(_RARE)
    assert want < 1e-20, "the construction must actually be small"

    lp = mix.log_inner_product(0, 1)
    assert math.isfinite(lp)
    assert mix.inner_product(0, 1) > 0.0
    assert mix.inner_product(0, 1) == pytest.approx(want, rel=1e-11)


def test_one_shared_sequence_gives_a_union_one_short_of_the_sum():
    """`|A u B| = |A| + |B| - |A n B|`, with the intersection known to be 1.

    A half-billion-sequence union counted exactly, against arithmetic. The
    bitmask sweep has to attribute that single shared sequence to BOTH members
    and still count it once; a sweep that dropped it or double-counted it would
    be off by exactly one, which is the error a large total hides best.
    """
    a, b = _one_shared_sequence()
    size = Mixture([a, b]).support_size()
    assert size.exact
    assert size.value == a.path_count() + b.path_count() - 1.0


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7, 8])
def test_a_k_way_product_of_one_model_is_its_own_mellin_transform(small, k):
    """`<P,..,P>` over k copies is `sum_s P(s)^k`, which log_mellin gives.

    The k >= 3 path is not the k = 2 path run twice -- it sums k weight layers
    into one accumulator -- so it needs its own cross-check, and the Mellin
    transform provides one at every k for free.
    """
    mix = Mixture([small] * k)
    got = mix.log_inner_product(*range(k))
    assert got == pytest.approx(small.log_mellin(float(k)), rel=1e-12)


def test_the_gram_matrix_is_exactly_symmetric(tiny_a, tiny_b, tiny_c):
    """Not approximately: `G[i,j]` and `G[j,i]` must be the same float.

    The kernel sums the members' edge weights in the order it is handed them,
    so computing both triangles independently could differ in the last bit.
    One triangle is computed and mirrored, which makes symmetry structural.
    """
    g = Mixture([tiny_a, tiny_b, tiny_c]).gram_matrix()
    assert np.array_equal(g, g.T)
    assert np.all(np.diag(g) > 0.0)
