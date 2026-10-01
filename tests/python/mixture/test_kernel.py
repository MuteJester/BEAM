"""The product sweep, checked against machinery that already existed.

`<P_i, P_i> = sum_s P_i(s)^2 = 1 / D2(i)`, so running the new kernel on one
model against itself must reproduce `hill_number(2.0)`, which reaches the same
number down a completely separate path -- the Mellin transform of the tilted
partition function in math/diversity.c.

That is worth more than any brute-force check on a toy alphabet, because it
runs at full scale on a real model. A recursion that is subtly wrong about
rescaling, about which states carry mass, or about the sink terms will agree
with a small enumeration and disagree here. Under the prior every cell is
positive, so every state carries mass and nothing drops out of the sum.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest

from BEAM import BeamModel, Mixture
from BEAM import _cbeam as _c
from _tiny import MAX_LENGTH, POOL_A, POOL_B, strings, tiny_model

FOUNDATION = Path(".dev/runtime_profile/cache/foundation.beam")


def _foundation_or_skip():
    """The cached foundation model, or a skip that says which of the two
    reasons applies.

    Presence is no longer sufficient. The cache is a file this build may
    decline to read: this build reads only the format it writes, and refuses
    a file from any earlier development build -- which, since the prior
    replaced the strict and open worlds, is every cache written before it.
    That is a stale cache, not a kernel defect, and saying so is the
    difference between a skip somebody acts on and one they ignore. Matched
    on the refusal's own words, so any other load failure still raises.
    """
    if not FOUNDATION.exists():
        pytest.skip("the cached foundation model is not present")
    try:
        return BeamModel.from_file(str(FOUNDATION))
    except Exception as exc:                       # noqa: BLE001
        if "earlier development build" not in str(exc).lower():
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


# A pair built over disjoint residue pools with a very weak prior. Every string
# is still in both supports, but a residue outside a member's pool costs about
# (alpha / n)**(order + 1) there -- ~1e-186 at alpha = 1e-60 -- so every string
# is far-fetched under at least one of the two, and the length-L overlap falls
# by tens of orders of magnitude per residue. From length 4 up it is below the
# smallest double.
_TINY_ALPHA = 1e-60


def _far_pair():
    a = tiny_model(21, letters=POOL_A, alpha=_TINY_ALPHA)
    b = tiny_model(22, letters=POOL_B, alpha=_TINY_ALPHA)
    return a, b


def _log_sum_exp(values):
    values = np.asarray(values, dtype=np.float64)
    mx = values.max()
    return float(mx + math.log(np.exp(values - mx).sum()))


def test_a_tiny_overlap_survives_the_rescaling():
    """Per-length overlaps far below a double must come back finite and right.

    A product of per-edge probabilities over the layers is far below what a
    double holds, which is why the accumulator is rescaled every layer with the
    log of the scale carried alongside. Without that the sweep returns 0.0 --
    `-inf` in the log -- which is the answer a closed world would give for two
    members that share nothing, and is wrong for any two members now.

    Checked against brute force in the log domain: every string is scored by
    each member, the log-probabilities are added, and the per-length sums are
    taken by log-sum-exp. The deepest of them is around e**-1000, more than a
    hundred orders of magnitude below the smallest subnormal double, so no
    linear-scale shortcut could reproduce it.
    """
    a, b = _far_pair()
    seqs = list(strings())
    la = np.asarray(a.log_pseq_batch(seqs), dtype=np.float64)
    lb = np.asarray(b.log_pseq_batch(seqs), dtype=np.float64)
    assert np.isfinite(la).all() and np.isfinite(lb).all(), \
        "the prior must still reach every string at this alpha"
    lengths = np.array([len(s) for s in seqs])
    want = [_log_sum_exp((la + lb)[lengths == L])
            for L in range(1, MAX_LENGTH + 1)]
    assert want[-1] < -745.0, "the construction must actually underflow"

    total, by = _c.mixture_inner_product([a._g, b._g])
    got = np.frombuffer(by, dtype=np.float64)
    assert got[0] == -math.inf, "there is no length-0 sequence"
    for L in range(1, MAX_LENGTH + 1):
        assert math.isfinite(got[L])
        assert got[L] == pytest.approx(want[L - 1], rel=1e-12), f"length {L}"
    assert total == pytest.approx(_log_sum_exp(la + lb), rel=1e-12)
    assert Mixture([a, b]).log_inner_product(0, 1) == pytest.approx(
        _log_sum_exp(la + lb), rel=1e-12)


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
