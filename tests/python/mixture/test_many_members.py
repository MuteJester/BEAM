"""Members beyond 64, and the Gram matrix that makes them affordable.

The 64-member ceiling used to sit on `Mixture` itself, so a cohort-scale
mixture could not be built at all and every quantity went with it -- including
the three that have nothing to do with the bitmask the ceiling protects. These
tests pin the split: the union still refuses above 64, and nothing else does.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from BEAM import BeamModel, Mixture
from BEAM.errors import BeamValueError
from _tiny import brute_hill, mixture_pmf, tiny_model, whole_space


@pytest.fixture(scope="module")
def many():
    """Enough members to cross the union's ceiling, and cheap to build."""
    return [tiny_model(4000 + i, n=60, lo=3, hi=4) for i in range(70)]


def test_a_mixture_larger_than_the_union_ceiling_still_builds(many):
    """The ceiling belongs to one sweep, not to the object."""
    mix = Mixture(many)
    assert mix.n_members == 70


@pytest.mark.parametrize("q", [2.0, 3.0])
def test_integer_orders_work_beyond_the_union_ceiling(many, q):
    """The inner product never distinguishes its members, so it has no bound."""
    mix = Mixture(many[:66])
    assert mix.hill_number(q) > 0.0


def test_alpha_and_the_entropy_bounds_work_beyond_the_ceiling(many):
    """Neither touches a pair, let alone a bitmask; both are O(M)."""
    mix = Mixture(many)
    assert mix.alpha_diversity(2.0) > 0.0
    b = mix.entropy_bounds()
    assert b.lower <= b.upper
    assert b.upper - b.lower == pytest.approx(math.log(70), rel=1e-12)


def test_the_exact_union_still_refuses_above_sixty_four(many):
    """A uint64 carries 64 bits and that is structural, not a tuning knob.

    The message has to say which sweep is limited and what to do instead, or a
    caller reads it as "mixtures stop at 64" -- which is what the shared
    ceiling used to make true.
    """
    with pytest.raises(BeamValueError) as exc:
        Mixture(many).support_size()
    msg = str(exc.value)
    assert "support union" in msg
    assert "64-bit mask" in msg
    assert "bounds" in msg


def test_the_union_bounds_are_available_at_any_size(many):
    """What you get instead of the exact union, and it costs nothing."""
    s = Mixture(many).support_size(exact=False)
    assert s.exact is False and s.value is None
    assert s.lower == max(m.path_count() for m in many)
    assert s.upper == sum(m.path_count() for m in many)


def test_sixty_four_members_still_have_an_exact_union(many):
    """The boundary itself, from the inside.

    Every member spans the whole space, so the exact answer is known: a
    64-bit mask that dropped or double-counted a member would miss it.
    """
    s = Mixture(many[:64]).support_size()
    assert s.exact and s.value == whole_space()


# ── The Gram matrix ─────────────────────────────────────────────

def test_the_gram_matrix_equals_the_pairwise_assembly(tiny_a, tiny_b, tiny_c):
    """One call must give exactly what the pairs give one at a time."""
    mix = Mixture([tiny_a, tiny_b, tiny_c])
    g = mix.log_gram_matrix()
    for i, j in itertools.combinations_with_replacement(range(3), 2):
        want = Mixture([tiny_a, tiny_b, tiny_c]).log_inner_product(i, j)
        assert g[i, j] == want
        assert g[j, i] == want


def test_the_gram_matrix_is_exactly_symmetric(tiny_a, tiny_b, tiny_c):
    g = Mixture([tiny_a, tiny_b, tiny_c]).log_gram_matrix()
    assert np.array_equal(g, g.T)
    assert np.array_equal(np.exp(g), np.exp(g).T)


def test_the_gram_matrix_populates_the_cache(tiny_a, tiny_b):
    """So hill_number costs nothing after it, rather than sweeping again."""
    mix = Mixture([tiny_a, tiny_b])
    mix.log_gram_matrix()
    assert set(mix._ip_cache) == {(0, 0), (0, 1), (1, 1)}


def test_q2_through_the_driver_matches_the_q_tuple_expansion(
        tiny_a, tiny_b, tiny_c, sup_a, sup_b, sup_c):
    """q=2 takes the parallel path; it must agree with direct summation.

    The off-diagonal is DOUBLED rather than summed twice, which is the whole
    content of the expansion and the easiest thing to get wrong by hand.
    """
    w = (0.2, 0.5, 0.3)
    mix = Mixture([tiny_a, tiny_b, tiny_c], w)
    pmf = mixture_pmf([sup_a, sup_b, sup_c], w)
    assert mix.hill_number(2.0) == pytest.approx(brute_hill(pmf, 2.0), rel=1e-11)


def test_a_far_pair_reads_as_finite_in_the_log_gram(far_a, far_b):
    """No pair of members is disjoint any more, and the Gram matrix says so.

    The far pair shares ~4e-20 of mass. The log Gram matrix must carry that as
    the finite number the pairwise route gives, not collapse it to the `-inf` a
    closed world would have reported.
    """
    g = Mixture([far_a, far_b]).log_gram_matrix()
    assert np.isfinite(g).all()
    assert g[0, 1] == Mixture([far_a, far_b]).log_inner_product(0, 1)
    assert g[0, 1] < min(g[0, 0], g[1, 1]) - 30.0
