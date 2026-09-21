"""The q=0 corner: the union of the members' supports, and what it is not.

The trap this file exists to pin is that `merged()` is not the mixture.
Summing edge counts and refinalizing produces the model you would get by
TRAINING on the pooled corpus, and its support strictly contains the union
because a pooled path may splice an edge learned from one member onto an edge
learned from another. Anyone reviewing this package will check that first, so
it is checked here from both sides: the union is what direct enumeration says
it is, and it is smaller than the pooled model's support on a case where the
splice actually happens.
"""
from __future__ import annotations

import pytest

from BEAM import Mixture
from BEAM.mixture import MixtureTooLarge
from _tiny import support, tiny_model


def test_the_union_is_what_enumeration_says_it_is(tiny_a, tiny_b, sup_a, sup_b):
    size = Mixture([tiny_a, tiny_b]).support_size()
    assert size.exact
    assert size.value == float(len(set(sup_a) | set(sup_b)))


def test_three_member_union_is_what_enumeration_says_it_is(
        tiny_a, tiny_b, tiny_c, sup_a, sup_b, sup_c):
    """Three members means masks with three bits, not just pairwise overlap."""
    size = Mixture([tiny_a, tiny_b, tiny_c]).support_size()
    assert size.value == float(len(set(sup_a) | set(sup_b) | set(sup_c)))


def test_the_union_sits_inside_its_own_bounds(tiny_a, tiny_b, tiny_c):
    size = Mixture([tiny_a, tiny_b, tiny_c]).support_size()
    assert size.lower <= size.value <= size.upper
    assert size.lower == max(m.path_count() for m in (tiny_a, tiny_b, tiny_c))
    assert size.upper == sum(m.path_count() for m in (tiny_a, tiny_b, tiny_c))


def test_disjoint_members_attain_the_upper_bound(far_a, far_b):
    """Nothing shared means the union is exactly the sum. The bound is tight."""
    size = Mixture([far_a, far_b]).support_size()
    assert size.value == size.upper
    assert size.value == far_a.path_count() + far_b.path_count()


def test_identical_members_attain_the_lower_bound(tiny_a):
    size = Mixture([tiny_a, tiny_a, tiny_a]).support_size()
    assert size.value == size.lower == tiny_a.path_count()


# ── The trap ────────────────────────────────────────────────────

def test_the_union_is_never_larger_than_the_pooled_model(tiny_a, tiny_b):
    """`D0(union) <= D0(merged)`, the invariant that holds either way round."""
    union = Mixture([tiny_a, tiny_b]).support_size().value
    assert union <= tiny_a.merged(tiny_b).path_count()


def test_the_pooled_model_spells_sequences_no_member_spells(tiny_a, tiny_b,
                                                            sup_a, sup_b):
    """The inequality above is STRICT here, and the excess is recombination.

    A pooled model reaches sequences that splice one member's edges onto
    another's. This is the whole reason the mixture is not implemented in terms
    of the count algebra, so it is asserted on a case where the gap is real
    rather than only documented.
    """
    union = Mixture([tiny_a, tiny_b]).support_size().value
    pooled = tiny_a.merged(tiny_b).path_count()
    assert union < pooled
    assert union == float(len(set(sup_a) | set(sup_b)))

    # And the extra sequences are genuinely outside both members' supports.
    merged = tiny_a.merged(tiny_b)
    both = set(sup_a) | set(sup_b)
    spliced = [s for s in support(merged) if s not in both]
    assert spliced, "the fixtures must actually admit a splice"
    for s in spliced[:5]:
        assert tiny_a.log_pseq(s) == float("-inf")
        assert tiny_b.log_pseq(s) == float("-inf")


# ── Refusing rather than degrading ──────────────────────────────

def test_bounds_only_costs_nothing_and_says_so(tiny_a, tiny_b):
    size = Mixture([tiny_a, tiny_b]).support_size(exact=False)
    assert size.exact is False
    assert size.value is None
    assert size.lower <= size.upper


def test_an_impossible_cap_is_refused_and_not_degraded(tiny_a, tiny_b, tiny_c):
    """Outgrowing the ceiling raises. It never returns a partial count.

    A support union that quietly stopped counting would be indistinguishable
    from a small one, and it is exactly the number a reader would trust.
    """
    mix = Mixture([tiny_a, tiny_b, tiny_c])
    with pytest.raises(MixtureTooLarge) as exc:
        mix.support_size(cap=1)
    msg = str(exc.value)
    assert "pairs" in msg and "budget" in msg


def test_the_peak_population_is_reported(tiny_a, tiny_b, tiny_c):
    """What actually limits M is the surviving member-sets, so it is measured.

    The 64-member structural ceiling is almost never what binds; this figure is
    how a caller finds out where it really stands.
    """
    size = Mixture([tiny_a, tiny_b, tiny_c]).support_size()
    assert size.peak_states > 0
    # Three members can produce at most seven non-empty masks per state.
    assert size.peak_states <= 7 * tiny_a.n_states


def test_hill_number_at_q0_is_the_union(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    assert mix.hill_number(0.0) == mix.support_size().value


def test_a_member_that_never_makes_a_length_is_absent_from_it():
    """Length priors differ, and a member with zero prior contributes nothing.

    Built so one member produces only length 4 and the other only length 5: the
    union at each length must be exactly one member's support, not both.
    """
    short = tiny_model(61, letters="ACD", lo=4, hi=4)
    long_ = tiny_model(62, letters="ACD", lo=5, hi=5)
    assert short.observed_lengths == (4)
    assert long_.observed_lengths == (5)
    mix = Mixture([short, long_])
    size = mix.support_size()
    assert size.value == short.path_count() + long_.path_count()
    assert mix.member_weights(4) == pytest.approx([1.0, 0.0])
    assert mix.member_weights(5) == pytest.approx([0.0, 1.0])


def test_growing_the_table_gives_the_same_count_as_not_growing(tiny_a, tiny_b,
                                                              tiny_c, small):
    """The rehash path, checked against the same sweep that never rehashes.

    A ceiling the sweep derives for itself starts at 64 live pairs and doubles;
    a ceiling the caller names is allocated outright and never grows. So the
    same union computed both ways runs the rehash on one side and not the
    other, and the two must agree exactly -- a rehash that dropped or merged an
    entry would show up as a different count.

    Worth having because nothing else reaches it: the toy fixtures peak at 64
    pairs and the shared `small` model at a few hundred, which is why the
    starting size is small enough for an ordinary test to cross.
    """
    for members in ([tiny_a, tiny_b], [tiny_a, tiny_b, tiny_c]):
        mix = Mixture(members)
        grown = mix.support_size()                      # starts at 64, doubles
        preallocated = mix.support_size(cap=1 << 20)    # one allocation
        assert grown.value == preallocated.value
        assert grown.peak_states == preallocated.peak_states


def test_the_growth_path_runs_on_a_model_that_needs_it(small):
    """A case whose population is several times the starting allocation.

    `small` is the order-2 fixture the rest of the suite shares, so this is a
    real trellis rather than a toy: its peak is a few hundred pairs, which is
    several doublings past where the table begins.
    """
    mix = Mixture([small, small])
    s = mix.support_size()
    assert s.peak_states > 64, "this fixture no longer exercises growth"
    assert s.value == small.path_count()
    assert s.value == mix.support_size(cap=1 << 20).value
