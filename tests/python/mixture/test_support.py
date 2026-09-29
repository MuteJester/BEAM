"""The q=0 corner: the union of the members' supports, and what it is not.

Under the prior every member's support is every string over the alphabet up
to max_length, so the union is always that whole space and D0 is the same for
every mixture. The union sweep is still checked -- against enumeration, at its
bounds, and through its growth path -- because it is the only exact route to
D0 for a mixture and a sweep that double-counted a shared sequence would say
so here.

The trap this file exists to pin is that `merged()` is not the mixture.
Summing edge counts and refinalizing produces the model you would get by
TRAINING on the pooled corpus, which may splice an edge learned from one member
onto an edge learned from another. D0 can no longer show that -- both sides
span the whole space -- so it is shown on the probabilities instead: the pooled
model puts more mass on some sequences than EITHER member does, which no
mixture of them can.
"""
from __future__ import annotations

import math

import pytest

from BEAM import Mixture
from BEAM.mixture import MixtureTooLarge
from _tiny import MAX_LENGTH, support, tiny_model, whole_space


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


def test_members_that_share_almost_no_mass_still_share_every_string(far_a,
                                                                    far_b):
    """The union attains the LOWER bound even for the far pair.

    Built over disjoint residue pools, these two agree on essentially nothing
    -- their overlap is ~4e-20 of mass -- and under a closed world their union
    was the sum of the two. The prior gives each of them the whole space, so
    the union is that space, counted once, and not the sum.
    """
    size = Mixture([far_a, far_b]).support_size()
    assert size.value == size.lower == whole_space()
    assert size.value == far_a.path_count() == far_b.path_count()
    assert size.value < size.upper


def test_identical_members_attain_the_lower_bound(tiny_a):
    size = Mixture([tiny_a, tiny_a, tiny_a]).support_size()
    assert size.value == size.lower == tiny_a.path_count()


# ── The trap ────────────────────────────────────────────────────

def test_d0_cannot_tell_the_pooled_model_from_the_mixture(tiny_a, tiny_b):
    """Both span the whole space, so the trap has to be sprung elsewhere."""
    union = Mixture([tiny_a, tiny_b]).support_size().value
    assert union == tiny_a.merged(tiny_b).path_count() == whole_space()


def test_the_pooled_model_is_not_any_mixture_of_its_members(tiny_a, tiny_b,
                                                            sup_a, sup_b):
    """The pooled model outscores BOTH members on spliced sequences.

    A mixture can never do that: `pbar(s) = sum_i w_i P_i(s)` is at most
    `max_i P_i(s)` whatever the weights. The pooled model can, because it
    reaches sequences through edges learned from one member spliced onto edges
    learned from the other. That recombination is the whole reason the mixture
    is not implemented in terms of the count algebra, so it is asserted on a
    case where the excess is real rather than only documented.
    """
    merged = tiny_a.merged(tiny_b)
    pooled = support(merged)
    assert len(pooled) == len(sup_a) == len(sup_b)

    spliced = [s for s, p in pooled.items()
               if p > 1.01 * max(sup_a[s], sup_b[s])]
    assert spliced, "the fixtures must actually admit a splice"

    # The mixture sits at or below the better member on every one of them,
    # for weights at both extremes and in between, as the bound says it must.
    for w in ((0.5, 0.5), (0.01, 0.99), (0.99, 0.01)):
        mix = Mixture([tiny_a, tiny_b], w)
        for s in spliced[:20]:
            best = max(tiny_a.log_pseq(s), tiny_b.log_pseq(s))
            assert mix.log_pseq(s) <= best + 1e-12
            assert merged.log_pseq(s) > best + math.log(1.01)

    # And the two distributions differ in aggregate, not just on a few strings.
    assert merged.hill_number(2.0) != pytest.approx(
        Mixture([tiny_a, tiny_b]).hill_number(2.0), rel=1e-3)


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
    # Three members can produce at most seven non-empty masks per state. Under
    # the prior every member is live on every path, so only the full mask ever
    # survives and the population is at most one pair per state.
    assert size.peak_states <= 7 * tiny_a.n_states
    assert size.peak_states <= tiny_a.n_states


def test_hill_number_at_q0_is_the_union(tiny_a, tiny_b):
    mix = Mixture([tiny_a, tiny_b])
    assert mix.hill_number(0.0) == mix.support_size().value


def test_a_member_that_never_made_a_length_keeps_only_its_prior_share():
    """Length priors differ, and a member re-weights by its own within a length.

    Built so one member's corpus is all length 4 and the other's all length 5.
    Neither is absent from the other's length -- the length prior gives every
    length up to max_length `alpha_length / max_length` pseudo-counts -- but
    each is down to that share there. The within-length weights are checked
    against the prior's closed form, `(n_L + a/M) / (N + a)`, computed here
    from the corpus sizes rather than read back off the model.
    """
    # The length strength is FIXED here: the test is about how a mixture
    # re-weights its members within a length, and a strength each member
    # estimated for itself would make the two shares differ for a reason that
    # has nothing to do with that.
    n, a = 120, 10.0
    short = tiny_model(61, letters="ACG", n=n, lo=4, hi=4, alpha_length=a)
    long_ = tiny_model(62, letters="ACG", n=n, lo=5, hi=5, alpha_length=a)
    lengths = tuple(range(1, MAX_LENGTH + 1))
    assert short.observed_lengths == long_.observed_lengths == lengths

    seen = (n + a / MAX_LENGTH) / (n + a)
    unseen = (a / MAX_LENGTH) / (n + a)
    mix = Mixture([short, long_])
    assert mix.member_weights(4) == pytest.approx(
        [seen / (seen + unseen), unseen / (seen + unseen)], rel=1e-12)
    assert mix.member_weights(5) == pytest.approx(
        [unseen / (seen + unseen), seen / (seen + unseen)], rel=1e-12)
    # A length neither corpus made is the prior's alone, and equal for both.
    assert mix.member_weights(6) == pytest.approx([0.5, 0.5], rel=1e-12)

    # And both members span the whole space, so the union is counted once.
    size = mix.support_size()
    assert size.value == short.path_count() == whole_space()


def test_growing_the_table_gives_the_same_count_as_not_growing(tiny_a, tiny_b,
                                                              tiny_c, small):
    """The rehash path, checked against the same sweep that never rehashes.

    A ceiling the sweep derives for itself starts at 64 live pairs and doubles;
    a ceiling the caller names is allocated outright and never grows. So the
    same union computed both ways runs the rehash on one side and not the
    other, and the two must agree exactly -- a rehash that dropped or merged an
    entry would show up as a different count.

    Worth having because nothing else reaches it. Under the prior only the
    full member-set survives, so the population is one pair per context: the
    toy fixtures peak at 25, below where the table starts, and never rehash;
    the shared `small` model peaks at 400, three doublings past it. Both kinds
    are run, so the comparison covers a sweep that grows and ones that do not.
    """
    for members in ([tiny_a, tiny_b], [tiny_a, tiny_b, tiny_c],
                    [small, small]):
        mix = Mixture(members)
        grown = mix.support_size()                      # starts at 64, doubles
        preallocated = mix.support_size(cap=1 << 20)    # one allocation
        assert grown.value == preallocated.value
        assert grown.peak_states == preallocated.peak_states


def test_the_growth_path_runs_on_a_model_that_needs_it(small):
    """A case whose population is several times the starting allocation.

    `small` is the order-2 fixture the rest of the suite shares, so this is a
    real trellis rather than a toy: its peak is 400 pairs, one per amino-acid
    context, which is several doublings past where the table begins.
    """
    mix = Mixture([small, small])
    s = mix.support_size()
    assert s.peak_states > 64, "this fixture no longer exercises growth"
    assert s.value == small.path_count()
    assert s.value == mix.support_size(cap=1 << 20).value
