"""Every mixture quantity against direct summation over the whole support.

This is the file that actually proves the dynamic programs correct. Everything
else in this group is an identity, an invariant or a smoke test; only here is
each number computed a second time by a method that shares no code with the
first -- a Python dict of probabilities and a plain sum over it.

The models are deliberately tiny (four residues, length 3 to 5) so their entire
support fits in memory. That is not a weaker test than one on a real model: a
forward recursion that is wrong is wrong at every scale, and at this scale the
answer can be checked rather than merely compared against itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import Mixture
from _tiny import brute_entropy, brute_hill, mixture_pmf, support


# The enumeration underpins every assertion below, so it is checked first and
# on its own. If `support` missed part of the support, every comparison in this
# file would agree with the same omission twice and pass.
def test_the_enumeration_is_the_whole_support(tiny_a, sup_a):
    assert len(sup_a) == tiny_a.path_count()
    assert sum(sup_a.values()) == pytest.approx(1.0, abs=1e-12)


def test_the_mixture_pmf_is_a_distribution(sup_a, sup_b):
    pmf = mixture_pmf([sup_a, sup_b], [0.3, 0.7])
    assert sum(pmf.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.3, 0.7), (0.9, 0.1)])
def test_log_pseq_matches_direct_summation(tiny_a, tiny_b, sup_a, sup_b,
                                           weights):
    mix = Mixture([tiny_a, tiny_b], weights)
    pmf = mixture_pmf([sup_a, sup_b], weights)
    for s, p in pmf.items():
        assert mix.log_pseq(s) == pytest.approx(math.log(p), rel=1e-12)


def test_log_pseq_batch_agrees_with_the_scalar_form(tiny_a, tiny_b, sup_a):
    mix = Mixture([tiny_a, tiny_b])
    seqs = sorted(sup_a)[:40]
    batch = mix.log_pseq_batch(seqs)
    assert batch == pytest.approx([mix.log_pseq(s) for s in seqs], rel=1e-13)


def test_inner_product_matches_direct_summation(tiny_a, tiny_b, sup_a, sup_b):
    """`<P_a, P_b> = sum_s P_a(s) P_b(s)`, over the sequences both spell."""
    mix = Mixture([tiny_a, tiny_b])
    shared = set(sup_a) & set(sup_b)
    assert shared, "the fixtures must overlap for this to be a real test"
    want = sum(sup_a[s] * sup_b[s] for s in shared)
    assert mix.inner_product(0, 1) == pytest.approx(want, rel=1e-12)


def test_three_way_inner_product_matches_direct_summation(
        tiny_a, tiny_b, tiny_c, sup_a, sup_b, sup_c):
    """The q >= 3 machinery is a k-way product, not a chain of pairs."""
    mix = Mixture([tiny_a, tiny_b, tiny_c])
    shared = set(sup_a) & set(sup_b) & set(sup_c)
    assert shared
    want = sum(sup_a[s] * sup_b[s] * sup_c[s] for s in shared)
    assert mix.inner_product(0, 1, 2) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("q", [2, 3, 4])
@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.25, 0.75)])
def test_hill_numbers_match_direct_summation(tiny_a, tiny_b, sup_a, sup_b,
                                             q, weights):
    """The whole q-tuple expansion, checked against `sum_s pbar(s)^q`."""
    mix = Mixture([tiny_a, tiny_b], weights)
    pmf = mixture_pmf([sup_a, sup_b], weights)
    assert mix.hill_number(q) == pytest.approx(brute_hill(pmf, q), rel=1e-11)


def test_hill_number_at_q0_matches_the_enumerated_union(tiny_a, tiny_b,
                                                        sup_a, sup_b):
    mix = Mixture([tiny_a, tiny_b])
    assert mix.hill_number(0.0) == float(len(set(sup_a) | set(sup_b)))


def test_three_member_hill_matches_direct_summation(
        tiny_a, tiny_b, tiny_c, sup_a, sup_b, sup_c):
    w = (0.2, 0.5, 0.3)
    mix = Mixture([tiny_a, tiny_b, tiny_c], w)
    pmf = mixture_pmf([sup_a, sup_b, sup_c], w)
    assert mix.hill_number(2.0) == pytest.approx(brute_hill(pmf, 2.0), rel=1e-11)
    assert mix.hill_number(0.0) == float(len(pmf))


def test_length_distribution_matches_direct_summation(tiny_a, tiny_b,
                                                      sup_a, sup_b):
    w = (0.3, 0.7)
    mix = Mixture([tiny_a, tiny_b], w)
    pmf = mixture_pmf([sup_a, sup_b], w)
    want: dict[int, float] = {}
    for s, p in pmf.items():
        want[len(s)] = want.get(len(s), 0.0) + p
    got = mix.length_distribution
    assert sorted(got) == sorted(want)
    for L in want:
        assert got[L] == pytest.approx(want[L], rel=1e-12)


def test_member_weights_within_a_length_match_direct_summation(
        tiny_a, tiny_b, sup_a, sup_b):
    """The renormalised within-length weights, which are NOT `w`.

    Using `w` unchanged still yields a normalised distribution and plausible
    numbers, so this is checked against the only thing that can catch it: the
    share of the length-L mass each member actually contributes.
    """
    w = (0.3, 0.7)
    mix = Mixture([tiny_a, tiny_b], w)
    for L in mix.observed_lengths:
        per = [sum(p for s, p in sup.items() if len(s) == L)
               for sup in (sup_a, sup_b)]
        total = w[0] * per[0] + w[1] * per[1]
        want = np.array([w[0] * per[0] / total, w[1] * per[1] / total])
        assert mix.member_weights(L) == pytest.approx(want, rel=1e-12)
        # And they are genuinely different from the unconditional weights on at
        # least one length, or this test would pass on a wrong implementation.
    assert any(not np.allclose(mix.member_weights(L), w, atol=1e-6)
               for L in mix.observed_lengths)


def test_entropy_bounds_bracket_the_true_entropy(tiny_a, tiny_b, sup_a, sup_b):
    w = (0.4, 0.6)
    mix = Mixture([tiny_a, tiny_b], w)
    h = brute_entropy(mixture_pmf([sup_a, sup_b], w))
    b = mix.entropy_bounds()
    assert b.lower <= h <= b.upper
    assert b.brackets(h)
    assert b.width == pytest.approx(b.weight_entropy, rel=1e-15)


def test_the_entropy_estimator_converges_on_the_true_value(
        tiny_a, tiny_b, sup_a, sup_b):
    """Unbiased, so it must close on the brute-force value as n grows.

    Asserted as a z-score against the reported standard error rather than a
    fixed tolerance: the claim being tested is that the error bar means what it
    says, which a raw tolerance would not check.
    """
    w = (0.4, 0.6)
    mix = Mixture([tiny_a, tiny_b], w)
    h = brute_entropy(mixture_pmf([sup_a, sup_b], w))
    est = mix.entropy_estimate(40000, seed=17)
    z = (est.value - h) / est.stderr
    assert abs(z) < 4.0, f"estimate {est.value} vs exact {h}, z={z}"
    assert est.bounds.brackets(est.value)
    assert est.exact is False
