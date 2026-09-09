"""Cases where the answer is known exactly in advance.

Three of them, and between them they pin the whole beta-diversity apparatus:

  - a model mixed with ITSELF is that model, at every order;
  - members that share nothing are M effectively distinct repertoires;
  - members that are the same distribution are 1.

The last two are the endpoints of `1 <= D_beta <= M`, so a formula that is
wrong anywhere in between is very likely to be wrong at one of them -- and the
particular error this guards against, averaging the members' diversities
arithmetically instead of by the power mean of order 1-q, IS wrong at both.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import Mixture
from _tiny import brute_entropy, mixture_pmf, support, tiny_model


# ── A model mixed with itself ───────────────────────────────────

@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.3, 0.7), (0.99, 0.01)])
@pytest.mark.parametrize("q", [0.0, 2.0, 3.0])
def test_a_model_mixed_with_itself_reproduces_it(tiny_a, weights, q):
    """`pbar = sum_i w_i P = P` for any weights, so every order must agree.

    Aliasing the same model into both slots is legitimate here in a way it is
    not for `add` and `subtract`: nothing in this package mutates a member, so
    there is no source-equals-destination hazard to refuse.
    """
    mix = Mixture([tiny_a, tiny_a], weights)
    assert mix.hill_number(q) == pytest.approx(tiny_a.hill_number(q), rel=1e-12)


def test_a_model_mixed_with_itself_reproduces_its_scores(tiny_a, sup_a):
    mix = Mixture([tiny_a, tiny_a], (0.3, 0.7))
    for s in sorted(sup_a)[:60]:
        assert mix.log_pseq(s) == pytest.approx(tiny_a.log_pseq(s), rel=1e-13)


def test_a_model_mixed_with_itself_has_the_entropy_it_had(tiny_a):
    """Both bounds collapse: `sum w_i H_i` is `H`, and the true value is too."""
    mix = Mixture([tiny_a, tiny_a], (0.3, 0.7))
    b = mix.entropy_bounds()
    assert b.lower == pytest.approx(tiny_a.entropy(), rel=1e-13)
    est = mix.entropy_estimate(20000, seed=5)
    assert abs((est.value - tiny_a.entropy()) / est.stderr) < 4.0


# ── Members that share nothing ──────────────────────────────────

def test_the_disjoint_fixtures_really_are_disjoint(far_a, far_b):
    """Asserted, not assumed: the tests below mean nothing without it."""
    sa, sb = support(far_a), support(far_b)
    assert sa and sb
    assert not (set(sa) & set(sb))


@pytest.mark.parametrize("q", [0.0, 2.0])
def test_disjoint_members_give_beta_equal_to_M(far_a, far_b, q):
    """The upper end of `1 <= D_beta <= M`, exactly.

    Exact at these orders because gamma is exact at these orders. The naive
    arithmetic mean of the members' D_q would put beta strictly below 2 here
    whenever the members differ in diversity; see the test below.
    """
    mix = Mixture([far_a, far_b])
    r = mix.beta_diversity(q)
    assert r.exact
    assert r.beta == pytest.approx(2.0, rel=1e-10)


def test_disjoint_members_give_beta_equal_to_M_at_q1(far_a, far_b):
    """At q=1 gamma has no closed form, so the EXACT statement is the bound.

    For disjoint members `H(pbar) = sum_i w_i H_i + H(w)` exactly -- the upper
    bound is attained -- so `beta = exp(H(w)) = M`. That is checked three ways:
    the bound equals M, brute-force enumeration lands on the bound, and the
    sampled estimate agrees within its own error bar.
    """
    mix = Mixture([far_a, far_b])
    r = mix.beta_diversity(1.0, samples=40000, seed=23)
    assert not r.exact and r.beta is None
    assert r.upper == pytest.approx(2.0, rel=1e-12)

    b = mix.entropy_bounds()
    pmf = mixture_pmf([support(far_a), support(far_b)], (0.5, 0.5))
    assert brute_entropy(pmf) == pytest.approx(b.upper, rel=1e-12)

    assert abs((r.estimate - 2.0) / r.stderr) < 4.0


def test_disjoint_members_give_a_zero_inner_product(far_a, far_b):
    """`-inf` in the log domain, exactly 0.0 on the linear one. An answer."""
    mix = Mixture([far_a, far_b])
    assert mix.log_inner_product(0, 1) == -math.inf
    assert mix.inner_product(0, 1) == 0.0


def test_three_disjoint_members_give_beta_equal_to_three():
    """M = 3, to show the 2 above is not a coincidence of pairs.

    Built from its own three-way split of the residues rather than from the
    module fixtures: `far_a` and `far_b` use two pools between them, and a
    third member disjoint from both needs a third pool.
    """
    a = tiny_model(41, letters="AC")
    b = tiny_model(42, letters="DG")
    c = tiny_model(43, letters="FH")
    sa, sb, sc = support(a), support(b), support(c)
    assert not (set(sa) & set(sb)) and not (set(sa) & set(sc))
    assert not (set(sb) & set(sc))
    mix = Mixture([a, b, c])
    assert mix.beta_diversity(0.0).beta == pytest.approx(3.0, rel=1e-10)
    assert mix.beta_diversity(2.0).beta == pytest.approx(3.0, rel=1e-10)
    assert mix.beta_diversity(1.0).upper == pytest.approx(3.0, rel=1e-12)


# ── Members that are the same distribution ──────────────────────

@pytest.mark.parametrize("q", [0.0, 2.0, 3.0])
def test_identical_members_give_beta_equal_to_one(tiny_a, q):
    mix = Mixture([tiny_a, tiny_a, tiny_a])
    r = mix.beta_diversity(q)
    assert r.exact
    assert r.beta == pytest.approx(1.0, rel=1e-12)


def test_identical_members_give_beta_equal_to_one_at_q1(tiny_a):
    """The lower end of the bracket is attained, and it is exactly 1."""
    mix = Mixture([tiny_a, tiny_a, tiny_a])
    r = mix.beta_diversity(1.0, samples=20000, seed=29)
    assert r.lower == 1.0
    assert abs((r.estimate - 1.0) / r.stderr) < 4.0


# ── The power mean, and the trap it replaces ────────────────────

@pytest.mark.parametrize("q,name", [(0.0, "arithmetic"), (1.0, "geometric"),
                                    (2.0, "harmonic")])
def test_alpha_is_the_power_mean_of_order_one_minus_q(tiny_a, tiny_b, q, name):
    """Arithmetic at q=0, geometric at q=1, harmonic at q=2, by hand."""
    w = np.array([0.3, 0.7])
    mix = Mixture([tiny_a, tiny_b], w)
    d = np.array([tiny_a.hill_number(q), tiny_b.hill_number(q)])
    want = {"arithmetic": lambda: float(np.dot(w, d)),
            "geometric": lambda: float(np.exp(np.dot(w, np.log(d)))),
            "harmonic": lambda: float(1.0 / np.dot(w, 1.0 / d))}[name]()
    assert mix.alpha_diversity(q) == pytest.approx(want, rel=1e-12)


def test_the_naive_arithmetic_mean_would_understate_beta(far_a, far_b):
    """Why `alpha_diversity` exists rather than being left to the caller.

    On disjoint members beta must be exactly 2. Averaging the members' D2
    arithmetically instead of harmonically returns something strictly smaller
    whenever they differ in diversity, and it looks entirely plausible -- a
    measured two-donor case returned 1.325. The point of this test is that the
    correct helper is shipped and the incorrect one is demonstrably wrong on a
    case with a known answer.
    """
    mix = Mixture([far_a, far_b])
    d2 = np.array([far_a.hill_number(2.0), far_b.hill_number(2.0)])
    assert d2[0] != pytest.approx(d2[1], rel=1e-3), \
        "the fixtures must differ in diversity or the trap cannot fire"

    gamma = mix.hill_number(2.0)
    assert gamma / mix.alpha_diversity(2.0) == pytest.approx(2.0, rel=1e-10)

    naive = gamma / float(d2.mean())
    assert naive < 2.0 - 1e-6
