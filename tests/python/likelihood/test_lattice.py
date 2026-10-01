"""The saddlepoint for a sum of independent categorical units, against exact
convolution.

Every approximate piece of the repertoire likelihood -- how many unique, how
many shared, the abundance pattern -- is the probability that independent
units land in given category totals. This file pins how close the
saddlepoint gets, and that the cases it should answer exactly it does.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from BEAM.likelihood.lattice import categorical_sum_log_pmf
from _brute import SLACK, exact_categorical_sum


def _call(unit_probs, counts):
    q = np.asarray(unit_probs, dtype=float)
    with np.errstate(divide="ignore"):
        return categorical_sum_log_pmf(np.log(q), np.ones(len(q)),
                                       np.asarray(counts, dtype=float))


_EXACT = 1e-9


def _check(unit_probs, key, exact_prob):
    """A value flagged exact must BE exact; otherwise it must lie within the
    error the function itself reports -- ``sum_j 1/(12 v_j)`` over the
    categories its normal term covered -- give or take ``SLACK``, since that is
    a leading-order estimate and not a ceiling."""
    got, is_exact, err = _call(unit_probs, key)
    tol = _EXACT if is_exact else SLACK * err
    assert got == pytest.approx(math.log(exact_prob), abs=tol), (key, is_exact, err)
    return is_exact


@pytest.fixture(scope="module")
def binary_units():
    rng = random.Random(3)
    ps = [rng.uniform(0.001, 0.3) for _ in range(300)]
    return [(1 - p, p) for p in ps]


@pytest.fixture(scope="module")
def three_way_units():
    rng = np.random.default_rng(4)
    return [tuple(r) for r in rng.dirichlet([3.0, 1.0, 0.6], size=70)]


def test_a_poisson_binomial_matches_exact_convolution(binary_units):
    exact = exact_categorical_sum(binary_units)
    mean = sum(p for _, p in binary_units)
    flags = []
    for n in [1, 2, 5, 20, int(mean) - 15, int(mean), int(mean) + 15, 120]:
        key = (len(binary_units) - n, n)
        flags.append(_check(binary_units, key, exact[key]))
    assert not all(flags)     # the bulk goes through the saddlepoint


def test_three_categories_match_exact_convolution(three_way_units):
    exact = exact_categorical_sum(three_way_units)
    checked = 0
    for key, prob in exact.items():
        if min(key) < 1 or prob < 1e-250:
            continue
        _check(three_way_units, key, prob)
        checked += 1
    assert checked > 500


def test_an_empty_category_is_conditioned_on_exactly(three_way_units):
    exact = exact_categorical_sum(three_way_units)
    for key in [(50, 20, 0), (50, 0, 20), (0, 35, 35)]:
        _check(three_way_units, key, exact[key])


def test_every_unit_in_one_category_is_exact(three_way_units):
    key = (70, 0, 0)
    got, is_exact, _ = _call(three_way_units, key)
    assert is_exact
    expected = sum(math.log(q[0]) for q in three_way_units)
    assert got == pytest.approx(expected, abs=1e-10)


def test_a_count_no_unit_can_reach_is_impossible():
    units = [(0.5, 0.5, 0.0)] * 10
    got, is_exact, _ = _call(units, (5, 4, 1))
    assert got == -math.inf and is_exact


@pytest.fixture(scope="module")
def one_rare_route():
    """40 ordinary units, and two that can reach a third category only with
    probability 1e-25 and 1e-40. Observing that category at all forces the
    saddlepoint so far into the tail that the category's count is almost
    fixed, which is where a normal term stops meaning anything."""
    rng = random.Random(6)
    units = []
    for _ in range(40):
        p = rng.uniform(0.1, 0.5)
        units.append((1 - p, p, 0.0))
    units.append((0.5, 0.5 - 1e-25, 1e-25))
    units.append((0.7, 0.3 - 1e-40, 1e-40))
    return units


@pytest.mark.parametrize("key", [(25, 16, 1), (20, 21, 1), (22, 18, 2)])
def test_a_nearly_unreachable_category_matches_exact_convolution(
        one_rare_route, key):
    exact = exact_categorical_sum(one_rare_route)
    _check(one_rare_route, key, exact[key])


def test_two_nearly_fixed_categories_match_exact_convolution():
    units = [(0.6, 0.4, 0.0, 0.0)] * 30 + [(0.5, 0.5 - 2e-20, 1e-20, 1e-20),
                                           (0.4, 0.6 - 1e-30, 0.0, 1e-30),
                                           (1e-18, 0.0, 1.0 - 1e-18, 0.0)]
    exact = exact_categorical_sum(units)
    for key in [(18, 12, 2, 1), (17, 13, 1, 2)]:
        _check(units, key, exact[key])


def test_real_multiplicities_behave_like_repeated_units(binary_units):
    """The spectrum's multiplicities are real numbers of sequences per bin, so
    two copies of a unit must count exactly as that unit with weight two."""
    once = binary_units[:40]
    q = np.log(np.asarray(once))
    doubled, _, _ = categorical_sum_log_pmf(np.vstack([q, q]), np.ones(80),
                                            np.array([70.0, 10.0]))
    weighted, _, _ = categorical_sum_log_pmf(q, np.full(40, 2.0),
                                             np.array([70.0, 10.0]))
    assert weighted == pytest.approx(doubled, abs=1e-9)


def test_the_counts_must_account_for_every_unit():
    with pytest.raises(ValueError):
        _call([(0.5, 0.5)] * 10, (3, 3))


def _log_pois(k, lam):
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def _log_binom(k, n, p):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def test_a_huge_implied_rest_keeps_the_small_counts():
    """A spectrum can hold 1e24 sequences. 1e24 - 12 is 1e24 in float64, so a
    total formed by subtraction has already lost the 12; the rest is passed as
    implied and never formed. Three bins of 1e23-scale units that each land in
    category 1 with probability 1e-23 are Poisson(10) to within 1e-23, plus
    five ordinary units: the exact answer is a short convolution."""
    q = 1e-23
    lq = np.array([[math.log1p(-q), math.log(q)]] * 3
                  + [[math.log(0.4), math.log(0.6)]])
    mult = np.array([4e23, 3e23, 3e23, 5.0])
    for x in (4, 12, 25):
        exact = math.log(sum(math.exp(_log_pois(x - k, 10.0)
                                      + _log_binom(k, 5, 0.6))
                             for k in range(0, min(5, x) + 1)))
        got, is_exact, err = categorical_sum_log_pmf(lq, mult,
                                                     np.array([0.0, x]), rest=0)
        tol = _EXACT if is_exact else SLACK * err
        assert got == pytest.approx(exact, abs=tol), x


def test_an_implied_rest_agrees_with_an_explicit_one(three_way_units):
    q = np.log(np.asarray(three_way_units))
    key = np.array([45.0, 17.0, 8.0])
    explicit, _, _ = categorical_sum_log_pmf(q, np.ones(70), key)
    implied, _, _ = categorical_sum_log_pmf(q, np.ones(70),
                                            np.array([0.0, 17.0, 8.0]), rest=0)
    assert implied == pytest.approx(explicit, abs=1e-12)
