"""Exact occupancy quantities from power sums, against enumeration.

`ExactOccupancy` computes sums over every sequence of functions of P(s) --
expected richness, occupancy counts, sharing, private counts -- from the
power sums M(q) = sum_s P(s)^q alone: exact real series for small depths and
Mellin-Barnes contours on vertical lines for the rest. No binned spectrum.

Every value comes with a RIGOROUS relative bound, and the tests hold each one
to the direct sum over an enumerated support with that bound -- not to a
tolerance chosen here. A bound that is too loose to be useful fails too.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.analytics.exact_occupancy import BoundedValue, ExactOccupancy


def _enumerated(alphabet):
    if alphabet == "nt":
        rng = random.Random(5)
        seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
                + "C" for _ in range(400)]
        ab = [rng.randint(1, 40) for _ in seqs]
        m = BeamModel(seqs, abundances=ab, alphabet="nt", order=3, max_length=6)
        every = ["".join(t) for L in range(1, 7)
                 for t in itertools.product("ACGTN", repeat=L)]
    else:
        rng = random.Random(11)
        seqs = ["C" + "".join(rng.choice("ASGLT") for _ in range(rng.randint(0, 2)))
                + "F" for _ in range(300)]
        ab = [rng.randint(1, 9) for _ in seqs]
        m = BeamModel(seqs, abundances=ab, order=2, max_length=4)
        every = ["".join(t) for L in range(1, 5)
                 for t in itertools.product("ACDEFGHIKLMNPQRSTVWY", repeat=L)]
    p = np.exp(np.asarray(m.log_pseq_batch(every), dtype=np.float64))
    return m, p.astype(np.longdouble)


@pytest.fixture(scope="module", params=["nt", "aa"])
def case(request):
    m, p = _enumerated(request.param)
    return m, p, ExactOccupancy(m, rel_tol=1e-10)


def _miss(p, n):
    """(1 - p)^n in long double, safe for tiny p and huge n."""
    return np.exp(np.longdouble(n) * np.log1p(-p))


def _seen(p, n):
    """1 - (1 - p)^n, as -expm1: the direct difference loses every digit
    that 1 and (1-p)^n share, which for tiny p is all of them."""
    return -np.expm1(np.longdouble(n) * np.log1p(-p))


def _log_binom(n, j):
    """ln C(n, j) for integer j, as a sum of logs: lgamma(n+1) at n = 1e8 is
    1.7e9 and a difference of two of them is good to 1e-7 at best."""
    return (float(np.sum(np.log(np.longdouble(n) - np.arange(j, dtype=np.longdouble))))
            - math.lgamma(j + 1))


def _held(v: BoundedValue, exact: float, tol: float = 1e-10):
    """The value is within its own bound of the truth, and the bound is useful."""
    assert isinstance(v, BoundedValue)
    assert abs(v.value - exact) <= v.bound * abs(exact) + 1e-300, (v, exact)
    assert v.bound <= tol, v


DEPTHS = [0.5, 1.0, 3.0, 17.5, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e13]


def test_richness(case):
    m, p, eo = case
    for n in DEPTHS:
        exact = float(np.sum(_seen(p, n)))
        _held(eo.richness(n), exact)


def test_richness_corners(case):
    m, p, eo = case
    assert eo.richness(0.0).value == 0.0
    _held(eo.richness(1.0), float(np.sum(p)))          # R(1) is the mass
    d0 = float(p.size)
    big = eo.richness(1e40)                            # every sequence drawn
    assert big.value == pytest.approx(d0, rel=1e-10)


def test_richness_many_depths_at_once(case):
    m, p, eo = case
    ns = np.geomspace(1.0, 1e12, 40)
    vals = eo.richness_many(ns)
    for n, v in zip(ns, vals):
        _held(v, float(np.sum(_seen(p, n))))


def test_frequency(case):
    """E[# sequences drawn exactly j times in n draws] = sum C(n,j) p^j (1-p)^{n-j}."""
    m, p, eo = case
    for n in (5.0, 50.0, 1e3, 1e5, 1e8):
        for j in (1, 2, 3, 5):
            if j > n:
                continue
            logc = _log_binom(n, j)
            exact = float(np.sum(np.exp(np.longdouble(logc) + j * np.log(p)
                                        + np.longdouble(n - j) * np.log1p(-p))))
            _held(eo.frequency(n, j), exact, tol=1e-9)


def test_frequencies_add_up_to_richness(case):
    """Seen at least once = sum over j >= 1 of seen exactly j times (n small)."""
    m, p, eo = case
    n = 6.0
    total = sum(eo.frequency(n, j).value for j in range(1, 7))
    assert total == pytest.approx(eo.richness(n).value, rel=1e-10)


def test_richness_difference_keeps_its_digits(case):
    """R(n1) - R(n2) for n1 close to n2, where subtracting two R's would
    lose every digit the two share."""
    m, p, eo = case
    for n2, gap in ((1e6, 1.0), (1e9, 1e3), (1e12, 7.0)):
        exact = float(np.sum(_miss(p, n2) * _seen(p, gap)))
        _held(eo.richness_difference(n2 + gap, n2), exact, tol=1e-9)


def test_sharing(case):
    """E[# sequences in both of two independent repertoires of depths d1, d2]."""
    m, p, eo = case
    for d1, d2 in ((1e3, 1e3), (1e3, 1e6), (5e5, 2e8)):
        exact = float(np.sum(_seen(p, d1) * _seen(p, d2)))
        _held(eo.sharing(d1, d2), exact, tol=1e-9)


def test_private_in_a_cohort(case):
    """E[# sequences seen in exactly one repertoire], unequal depths."""
    m, p, eo = case
    rng = np.random.default_rng(3)
    depths = rng.integers(100, 100_000, size=25).astype(float)
    T = depths.sum()
    exact = float(np.sum([np.sum(_seen(p, d) * _miss(p, T - d)) for d in depths]))
    _held(eo.private(depths), exact, tol=1e-9)


def test_equal_depth_cohort(case):
    """N repertoires of one depth, N up to 1e10 where R(N d) - R((N-1) d)
    differ in their tenth digit."""
    m, p, eo = case
    for depth, N in ((1e3, 10), (1e4, 768), (1e3, 1e10)):
        seen_exact = float(np.sum(_seen(p, N * depth)))
        one = _seen(p, depth) * _miss(p, (N - 1) * depth)
        priv_exact = float(N * np.sum(one))
        c = eo.equal_depth_cohort(depth, N)
        _held(c["seen"], seen_exact)
        _held(c["private"], priv_exact, tol=1e-9)


def test_bounded_value_does_not_pass_for_a_float(case):
    m, p, eo = case
    v = eo.richness(10.0)
    with pytest.raises(TypeError):
        float(v)
    assert v.value > 0 and 0 <= v.bound < 1


def test_lines_are_reused(case):
    m, p, eo = case
    eo.richness(1e6)
    before = len(eo.lines)
    for n in (1.1e6, 2e6, 5e6):
        eo.richness(n)
    assert len(eo.lines) == before


def test_an_extended_line_matches_a_computed_one(case):
    """Extending a line appends the nodes a longer line would have computed."""
    m, p, eo = case
    fresh = ExactOccupancy(m, rel_tol=1e-10)
    short = fresh._line(0.5, 0.1, 5.0)
    n_short = short.t.size
    long_ = fresh._line(0.5, 0.1, 12.0)
    assert long_ is short and long_.t.size > n_short     # extended in place
    direct = m.log_mellin_line(0.5, 0.0, -0.1, long_.t.size)
    ref = m.log_mellin(0.5)
    d = np.abs(np.exp(long_.log_m - ref) - np.exp(direct - ref))
    assert d.max() <= 1e-13


def test_rejects_bad_depths(case):
    m, p, eo = case
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            eo.richness(bad)
    with pytest.raises(ValueError):
        eo.frequency(10.0, 0)
