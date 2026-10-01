"""The Mellin transform along a vertical line: ``M(re + i(start + k step))``.

The contour a Mellin-Barnes integral runs along is a vertical line sampled at
equal steps. There every order shares its real part, so each edge's modulus
factor is taken once for the whole line, and each order's rotation is the
previous one's times a fixed factor, recomputed directly every 32 orders.

That recurrence is exact algebra and not exact arithmetic, so this function is
held to :meth:`log_mellin_complex` -- the exact reference, which takes every
order from scratch -- to rounding rather than to the bit. What IS held to the
bit is everything the line itself determines: an order's value depends on the
line and its index only, not on how many orders follow it, the thread count
or the memory budget.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel


def _nt_model():
    rng = random.Random(5)
    seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
            + "C" for _ in range(400)]
    abund = [rng.randint(1, 40) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=3, max_length=6)
    every = ["".join(t) for L in range(1, 7)
             for t in itertools.product("ACGTN", repeat=L)]
    return m, np.asarray(m.log_pseq_batch(every), dtype=np.float64)


def _aa_model():
    rng = random.Random(11)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["C" + "".join(rng.choice("ASGLT") for _ in range(rng.randint(0, 1)))
            + "F" for _ in range(300)]
    abund = [rng.randint(1, 9) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, order=2, max_length=3)
    every = ["".join(t) for L in range(1, 4)
             for t in itertools.product(aa, repeat=L)]
    return m, np.asarray(m.log_pseq_batch(every), dtype=np.float64)


@pytest.fixture(scope="module", params=["nt", "aa"])
def enumerated(request):
    return _nt_model() if request.param == "nt" else _aa_model()


#: Lines on both sides of Re q = 0, stepping up and down, long enough to cross
#: several recurrence blocks (32 orders each) and to reach |Im q| ~ 40.
LINES = [(0.5, -20.0, 0.25, 161), (-0.4, 0.0, 0.3, 90), (1.5, 30.0, -0.7, 70)]


def _orders(re, start, step, n):
    return [complex(re, start + k * step) for k in range(n)]


def test_agrees_with_the_exact_reference(enumerated):
    m, lp = enumerated
    for re, start, step, n in LINES:
        got = m.log_mellin_line(re, start, step, n)
        want = m.log_mellin_complex(_orders(re, start, step, n))
        ref = m.log_mellin(re)                  # M(Re q), the sum of moduli
        d = np.abs(np.exp(got - ref) - np.exp(want - ref))
        assert d.max() <= 1e-12, (re, start, step, d.max())


def test_matches_brute_force(enumerated):
    m, lp = enumerated
    for re, start, step, n in LINES:
        got = m.log_mellin_line(re, start, step, n)
        q = np.array(_orders(re, start, step, n))
        terms = np.exp(q[:, None] * lp[None, :].astype(np.complex128))
        want = terms.sum(axis=1)
        scale = np.abs(terms).sum(axis=1)
        err = np.abs(np.exp(got) - want) / scale
        assert err.max() <= 1e-12, (re, start, step, err.max())


def test_an_order_does_not_depend_on_the_length_of_the_line(enumerated):
    m, _ = enumerated
    long = m.log_mellin_line(0.5, -20.0, 0.25, 161)
    for n in (1, 20, 32, 33, 100):
        short = m.log_mellin_line(0.5, -20.0, 0.25, n)
        assert np.array_equal(short.view(np.uint64), long[:n].view(np.uint64)), n


def test_threads_and_budget_do_not_change_the_bits(enumerated):
    m, _ = enumerated
    want = m.log_mellin_line(0.5, -20.0, 0.25, 161)
    before_t, before_b = BEAM.threads(), BEAM.memory_budget()
    try:
        BEAM.set_threads(1)
        a = m.log_mellin_line(0.5, -20.0, 0.25, 161)
        BEAM.set_threads(4)
        b = m.log_mellin_line(0.5, -20.0, 0.25, 161)
        # small enough to split the line into several blocks
        BEAM.set_memory_budget(2 << 20)
        c = m.log_mellin_line(0.5, -20.0, 0.25, 161)
    finally:
        BEAM.set_threads(before_t)
        BEAM.set_memory_budget(before_b)
    for x in (a, b, c):
        assert np.array_equal(x.view(np.uint64), want.view(np.uint64))


def test_order_zero_on_the_line_is_the_real_transform(enumerated):
    m, _ = enumerated
    v = m.log_mellin_line(0.7, 0.0, 1.0, 1)[0]
    assert v.real == pytest.approx(m.log_mellin(0.7), rel=1e-12, abs=1e-12)
    assert abs(v.imag) <= 1e-12


def test_calling_contract(enumerated):
    m, _ = enumerated
    out = m.log_mellin_line(0.5, 0.0, 0.1, 0)
    assert out.shape == (0,) and out.dtype == np.complex128
    for bad in ((math.nan, 0.0, 0.1, 3), (0.5, math.inf, 0.1, 3),
                (0.5, 0.0, math.nan, 3)):
        with pytest.raises((ValueError, BEAM.errors.BeamError)):
            m.log_mellin_line(*bad)
    with pytest.raises((ValueError, TypeError, BEAM.errors.BeamError)):
        m.log_mellin_line(0.5, 0.0, 0.1, -1)


def test_mellin_barnes_richness_on_the_line():
    """Expected richness from one line of M, against the enumerated sum."""
    from scipy.special import loggamma
    m, lp = _nt_model()
    c, h, T = -0.5, 0.1, 40.0
    n_nodes = int(round(2 * T / h)) + 1
    t = -T + h * np.arange(n_nodes)
    z = c + 1j * t
    logM = m.log_mellin_line(-c, T, -h, n_nodes)    # M(-z): re = -c, im = -t
    for n in (1e1, 1e3, 1e6, 1e9):
        vals = np.exp(loggamma(z) - z * math.log(n) + logM)
        got = float(-(vals.sum() * h).real / (2 * math.pi))
        want = float(np.sum(-np.expm1(-n * np.exp(lp))))
        assert got == pytest.approx(want, rel=1e-6), n
