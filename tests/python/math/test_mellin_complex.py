"""The Mellin transform at complex orders, ``ln sum_s P(s)^q`` for complex q.

Why complex orders at all: a sum over sequences of a function of P that does
not factor along the sequence -- expected richness ``sum_s 1 - e^{-n P(s)}``,
the occupancy counts, pairwise sharing -- is a contour integral of the power
sum ``M(q) = sum_s P(s)^q`` along a vertical line in the complex plane
(Mellin-Barnes). ``P(s)^q`` DOES factor along the sequence at any q, so M is
an exact DP at every point of that line, and the binned spectrum is not needed
for those quantities.

Every test here holds the DP against a direct sum over an enumerated support.
"""
from __future__ import annotations

import cmath
import itertools
import math
import random

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel

#: Complex orders spanning both half-planes of Re q, the real axis, the
#: imaginary axis, and imaginary parts large enough for the phases of
#: different sequences to cancel.
ORDERS = [0.5 + 0.0j, 1.0 + 0.0j, 0.5 + 3.0j, 0.5 - 3.0j, 0.3 + 11.0j,
          1.0 + 0.7j, 2.0 - 1.5j, -0.4 + 2.0j, 0.0 + 1.0j, 0.0 + 25.0j]


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


def _brute(lp, q):
    """M(q) by direct summation, and the sum of |terms| it is judged against.

    With complex q the terms carry phases and cancel, so |M| can be far below
    the sum of moduli; an error is only meaningful relative to what was
    summed, which is M(Re q)."""
    terms = np.exp(q * lp.astype(np.complex128))
    return complex(terms.sum()), float(np.abs(terms).sum())


def test_log_mellin_complex_matches_brute_force(enumerated):
    m, lp = enumerated
    assert np.isfinite(lp).all()
    got = m.log_mellin_complex(ORDERS)
    assert got.dtype == np.complex128 and got.shape == (len(ORDERS),)
    for q, lm in zip(ORDERS, got):
        want, scale = _brute(lp, q)
        assert abs(cmath.exp(lm) - want) <= 1e-12 * scale, (q, cmath.exp(lm), want)


def test_real_orders_agree_with_the_real_transform(enumerated):
    m, _ = enumerated
    qs = [0.0, 0.5, 1.0, 2.0, -0.3]
    got = m.log_mellin_complex([complex(q) for q in qs])
    for q, lm in zip(qs, got):
        assert lm.real == pytest.approx(m.log_mellin(q), rel=1e-12, abs=1e-12), q
        assert abs(lm.imag) <= 1e-12, q


def test_conjugate_order_gives_conjugate_value(enumerated):
    """P^conj(q) = conj(P^q) for real P, so M(conj q) = conj M(q)."""
    m, _ = enumerated
    qs = [0.5 + 3.0j, 0.3 + 11.0j, 2.0 - 1.5j]
    a = m.log_mellin_complex(qs)
    b = m.log_mellin_complex([q.conjugate() for q in qs])
    for x, y in zip(a, b):
        assert cmath.exp(x) == pytest.approx(cmath.exp(y).conjugate(), rel=1e-12)


def test_order_zero_counts_the_support(enumerated):
    m, lp = enumerated
    lm = m.log_mellin_complex([0j])[0]
    assert lm.real == pytest.approx(math.log(lp.size), rel=1e-12)
    assert abs(lm.imag) <= 1e-12


def test_batch_is_bit_identical_to_single_calls(enumerated):
    """Which orders share a traversal must not change any one of them."""
    m, _ = enumerated
    batch = m.log_mellin_complex(ORDERS)
    for q, v in zip(ORDERS, batch):
        one = m.log_mellin_complex([q])[0]
        assert one.real == v.real and one.imag == v.imag, q


def test_thread_count_does_not_change_the_bits(enumerated):
    m, _ = enumerated
    before = BEAM.threads()
    try:
        BEAM.set_threads(1)
        a = m.log_mellin_complex(ORDERS)
        BEAM.set_threads(4)
        b = m.log_mellin_complex(ORDERS)
    finally:
        BEAM.set_threads(before)
    assert np.array_equal(a.view(np.float64), b.view(np.float64))


def test_accepts_python_and_numpy_scalars_and_real_input(enumerated):
    m, _ = enumerated
    a = m.log_mellin_complex(np.array([0.5 + 3.0j]))
    b = m.log_mellin_complex([0.5 + 3.0j])
    c = m.log_mellin_complex([0.5])                     # a real order
    assert a[0] == b[0]
    assert c[0].real == pytest.approx(m.log_mellin(0.5), rel=1e-12)


def test_empty_input_gives_empty_output(enumerated):
    m, _ = enumerated
    out = m.log_mellin_complex([])
    assert out.shape == (0,) and out.dtype == np.complex128


def test_rejects_non_finite_orders(enumerated):
    m, _ = enumerated
    for bad in (complex(math.nan, 1.0), complex(1.0, math.inf)):
        with pytest.raises((ValueError, BEAM.errors.BeamError)):
            m.log_mellin_complex([bad])


def test_mellin_barnes_richness_from_complex_orders():
    """The reason this exists: expected richness from M alone, no spectrum.

    R(n) = sum_s 1 - e^{-n P(s)} = -(1/2pi) int Gamma(c+it) n^-(c+it)
    M(-(c+it)) dt for -1 < c < 0. Trapezoid on c = -1/2, against the direct
    sum over the enumerated support."""
    from scipy.special import loggamma
    m, lp = _nt_model()
    c, h, T = -0.5, 0.1, 40.0
    t = np.arange(-T, T + h / 2, h)
    z = c + 1j * t
    logM = m.log_mellin_complex(list(-z))
    for n in (1e1, 1e3, 1e6, 1e9):
        vals = np.exp(loggamma(z) - z * math.log(n) + logM)
        got = float(-(vals.sum() * h).real / (2 * math.pi))
        want = float(np.sum(-np.expm1(-n * np.exp(lp))))
        assert got == pytest.approx(want, rel=1e-6), n
