"""ln Gamma(w) - ln Gamma(w + z) in C, against 30 significant digits of it.

The contour kernels of `ExactOccupancy` are built from this ratio at every
node of a line and every depth asked for: w = n + 1 runs from 1 to 1e82, z
from the line's real part to far out along it. Taking it in numpy, a
ninety-term series with a full matrix per term, had become most of the
non-sweep time on the figure steps.

What the bounds assume of it is an ABSOLUTE error of a few rounding units of
its size, |z| ln(w + |z|), and that is what is held here -- the imaginary
part only modulo 2 pi, since only exp of it is ever used.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

mpmath = pytest.importorskip("mpmath")

from BEAM.analytics.exact_occupancy import _log_gamma_ratio_c  # noqa: E402

WS = [1.0, 1.5, 2.0, 7.3, 14.9, 15.0, 15.1, 100.0, 1e4, 1e8, 1e20, 1e50, 1e82]
ZS = [-0.9 + 0.1j, -0.5 + 0j, -0.5 + 1j, 0.5 + 0j, 0.3 + 40j, 0.7 - 5j,
      -0.2 + 1e3j, -0.5 + 1e6j, 1e-8j, -0.75 + 3.25j, 1.0 + 0.5j, 0.9 - 250j]


def _reference(w, z):
    # ln Gamma(w) is about w ln w, so 30 digits of the DIFFERENCE need 30
    # beyond that: at w = 1e50 a 30-digit reference returns 0.
    mpmath.mp.dps = 30 + int(math.log10(w * (1.0 + math.log(w)) + 1.0))
    v = mpmath.loggamma(mpmath.mpf(w)) - mpmath.loggamma(mpmath.mpf(w) + mpmath.mpc(z))
    return complex(v)


def test_matches_thirty_digits_everywhere_it_is_used():
    w = np.array(WS)
    z = np.array(ZS)
    got = _log_gamma_ratio_c(w, z)
    assert got.shape == (z.size, w.size)
    worst = 0.0
    for i, zz in enumerate(ZS):
        for k, ww in enumerate(WS):
            want = _reference(ww, zz)
            scale = 1.0 + abs(zz) * (1.0 + math.log(ww + abs(zz)))
            d_re = abs(got[i, k].real - want.real)
            d_im = abs(math.remainder(got[i, k].imag - want.imag, 2 * math.pi))
            err = max(d_re, d_im) / scale
            worst = max(worst, err)
            assert err <= 8 * np.finfo(float).eps, (ww, zz, got[i, k], want)
    print(f"worst {worst:.2e} of the scale")


def test_agrees_with_the_numpy_form_it_replaces():
    """To the numpy form's own error: against 30 digits on this grid that
    form is off by up to 5.8e-15 of the scale (at w = 40, z = -0.4), the C one
    by 2.1e-16."""
    from BEAM.analytics.exact_occupancy import _log_ratio_matrix_numpy
    w = np.array([1.0, 3.0, 40.0, 1e3, 1e6, 1e12, 1e40])
    t = np.linspace(0.0, 60.0, 61)
    z = -0.4 + 1j * t
    a = _log_gamma_ratio_c(w, z)
    b = _log_ratio_matrix_numpy(w, z)
    scale = 1.0 + np.abs(z)[:, None] * (1.0 + np.log(w[None, :] + np.abs(z)[:, None]))
    d = np.abs(np.exp(1j * (a.imag - b.imag)) - 1.0) + np.abs(a.real - b.real)
    assert np.max(d / scale) <= 32 * np.finfo(float).eps


def test_zero_is_zero():
    got = _log_gamma_ratio_c(np.array([1.0, 1e82]), np.array([0j]))
    assert np.all(got == 0)


# The cohort private kernel's exponent: how much the ratio moves when w drops
# by d. Tiny next to either ratio -- d |z| / w against |z| ln w -- so it cannot
# be their difference; the Taylor series in d it replaces needed a matrix per
# power of d.
SHIFT_WS = [200.0, 1e3, 1e6, 1e12, 1e40, 1e82]
SHIFT_ZS = [-0.02 + 0j, -0.5 + 0j, -0.5 + 1j, -0.9 + 0.1j, -0.3 + 40j,
            -0.7 - 5j, -0.2 + 1e3j, -0.5 + 1e5j, -0.75 + 3.25j]


def _shift_reference(w, d, z):
    # Four log-gammas of size w ln w whose sum is d |z| / w: enough digits
    # for both.
    big = math.log10(w * (1.0 + math.log(w)) + 1.0)
    mpmath.mp.dps = 30 + int(2 * big)
    W, V, Z = mpmath.mpf(w), mpmath.mpf(w) - mpmath.mpf(d), mpmath.mpc(z)
    v = (mpmath.loggamma(V) - mpmath.loggamma(V + Z)
         - mpmath.loggamma(W) + mpmath.loggamma(W + Z))
    return complex(v)


def test_the_shifted_ratio_keeps_its_digits():
    """[ln G(w-d) - ln G(w-d+z)] - [ln G(w) - ln G(w+z)] to a few rounding
    units of d |z| / (w - d), its size where |z| << w and what bounds its
    error everywhere -- the imaginary part modulo 2 pi."""
    from BEAM.analytics.exact_occupancy import _log_gamma_ratio_shift_c
    worst = 0.0
    for w in SHIFT_WS:
        ds = sorted({1.0, 37.0, 1e-3 * w, 0.25 * w})
        z = np.array(SHIFT_ZS)
        got = _log_gamma_ratio_shift_c(w, np.array(ds), z)
        assert got.shape == (z.size, len(ds))
        for i, zz in enumerate(SHIFT_ZS):
            for k, d in enumerate(ds):
                want = _shift_reference(w, d, zz)
                scale = abs(want) + d * abs(zz) / (w - d)
                d_re = abs(got[i, k].real - want.real)
                d_im = abs(math.remainder(got[i, k].imag - want.imag, 2 * math.pi))
                err = max(d_re, d_im) / scale
                worst = max(worst, err)
                assert err <= 8 * np.finfo(float).eps, (w, d, zz, got[i, k], want)
    print(f"worst {worst:.2e} of the scale")


def test_the_shifted_ratio_refuses_a_small_shifted_argument():
    """Stirling's series needs w - d >= 15; below that it says so with NaN
    rather than return a number with no digits."""
    from BEAM.analytics.exact_occupancy import _log_gamma_ratio_shift_c
    got = _log_gamma_ratio_shift_c(200.0, np.array([1.0, 190.0]), np.array([-0.5 + 1j]))
    assert np.isfinite(got[0, 0]) and np.isnan(got[0, 1].real)
