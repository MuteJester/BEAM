"""The tilted zero-truncated-Poisson bin moments, in C, against numpy.

Every Newton step of ``counts_given_set_log_pmf`` needs, for each seen
sequence, the tilted probability of each abundance bin and the first two
moments of its count. The numpy form builds a (sequences x counts) matrix per
step and scatters it into bins with ``np.add.at``; on a repertoire of 100,000
sequences with bins up to 101 reads that was 11 of the 12 seconds a donor
cost. The C kernel accumulates each sequence's bins in one pass over the
counts, with no matrix, so only the REDUCTIONS over sequences cross back --
which is all the caller ever used.

What is held here is that the two agree. The quantities themselves are held
against exact convolution by ``test_fixed_total.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import _cbeam as _c
from BEAM.likelihood.fixed_total import _Units

#: Sequences the model makes almost impossible, ordinary ones, and clones so
#: large that the Poisson mass sits far above the top bin.
LAMS = np.array([1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 1.7, 3.0, 8.0,
                 25.0, 60.0, 120.0, 300.0])
EDGES = [(1, 2, 3, 4, 5, 6, 11, 101), (1, 2, 4), (1,), (1, 2, 3, 4, 5, 6, 11)]


def _numpy_reduction(units, t_bins, tau, allowed):
    """What the Newton step actually consumes, from the numpy moments."""
    logZ, Pb, Ec, Ec2, Rb = units.moments(t_bins, tau, allowed)
    S = np.column_stack([Pb, Ec])
    return dict(logZ=float(np.sum(logZ)), Pb=Pb.sum(axis=0), Rb=Rb.sum(axis=0),
                Ec=float(Ec.sum()), Ec2=float(Ec2.sum()), gram=S.T @ S)


def _c_reduction(lam, edges, t_bins, tau, allowed):
    raw = _c.ztp_bin_reduce(np.ascontiguousarray(lam, dtype=float),
                            np.ascontiguousarray(t_bins, dtype=float),
                            np.asarray(edges, dtype=float),
                            np.asarray(allowed, dtype=float), float(tau))
    v = np.frombuffer(raw, dtype=float)
    B = len(edges)
    return dict(logZ=v[0], Pb=v[1:1 + B].copy(), Rb=v[1 + B:1 + 2 * B].copy(),
                Ec=v[1 + 2 * B], Ec2=v[2 + 2 * B],
                gram=v[3 + 2 * B:].reshape(B + 1, B + 1).copy())


#: ``logZ`` is a difference of terms the size of the series' own peak, which
#: for a clone with a Poisson mean of 300 is about 1,000 nats. Its rounding
#: floor is therefore around 1e-13, and at no tilt its true value is exactly
#: zero -- so it is held to an ABSOLUTE tolerance. It enters the saddlepoint
#: as a sum of thousands of nats, where 1e-9 is nothing.
_ATOL = {"logZ": 1e-9}


def _close(a, b, tol=1e-9, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.maximum(np.abs(a), np.abs(b))
    return np.all(np.abs(a - b) <= atol + tol * np.maximum(scale, 1e-12))


@pytest.mark.parametrize("edges", EDGES)
@pytest.mark.parametrize("tau", [0.0, 0.7, -1.3])
def test_the_reductions_match_numpy_without_a_tilt_on_the_bins(edges, tau):
    units = _Units(LAMS, edges)
    B = len(edges)
    allowed = np.ones(B, dtype=bool)
    want = _numpy_reduction(units, np.zeros(B), tau, allowed)
    got = _c_reduction(LAMS, edges, np.zeros(B), tau, allowed)
    for k in want:
        assert _close(want[k], got[k], atol=_ATOL.get(k, 0.0)), (k, want[k], got[k])


@pytest.mark.parametrize("edges", [EDGES[0], EDGES[1]])
def test_the_reductions_match_numpy_with_a_tilt(edges):
    rng = np.random.default_rng(4)
    units = _Units(LAMS, edges)
    B = len(edges)
    allowed = np.ones(B, dtype=bool)
    for _ in range(5):
        t_bins = rng.normal(0.0, 2.0, size=B)
        tau = float(rng.normal(0.0, 0.8))
        want = _numpy_reduction(units, t_bins, tau, allowed)
        got = _c_reduction(LAMS, edges, t_bins, tau, allowed)
        for k in want:
            assert _close(want[k], got[k], atol=_ATOL.get(k, 0.0)), (k, t_bins, tau, want[k], got[k])


def test_a_disallowed_bin_takes_no_mass():
    edges = (1, 2, 3, 4, 5, 6, 11, 101)
    B = len(edges)
    units = _Units(LAMS, edges)
    allowed = np.array([True, False, True, True, False, True, True, False])
    t_bins = np.zeros(B)
    want = _numpy_reduction(units, t_bins, 0.3, allowed)
    got = _c_reduction(LAMS, edges, t_bins, 0.3, allowed)
    for k in want:
        assert _close(want[k], got[k], atol=_ATOL.get(k, 0.0)), (k, want[k], got[k])
    assert got["Pb"][1] == 0.0 and got["Pb"][4] == 0.0 and got["Pb"][7] == 0.0


def test_the_expected_bins_match_numpy():
    from BEAM.likelihood.fixed_total import expected_bins_given_reads
    edges = (1, 2, 3, 4, 5, 6, 11, 101)
    rng = np.random.default_rng(11)
    lam = np.concatenate([LAMS, rng.gamma(0.3, 0.5, size=400)])
    total = int(round(float(np.sum(lam / -np.expm1(-lam))))) + 30
    want = expected_bins_given_reads(lam, total, edges)
    raw = _c.ztp_bin_expect(np.ascontiguousarray(lam), np.asarray(edges, float),
                            float(total))
    got = np.frombuffer(raw, dtype=float)
    assert got.size == len(edges)
    assert _close(want, got, tol=1e-8), (want, got)
    assert abs(got.sum() - lam.size) <= 1e-6 * lam.size


def test_one_sequence_is_a_distribution():
    edges = (1, 2, 3, 4, 5, 6, 11, 101)
    B = len(edges)
    got = _c_reduction(np.array([2.5]), edges, np.zeros(B), 0.0, np.ones(B, bool))
    assert abs(got["Pb"].sum() - 1.0) <= 1e-12
    # E[c] of a zero-truncated Poisson(2.5) is lam / (1 - e^-lam)
    assert got["Ec"] == pytest.approx(2.5 / -math.expm1(-2.5), rel=1e-10)
