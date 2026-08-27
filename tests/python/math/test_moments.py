"""Surprisal moments, pooled and per length.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Surprisal moments
# ═══════════════════════════════════════════════════════════════

def test_surprisal_mean_is_the_entropy(model):
    """At q=1 the mean surprisal IS the entropy, computed by a different DP."""
    mu = model.surprisal_moments(q=1.0, order=1)
    assert mu[0] == pytest.approx(1.0, abs=1e-12)
    assert mu[1] == pytest.approx(model.entropy(), rel=1e-9)


def test_surprisal_moments_match_brute_force():
    """Exhaustive: every length-3 and length-4 sequence, weighted by its own
    probability. This pins the jet DP against a direct sum."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)

    for q in (1.0, 2.0):
        wsum = 0.0
        acc = np.zeros(5)
        for L in (3, 4):
            for t in itertools.product(AA, repeat=L):
                lp = m.log_pseq("".join(t))
                if not math.isfinite(lp):
                    continue
                w = math.exp(q * lp)
                u = -lp
                wsum += w
                for k in range(5):
                    acc[k] += w * u ** k
        want = acc / wsum
        got = m.surprisal_moments(q=q, order=4)
        for k in range(5):
            assert got[k] == pytest.approx(want[k], rel=1e-8), (q, k)


def test_surprisal_cumulants_match_moments(model):
    """Cumulants derived from the moments by the standard recursion."""
    mu = model.surprisal_moments(q=1.0, order=4)
    k = model.surprisal_cumulants(q=1.0, order=4)
    assert k[0] == pytest.approx(mu[1], rel=1e-9)                      # mean
    assert k[1] == pytest.approx(mu[2] - mu[1] ** 2, rel=1e-8)         # variance
    third = mu[3] - 3 * mu[1] * mu[2] + 2 * mu[1] ** 3
    assert k[2] == pytest.approx(third, rel=1e-7)


def test_surprisal_variance_matches_sampling(model):
    """The exact variance must agree with the sample variance of draws from
    the model itself, which is an independent route to the same number."""
    draws = model.sample(30000, seed=11)
    u = np.array([-model.log_pseq(s) for s in draws])
    stats = model.surprisal_stats(q=1.0)
    assert u.mean() == pytest.approx(stats["mean"], rel=0.02)
    assert u.var() == pytest.approx(stats["variance"], rel=0.15)


def test_cgf_is_the_shifted_mellin(model):
    for t in (-0.5, 0.0, 0.5):
        assert model.cgf(t) == pytest.approx(model.log_mellin(1.0 + t), rel=1e-12)


def test_cgf_derivative_at_zero_is_minus_entropy(model):
    """d/dt ln E[exp(t ln P)] at t=0 is E[ln P] = -H. A finite difference on
    the Mellin transform must reproduce the entropy DP."""
    h = 1e-4
    slope = (model.cgf(h) - model.cgf(-h)) / (2 * h)
    assert slope == pytest.approx(-model.entropy(), rel=1e-4)


# ═══════════════════════════════════════════════════════════════
# Per-length tilted moments
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
def test_length_moments_mix_to_the_pooled_moments(model, q):
    """`mu[k] == sum_L share[L] * moments[L, k]`.

    The identity the unmixed form exists to make checkable: slot 0 carries
    each length's SHARE of the q-tilted law rather than the constant 1 a raw
    jet would put there, precisely so the mixture can be verified from the
    output alone against `surprisal_moments`, which is a separate DP.
    """
    order = 4
    lm = model.length_moments(q=q, order=order)
    mu = model.surprisal_moments(q=q, order=order)
    share, moments = lm["share"], lm["moments"]

    assert share.sum() == pytest.approx(1.0, rel=1e-12)
    for k in range(order + 1):
        assert float((share * moments[:, k]).sum()) == pytest.approx(
            float(mu[k]), rel=1e-9)


def test_length_moments_share_at_q_one_is_the_length_prior(model):
    lm = model.length_moments(q=1.0, order=1)
    pmf = model.length_pmf()
    assert np.allclose(lm["share"][:len(pmf)], pmf, rtol=1e-9, atol=1e-15)


def test_length_moments_conditional_mean_matches_length_marginals(model):
    """A second, independent C route to E[-ln P | L]."""
    lm = model.length_moments(q=1.0, order=1)
    marg = model.length_marginals()
    for L in model.observed_lengths:
        assert lm["moments"][L, 1] == pytest.approx(
            float(marg["mean_surprisal"][L]), rel=1e-9)


def test_length_moments_ungenerated_lengths_are_zero(model):
    lm = model.length_moments(q=1.0, order=2)
    for L in range(len(lm["share"])):
        if L not in model.observed_lengths:
            assert lm["share"][L] == 0.0


def test_length_derivatives_q0_still_is_the_per_length_support(model):
    """The generalised route must not have broken the q=0 corner the capacity
    tiers read."""
    per = model.pseq_analysis().length_derivatives(0.0, 0)
    for L, jet in per.items():
        assert float(jet[0]) == pytest.approx(model.path_count(L), rel=1e-9)


def test_length_derivatives_q1_still_decomposes_the_entropy(model):
    per = model.pseq_analysis().length_derivatives(1.0, 2)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(1.0,
                                                                  rel=1e-12)
    assert -sum(float(j[1]) for j in per.values()) == pytest.approx(
        model.entropy(), rel=1e-9)


# ════════════════════════════════════════════════════════════════════════
