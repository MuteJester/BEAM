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
    """Exhaustive: every sequence the model can produce, weighted by its own
    probability. This pins the jet DP against a direct sum.

    Every string over the whole alphabet up to max_length is possible, so the
    space is nucleotide and short (19,530 strings), and the high moments feel
    the improbable tail as well as the corpus-like bulk."""
    import itertools
    rng = random.Random(5)
    seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
            + "C" for _ in range(400)]
    abund = [rng.randint(1, 40) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=3, max_length=6)
    every = ["".join(t) for L in range(1, 7)
             for t in itertools.product("ACGTN", repeat=L)]
    lp = np.asarray(m.log_pseq_batch(every), dtype=np.float64)
    assert np.isfinite(lp).all(), "the prior leaves no string impossible"

    for q in (1.0, 2.0):
        w = np.exp(q * lp)
        u = -lp
        want = np.array([(w * u ** k).sum() for k in range(5)]) / w.sum()
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


def test_length_moments_share_is_zero_only_at_length_zero(model):
    """Every length 1..max_length is generated -- the length prior gives even
    the ones the corpus never had a pseudo-count -- so slot 0 is the only
    empty one, and it must be exactly empty rather than merely small."""
    lm = model.length_moments(q=1.0, order=2)
    share = lm["share"]
    assert len(share) == int(model.config["max_length"]) + 1
    assert share[0] == 0.0
    assert (share[1:] > 0.0).all()
    assert tuple(model.observed_lengths) == tuple(range(1, len(share)))


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
