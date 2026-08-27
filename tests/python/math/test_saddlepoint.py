"""The length mixture and the saddlepoint inversion.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Length mixture and saddlepoint
# ═══════════════════════════════════════════════════════════════

def test_surprisal_distribution_components(small):
    d = small.surprisal_distribution()
    assert d["weight"].sum() == pytest.approx(1.0, rel=1e-12)
    # The mixture mean must be the entropy.
    assert float((d["weight"] * d["mean"]).sum()) == pytest.approx(
        small.entropy(), rel=1e-9)
    # Law of total variance against the exact global variance.
    var = small.surprisal_stats()["variance"]
    within = float((d["weight"] * d["variance"]).sum())
    between = float((d["weight"] * (d["mean"] - small.entropy()) ** 2).sum())
    assert within + between == pytest.approx(var, rel=1e-6)


def test_surprisal_cdf_is_a_distribution(small):
    lo, hi = small.surprisal_range()
    xs = np.linspace(lo, hi, 40)
    c = small.surprisal_cdf(xs)
    assert np.all(np.diff(c) >= -1e-12), "must be non-decreasing"
    assert 0.0 <= c[0] <= 0.2
    assert c[-1] > 0.95


def test_saddlepoint_reports_non_convergence(small):
    """A point beyond the reachable tilt must be flagged, not answered from a
    tilt that does not solve the saddlepoint equation. Silently returning a
    confident wrong number is the failure mode this guards."""
    lo, hi = small.surprisal_range()
    deep = small.saddlepoint(lo + 1e-6)
    assert deep["converged"] is False, "the flag is the point of this test"
    assert deep["cdf"] == pytest.approx(0.0, abs=1e-9)
    assert deep["pdf"] == pytest.approx(0.0, abs=1e-9)
    far = small.saddlepoint(hi - 1e-6)
    assert far["converged"] is False
    assert far["cdf"] == pytest.approx(1.0, abs=1e-9)
    # and inside the bulk it must converge, and be flagged as having done so
    mu = float(small.surprisal_moments(order=1)[1])
    bulk = small.saddlepoint(mu)
    assert bulk["converged"] is True
    assert 0.0 < bulk["cdf"] < 1.0 and bulk["pdf"] > 0.0


def test_saddlepoint_solves_the_right_equation(small):
    """At the solved t, K'(t) must equal the requested surprisal. This is the
    defining property, and it uses the exact CGF derivatives."""
    mu = float(small.surprisal_moments(order=1)[1])
    for x in (mu - 0.5, mu + 0.8):
        sp = small.saddlepoint(x)
        assert sp["converged"]
        jet = small.cgf_jet(sp["t"])
        assert jet["K1"] == pytest.approx(x, abs=1e-6)


def test_saddlepoint_cdf_is_monotone_and_bracketed(small):
    """Sampled across the BULK of the distribution, not at its extremes: a
    saddlepoint arbitrarily close to the minimum surprisal needs an unbounded
    tilt, and the solve is deliberately clamped rather than allowed to run
    into a degenerate sweep."""
    mu = float(small.surprisal_moments(order=1)[1])
    sd = math.sqrt(small.surprisal_stats()["variance"])
    xs = np.linspace(mu - 1.5 * sd, mu + 1.5 * sd, 7)
    sp = small.saddlepoint(xs)
    assert np.all(np.diff(sp["cdf"]) >= -1e-9)
    assert np.all((sp["cdf"] >= 0) & (sp["cdf"] <= 1))
    assert np.all(sp["pdf"] >= 0)


def test_saddlepoint_is_exactly_zero_outside_the_support(small):
    """Outside the range no sequence exists, so the answer is exact, not an
    extrapolation."""
    lo, hi = small.surprisal_range()
    below = small.saddlepoint(lo - 1.0)
    above = small.saddlepoint(hi + 1.0)
    assert below["cdf"] == 0.0 and below["pdf"] == 0.0
    assert above["cdf"] == 1.0


def test_saddlepoint_agrees_with_sampling(small):
    """The approximation is only in the inversion, so it should track the
    empirical distribution of draws from the model closely."""
    draws = small.sample(20000, seed=17)
    u = np.sort(np.array([-small.log_pseq(s) for s in draws]))
    for frac in (0.25, 0.5, 0.75):
        x = float(np.quantile(u, frac))
        got = small.saddlepoint(x)["cdf"]
        assert got == pytest.approx(frac, abs=0.06), (frac, got)


def test_cgf_jet_matches_the_separate_routes(small):
    """One sweep must give the same K, K' and K'' as three separate calls."""
    for t in (-0.3, 0.0, 0.4):
        jet = small.cgf_jet(t)
        q = 1.0 - t
        assert jet["K"] == pytest.approx(small.log_mellin(q), rel=1e-9)
        k = small.surprisal_cumulants(q=q, order=2)
        assert jet["K1"] == pytest.approx(k[0], rel=1e-9)
        assert jet["K2"] == pytest.approx(k[1], rel=1e-6)


# ═══════════════════════════════════════════════════════════════
