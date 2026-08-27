"""Generation and the simulation container.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Generation ───────────────────────────────────────────────────

def test_sample_is_reproducible(model):
    assert model.sample(20, seed=7) == model.sample(20, seed=7)
    assert model.sample(20, seed=7) != model.sample(20, seed=8)


def test_sample_respects_the_model(model):
    """Draws must be scoreable and follow the length prior."""
    draws = model.sample(400, seed=3)
    assert all(math.isfinite(model.log_pseq(s)) for s in draws)
    pmf = model.length_pmf()
    for s in draws:
        assert pmf[len(s)] > 0


def test_sample_fixed_length(model):
    draws = model.sample(50, length=13, seed=2)
    assert all(len(s) == 13 for s in draws)


def test_sample_frequencies_match_probabilities():
    """Exact trellis sampling, so empirical frequencies must converge to the
    model's own probabilities."""
    seqs, abund = make_corpus(n=300, seed=4, lo=4, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    draws = m.sample(60_000, length=4, seed=5)
    from collections import Counter
    obs = Counter(draws)
    top = [s for s, _ in obs.most_common(10)]
    for s in top:
        want = math.exp(m.log_pseq(s) - m.log_length_prior(4))
        got = obs[s] / len(draws)
        assert got == pytest.approx(want, rel=0.15), (s, got, want)


# Simulation container
# ═══════════════════════════════════════════════════════════════

def test_simulate_returns_a_simulation_result(model):
    from BEAM._simulation_result import SimulationResult
    sim = model.simulate(50, seed=4)
    assert isinstance(sim, SimulationResult)
    assert len(sim.sequences) == 50
    assert len(sim.log_probs) == 50
    for s, lp, nt in zip(sim.sequences, sim.log_probs, sim.n_tokens):
        assert model.log_pseq(s) == pytest.approx(lp, abs=1e-12)
        assert nt == len(s)


def test_simulate_is_reproducible(model):
    a = model.simulate(20, seed=9)
    b = model.simulate(20, seed=9)
    assert a.sequences == b.sequences


# ═══════════════════════════════════════════════════════════════
