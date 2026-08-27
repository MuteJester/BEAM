"""Per-length marginals.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Per-length marginals
# ═══════════════════════════════════════════════════════════════

def test_length_marginals_decompose_the_totals(model):
    lm = model.length_marginals()
    assert lm["counting"].sum() == pytest.approx(model.path_count(), rel=1e-9)
    assert lm["generated"].sum() == pytest.approx(1.0, rel=1e-12)


def test_length_marginals_generated_is_the_length_prior(model):
    lm = model.length_marginals()
    assert np.abs(lm["generated"] - model.length_pmf()).max() < 1e-12


def test_length_mean_surprisal_reconstructs_the_entropy(model):
    """The entropy is the length-weighted mean of the per-length means. If the
    per-length numbers were wrong this identity would not hold."""
    lm = model.length_marginals()
    h = float((lm["generated"] * lm["mean_surprisal"]).sum())
    assert h == pytest.approx(model.entropy(), rel=1e-9)


def test_length_mean_surprisal_matches_sampling(model):
    """Draw at a fixed length and compare the empirical mean surprisal."""
    lm = model.length_marginals()
    L = int(np.argmax(lm["generated"]))
    draws = model.sample(20000, length=L, seed=5)
    u = np.array([-model.log_pseq(s) for s in draws])
    assert u.mean() == pytest.approx(lm["mean_surprisal"][L], rel=0.02)


# ═══════════════════════════════════════════════════════════════
