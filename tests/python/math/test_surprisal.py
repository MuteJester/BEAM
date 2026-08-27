"""Surprisal range and depth attribution.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Surprisal range
# ═══════════════════════════════════════════════════════════════

def test_surprisal_range_brackets_every_sequence():
    """Exhaustive: no sequence may fall outside the reported range, and both
    ends must be attained."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    lo, hi = m.surprisal_range()

    seen_lo, seen_hi = math.inf, -math.inf
    for L in (3, 4):
        for t in itertools.product(AA, repeat=L):
            lp = m.log_pseq("".join(t))
            if not math.isfinite(lp):
                continue
            u = -lp
            seen_lo = min(seen_lo, u)
            seen_hi = max(seen_hi, u)
    assert lo == pytest.approx(seen_lo, rel=1e-9)
    assert hi == pytest.approx(seen_hi, rel=1e-9)


def test_dynamic_range_is_positive(model):
    lo, hi = model.surprisal_range()
    assert hi > lo > 0
    assert model.dynamic_range() == pytest.approx((hi - lo) / math.log(10), rel=1e-12)


def test_top_sequence_attains_the_minimum_surprisal(model):
    """The most probable sequence must sit exactly at the low end of the range,
    which links the top-k search and the min-plus sweep."""
    lo, _ = model.surprisal_range()
    (_, best_lp), = model.top_k_sequences(1)
    assert -best_lp == pytest.approx(lo, rel=1e-9)


# ═══════════════════════════════════════════════════════════════
# Surprisal by depth
# ═══════════════════════════════════════════════════════════════

def test_surprisal_by_depth_reconstructs_the_mean(model):
    """The decomposition must add up EXACTLY to the mean surprisal.

    This is the whole point of reporting a breakdown: if the layer terms, the
    sink term and the per-length normalisation did not sum to the entropy,
    something would be lost or double-counted and the picture would mislead.
    """
    d = model.surprisal_by_depth(q=1.0)
    # rel 1e-9, not the 1e-6 this used to allow: the forward-backward DP is
    # exact to ~1e-13, so a loose tolerance here would hide a real regression.
    assert d["total"] == pytest.approx(model.entropy(), rel=1e-9)
    # `total` is defined in Python as this sum, so on its own the identity
    # below is tautological; it is kept as a guard on the packing. The check
    # with teeth is that the same identity holds under a TILT, where the
    # length weights, the posterior and the constant all change together.
    assert d["by_depth"].sum() + d["length_constant"] == pytest.approx(
        d["total"], rel=1e-12)
    tilted = model.surprisal_by_depth(q=2.0)
    want = float(model.surprisal_moments(q=2.0, order=1)[1])
    assert tilted["total"] == pytest.approx(want, rel=1e-9)
    assert tilted["total"] != pytest.approx(d["total"], rel=1e-3), (
        "q=2 must actually tilt; if it did not, the q argument is ignored")


def test_surprisal_by_depth_finds_the_junction(model):
    """The conserved start costs almost nothing and the middle carries the
    information. If the two anchors were confused this would invert."""
    d = model.surprisal_by_depth(q=1.0)["by_depth"]
    assert d[0] < 0.05, "position 0 is conserved and must be nearly free"
    assert d[5:9].mean() > d[0] * 5, "the junction must dominate"


# ═══════════════════════════════════════════════════════════════
