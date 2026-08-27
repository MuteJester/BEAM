"""Exact best-first top-k.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Top-k
# ═══════════════════════════════════════════════════════════════

def test_top_k_is_sorted_and_scores_correctly(model):
    top = model.top_k_sequences(25)
    assert len(top) == 25
    lps = [lp for _, lp in top]
    assert lps == sorted(lps, reverse=True)
    for seq, lp in top:
        assert model.log_pseq(seq) == pytest.approx(lp, abs=1e-9)


def test_top_k_is_exact_against_exhaustive_enumeration():
    """The claim is that this is exact, not a beam, so it is checked against a
    full enumeration of a short length. A beam would silently miss sequences
    whose prefix looked unpromising."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)

    every = []
    for L in (3, 4):
        for t in itertools.product(AA, repeat=L):
            s = "".join(t)
            lp = m.log_pseq(s)
            if math.isfinite(lp):
                every.append((s, lp))
    every.sort(key=lambda kv: -kv[1])

    got = m.top_k_sequences(30)
    assert len(got) == 30
    for (gs, glp), (ws, wlp) in zip(got, every[:30]):
        assert glp == pytest.approx(wlp, abs=1e-9)
        assert gs == ws


def test_top_k_at_a_fixed_length(model):
    top = model.top_k_sequences(10, length=13)
    assert len(top) == 10
    assert all(len(s) == 13 for s, _ in top)


def test_top_k_beats_random_draws(model):
    """The best sequence must be at least as probable as anything sampled."""
    (_, best), = model.top_k_sequences(1)
    draws = model.sample(2000, seed=3)
    assert best >= max(model.log_pseq(s) for s in draws) - 1e-9


def test_top_k_more_than_exists():
    """Asking for more than the model can produce returns what there is.

    With every table cell positive the support is 20**L, so the only way to
    have fewer than 500 sequences is a genuinely short model: length 2 gives
    exactly 400.
    """
    m = BeamModel(["CA", "CF", "SA"], order=2, max_length=3, strict=False)
    top = m.top_k_sequences(500)
    assert len(top) == 400, "20**2 sequences exist and all should come back"
    assert all(math.isfinite(lp) for _, lp in top)
    assert len({s for s, _ in top}) == 400, "and they must be distinct"


# ═══════════════════════════════════════════════════════════════
