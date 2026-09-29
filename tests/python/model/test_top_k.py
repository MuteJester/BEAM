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

# A search over every length runs one best-first search per length, and under
# the prior every length up to max_length holds sequences; the pooled searches
# therefore use `small` (max_length 10) rather than `model` (27).

def test_top_k_is_sorted_and_scores_correctly(small):
    top = small.top_k_sequences(25)
    assert len(top) == 25
    lps = [lp for _, lp in top]
    assert lps == sorted(lps, reverse=True)
    for seq, lp in top:
        assert small.log_pseq(seq) == pytest.approx(lp, abs=1e-9)


def test_top_k_is_exact_against_exhaustive_enumeration():
    """The claim is that this is exact, not a beam, so it is checked against a
    full enumeration of the whole support: every string of length 1 to 3,
    8,420 of them. A beam would silently miss sequences whose prefix looked
    unpromising, and a search confined to the corpus's lengths would miss the
    unseen lengths the prior ranks among them."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=2, hi=3)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=3)

    strings = ["".join(t) for L in (1, 2, 3)
               for t in itertools.product(AA, repeat=L)]
    scores = m.log_pseq_batch(strings)
    assert np.isfinite(scores).all()
    every = sorted(zip(strings, scores.tolist()), key=lambda kv: -kv[1])

    got = m.top_k_sequences(30)
    assert len(got) == 30
    assert [lp for _, lp in got] == pytest.approx(
        [lp for _, lp in every[:30]], abs=1e-9)
    # Exact ties are real under the prior -- two residues a row never saw are
    # priced identically -- so which of a tied pair is listed first is the
    # search's choice. What is not: every sequence scoring strictly above the
    # 30th is returned, and every one returned scores what it says.
    cutoff = every[29][1]
    must = {s for s, lp in every if lp > cutoff + 1e-9}
    assert must <= {s for s, _ in got}
    assert len({s for s, _ in got}) == 30
    for s, lp in got:
        assert m.log_pseq(s) == pytest.approx(lp, abs=1e-9)


def test_top_k_at_a_fixed_length(model):
    top = model.top_k_sequences(10, length=13)
    assert len(top) == 10
    assert all(len(s) == 13 for s, _ in top)


def test_top_k_beats_random_draws(small):
    """The best sequence must be at least as probable as anything sampled."""
    (_, best), = small.top_k_sequences(1)
    draws = small.sample(2000, seed=3)
    assert best >= max(small.log_pseq(s) for s in draws) - 1e-9


def test_top_k_more_than_exists():
    """Asking for more than the model can produce returns what there is.

    With every table cell positive the support is every string up to
    max_length, so the only way to have fewer than 500 is a genuinely short
    model: max_length 2 gives exactly 20 + 400 = 420.
    """
    m = BeamModel(["CA", "CF", "SA"], order=2, max_length=2)
    top = m.top_k_sequences(500)
    assert len(top) == 420, "20 + 20**2 sequences exist and all should come back"
    assert all(math.isfinite(lp) for _, lp in top)
    assert len({s for s, _ in top}) == 420, "and they must be distinct"
    assert math.fsum(math.exp(lp) for _, lp in top) == pytest.approx(
        1.0, abs=1e-12), "and together they are all the mass there is"


# ═══════════════════════════════════════════════════════════════
