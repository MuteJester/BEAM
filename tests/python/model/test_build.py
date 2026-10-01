"""Degenerate corpora.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel
from beam_model import reference_build, reference_scores

from _helpers import AA, J_MOTIFS, make_corpus

# Degenerate corpora
#
# Every one of these builds a model whose tables are almost entirely prior,
# which is exactly where an off-by-one in the smoothing or the length prior
# stops being visible in the aggregate numbers. The check is always the same
# and always exhaustive: sum P(s|L) over every sequence of that length must
# be 1.
# ═══════════════════════════════════════════════════════════════

def _length_rule(counts, max_length, length, alpha_length):
    """pi(L) = (n_L + alpha_length / max_length) / (N + alpha_length), with the
    strength the model estimated (its estimate is checked on its own in
    test_hierarchical_prior; this checks the rule applied to it)."""
    n = sum(counts.values())
    return ((counts.get(length, 0) + alpha_length / max_length)
            / (n + alpha_length))


def _sum_conditional(m, length):
    """Exhaustive sum of P(s | L) over every length-`L` peptide."""
    import itertools
    every = ["".join(t) for t in itertools.product(AA, repeat=length)]
    lp = m.log_pseq_batch(every) - m.log_length_prior(length)
    assert np.isfinite(lp).all(), "the prior leaves no sequence impossible"
    return float(np.exp(lp).sum())


def test_single_training_sequence_normalises():
    """One sequence, so every count is 1 and every table row is dominated by
    its prior. The normalisation must still be exact."""
    m = BeamModel(["CASF"], order=2, max_length=6)
    assert m.n_observed == 1 and m.n_skipped == 0
    assert m.length_pmf()[4] == pytest.approx(
        _length_rule({4: 1}, 6, 4, m.config["alpha_length"]), rel=1e-12)
    assert _sum_conditional(m, 4) == pytest.approx(1.0, abs=1e-9)
    assert math.isfinite(m.log_pseq("CASF"))
    assert math.isfinite(m.log_pseq("WWWW")), "the prior leaves nothing dead"
    assert m.log_pseq("CASF") > m.log_pseq("WWWW")


def test_all_identical_sequences_normalise():
    """A degenerate empirical distribution: one sequence repeated. The counts
    concentrate on a single path and everything else lives on the prior."""
    m = BeamModel(["CASF"] * 50, order=2, max_length=6)
    assert m.n_observed == 50
    assert _sum_conditional(m, 4) == pytest.approx(1.0, abs=1e-9)
    # The observed sequence must still be the most probable one.
    (best, _), = m.top_k_sequences(1)
    assert best == "CASF"


def test_length_one_sequences_normalise():
    """Length 1 is the shortest thing the model can represent, and the one
    where the backward expert's lag never gets a chance to settle during the
    walk: every residue's backward score is paid at the sink instead."""
    m = BeamModel(["C", "A", "S", "C"], order=2, max_length=4)
    assert m.length_pmf()[1] == pytest.approx(
        _length_rule({1: 4}, 4, 1, m.config["alpha_length"]), rel=1e-12)
    assert _sum_conditional(m, 1) == pytest.approx(1.0, abs=1e-12)
    assert m.path_count(1) == pytest.approx(20.0)
    top = m.top_k_sequences(20)
    assert len(top) == 20 and len({s for s, _ in top}) == 20
    assert all(len(s) == 1 for s, _ in top)


def test_one_length_carries_a_single_sequence():
    """A length seen exactly once, alongside a length seen many times. Its
    prior is small but non-zero, so it must have a partition function, appear
    in the marginals, and normalise like any other."""
    m = BeamModel(["CAF"] * 60 + ["CASF"], order=2, max_length=5)
    pmf = m.length_pmf()
    assert pmf[4] == pytest.approx(_length_rule({3: 60, 4: 1}, 5, 4,
                                                m.config["alpha_length"]),
                                   rel=1e-12)
    assert math.isfinite(m.log_partition(4))
    assert _sum_conditional(m, 4) == pytest.approx(1.0, abs=1e-9)
    lm = m.length_marginals()
    assert lm["generated"][4] == pytest.approx(pmf[4], rel=1e-12)
    assert lm["counting"][4] == pytest.approx(20.0 ** 4)


def test_max_length_two_is_the_smallest_model():
    """Two is the floor `max_length` accepts. It has to work, not merely be
    accepted: the depth axis is one slot per position, so a two-residue model
    exercises both an anchored and a settled position with nothing in
    between."""
    m = BeamModel(["CA", "CF", "SA"], order=2, max_length=2)
    assert m.config["max_length"] == 2
    assert _sum_conditional(m, 2) == pytest.approx(1.0, abs=1e-12)
    assert m.path_count(2) == pytest.approx(400.0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_degenerate_corpus_matches_the_reference(order):
    """The independent reference, on a corpus small enough that almost every
    table cell is the prior and nothing else. The agreement tests elsewhere
    all use a corpus with hundreds of sequences, where a normalisation bug is
    diluted."""
    seqs = ["CASF", "CASSF", "CASF", "CAF"]
    m = BeamModel(seqs, order=order, max_length=6)
    ref = reference_build(seqs, order=order, max_len=6,
                          alpha=m.config["alpha"],
                          alpha_length=m.config["alpha_length"])
    # Junk, and two lengths the corpus never had, which only the prior prices.
    probe = ["CASF", "CASSF", "CAF", "WWWW", "YYYYY", "W", "CASSQF"]
    got = np.array([m.log_pseq(s) for s in probe])
    want = reference_scores(ref, probe)
    assert np.abs(got - want).max() < 1e-9, (order, got, want)


# ═══════════════════════════════════════════════════════════════
