"""Model arithmetic: add, subtract, and their guards.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Counts are additive ──────────────────────────────────────────

def test_subtract_is_additive_inverse(corpus):
    """Holding a cohort out of a foundation model must be a subtraction, not a
    recount of the corpus. That only works if counts are additive."""
    seqs, abund = corpus
    a_seqs, a_ab = seqs[:1000], abund[:1000]
    b_seqs, b_ab = seqs[1000:], abund[1000:]

    full = BeamModel(seqs, abundances=abund, order=2)
    part = BeamModel(b_seqs, abundances=b_ab, order=2)
    only_a = BeamModel(a_seqs, abundances=a_ab, order=2)

    full.subtract(part)
    probe = [s for s in a_seqs[:50]]
    got = np.array([full.log_pseq(s) for s in probe])
    want = np.array([only_a.log_pseq(s) for s in probe])
    assert np.abs(got - want).max() < 1e-9


# Count addition
# ═══════════════════════════════════════════════════════════════

def test_add_is_the_inverse_of_subtract(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:800], abundances=abund[:800], order=2)
    b = BeamModel(seqs[800:], abundances=abund[800:], order=2)
    whole = BeamModel(seqs, abundances=abund, order=2)

    a.add(b)
    probe = seqs[:60]
    assert np.abs(np.array([a.log_pseq(s) for s in probe])
                  - np.array([whole.log_pseq(s) for s in probe])).max() < 1e-9

    a.subtract(b)
    only_a = BeamModel(seqs[:800], abundances=abund[:800], order=2)
    assert np.abs(np.array([a.log_pseq(s) for s in probe])
                  - np.array([only_a.log_pseq(s) for s in probe])).max() < 1e-9


def test_add_accumulates_bookkeeping(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:500], abundances=abund[:500], order=2)
    b = BeamModel(seqs[500:900], abundances=abund[500:900], order=2)
    wa, na = a.total_weight, a.n_observed
    a.add(b)
    assert a.n_observed == na + b.n_observed
    assert a.total_weight == pytest.approx(wa + b.total_weight)


# ═══════════════════════════════════════════════════════════════
# Set operations: guards and bookkeeping
# ═══════════════════════════════════════════════════════════════

def test_operand_must_be_a_distinct_edge(corpus):
    """`add`/`subtract` alias source and destination in C, so passing the
    model itself doubles or zeroes its own counts. Subtracting a model from
    itself additionally empties the tables, at which point the re-finalize
    fails and there is nothing left to roll back to."""
    seqs, abund = corpus
    m = BeamModel(seqs[:200], abundances=abund[:200], order=2)
    with pytest.raises(ValueError, match="itself"):
        m.add(m)
    with pytest.raises(ValueError, match="itself"):
        m.subtract(m)
    with pytest.raises(TypeError):
        m.add("not a model")
    # and the guard must not have disturbed the model
    assert math.isfinite(m.log_pseq(seqs[0]))


def test_compact_disables_the_count_operations(corpus):
    """`compact` releases the counts, so everything that reads them has to
    say so. `n_nonzero_cells` reported 0, which is indistinguishable from a
    genuinely empty model."""
    seqs, abund = corpus
    m = BeamModel(seqs[:200], abundances=abund[:200], order=2)
    other = BeamModel(seqs[200:300], abundances=abund[200:300], order=2)
    assert m.n_nonzero_cells > 0
    m.compact()
    with pytest.raises(RuntimeError, match="compact"):
        m.n_nonzero_cells
    with pytest.raises(RuntimeError, match="compact"):
        m.add(other)
    with pytest.raises(RuntimeError, match="compact"):
        m.subtract(other)


def test_subtract_restores_the_total_weight(corpus):
    """`total_weight` is the bookkeeping number that survives a subtraction
    correctly, and the one the docstring points callers at."""
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2)
    b = BeamModel(seqs[400:700], abundances=abund[400:700], order=2)
    w0 = a.total_weight
    a.add(b)
    assert a.total_weight == pytest.approx(w0 + b.total_weight)
    a.subtract(b)
    assert a.total_weight == pytest.approx(w0)


def test_subtract_restores_the_sequence_count(corpus):
    """`add` and `subtract` must be exact inverses on the BOOKKEEPING too, not
    just on the counts.

    They were not: `add` adjusted `n_observed` and `n_skipped` while
    `subtract` adjusted only `total_weight`, so every leave-one-cohort-out
    workflow — the thing `subtract` exists for — reported a corpus size that
    was too large. The two functions live in different files, which is how it
    went unnoticed.
    """
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2)
    b = BeamModel(seqs[400:700], abundances=abund[400:700], order=2)
    n0 = a.n_observed
    a.add(b)
    assert a.n_observed == n0 + b.n_observed
    a.subtract(b)
    assert a.n_observed == n0


# ════════════════════════════════════════════════════════════════════════
# Non-destructive set operations
# ════════════════════════════════════════════════════════════════════════


def test_copy_is_independent(corpus):
    """`subtract` is destructive and the counts are the only copy there is, so
    the leave-one-out workflow needs a real clone, not a second reference."""
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    hold = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    before = base.log_pseq_batch(seqs[:50])

    clone = base.copy()
    assert np.array_equal(clone.log_pseq_batch(seqs[:50]), before)
    clone.subtract(hold)
    assert not np.array_equal(clone.log_pseq_batch(seqs[:50]), before)
    assert np.array_equal(base.log_pseq_batch(seqs[:50]), before)


def test_without_a_model_equals_an_explicit_subtract(corpus):
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    hold = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)

    held_out = base.without(hold)
    manual = base.copy()
    manual.subtract(hold)
    assert np.array_equal(held_out.log_pseq_batch(seqs[:50]),
                          manual.log_pseq_batch(seqs[:50]))
    assert base.total_weight == pytest.approx(float(sum(abund[:800])))


def test_without_sequences_builds_the_matching_operand(corpus):
    seqs, abund = corpus
    base = BeamModel(seqs[:800], abundances=abund[:800], order=2, max_length=20)
    a = base.without(seqs[:200], abundances=abund[:200])
    b = base.without(BeamModel(seqs[:200], abundances=abund[:200], order=2,
                          max_length=20))
    assert np.array_equal(a.log_pseq_batch(seqs[:50]),
                          b.log_pseq_batch(seqs[:50]))
    assert a.total_weight == pytest.approx(
        base.total_weight - float(sum(abund[:200])))


def test_merged_does_not_mutate_either_operand(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2, max_length=20)
    b = BeamModel(seqs[400:800], abundances=abund[400:800], order=2, max_length=20)
    wa, wb = a.total_weight, b.total_weight

    merged = a.merged(b)
    assert merged.total_weight == pytest.approx(wa + wb)
    assert a.total_weight == wa
    assert b.total_weight == wb

    inplace = a.copy()
    inplace.add(b)
    assert np.array_equal(merged.log_pseq_batch(seqs[:50]),
                          inplace.log_pseq_batch(seqs[:50]))


def test_without_rejects_abundances_for_a_model_operand(corpus):
    seqs, abund = corpus
    a = BeamModel(seqs[:400], abundances=abund[:400], order=2, max_length=20)
    b = BeamModel(seqs[:100], abundances=abund[:100], order=2, max_length=20)
    with pytest.raises(TypeError, match="already carries its own weights"):
        a.without(b, abundances=[1.0])


# ════════════════════════════════════════════════════════════════════════
