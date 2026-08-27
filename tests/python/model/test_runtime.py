"""Thread controls and parallel determinism.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, ANALYSIS_BINS)

# Parallelism
#
# The cap is process-wide, so every test here restores it. The property that
# makes it safe to use is that results do not depend on it: every reduction
# is order-independent or summed in a fixed index order.
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def restore_threads():
    from BEAM import threads, set_threads
    before = threads()
    yield
    set_threads(before)


def test_thread_cap_round_trips(model, restore_threads):
    from BEAM import effective_threads, threads, set_threads
    set_threads(2)
    assert threads() == 2
    assert model.threads == 2
    assert 1 <= effective_threads() <= 2
    assert model.effective_threads == effective_threads()

    model.threads = 1
    assert threads() == 1
    assert model.effective_threads == 1

    set_threads(0)
    assert threads() == 0
    assert model.effective_threads >= 1


def test_thread_cap_treats_negative_as_default(model, restore_threads):
    from BEAM import threads
    model.threads = -4
    assert threads() == 0


def test_results_do_not_depend_on_the_thread_count(model, corpus,
                                                   restore_threads):
    """The whole point of a cap being safe to set: pinning it for a benchmark
    must change the timing and not the answer. Both the batch scorer and an
    analytic that reduces over the trellis are checked."""
    seqs, _ = corpus
    probe = seqs[:300]

    model.threads = 1
    one_scores = model.log_pseq_batch(probe)
    one_entropy = model.entropy()
    one_mellin = model.log_mellin(2.0)

    model.threads = 4
    assert np.array_equal(model.log_pseq_batch(probe), one_scores)
    assert model.entropy() == one_entropy
    assert model.log_mellin(2.0) == one_mellin


def test_atoms_feed_the_publicness_model(small):
    """The atom pair is what a sharing prediction consumes, so it has to be
    accepted by `PublicnessModel.expected_counts` as it comes off the
    spectrum — no reshaping, no dtype surgery at the call site."""
    from BEAM import PublicnessModel

    p, c = small.pseq_analysis().atoms(ANALYSIS_BINS).as_arrays()
    depths = np.arange(100, 400, dtype=np.int64)
    pb = PublicnessModel(depths, depth_bins=8)
    out = pb.expected_counts(p, c, levels=np.array([1, 5, len(depths) + 1]))
    counts = np.asarray(out["expected_counts"], dtype=np.float64)
    assert counts.size == 2
    assert np.all(np.isfinite(counts)) and np.all(counts >= 0)


# ════════════════════════════════════════════════════════════════════════
