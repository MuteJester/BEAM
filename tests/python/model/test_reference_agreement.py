"""Agreement with the independent reference implementation.
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

# ── Agreement with the independent reference ─────────────────────

@pytest.mark.parametrize("order", [2, 3, 4])
def test_scores_match_reference(corpus, order):
    """The whole pipeline — counting, hierarchical backoff, mixing, the
    partition function — reproduced from the specification independently."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=order, strict=False)
    probe = seqs[:40] + ["CASSLGQGAEQYF", "CASSLGQGAEQYW"]
    probe = [s for s in probe if 11 <= len(s) <= 18]

    ref = reference_build(seqs, abund, order=order)
    want = reference_scores(ref, probe)
    got = np.array([m.log_pseq(s) for s in probe])

    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-9, (
        f"order {order}: worst deviation {np.abs(got - want).max():.3e}")


def test_partition_matches_reference(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=3, strict=False)
    ref = reference_build(seqs, abund, order=3)
    for length in range(11, 19):
        assert abs(m.log_partition(length) - float(ref.logZ(length))) < 1e-9


def test_path_count_matches_reference(corpus):
    """Support size over the trellis, against the reference DP."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2, strict=False)
    ref = reference_build(seqs, abund, order=2)
    import beam_model as R
    want = R.reference_path_counts(ref, range(11, 19))
    for length in range(11, 19):
        assert m.path_count(length) == pytest.approx(want[length], rel=1e-12)
