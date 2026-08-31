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

from _helpers import AA, J_MOTIFS, make_corpus, make_nt_corpus

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


# ── The same agreement, over the nucleotide alphabet ─────────────
#
# The reference is built from the specification with no code shared with
# src/native/, so these check the radix-6 packing, backoff, mixing and
# partition function against an independent implementation rather than
# against the C's own opinion of itself. Nothing else in the suite can catch a
# dimension that is consistently wrong in both places.


@pytest.fixture(scope="module")
def nt_corpus():
    return make_nt_corpus()


@pytest.mark.parametrize("order", [2, 3, 4])
def test_nucleotide_scores_match_reference(nt_corpus, order):
    seqs, abund = nt_corpus
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=order,
                  max_length=36, depth_cap=8, strict=False)
    probe = [s for s in seqs[:40] if len(s) <= 36]

    ref = reference_build(seqs, abund, order=order, max_len=36, depth_cap=8,
                          alphabet="nt")
    want = reference_scores(ref, probe)
    got = np.array([m.log_pseq(s) for s in probe])

    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-9, (
        f"order {order}: worst deviation {np.abs(got - want).max():.3e}")


def test_nucleotide_partition_matches_reference(nt_corpus):
    seqs, abund = nt_corpus
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=3,
                  max_length=36, depth_cap=8, strict=False)
    ref = reference_build(seqs, abund, order=3, max_len=36, depth_cap=8,
                          alphabet="nt")
    for length in range(21, 26):
        assert abs(m.log_partition(length) - float(ref.logZ(length))) < 1e-9


def test_nucleotide_path_count_matches_reference(nt_corpus):
    """Support size over the radix-6 trellis, against the reference DP."""
    seqs, abund = nt_corpus
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=2,
                  max_length=36, depth_cap=8, strict=False)
    ref = reference_build(seqs, abund, order=2, max_len=36, depth_cap=8,
                          alphabet="nt")
    import beam_model as R
    want = R.reference_path_counts(ref, range(21, 26))
    for length in range(21, 26):
        assert m.path_count(length) == pytest.approx(want[length], rel=1e-12)
