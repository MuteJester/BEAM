"""Agreement with the independent reference implementation.

`tests/reference/beam_model.py` is written from the specification with no code
shared with src/native/: counting, the hierarchical prior, both anchors, the
length prior and the partition function. Scores, Z(L) and path counts are
compared against it over seen sequences, junk, and lengths the corpus never
had. The prior prices every one of those, so every one of them is checked.
"""
from __future__ import annotations

import numpy as np
import pytest

import beam_model as R
from BEAM import BeamModel
from beam_model import reference_build, reference_scores

from _helpers import make_corpus, make_nt_corpus

# ── Agreement with the independent reference ─────────────────────


@pytest.fixture(scope="module")
def aa(corpus):
    """`order -> (sequences, model, reference)`, each built once.

    The reference is plain numpy over a dense trellis, and it sweeps every
    length up to max_length while it builds. That is seconds at orders 2 and
    3 on the shared corpus and minutes at order 4, so order 4 gets a corpus
    of short sequences instead: the arithmetic under test does not depend on
    how long the sequences are, only on every table level being exercised.
    """
    built = {}

    def get(order):
        if order not in built:
            if order == 4:
                seqs, abund = make_corpus(n=600, seed=31, lo=5, hi=7)
                max_length = 7
            else:
                seqs, abund = corpus                # lengths 11..18
                max_length = 20
            m = BeamModel(seqs, abundances=abund, order=order,
                          max_length=max_length)
            ref = reference_build(seqs, abund, order=order, max_len=max_length)
            built[order] = (seqs, m, ref)
        return built[order]
    return get


def _aa_probe(seqs, max_length):
    """Training sequences, a near miss, junk, and unseen lengths."""
    extra = ["CASSLGQGAEQYW", "WWWWW", "C", "CA", "Y" * max_length,
             "CASS" + "W" * (max_length - 5) + "F"]
    return seqs[:40] + [s for s in extra if len(s) <= max_length]


@pytest.mark.parametrize("order", [2, 3, 4])
def test_scores_match_reference(aa, order):
    """The whole pipeline -- counting, the prior, both anchors, the partition
    function -- reproduced from the specification independently."""
    seqs, m, ref = aa(order)
    probe = _aa_probe(seqs, m.config["max_length"])
    want = reference_scores(ref, probe)
    got = np.array([m.log_pseq(s) for s in probe])

    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-9, (
        f"order {order}: worst deviation {np.abs(got - want).max():.3e}")


@pytest.mark.parametrize("order", [2, 3, 4])
def test_partition_matches_reference(aa, order):
    """Every length, observed or not: each has a share of the length prior,
    so each has a partition function to get right."""
    _, m, ref = aa(order)
    for length in range(1, m.config["max_length"] + 1):
        assert abs(m.log_partition(length) - float(ref.logZ(length))) < 1e-9, \
            length


def test_path_count_matches_reference(aa):
    """Support size over the trellis, against the reference DP. Every cell is
    positive, so both must also be exactly the 20**L strings of the length."""
    _, m, ref = aa(2)
    lengths = range(1, m.config["max_length"] + 1)
    want = R.reference_path_counts(ref, lengths)
    for length in lengths:
        assert m.path_count(length) == pytest.approx(want[length], rel=1e-12)
        assert m.path_count(length) == pytest.approx(20.0 ** length,
                                                     rel=1e-12)


# ── The same agreement, over the nucleotide alphabet ─────────────
#
# These check the radix-6 packing, the prior, mixing and the partition
# function against the independent implementation rather than against the
# C's own opinion of itself. Nothing else in the suite can catch a dimension
# that is consistently wrong in both places.

NT_MAX_LENGTH = 36


@pytest.fixture(scope="module")
def nt():
    seqs, abund = make_nt_corpus()
    built = {}

    def get(order):
        if order not in built:
            m = BeamModel(seqs, abundances=abund, alphabet="nt", order=order,
                          max_length=NT_MAX_LENGTH)
            ref = reference_build(seqs, abund, order=order,
                                  max_len=NT_MAX_LENGTH, alphabet="nt")
            built[order] = (seqs, m, ref)
        return built[order]
    return get


@pytest.mark.parametrize("order", [2, 3, 4])
def test_nucleotide_scores_match_reference(nt, order):
    seqs, m, ref = nt(order)
    # N is a residue, so a run of it is junk the model must still price; and
    # both implementations fold case, so a lower-case probe must agree too.
    probe = seqs[:40] + ["A", "ACGT", "N" * 30, "T" * NT_MAX_LENGTH,
                         "tgtgccagcagc"]
    want = reference_scores(ref, probe)
    got = np.array([m.log_pseq(s) for s in probe])

    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-9, (
        f"order {order}: worst deviation {np.abs(got - want).max():.3e}")


@pytest.mark.parametrize("order", [2, 3, 4])
def test_nucleotide_partition_matches_reference(nt, order):
    _, m, ref = nt(order)
    for length in range(1, NT_MAX_LENGTH + 1):
        assert abs(m.log_partition(length) - float(ref.logZ(length))) < 1e-9, \
            length


def test_nucleotide_path_count_matches_reference(nt):
    """Support size over the radix-6 trellis, against the reference DP: the
    5**L strings over A, C, G, T and N."""
    _, m, ref = nt(2)
    lengths = range(1, NT_MAX_LENGTH + 1)
    want = R.reference_path_counts(ref, lengths)
    for length in lengths:
        assert m.path_count(length) == pytest.approx(want[length], rel=1e-12)
        assert m.path_count(length) == pytest.approx(5.0 ** length, rel=1e-12)
