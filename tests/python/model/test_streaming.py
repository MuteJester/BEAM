"""Streaming construction through the builder.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Streaming construction
#
# The property that makes a streamed build usable is that it is not an
# approximation of a one-shot build: counts are additive and finalize
# recomputes from scratch, so the two must agree BIT FOR BIT. Anything
# weaker and a convergence curve measured from snapshots is measuring the
# streaming path rather than the model.
# ════════════════════════════════════════════════════════════════════════


def test_builder_is_byte_identical_to_a_one_shot_build(corpus):
    seqs, abund = corpus
    one = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    b = BeamModel.builder(order=2, max_length=20)
    for i in range(0, len(seqs), 137):          # deliberately ragged batches
        b.observe(seqs[i:i + 137], abund[i:i + 137])
    streamed = b.build()

    probe = seqs[:200]
    assert np.array_equal(streamed.log_pseq_batch(probe),
                          one.log_pseq_batch(probe))
    assert streamed.total_weight == one.total_weight
    assert streamed.n_observed == one.n_observed
    assert streamed.n_skipped == one.n_skipped


def test_builder_snapshot_is_independent_of_later_batches(corpus):
    """A convergence curve measures a snapshot and keeps ingesting. If the
    snapshot shared the builder's counts, every earlier checkpoint would
    silently become the final model."""
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    b.observe(seqs[:300], abund[:300])
    early = b.snapshot()
    early_scores = early.log_pseq_batch(seqs[:50])
    early_weight = early.total_weight

    b.observe(seqs[300:], abund[300:])
    late = b.build()

    assert np.array_equal(early.log_pseq_batch(seqs[:50]), early_scores)
    assert early.total_weight == early_weight
    assert late.total_weight > early_weight
    assert not np.array_equal(late.log_pseq_batch(seqs[:50]), early_scores)

    # And the snapshot really is the model that batch alone would have built.
    reference = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    assert np.array_equal(early_scores, reference.log_pseq_batch(seqs[:50]))


def test_builder_tracks_its_running_totals(corpus):
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    assert b.n_observed == 0
    b.observe(seqs[:100], abund[:100])
    assert b.n_observed == 100
    assert b.total_weight == pytest.approx(float(sum(abund[:100])))
    b.observe(seqs[100:250], abund[100:250])
    assert b.n_observed == 250


def test_builder_is_spent_after_build(corpus):
    seqs, abund = corpus
    b = BeamModel.builder(order=2, max_length=20)
    b.observe(seqs[:100], abund[:100])
    b.build()
    with pytest.raises(RuntimeError, match="already been build"):
        b.observe(seqs[100:200])
    with pytest.raises(RuntimeError, match="already been build"):
        b.build()


def test_builder_rejects_mismatched_abundances():
    b = BeamModel.builder(order=2, max_length=10)
    with pytest.raises(ValueError, match="abundances has 2 entries"):
        b.observe(["CASSF", "CASSY", "CASSW"], [1.0, 2.0])


def test_from_sequence_file_matches_a_direct_build(tmp_path, corpus):
    seqs, abund = corpus
    path = tmp_path / "corpus.tsv"
    path.write_text("".join(f"{s}\t{a}\n" for s, a in zip(seqs, abund)))

    from_file = BeamModel.from_sequence_file(path, order=2, max_length=20,
                                        chunk_size=97)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert from_file.total_weight == direct.total_weight
    assert np.array_equal(from_file.log_pseq_batch(seqs[:100]),
                          direct.log_pseq_batch(seqs[:100]))


def test_from_sequence_file_without_weights(tmp_path, corpus):
    """A bare list of sequences, and a two-column file read with the weight
    column switched off, must give the same unweighted model."""
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs))
    from_file = BeamModel.from_sequence_file(path, abundance_column=None,
                                        order=2, max_length=20)
    direct = BeamModel(seqs, order=2, max_length=20)
    assert from_file.total_weight == direct.total_weight
    assert np.array_equal(from_file.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


def test_from_sequence_file_skips_a_header(tmp_path):
    seqs = ["CASSLGF", "CASSYEQF", "CASSWTF"]
    path = tmp_path / "h.tsv"
    path.write_text("cdr3_aa\tcount\n" + "".join(f"{s}\t2\n" for s in seqs))
    m = BeamModel.from_sequence_file(path, has_header=True, order=2, max_length=10)
    assert m.n_observed == 3
    assert m.total_weight == pytest.approx(6.0)


def test_from_dataframe_matches_a_direct_build(corpus):
    seqs, abund = corpus
    table = {"cdr3_aa": seqs, "total_abundance": abund}
    from_df = BeamModel.from_dataframe(table, abundance_column="total_abundance",
                                  order=2, max_length=20)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert np.array_equal(from_df.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


def test_from_dataframe_names_the_missing_column(corpus):
    seqs, _ = corpus
    with pytest.raises(KeyError, match="no column 'junction_aa'"):
        BeamModel.from_dataframe({"cdr3_aa": seqs}, sequence_column="junction_aa",
                            order=2, max_length=20)


def test_from_parquet_matches_a_direct_build(tmp_path, corpus):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    seqs, abund = corpus
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"cdr3_aa": seqs,
                             "total_abundance": [float(a) for a in abund]}),
                   path)
    from_pq = BeamModel.from_parquet(path, abundance_column="total_abundance",
                                batch_size=211, order=2, max_length=20)
    direct = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert from_pq.total_weight == pytest.approx(direct.total_weight)
    assert np.array_equal(from_pq.log_pseq_batch(seqs[:50]),
                          direct.log_pseq_batch(seqs[:50]))


# ════════════════════════════════════════════════════════════════════════
