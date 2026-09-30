"""Scoring straight from Arrow buffers gives exactly `log_pseq_batch`'s scores.

`log_pseq_arrow` reads a pyarrow string column's offsets and bytes directly --
no Python string per sequence, no Python float per score -- which is where
scoring a large parquet column spent most of its time. Every score must be
BIT-identical to the string path on the same sequences, including the ones
the model cannot read (empty, too long, outside the alphabet), whatever the
array's layout: 32- or 64-bit offsets, a slice that starts mid-buffer, a
chunked column. Nulls are not sequences and score NaN.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from BEAM import BeamModel

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def model():
    rng = random.Random(4)
    seqs = ["CAS" + "".join(rng.choice(AA) for _ in range(rng.randint(3, 12))) + "F"
            for _ in range(3000)]
    return BeamModel(seqs, abundances=[rng.randint(1, 20) for _ in seqs],
                     order=2, max_length=18)


@pytest.fixture(scope="module")
def sequences():
    rng = random.Random(9)
    out = ["CAS" + "".join(rng.choice(AA) for _ in range(rng.randint(0, 15))) + "F"
           for _ in range(2500)]
    # the ones the model cannot read, and the edges of what it can
    out += ["", "C", "A" * 18, "A" * 19, "CASSXF", "cassf", "CAS SF", "CAS*F",
            "CASSéF", "Z", "B" * 3]
    rng.shuffle(out)
    return out


def _same(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("kind", [pa.string(), pa.large_string()])
def test_bit_identical_to_the_string_path(model, sequences, kind):
    want = model.log_pseq_batch(sequences)
    got = model.log_pseq_arrow(pa.array(sequences, type=kind))
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    assert _same(got, want)
    assert np.isneginf(got).any() and np.isfinite(got).any()


def test_a_slice_that_starts_mid_buffer(model, sequences):
    arr = pa.array(sequences)
    for lo, hi in [(7, 900), (1234, 2511), (0, 1), (100, 100)]:
        assert _same(model.log_pseq_arrow(arr.slice(lo, hi - lo)),
                     model.log_pseq_batch(sequences[lo:hi]))


def test_a_chunked_column(model, sequences):
    col = pa.chunked_array([pa.array(sequences[:1000]), pa.array(sequences[1000:1003]),
                            pa.array([], type=pa.string()), pa.array(sequences[1003:])])
    assert _same(model.log_pseq_arrow(col), model.log_pseq_batch(sequences))


def test_nulls_score_nan(model):
    got = model.log_pseq_arrow(pa.array(["CASSF", None, "CASRF", None]))
    want = model.log_pseq_batch(["CASSF", "CASRF"])
    assert np.isnan(got[[1, 3]]).all()
    assert _same(got[[0, 2]], want)


def test_log_pseq_parquet_is_the_string_path_over_the_column(model, sequences, tmp_path):
    path = tmp_path / "seqs.parquet"
    pq.write_table(pa.table({"cdr3_aa": sequences}), path, row_group_size=700)
    got = model.log_pseq_parquet(path, batch_size=512)
    assert _same(got, model.log_pseq_batch(sequences))
