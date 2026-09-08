"""Reading a parquet file must not cost memory proportional to its size.

`_iter_parquet` streams a file once, in order. The pyarrow default coalesces
each row group's column chunks into buffers held for the life of the file --
right for high-latency storage, wrong for a single ordered pass, where it
grows with the file and is never read twice. On the 927M-row corpus that is
about 5 GB, the larger part of a foundation build's peak.

The invariant is stated as a SHAPE rather than a number: the memory a read
holds must not grow with how many row groups the file has. A threshold in
megabytes would be a recording of this machine's pyarrow; "flat in the size of
the file" is the property that actually has to hold.

Each test also checks that it CAN see the difference, by reading the same
files the eager way. A bound that nothing can violate proves nothing about
the code it guards.
"""
from __future__ import annotations

import gc
import random

import pytest

from BEAM.io.readers import _iter_parquet, _open_parquet_for_streaming

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROWS_PER_GROUP = 200_000
AA = "ACDEFGHIKLMNPQRSTVWY"


def _write(path, n_groups, rows_per_group=ROWS_PER_GROUP):
    """A parquet file with a known number of row groups.

    The sequences must be genuinely varied. A first version of this fixture
    repeated twenty distinct strings, which parquet dictionary-encodes down to
    almost nothing -- the column chunks were then too small for either reader
    to hold a measurable amount, and the detector test below correctly refused
    to certify a fixture that could not show the difference.
    """
    rng = random.Random(20260908)
    block = ["C" + "".join(rng.choice(AA) for _ in range(12)) + "F"
             for _ in range(rows_per_group)]
    table = pa.table({"cdr3_aa": block * n_groups,
                      "total_abundance": [1.0] * (rows_per_group * n_groups)})
    pq.write_table(table, path, row_group_size=rows_per_group)
    assert pq.ParquetFile(path).metadata.num_row_groups == n_groups
    return path


def _peak_pool(reader):
    """Highest Arrow pool occupancy reached while draining `reader`."""
    gc.collect()
    base = pa.total_allocated_bytes()
    peak = 0
    for seqs, ab in reader:
        peak = max(peak, pa.total_allocated_bytes() - base)
        del seqs, ab
    return peak


def _peak_pool_eager(path):
    """The same drain, through a reader that keeps its buffers."""
    gc.collect()
    base = pa.total_allocated_bytes()
    peak = 0
    pf = pq.ParquetFile(path, pre_buffer=True)
    for batch in pf.iter_batches(batch_size=ROWS_PER_GROUP,
                                 columns=["cdr3_aa", "total_abundance"]):
        batch.column("cdr3_aa").to_pylist()
        peak = max(peak, pa.total_allocated_bytes() - base)
        del batch
    return peak


def _drain(path):
    return _iter_parquet(path, "cdr3_aa", "total_abundance", ROWS_PER_GROUP)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("parquet")
    return (_write(d / "small.parquet", 4), _write(d / "large.parquet", 8))


def test_the_memory_a_read_holds_does_not_grow_with_the_file(files):
    """Twice the row groups must not cost meaningfully more memory."""
    small, large = files
    peak_small = _peak_pool(_drain(small))
    peak_large = _peak_pool(_drain(large))

    assert peak_small > 0, "measured nothing; the probe is not working"
    # Doubling the row groups may cost a little (allocator behaviour is not
    # perfectly reproducible) but must not scale with the file.
    assert peak_large < peak_small * 1.5, (
        f"holding {peak_large:,} bytes for 8 row groups against "
        f"{peak_small:,} for 4 -- the read is accumulating"
    )


def test_the_eager_reader_does_grow_with_the_file(files):
    """The detector detects.

    Without this, the bound above would pass on any reader at all, including
    one that had quietly gone back to accumulating.
    """
    small, large = files
    eager_small = _peak_pool_eager(small)
    eager_large = _peak_pool_eager(large)
    assert eager_large > eager_small * 1.5, (
        f"the eager reader held {eager_large:,} for 8 row groups against "
        f"{eager_small:,} for 4; if it no longer accumulates, the streaming "
        f"reader's guarantee is untestable this way and this file needs "
        f"rewriting rather than deleting"
    )


def test_the_streaming_reader_holds_less_than_the_eager_one(files):
    _, large = files
    assert _peak_pool(_drain(large)) < _peak_pool_eager(large)


def test_streaming_and_eager_reads_return_the_same_rows(files):
    """The saving must not have come from reading less.

    Compares against pyarrow read whole, so the reference shares no code with
    the reader under test beyond pyarrow itself.
    """
    _, large = files
    want = pq.read_table(large, columns=["cdr3_aa", "total_abundance"])
    want_seqs = want.column("cdr3_aa").to_pylist()
    want_ab = [float(x) for x in want.column("total_abundance").to_pylist()]

    got_seqs, got_ab = [], []
    for seqs, ab in _drain(large):
        got_seqs.extend(seqs)
        got_ab.extend(ab)

    assert got_seqs == want_seqs
    assert got_ab == want_ab


def test_the_opener_falls_back_when_pre_buffer_is_unknown():
    """An older pyarrow loses the saving, not the ability to read.

    pyarrow is optional here and declares no version floor, so a reader that
    raised on an unrecognised keyword would turn a memory optimisation into a
    hard failure for anyone on a build predating it.
    """
    class OldParquet:
        def __init__(self, path, **kw):
            if kw:
                raise TypeError(f"unexpected keyword arguments {sorted(kw)}")
            self.path = path

    class OldModule:
        ParquetFile = OldParquet

    got = _open_parquet_for_streaming(OldModule, "somewhere.parquet")
    assert isinstance(got, OldParquet)
    assert got.path == "somewhere.parquet"


def test_the_opener_asks_for_no_pre_buffering_when_it_can():
    seen = {}

    class NewParquet:
        def __init__(self, path, **kw):
            seen.update(kw)

    class NewModule:
        ParquetFile = NewParquet

    _open_parquet_for_streaming(NewModule, "somewhere.parquet")
    assert seen == {"pre_buffer": False}
