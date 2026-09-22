"""Weights given as a float64 buffer must build exactly the model a list does.

The abundance column of a parquet corpus is a contiguous integer buffer in the
file. It used to reach C as Python objects four times over: `to_pylist` built
them, `_iter_parquet` rebuilt them with `float(x)`, `Builder.observe` rebuilt
them again, and the binding then called `PyFloat_AsDouble` on each -- 926,909,993
objects created and destroyed per foundation build to move numbers that were
already contiguous doubles.

`observe` now passes a C-contiguous float64 buffer straight through. That is a
second code path into the one operation whose output everything else depends
on, so these tests compare it against the list path rather than merely
exercising it: same sequences, same weights, same model, checked on the
numbers the model actually reports.
"""
from __future__ import annotations

import sys

import numpy as np

import pytest

from BEAM import Builder

SEQS = ["CASSLGQAYEQYF", "CASSPGQGAYEQYF", "CASRDRTGNGYTF",
        "CASSQDRGTEAFF", "CASSLLGQGYNEQFF", "CASSSDPVGNQPQHF"]
W = [2.0, 3.0, 4.0, 1.0, 7.0, 5.0]


def _build(weights):
    b = Builder(order=3, max_length=20)
    b.observe(SEQS, weights)
    return b.build()


def _describe(m):
    """Everything the model will say about itself, to full precision."""
    return (m.total_weight, m.n_observed, m.n_skipped, m.memory_bytes,
            tuple(m.log_pseq(s).hex() for s in SEQS))


def test_a_float64_array_builds_the_same_model_as_a_list():
    assert _describe(_build(W)) == _describe(_build(np.array(W, dtype=np.float64)))


class _OnlyABuffer:
    """A C-contiguous float64 buffer that cannot be iterated.

    A `memoryview` is useless as a detector here: it IS iterable, so the
    element-by-element fallback consumes it and produces the same answer. That
    was the first version of this test and it passed with the fast path
    compiled out, which is to say it tested nothing. This exposes the buffer
    protocol (PEP 688) and nothing else, so a fallback has no way to read it.
    """

    def __init__(self, a):
        self._a = a

    def __buffer__(self, flags):
        return memoryview(self._a)

    def __len__(self):
        return len(self._a)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="the fixture exposes the buffer protocol through __buffer__, "
           "which is PEP 688 and 3.12+. On 3.11 memoryview() cannot see "
           "it, so the object is neither a buffer nor iterable and the "
           "test cannot be constructed. The behaviour it guards is "
           "unaffected: verified on ig05 at 3.11.13 with numpy 2.4.6, a "
           "contiguous float64 ndarray and an array('d') both take the "
           "fast path, and a list and a strided slice both materialise.",
)
def test_the_buffer_path_is_actually_taken():
    o = _OnlyABuffer(np.array(W, dtype=np.float64))
    with pytest.raises(TypeError):
        iter(o)                                   # the detector is sound
    assert _describe(_build(o)) == _describe(_build(W))


def test_a_memoryview_builds_the_same_model():
    a = np.array(W, dtype=np.float64)
    mv = memoryview(a)
    assert mv.format == "d" and mv.c_contiguous
    assert _describe(_build(mv)) == _describe(_build(W))


@pytest.mark.parametrize("dtype", ["int64", "float32", "int32"])
def test_other_dtypes_still_build_the_same_model(dtype):
    """Anything not float64 takes the original element-by-element path."""
    assert _describe(_build(np.array(W, dtype=dtype))) == _describe(_build(W))


def test_a_non_contiguous_array_still_builds_the_same_model():
    """A strided view is not a buffer C can memcpy; it must not be treated as one."""
    wide = np.zeros(len(W) * 2, dtype=np.float64)
    wide[::2] = W
    view = wide[::2]
    assert not view.flags.c_contiguous
    assert _describe(_build(view)) == _describe(_build(W))


def test_a_mismatched_buffer_length_is_still_refused():
    with pytest.raises(ValueError, match="abundances"):
        _build(np.array(W[:-1], dtype=np.float64))


def test_a_list_of_python_floats_is_unchanged():
    m = _build(W)
    assert m.total_weight == sum(W)
    assert m.n_observed == len(SEQS)
