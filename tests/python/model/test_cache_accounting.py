"""What a model reports about its memory must include what it is actually holding.

`memory_bytes` reports the model's tables and is exact to within 0.2 MB of
what the build allocates. The per-length live cache is not in that figure --
it is a memoisation, droppable and rebuildable, so folding it in would make
the number mean two things and change with when it was asked.

But it is real memory, it appears only once an analytic has run, and it is
bounded by a quarter of the memory budget: a gigabyte at the default. Measured
before it was exposed, a model reporting 13.8 MB was holding 29.2 MB, and the
caller had no way to see the difference or to give it back.
"""
from __future__ import annotations

import random

import pytest

from BEAM import Builder

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def model():
    random.seed(11)
    seqs = ["C" + "".join(random.choice(AA) for _ in range(random.randint(8, 15))) + "F"
            for _ in range(8000)]
    b = Builder(order=4, max_length=20)
    b.observe(seqs, [1.0 + random.randint(0, 20) for _ in seqs])
    return b.build()


def test_the_cache_is_empty_until_an_analytic_runs(model):
    model.drop_caches()
    assert model.live_cache_bytes == 0


def test_an_analytic_fills_the_cache_and_memory_bytes_does_not_move(model):
    model.drop_caches()
    tables = model.memory_bytes
    model.hill_number(1.0)
    assert model.live_cache_bytes > 0, "the analytic cached nothing to report"
    assert model.memory_bytes == tables, \
        "memory_bytes moved; it must describe the tables, not the cache"


def test_drop_caches_releases_it(model):
    model.hill_number(1.0)
    assert model.live_cache_bytes > 0
    model.drop_caches()
    assert model.live_cache_bytes == 0


def test_drop_caches_changes_no_result(model):
    """The cache is a memoisation. Dropping it may cost time, never accuracy."""
    model.hill_number(1.0)
    before = [model.hill_number(q).hex() for q in (0.0, 1.0, 2.0)]
    before.append(model.log_pseq("CASSLGQAYEQYF").hex())
    model.drop_caches()
    after = [model.hill_number(q).hex() for q in (0.0, 1.0, 2.0)]
    after.append(model.log_pseq("CASSLGQAYEQYF").hex())
    assert before == after


def test_dropping_twice_is_safe(model):
    model.drop_caches()
    model.drop_caches()
    assert model.live_cache_bytes == 0


def test_dropping_before_any_analytic_is_safe():
    random.seed(3)
    seqs = ["C" + "".join(random.choice(AA) for _ in range(10)) + "F"
            for _ in range(200)]
    b = Builder(order=3, max_length=20)
    b.observe(seqs, [1.0] * len(seqs))
    m = b.build()
    m.drop_caches()                       # never analysed: no cache exists
    assert m.live_cache_bytes == 0
    assert m.hill_number(1.0) > 0
