"""What `expected_counts` holds, and what it says about it.

This analytic was the largest unbudgeted allocation left in the library. Six
hash maps accumulate one row per `(depth, context)` the trellis reaches, and on
the published foundation model they measured 896 MB against an 806 MB model,
peaking at 1,566 MB while the answer was copied out of them. No budget saw any
of it and no returned value mentioned it, so the only way to discover the
figure was to watch the machine.

It could not be bounded the way every other sweep here is. `positional_
marginals` multiplies `(L + 1) * n_states` and refuses before it starts; the
accumulators have no such expression, because their size is a property of the
ANSWER -- which cells the model actually uses -- and the only shape-derived
bound is the dense table, 148 million cells at order 4. A guard built on that
would refuse every call.

So the size is measured instead, charged as it is taken, and reported as
``working_set_bytes``. What is pinned here is that the figure is real: it moves
with the work, it covers the answer it came back with, and it does not move
when the budget does.

The refusal itself is pinned in `test_memory_budget.py`, where
``expected_counts`` now sits alongside every other heavy entry point.
"""
from __future__ import annotations

import numpy as np
import pytest

import BEAM
from BEAM import BudgetExceeded


@pytest.fixture(scope="module")
def model():
    """Several live lengths, so pooling is strictly more work than one length.

    max_length is the corpus's longest (15): every length up to it is live
    under the prior, so a larger one would only add lengths the corpus never
    had to every pooled sweep below."""
    rng = np.random.default_rng(11)
    seqs = ["CASS" + "".join(rng.choice(list("AGLSTY"), n)) + "EQYF"
            for n in (3, 4, 5, 6, 7) for _ in range(12)]
    return BEAM.BeamModel(seqs, order=3, max_length=15)


def test_the_working_set_is_reported_and_is_not_zero(model):
    ec = model.expected_counts()
    assert "working_set_bytes" in ec, (
        "the one figure a caller cannot compute for itself is missing")
    assert isinstance(ec["working_set_bytes"], int)
    assert ec["working_set_bytes"] > 0


def test_the_working_set_covers_the_answer_it_returned(model):
    """The blocks are part of what the call held, not something extra.

    A figure that excluded the answer would be an underestimate of exactly the
    part a caller keeps, which is the worst part to leave out.
    """
    ec = model.expected_counts()
    answer = sum(b["counts"].nbytes + b["rows"].nbytes
                 for k, b in ec.items() if isinstance(b, dict))
    assert answer > 0, "the model produced no rows, so this proves nothing"
    assert ec["working_set_bytes"] >= answer


def test_the_working_set_tracks_the_work(model):
    """Pooling reaches strictly more cells than a shorter length does.

    A constant large enough to pass the two checks above would satisfy them and
    mean nothing; this is what makes the figure a measurement.

    It used to be "than ANY single length", which the prior has made false.
    Every string of every length is possible now, so the longest length alone
    reaches every (depth, context) any shorter one does, and its figure is the
    pooled one (measured equal). What moves the figure is how deep the cells
    go, so the comparison is with the shortest length the corpus had (11),
    four residues shallower than the pooled answer -- and the longest must
    not exceed the pool it belongs to.
    """
    short = model.expected_counts(length=11)
    longest = model.expected_counts(length=max(model.observed_lengths))
    every = model.expected_counts()
    assert every["working_set_bytes"] > short["working_set_bytes"]
    assert longest["working_set_bytes"] <= every["working_set_bytes"]


def test_the_working_set_does_not_move_with_the_budget(model):
    """It is a property of the answer, not of the setting.

    The budget decides whether the call is allowed to run. It does not change
    which cells the model uses, so a figure that moved with it would be
    reporting the limit rather than the usage.
    """
    needed = model.expected_counts()["working_set_bytes"]
    before = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(8 * needed)
        wide = model.expected_counts()["working_set_bytes"]
        # Just enough room, so the two budgets are as far apart as they can be
        # while both still admitting the call.
        BEAM.set_memory_budget(needed + 4096)
        narrow = model.expected_counts()["working_set_bytes"]
    finally:
        BEAM.set_memory_budget(before)
    assert wide == narrow == needed


def test_a_budget_below_the_working_set_refuses_rather_than_allocating(model):
    """And the refusal says what refused and how to get past it.

    The measured figure is what makes this test possible to write honestly: the
    budget is set from what the call reported holding, so this is not a guess
    at a number that happens to be too small.
    """
    needed = model.expected_counts()["working_set_bytes"]
    before = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(needed // 2)
        with pytest.raises(BudgetExceeded) as e:
            model.expected_counts()
    finally:
        BEAM.set_memory_budget(before)
    msg = str(e.value)
    assert "set_memory_budget" in msg, (
        "the refusal does not name the way out")
    assert "working set" in msg or "backward field" in msg, (
        f"the refusal does not name what did not fit: {msg}")


def test_the_refusal_is_a_memory_error(model):
    """`BudgetExceeded` is a `MemoryError`, so `except MemoryError` catches it.

    A caller draining a queue of repertoires handles running out of memory
    already; a refusal that needed its own except clause would go unhandled in
    exactly the loop it exists to protect.
    """
    needed = model.expected_counts()["working_set_bytes"]
    before = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(needed // 2)
        with pytest.raises(MemoryError):
            model.expected_counts()
    finally:
        BEAM.set_memory_budget(before)
