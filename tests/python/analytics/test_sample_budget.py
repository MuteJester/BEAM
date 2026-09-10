"""Drawing more sequences than fit must refuse, not exhaust the machine.

`sample` allocates its result buffer in the binding, BEFORE the library runs
its budget check, and `beam_model_sample` allocated two more arrays of eight
bytes per draw that nothing charged for. So a request the budget would have
refused had already taken `n * stride` bytes: at 2**28 draws about 3.5 GB, and
the failure arrived as `MemoryError` with an EMPTY message -- or as the machine
going down, which is how it was found.

These pin the refusal, and pin that it says something useful. A bare
MemoryError is not a diagnosis: it names no quantity the caller controls.
"""
from __future__ import annotations

import random

import pytest

import BEAM
from BEAM import Builder
from BEAM.errors import BeamError, BudgetExceeded

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def budget():
    """Restore the process-wide budget however the test leaves it."""
    was = BEAM.memory_budget()
    yield
    BEAM.set_memory_budget(was)


@pytest.fixture(scope="module")
def model():
    random.seed(17)
    seqs = ["C" + "".join(random.choice(AA) for _ in range(8)) + "F"
            for _ in range(2000)]
    b = Builder(order=3, max_length=12)
    b.observe(seqs, [1.0] * len(seqs))
    return b.build()


def test_a_draw_that_fits_still_works(model):
    out = model.sample(500)
    assert len(out) == 500
    assert all(s.startswith("C") for s in out)


def test_a_draw_far_past_the_budget_is_refused(model):
    with pytest.raises(BudgetExceeded) as e:
        model.sample(400_000_000)
    msg = str(e.value)
    # The refusal must name what the caller asked for and what the limit is,
    # or it is no better than the MemoryError it replaced.
    assert "400000000" in msg, msg
    assert "budget" in msg, msg
    assert "set_memory_budget" in msg, msg


def test_the_refusal_tracks_the_budget(model, budget):
    """The same draw count is fine under one budget and refused under another."""
    BEAM.set_memory_budget(4 << 30)
    n = 2_000_000
    assert len(model.sample(n)) == n          # 26 MB of result: comfortable

    BEAM.set_memory_budget(8 << 20)           # 8 MB: the same draw cannot fit
    with pytest.raises(BudgetExceeded) as e:
        model.sample(n)
    assert "budget" in str(e.value)


def test_a_small_draw_survives_a_small_budget(model, budget):
    BEAM.set_memory_budget(8 << 20)
    assert len(model.sample(100)) == 100


def test_the_refusal_is_not_a_bare_memory_error(model):
    """MemoryError carries no message and no quantity. That was the bug."""
    with pytest.raises(BudgetExceeded) as e:
        model.sample(2 ** 31)
    # The point was never "not a MemoryError" -- BudgetExceeded IS one, and
    # should be. The point is that it is TYPED and carries a message, so a
    # caller can catch it precisely and read what to change.
    assert isinstance(e.value, BeamError), "not part of the error hierarchy"
    assert str(e.value).strip(), "the refusal carried no message"
