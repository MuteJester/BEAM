"""A number a caller supplies must be rejected, not truncated or allocated.

Two failures, both observed:

`PyArg_ParseTuple`'s "I" takes a Python int MODULO 2**32 with no error, so
`length_derivatives(q, 2**32 + 2)` was answered as order 2 -- a well-formed
table for a question nobody asked, returned with no indication. That is the
worst kind, because nothing about the result says it is wrong.

And a count that parses can still not fit: `top_k(2**28)` succeeded while
reserving about 9 GB, and `sample(2**28)` reported `MemoryError` carrying an
empty string -- no quantity, nothing the caller could act on.

These go through `_cbeam` directly where they must, because that is the layer
where truncation happened; the facade's own guard is tested separately below.
"""
from __future__ import annotations

import random

import pytest

import BEAM
from BEAM import Builder, _cbeam as _c
from BEAM.errors import BudgetExceeded

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(scope="module")
def model():
    random.seed(23)
    seqs = ["C" + "".join(random.choice(AA) for _ in range(8)) + "F"
            for _ in range(500)]
    b = Builder(order=3, max_length=12)
    b.observe(seqs, [1.0] * len(seqs))
    return b.build()


@pytest.fixture
def budget():
    was = BEAM.memory_budget()
    yield
    BEAM.set_memory_budget(was)


# ── the truncation ──────────────────────────────────────────────────────────

def test_an_order_that_fits_is_still_answered(model):
    r = _c.length_derivatives(model._g, 1.0, 2)
    assert r is not None


@pytest.mark.parametrize("order", [2 ** 32, 2 ** 32 + 2, 2 ** 33, 2 ** 64])
def test_an_order_past_the_word_is_refused_not_wrapped(model, order):
    """2**32 + 2 must not come back as order 2's answer."""
    with pytest.raises(ValueError) as e:
        _c.length_derivatives(model._g, 1.0, order)
    # The message must name the value the caller passed, not its remainder.
    assert str(order) in str(e.value), str(e.value)


def test_an_order_above_the_domain_is_refused(model):
    with pytest.raises(ValueError):
        _c.length_derivatives(model._g, 1.0, 17)


def test_a_negative_order_is_refused(model):
    with pytest.raises(ValueError):
        _c.length_derivatives(model._g, 1.0, -1)


def test_a_float_order_is_refused_rather_than_truncated(model):
    """2.9 must not become 2."""
    with pytest.raises(TypeError):
        _c.length_derivatives(model._g, 1.0, 2.9)


# ── the budget ──────────────────────────────────────────────────────────────

def test_top_k_refuses_what_will_not_fit(model):
    with pytest.raises(BudgetExceeded) as e:
        _c.top_k(model._g, 2 ** 28, 10)
    msg = str(e.value)
    assert "268435456" in msg and "budget" in msg, msg


def test_top_k_that_fits_still_runs(model):
    out = _c.top_k(model._g, 5, 10)
    assert out is not None


def test_the_top_k_refusal_tracks_the_budget(model, budget):
    BEAM.set_memory_budget(8 << 20)
    with pytest.raises(BudgetExceeded):
        _c.top_k(model._g, 5_000_000, 10)


# ── the facade's own guard ──────────────────────────────────────────────────

def test_the_facade_bounds_magnitude_not_only_sign(model):
    """It rejected -1 and let 2**32 + 5 through as a request for 5."""
    with pytest.raises(ValueError, match="non-negative"):
        model.sample(-1)
    with pytest.raises(ValueError, match="4294967295"):
        model.sample(2 ** 32 + 5)
