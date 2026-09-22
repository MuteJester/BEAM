"""Many Mellin orders in one pass, and the Hill profile built on it.

`log_mellin_many` is held to `log_mellin` bit for bit, because making it faster
is its only reason to exist. `hill_profile` is held to `hill_numbers` to
machine precision and no closer, because `hill_number` reaches the same sum
through a different sweep that rounds differently; its two corners, `q = 0`
and `q = 1`, take `hill_number`'s own routes and are held to it exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import BeamModel, memory_budget, set_memory_budget, set_threads
from BEAM.errors import BeamTypeError, BeamValueError

from _helpers import make_corpus, make_nt_corpus

ORDERS = np.round(np.arange(0.0, 3.0001, 0.1), 4)
STRESS = np.array([1e-3, 0.5, 0.99, 0.999, 1.001, 1.01, 1.5, 5.0, -0.5, -2.0])


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


@pytest.fixture(scope="module", params=["aa-strict", "aa-open", "nt-strict"])
def m(request):
    if request.param == "nt-strict":
        s, a = make_nt_corpus(n=600)
        return BeamModel(s, abundances=a, alphabet="nt", order=4)
    s, a = make_corpus(n=600, seed=31, lo=6, hi=11)
    return BeamModel(s, abundances=a, order=2, max_length=12,
                     strict=request.param == "aa-strict")


def test_log_mellin_many_is_log_mellin_bit_for_bit(m):
    qs = np.concatenate([ORDERS, STRESS])
    want = np.array([m.log_mellin(q) for q in qs])
    assert (bits(m.log_mellin_many(qs)) == bits(want)).all()


def test_no_schedule_and_no_company_moves_a_bit(m):
    """One thread, a budget that splits the orders into blocks, a budget that
    runs one single-order sweep at a time, and the orders reversed: four
    different paths through the driver, one set of bits."""
    qs = np.concatenate([ORDERS, STRESS])
    want = bits([m.log_mellin(q) for q in qs])
    S, R = m.n_states, (5 if m.config["alphabet"] == "nt" else 20)
    one_sweep = S * R * 8 + S * 8 + 2 * S * 8 + S + 60
    got = {}
    old = memory_budget()
    try:
        set_threads(1)
        got["one thread"] = m.log_mellin_many(qs)
        set_threads(0)
        set_memory_budget(4 * (one_sweep + 2 * 2 * S * 8))
        got["blocks"] = m.log_mellin_many(qs)
        set_memory_budget(one_sweep)
        got["one sweep"] = m.log_mellin_many(qs)
    finally:
        set_threads(0)
        set_memory_budget(old)
    got["reversed"] = m.log_mellin_many(qs[::-1])[::-1]
    for how, v in got.items():
        assert (bits(v) == want).all(), how


def test_hill_profile_agrees_with_hill_numbers_to_machine_precision(m):
    """1e-12 relative, on the default grid and on every order at least 0.01
    from 1. Measured worst on the default grid: 1.05e-14."""
    qs = np.concatenate([ORDERS, STRESS[np.abs(STRESS - 1.0) >= 0.01]])
    hp, hn = m.hill_profile(qs), m.hill_numbers(qs)
    assert np.max(np.abs(hp - hn) / np.abs(hn)) <= 1e-12


def test_near_one_the_difference_is_conditioning_not_disagreement(m):
    """Within 1e-3 of q = 1 the RELATIVE difference in D exceeds 1e-12 --
    1.33e-12 measured at q = 1.001 on the open-world fixture -- and this test
    does not pretend otherwise. D = exp(ln M / (1 - q)) multiplies any rounding
    in ln M by 1 / |1 - q|, a thousand here, and both routes pay it. What the
    two actually disagree on, ln M itself, stays at a few units in the last
    place, which is what this asserts."""
    qs = np.concatenate([ORDERS, STRESS])
    hp, hn = m.hill_profile(qs), m.hill_numbers(qs)
    gen = (qs != 0.0) & (np.abs(qs - 1.0) >= 1e-12)
    d_ln_m = np.abs(np.log(hp[gen]) - np.log(hn[gen])) * np.abs(1.0 - qs[gen])
    assert d_ln_m.max() <= 1e-13


def test_the_corners_are_hill_number_bit_for_bit(m):
    qs = np.array([0.0, 1.0, 1.0 + 1e-13, 2.0])
    hp = m.hill_profile(qs)
    want = [m.hill_number(0.0), m.hill_number(1.0), m.hill_number(1.0 + 1e-13)]
    assert (bits(hp[:3]) == bits(want)).all()


def test_the_profile_is_non_increasing(m):
    h = m.hill_profile(ORDERS)
    assert np.all(np.diff(h) <= 1e-9 * h[:-1])


def test_orders_are_checked_before_anything_runs(m):
    assert m.log_mellin_many([]).shape == (0,)
    assert m.hill_profile([]).shape == (0,)
    with pytest.raises(BeamValueError):
        m.log_mellin_many([0.5, math.nan])
    with pytest.raises(BeamValueError):
        m.hill_profile([math.inf])
    with pytest.raises(BeamValueError):
        m.log_mellin_many(np.ones((2, 2)))
    with pytest.raises(BeamTypeError):
        m.log_mellin_many(2.0)
    with pytest.raises(BeamTypeError):
        m.hill_profile("0.5")
    # A generator is an iterable of orders like any other.
    assert m.hill_profile(q for q in (0.5, 2.0)).shape == (2,)


def test_the_callers_array_is_not_what_the_c_reads(m):
    """The facade hands the C a private copy, so the caller's array is left
    exactly as it was -- dtype, values and writeability."""
    qs = np.array([0.5, 2.0, 3.0], dtype=np.float32)
    before = qs.copy()
    m.log_mellin_many(qs)
    assert qs.dtype == np.float32 and (qs == before).all()
