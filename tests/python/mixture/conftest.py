"""Fixtures for the mixture group.

Module-scoped, like the top-level conftest and for the same reason: building a
model is the expensive step and every test below wants the same two or three.
"""
from __future__ import annotations

import pytest

from _tiny import POOL_A, POOL_B, support, tiny_model


@pytest.fixture(scope="module")
def tiny_a():
    return tiny_model(1)


@pytest.fixture(scope="module")
def tiny_b():
    """Overlaps `tiny_a` heavily: same residue pool, different draw."""
    return tiny_model(2)


@pytest.fixture(scope="module")
def tiny_c():
    return tiny_model(3)


@pytest.fixture(scope="module")
def far_a():
    """Built over POOL_A only, so it shares no sequence with `far_b`."""
    return tiny_model(11, letters=POOL_A)


@pytest.fixture(scope="module")
def far_b():
    """Built over POOL_B only. Disjoint from `far_a` by construction."""
    return tiny_model(12, letters=POOL_B)


@pytest.fixture(scope="module")
def sup_a(tiny_a):
    return support(tiny_a)


@pytest.fixture(scope="module")
def sup_b(tiny_b):
    return support(tiny_b)


@pytest.fixture(scope="module")
def sup_c(tiny_c):
    return support(tiny_c)
