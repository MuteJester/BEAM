"""Fixtures for the mixture group.

Module-scoped, like the top-level conftest and for the same reason: building a
model is the expensive step and every test below wants the same two or three.
"""
from __future__ import annotations

import pytest

from _tiny import FAR_ALPHA, POOL_A, POOL_B, support, tiny_model


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
    """Built over POOL_A only, with a weak prior.

    It still spells every string `far_b` does -- every model spells every
    string now -- but the mass the two share is far below double precision.
    """
    return tiny_model(11, letters=POOL_A, alpha=FAR_ALPHA)


@pytest.fixture(scope="module")
def far_b():
    """Built over POOL_B only. Shares almost nothing with `far_a`."""
    return tiny_model(12, letters=POOL_B, alpha=FAR_ALPHA)


@pytest.fixture(scope="module")
def sup_a(tiny_a):
    return support(tiny_a)


@pytest.fixture(scope="module")
def sup_b(tiny_b):
    return support(tiny_b)


@pytest.fixture(scope="module")
def sup_c(tiny_c):
    return support(tiny_c)


@pytest.fixture(scope="module")
def sup_far_a(far_a):
    return support(far_a)


@pytest.fixture(scope="module")
def sup_far_b(far_b):
    return support(far_b)
