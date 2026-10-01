"""Fixtures for the likelihood group: one enumerable model and repertoires on it."""
from __future__ import annotations

import pytest

from _brute import repertoire, support, tiny_weighted_model


@pytest.fixture(scope="module")
def tiny():
    return tiny_weighted_model(5)


@pytest.fixture(scope="module")
def sup(tiny):
    return support(tiny)


@pytest.fixture(scope="module")
def rep(sup):
    """60 of the 3,905 supported sequences, one of them a 150-read clone, so the
    open-ended top count bin is occupied."""
    return repertoire(sup, 60, seed=8, big=150)


@pytest.fixture(scope="module")
def rep2(sup):
    return repertoire(sup, 45, seed=9)
