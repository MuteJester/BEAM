"""Fixtures shared by every group.

Module-scoped on purpose: building a model is the expensive step and the
analytics below hit the same one many times. There is one world now: every
model carries the hierarchical prior, so every sequence up to max_length is
possible and there is no separate open-world fixture to ask for.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def model(corpus):
    seqs, abund = corpus
    return BeamModel(seqs, abundances=abund, order=3)


@pytest.fixture(scope="module")
def small():
    """A deliberately small model for the analytics that cost a full trellis
    sweep per call. Order 2 and short sequences keep the saddlepoint and the
    calibration in seconds rather than minutes, without changing what is being
    checked."""
    seqs, abund = make_corpus(n=600, seed=31, lo=6, hi=9)
    return BeamModel(seqs, abundances=abund, order=2, max_length=10)
