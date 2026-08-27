"""Fixtures shared by every group.

Module-scoped on purpose: building a model is the expensive step and the
analytics below hit the same one many times. The `open_*` fixtures exist
because a handful of properties belong to the smoother rather than the
grammar and need backoff switched on; see their docstrings.
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
def open_model(corpus):
    """The order-3 model in the OPEN world.

    The default is closed-world: the model holds only what the training
    sequences spell. A handful of properties below are properties of the
    smoother rather than of the grammar -- graded scores for sequences the
    corpus never saw, a support of 20**L, agreement with the independent
    reference, which implements only the smoothed model. Those need backoff
    switched on, and say so by asking for this fixture instead of `model`.
    """
    seqs, abund = corpus
    return BeamModel(seqs, abundances=abund, order=3, strict=False)


@pytest.fixture(scope="module")
def open_small():
    """`small` in the open world. See `open_model`."""
    seqs, abund = make_corpus(n=600, seed=31, lo=6, hi=9)
    return BeamModel(seqs, abundances=abund, order=2, max_length=10,
                     strict=False)


@pytest.fixture(scope="module")
def small():
    """A deliberately small model for the analytics that cost a full trellis
    sweep per call. Order 2 and short sequences keep the saddlepoint and the
    calibration in seconds rather than minutes, without changing what is being
    checked."""
    seqs, abund = make_corpus(n=600, seed=31, lo=6, hi=9)
    return BeamModel(seqs, abundances=abund, order=2, max_length=10)
