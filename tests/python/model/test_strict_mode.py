"""Closed-world mode.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, STRICT_SEQS)

# ── Strict (closed-world) mode ──────────────────────────────────────────────
#
# The default model can never return zero: the order-0 base level adds one
# pseudo-count to every residue and the probabilities are floored before the
# log, so an unobserved transition arrives as a large finite negative rather
# than -inf. That is what makes D0 equal sum_L 20**L regardless of the data.
# `strict` turns off both, so the model spans only what it has seen.



def test_strict_spans_exactly_what_it_saw():
    """D0 is the count of training sequences, not 20**L."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    assert m.hill_number(0) == pytest.approx(3.0)
    assert m.hill_number(1) == pytest.approx(3.0)
    # Equally weighted, so each is exactly 1/3 and the entropy is ln 3.
    for s in STRICT_SEQS:
        assert math.exp(m.log_pseq(s)) == pytest.approx(1.0 / 3.0)
    assert m.entropy() == pytest.approx(math.log(3.0))


def test_strict_returns_zero_for_unspellable():
    """A sequence with no supporting path scores exactly zero, not merely small."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    # CASSW shares the CASS context but W was never emitted from it. In the
    # open-world model backoff would still price it; here there is no path.
    assert m.log_pseq("CASSW") == -math.inf
    assert m.log_pseq("WWWWW") == -math.inf
    assert math.exp(m.log_pseq("CASSW")) == 0.0


def test_open_world_never_returns_zero_for_an_observed_length():
    """The contrast the flag exists to make: the default floors instead."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=False)
    for q in ("CASSW", "WWWWW"):
        assert math.isfinite(m.log_pseq(q))
        assert m.log_pseq(q) < 0.0
    assert m.hill_number(0) == pytest.approx(float(20 ** 5))


def test_strict_is_still_a_proper_distribution():
    """Killing cells must not leak mass; the trellis renormalises over survivors."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    d = m.pgen_diagnostics()
    assert d["is_proper"]
    assert d["total_mass"] == pytest.approx(1.0)
    assert d["length_prior_mass"] == pytest.approx(1.0)


def test_strict_trims_the_trellis():
    """`trim`'s general path must run, so topology reports live structure only.

    With no dead cell the trim collapses to pure reachability and every state
    is live at every layer. Under strict the weights decide, and the trellis
    shrinks to the handful of states the corpus actually visited.
    """
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    t = m.topology(5)
    assert t["n_nodes"] < 100          # open-world would be 328,421
    assert m.path_count(5) == pytest.approx(3.0)
    top = m.top_k_sequences(5)
    assert {s for s, _ in top} == set(STRICT_SEQS)


def test_strict_builder_matches_one_shot():
    """The streamed path must build the same strict model as the one-shot path."""
    b = BeamModel.builder(max_length=6, strict=True)
    b.observe(STRICT_SEQS)
    streamed = b.build()
    one_shot = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    assert streamed.config == one_shot.config
    assert np.array_equal(streamed.log_pseq_batch(STRICT_SEQS),
                          one_shot.log_pseq_batch(STRICT_SEQS))
    assert streamed.hill_number(0) == one_shot.hill_number(0)


def test_strict_abundance_still_weights(corpus):
    """Zero probability is about support; abundance still shapes what survives."""
    m = BeamModel(["CASSF", "CASSY"], abundances=[9.0, 1.0],
             max_length=6, strict=True)
    assert math.exp(m.log_pseq("CASSF")) == pytest.approx(0.9)
    assert math.exp(m.log_pseq("CASSY")) == pytest.approx(0.1)
    assert m.hill_number(0) == pytest.approx(2.0)


def test_strict_survives_save_and_load(tmp_path):
    """`strict` must persist, and this failed silently when it did not.

    The file stores COUNTS, not finalized tables, so load re-finalizes. A
    strict model whose flag was dropped therefore came back with the add-one
    and the probability floor switched on again -- a third model, neither of
    the two on offer: kappa 0 so no backoff, but nothing dead, so D0 returned
    to 20**L and every floored cell contributed about 690 nats. On the 927M
    foundation that moved the surprisal range from ~230 nats to ~17,500 and
    made the spectrum grid useless, with no error anywhere.
    """
    m = BeamModel(STRICT_SEQS, max_length=6, strict=True)
    dst = tmp_path / "strict.model"
    m.save(dst)
    back = BeamModel.from_file(dst)

    assert back.config["strict"] is True
    assert back.config == m.config
    assert back.hill_number(0) == m.hill_number(0) == pytest.approx(3.0)
    assert back.log_pseq("CASSW") == -math.inf
    assert np.array_equal(back.log_pseq_batch(STRICT_SEQS),
                          m.log_pseq_batch(STRICT_SEQS))
    # The surprisal range is what blew up when the flag was lost.
    assert back.accumulator_max() == pytest.approx(m.accumulator_max())


def test_open_world_save_and_load_unchanged(tmp_path):
    """The reserved-word reuse must not disturb existing files."""
    m = BeamModel(STRICT_SEQS, max_length=6, strict=False)
    dst = tmp_path / "open.model"
    m.save(dst)
    back = BeamModel.from_file(dst)
    assert back.config["strict"] is False
    assert back.hill_number(0) == m.hill_number(0)
    assert np.array_equal(back.log_pseq_batch(STRICT_SEQS),
                          m.log_pseq_batch(STRICT_SEQS))
