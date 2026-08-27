"""Scoring behaviour.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Scoring behaviour ────────────────────────────────────────────

def test_batch_matches_scalar(model, corpus):
    seqs, _ = corpus
    batch = model.log_pseq_batch(seqs[:200])
    for i in range(200):
        assert batch[i] == pytest.approx(model.log_pseq(seqs[i]), abs=1e-12)


def test_explain_sums_to_the_score(model, corpus):
    """The per-residue decomposition must reconstruct the score exactly, so it
    can be trusted as an explanation rather than an illustration."""
    seqs, _ = corpus
    cfg = model.config
    wf, wb = cfg["weights"]
    for s in seqs[:20]:
        e = model.explain(s)
        raw = sum(wf * f + wb * b for f, b in zip(e["forward"], e["backward"]))
        want = raw + model.log_length_prior(len(s)) - model.log_partition(len(s))
        assert want == pytest.approx(e["log_pseq"], abs=1e-9)


def test_c_terminus_expert_is_doing_work(open_model):
    """Swapping the terminal residue for one no J gene produces must cost
    orders of magnitude. A single left-to-right chain cannot see this."""
    good = open_model.log_pseq("CASSLGQGAEQYF")
    bad = open_model.log_pseq("CASSLGQGAEQYW")
    assert math.isfinite(good) and math.isfinite(bad)
    assert good - bad > math.log(1000), (good, bad)


def test_rejects_non_amino_acid(model):
    assert model.log_pseq("CASSXQYF") == -math.inf
    assert model.log_pseq("") == -math.inf
    assert model.log_pseq("C" * 40) == -math.inf


def test_unobserved_length_is_impossible(model):
    """A length the model never saw has prior zero, so its sequences have
    probability zero. The prototype floors the prior at 1e-300 instead; that
    differs only outside the training length support, which the foundation
    model covers completely."""
    assert model.log_pseq("CASSF") == -math.inf
