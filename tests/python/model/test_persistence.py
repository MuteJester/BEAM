"""Save and load.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Persistence ──────────────────────────────────────────────────

def test_round_trip(tmp_path, corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    path = tmp_path / "model.edge"
    m.save(path)
    back = BeamModel.from_file(path)

    assert back.config == m.config
    assert back.n_observed == m.n_observed
    assert back.total_weight == pytest.approx(m.total_weight)
    for s in seqs[:50]:
        assert back.log_pseq(s) == pytest.approx(m.log_pseq(s), abs=1e-12)


def test_a_reload_refinalizes_under_the_stored_prior(tmp_path):
    """The file stores COUNTS, so every load re-finalizes, and the prior's
    strengths are what it re-finalizes under.

    Dropping a model-defining flag on this path has failed silently before:
    the reload came back as a third model, neither the one saved nor the
    default, and moved its surprisal range from ~230 nats to ~17,500 with no
    error anywhere. So the check here reaches past the config to the
    analytics that range feeds.
    """
    m = BeamModel(["CASSF", "CASSY", "CTTGW"], abundances=[5.0, 2.0, 1.0],
                  order=2, max_length=6, alpha=(0.2, 7.0, 90.0),
                  alpha_length=0.5)
    path = tmp_path / "prior.beam"
    m.save(path)
    back = BeamModel.from_file(path)

    assert back.config == m.config
    assert back.accumulator_max() == m.accumulator_max()
    assert back.entropy() == m.entropy()
    probe = ["CASSF", "CASSW", "WWWWW", "C"]
    assert np.array_equal(back.log_pseq_batch(probe), m.log_pseq_batch(probe))


def test_corrupt_file_is_rejected(tmp_path, corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    path = tmp_path / "model.edge"
    m.save(path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(OSError):
        BeamModel.from_file(path)
