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
