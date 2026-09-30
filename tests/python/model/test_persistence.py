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


# ── Loading under the stored strengths ───────────────────────────
#
# A model whose prior strengths are estimated from its counts stores the
# estimates. Re-deriving them on every load reproduced the same numbers at
# 48.5 s a load on the foundation (7.8 s without), so a file written by a
# FINALIZED model says so, and a load uses the strengths it stores. The tests
# plant a different strength in such a file: a load that re-estimated would
# not see it.

_FLAGS_AT, _ALPHA0_AT = 28, 72          # format-9 header offsets


def _crc32c(data: bytes) -> int:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    crc = 0xFFFFFFFF
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _rewrite(path, edit):
    data = bytearray(path.read_bytes())
    body, stored = data[:-4], int.from_bytes(data[-4:], "little")
    assert _crc32c(bytes(body)) == stored, "the test's CRC32C must match the file's"
    edit(body)
    path.write_bytes(bytes(body) + _crc32c(bytes(body)).to_bytes(4, "little"))


def _auto_model():
    rng = random.Random(3)
    seqs = ["CAS" + "".join(rng.choice(AA) for _ in range(rng.randint(3, 9))) + "F"
            for _ in range(400)]
    return BeamModel(seqs, abundances=[rng.randint(1, 9) for _ in seqs], order=2,
                     max_length=14)


def test_a_finalized_model_loads_under_its_stored_strengths(tmp_path):
    import struct
    m = _auto_model()
    path = tmp_path / "auto.beam"
    m.save(path)
    assert int.from_bytes(path.read_bytes()[_FLAGS_AT:_FLAGS_AT + 4], "little") & 1
    back = BeamModel.from_file(path)
    assert back.config == m.config
    probe = ["CASSF", "CASRGEQYF", "WWWW"]
    assert np.array_equal(back.log_pseq_batch(probe), m.log_pseq_batch(probe))

    def plant(body):
        body[_ALPHA0_AT:_ALPHA0_AT + 8] = struct.pack("<d", 3.25)
    _rewrite(path, plant)
    assert BeamModel.from_file(path).config["alpha"][0] == 3.25


def test_a_file_without_the_flag_still_reestimates(tmp_path):
    """Every file written before the flag existed: the strengths are derived
    from the counts again, whatever the header holds."""
    import struct
    m = _auto_model()
    path = tmp_path / "old.beam"
    m.save(path)

    def as_older_file(body):
        body[_FLAGS_AT:_FLAGS_AT + 4] = (0).to_bytes(4, "little")
        body[_ALPHA0_AT:_ALPHA0_AT + 8] = struct.pack("<d", 3.25)
    _rewrite(path, as_older_file)
    assert BeamModel.from_file(path).config["alpha"][0] == m.config["alpha"][0]
