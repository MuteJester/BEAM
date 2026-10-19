"""One file format: the one this build writes.

BEAM has not been released, so no file from an earlier development build is
owed a reader. Such a file must still be REFUSED properly: named as an older
format with the remedy, not reported as corruption or read at the wrong
offsets. `tests/fixtures/format6_sparse.beam` is a real file from such a build
and is kept for exactly that.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from BEAM import BeamModel, ModelFileError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SEQS = ["CASSLGQGAEQYF", "CASSPGTGELFF", "CASRDRGNTIYF", "CASSQETQYF"] * 3


def test_an_older_development_file_is_refused_with_the_remedy():
    with pytest.raises(ModelFileError) as info:
        BeamModel.from_file(FIXTURES / "format6_sparse.beam")
    msg = str(info.value)
    assert "format 6" in msg and "format 9" in msg, msg
    assert "Rebuild the model from its sequences" in msg, msg
    assert "checksum" not in msg


def test_the_format_word_decides_before_the_checksum(tmp_path):
    """A current file with only its format word changed is refused as a
    version, not as a checksum failure, because the format is read first."""
    p = tmp_path / "m.beam"
    BeamModel(SEQS).save(p)
    raw = bytearray(p.read_bytes())
    struct.pack_into("<I", raw, 4, 7)
    p.write_bytes(bytes(raw))
    with pytest.raises(ModelFileError, match="format 7"):
        BeamModel.from_file(p)


def test_a_newer_file_asks_for_a_newer_beam(tmp_path):
    """A file in a format newer than this build reads came from a later BEAM:
    the remedy is upgrading, not rebuilding from an earlier build."""
    p = tmp_path / "m.beam"
    BeamModel(SEQS).save(p)
    raw = bytearray(p.read_bytes())
    struct.pack_into("<I", raw, 4, 10)
    p.write_bytes(bytes(raw))
    with pytest.raises(ModelFileError) as info:
        BeamModel.from_file(p)
    msg = str(info.value)
    assert "format 10" in msg and "newer" in msg and "earlier" not in msg, msg


def test_a_current_file_round_trips(tmp_path):
    m = BeamModel(SEQS, abundances=range(1, len(SEQS) + 1), order=3,
                  alpha=(1.0, 2.0, 3.0, 4.0), alpha_length=6.0)
    p = tmp_path / "m.beam"
    m.save(p)
    back = BeamModel.from_file(p)
    assert back.config == m.config
    probe = SEQS + ["CASSWWWF", "Y"]
    assert np.array_equal(back.log_pseq_batch(probe), m.log_pseq_batch(probe))


def test_not_a_model_file_says_so(tmp_path):
    p = tmp_path / "x.beam"
    p.write_bytes(b"not a beam model at all")
    with pytest.raises(ModelFileError, match="not a BEAM model file"):
        BeamModel.from_file(p)
