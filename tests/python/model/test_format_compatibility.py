"""Two file formats are read: 9 and 10.

A model of at most 255 residues is still written as format 9, byte for byte as
0.9.0 wrote it; a longer one is stamped 10, which an older build refuses as a
newer format. The two differ only in the stamp. A file from an earlier
development build (below 9) must still be REFUSED properly: named as an older
format with the remedy, not reported as corruption or read at the wrong
offsets. `tests/fixtures/format6_sparse.beam` is a real file from such a build
and is kept for exactly that; `tests/fixtures/format9_short.beam` is a real
format 9 file written by the build before format 10 existed, kept to prove
every 0.9.0 file still loads.
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
    struct.pack_into("<I", raw, 4, 11)
    p.write_bytes(bytes(raw))
    with pytest.raises(ModelFileError) as info:
        BeamModel.from_file(p)
    msg = str(info.value)
    assert "format 11" in msg and "newer" in msg and "earlier" not in msg, msg


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


# Written by the build before format 10 existed (aa, order 2, max_length 27),
# with these scores recorded from that same build.
FORMAT9_PROBES = [
    ("CGWSNMGVMIMVVF", float.fromhex("-0x1.2c5e70ed95858p+5")),
    ("CVFNWRPIGWSVHFGF", float.fromhex("-0x1.4697a92ee0bb7p+5")),
    ("CKDDLSCDGPISF", float.fromhex("-0x1.0b9cc7cc36f0fp+5")),
    ("CDHVRQDYFADF", float.fromhex("-0x1.d8ea6e7ac7c9ep+4")),
    ("CNFHSLFDHCWLWF", float.fromhex("-0x1.223d646ef4ebcp+5")),
    ("CYMARHIRPF", float.fromhex("-0x1.96d5f5b06aab6p+4")),
    ("CRHRAKALMQGF", float.fromhex("-0x1.f79f3bc0a51acp+4")),
    ("CNCCARCFDQDAITWFWF", float.fromhex("-0x1.65a7652a02f7ap+5")),
    ("CCSVIGIMPDRDCEVF", float.fromhex("-0x1.4a60154588506p+5")),
    ("CIIMEDPYHQWEHGYF", float.fromhex("-0x1.46a6115785d2fp+5")),
    ("CVSWIGQCGPF", float.fromhex("-0x1.c088f54371c39p+4")),
    ("CAICVNDNMSWQQFPKTF", float.fromhex("-0x1.628ef244868efp+5")),
    ("CIVSIKCCLTF", float.fromhex("-0x1.b7664d2f64b9fp+4")),
    ("CTLNLIDPTRSF", float.fromhex("-0x1.f7f1e2f925ec0p+4")),
    ("CPYTLVRDAAF", float.fromhex("-0x1.c0a853b3d7915p+4")),
    ("CPINCQPFAYWRIHHQF", float.fromhex("-0x1.6122692d0b5c4p+5")),
    ("CVEKFQQGEWIF", float.fromhex("-0x1.f2518bda6d7aap+4")),
    ("CYHPYPPKETLECSQLIF", float.fromhex("-0x1.74ca8281f76f9p+5")),
    ("CHGCTADNRF", float.fromhex("-0x1.a2d082d83e530p+4")),
    ("CAQVRQMSWTTHF", float.fromhex("-0x1.0cbaa28639854p+5")),
]


def test_a_format_9_file_from_the_previous_build_still_loads():
    """Every 0.9.0 file keeps loading: the reader accepts 9 as well as 10."""
    path = FIXTURES / "format9_short.beam"
    assert struct.unpack_from("<I", path.read_bytes(), 4)[0] == 9
    back = BeamModel.from_file(path)
    assert back.config["max_length"] == 27
    got = back.log_pseq_batch([s for s, _ in FORMAT9_PROBES])
    assert np.array_equal(got, [v for _, v in FORMAT9_PROBES])


def test_a_format_9_file_is_saved_back_byte_for_byte(tmp_path):
    """A short model's bytes do not change: load, save, compare."""
    src = FIXTURES / "format9_short.beam"
    out = tmp_path / "resaved.beam"
    BeamModel.from_file(src).save(out)
    assert out.read_bytes() == src.read_bytes()


def test_a_short_model_is_still_saved_as_format_9(tmp_path):
    p = tmp_path / "m.beam"
    BeamModel(SEQS).save(p)
    assert struct.unpack_from("<I", p.read_bytes(), 4)[0] == 9
