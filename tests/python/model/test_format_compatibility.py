"""Files written by older builds must keep loading, exactly.

`tests/fixtures/format3_*.beam` were written by the last build that emitted
format 3, and `format3.json` records what that build read back out of them.
Nothing regenerates these: the moment a fixture is rewritten by the current
build it stops being evidence about the past and starts being a recording of
the present, which is the one thing a compatibility test must not be.

Two files because the two storage modes write different bytes -- sparse
writes occupied-row records, dense writes flat count blocks -- and a reader
can regress in one while the other still passes.

The comparison is on hex floats. A score that has drifted in the last bit is
a reader that has changed, and that is exactly what this exists to catch.
"""
import json
from pathlib import Path

import pytest

from BEAM import BeamModel

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
META = json.loads((FIXTURES / "format3.json").read_text())


@pytest.mark.parametrize("tag", ["sparse", "dense"])
def test_a_format_3_file_still_loads_with_the_same_values(tag):
    want = META[tag]
    path = FIXTURES / f"format3_{tag}.beam"
    assert path.stat().st_size == want["bytes"], (
        "the fixture itself changed size; it must never be regenerated"
    )

    m = BeamModel.from_file(str(path))

    got = [float(m.log_pseq(s)).hex() for s in want["probes"]]
    assert got == want["scores"], f"{tag}: a score moved since format 3"
    assert float(m.total_weight).hex() == want["total_weight"]
    assert int(m.n_nonzero_cells) == want["n_nonzero_cells"]


@pytest.mark.parametrize("tag", ["sparse", "dense"])
def test_a_format_3_file_survives_a_round_trip_through_the_current_format(tag, tmp_path):
    """Loading an old file and saving it again must not lose anything.

    The bytes are allowed to differ -- a newer format writes a newer layout --
    but everything they encode must survive, which is the property that makes
    an upgrade safe to perform in place.
    """
    want = META[tag]
    m = BeamModel.from_file(str(FIXTURES / f"format3_{tag}.beam"))
    out = tmp_path / "again"
    m.save(str(out))                      # save appends the .beam suffix
    again = BeamModel.from_file(str(out) + ".beam")

    got = [float(again.log_pseq(s)).hex() for s in want["probes"]]
    assert got == want["scores"], f"{tag}: a score moved through the round trip"
    assert float(again.total_weight).hex() == want["total_weight"]
    assert int(again.n_nonzero_cells) == want["n_nonzero_cells"]
