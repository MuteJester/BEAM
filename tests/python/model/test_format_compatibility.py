"""Files written by older builds must keep loading, exactly.

Each `tests/fixtures/formatN_*.beam` was written by the last build that emitted
format N, and `formatN.json` records what that build read back out of it.
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

# Every format this build must still read, with what the build that WROTE
# each one read back out of it. A new entry is added when the format changes,
# by the last build that emits the outgoing one -- never regenerated later,
# which would turn evidence about the past into a recording of the present.
LEGACY = {
    f"{fmt}_{tag}": (f"{fmt}_{tag}.beam", json.loads(
        (FIXTURES / f"{fmt}.json").read_text())[tag])
    for fmt, tags in (("format3", ("sparse", "dense")),
                      ("format4", ("sparse", "dense", "fractional")))
    for tag in tags
}


@pytest.mark.parametrize("case", sorted(LEGACY))
def test_an_older_file_still_loads_with_the_same_values(case):
    name, want = LEGACY[case]
    path = FIXTURES / name
    assert path.stat().st_size == want["bytes"], (
        "the fixture itself changed size; it must never be regenerated"
    )

    m = BeamModel.from_file(str(path))

    got = [float(m.log_pseq(s)).hex() for s in want["probes"]]
    assert got == want["scores"], f"{case}: a score moved"
    assert float(m.total_weight).hex() == want["total_weight"]
    assert int(m.n_nonzero_cells) == want["n_nonzero_cells"]


@pytest.mark.parametrize("case", sorted(LEGACY))
def test_an_older_file_survives_a_round_trip_through_the_current_format(case, tmp_path):
    """Loading an old file and saving it again must not lose anything.

    The bytes are allowed to differ -- a newer format writes a newer layout --
    but everything they encode must survive, which is the property that makes
    an upgrade safe to perform in place.
    """
    name, want = LEGACY[case]
    m = BeamModel.from_file(str(FIXTURES / name))
    out = tmp_path / "again"
    m.save(str(out))                      # save appends the .beam suffix
    again = BeamModel.from_file(str(out) + ".beam")

    got = [float(again.log_pseq(s)).hex() for s in want["probes"]]
    assert got == want["scores"], f"{case}: a score moved through the round trip"
    assert float(again.total_weight).hex() == want["total_weight"]
    assert int(again.n_nonzero_cells) == want["n_nonzero_cells"]
