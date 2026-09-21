"""Files written by older builds must be REFUSED, by name and with a reason.

Each `tests/fixtures/formatN_*.beam` was written by the last build that
emitted format N, and `formatN.json` records what that build read back out of
it. Nothing regenerates these: the moment a fixture is rewritten by the
current build it stops being evidence about the past and starts being a
recording of the present, which is the one thing a compatibility test must
not be. They are kept for exactly that reason, and what they are now evidence
OF is that this build declines to misread them.

Formats 1 to 5 were written with three experts per direction and a CLAMPED
depth index. They cannot be converted, and the reason is not effort: the clamp
merged every depth at or beyond it, so which depth an observation came from
was discarded when the model was trained. No table in such a file holds
order-k context at exact depth, and no transformation recovers it.

The danger this guards is not refusal, it is silence. An old file carries the
same magic, the same header size and, before the bump, the same format number
as a new one. Nothing would have rejected it; the payload extents would simply
have been wrong, and the reader would have reported a checksum failure -- a
corrupt-file diagnosis for a perfectly intact file. So the test is that the
refusal happens, names the format, and says why.
"""
import json
from pathlib import Path

import pytest

from BEAM import BeamModel

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

LEGACY = {
    f"{fmt}_{tag}": (f"{fmt}_{tag}.beam", int(fmt[-1]))
    for fmt, tags in (("format3", ("sparse", "dense")),
                      ("format4", ("sparse", "dense", "fractional")))
    for tag in tags
}


@pytest.mark.parametrize("case", sorted(LEGACY))
def test_an_older_file_is_refused_and_says_why(case):
    """Refused, naming the format and the reason, not merely failing."""
    name, fmt = LEGACY[case]
    with pytest.raises(Exception) as exc:
        BeamModel.from_file(FIXTURES / name)
    msg = str(exc.value)
    assert f"format {fmt}" in msg, msg
    assert "three experts" in msg, msg
    assert "retrain" in msg.lower(), msg


@pytest.mark.parametrize("case", sorted(LEGACY))
def test_the_refusal_is_not_a_checksum_complaint(case):
    """The fixtures are intact, and the reader must not say otherwise.

    Before the format bump an old file would have been accepted, read with
    the wrong extents and then failed its CRC, which blames the file for the
    reader's mistake. Nothing in the message may suggest the bytes are bad.
    """
    name, _ = LEGACY[case]
    with pytest.raises(Exception) as exc:
        BeamModel.from_file(FIXTURES / name)
    msg = str(exc.value).lower()
    assert "checksum" not in msg, msg
    assert "corrupt" not in msg, msg


# ── Format 6: read, not refused ────────────────────────────────────────────
#
# Format 6 is the one older format this build accepts, and the reason is that
# it is not an older MODEL. It is this model -- one table per direction, exact
# depth -- written before provenance existed. The bytes provenance now uses
# were `dead_coef[3]`, which format 6 wrote as three explicit zeros, so the
# payload is identical and the only thing missing is a build time.
#
# `format6_sparse.beam` was written by 3806d4c, the last commit that emitted
# format 6, from a worktree at that commit -- not by patching a format-7 file
# with the current build, which would have made it a recording of the present
# like every other fixture here warns against. `format6.json` is what THAT
# build read back out of it.
#
# This exists because the refusal above was briefly widened to swallow format
# 6, telling anyone holding one that their file came from "a build with three
# experts per direction and a capped depth index" and that they had to
# retrain. Every clause of that was false, and for a foundation-sized corpus
# it is an expensive thing to be told over 24 zero bytes.

def test_a_format_6_file_is_read_and_agrees_with_the_build_that_wrote_it():
    rec = json.loads((FIXTURES / "format6.json").read_text())
    m = BeamModel.from_file(FIXTURES / "format6_sparse.beam")

    assert m.n_observed == rec["n_observed"]
    assert m.n_sequences == rec["n_sequences"]
    assert m.total_weight == rec["total_weight"]
    assert m.n_states == rec["n_states"]
    assert m.n_nonzero_cells == rec["n_nonzero_cells"]
    assert [int(x) for x in m.observed_lengths] == rec["observed_lengths"]

    # The scores are the point: an accepted format that read the payload at
    # the wrong extents would still satisfy the counters above.
    for s, want in rec["log_pseq"].items():
        assert float(m.log_pseq(s)) == want, s
    assert float(m.path_count()) == rec["path_count"]
    assert float(m.entropy()) == rec["entropy"]


def test_a_format_6_file_reports_its_build_time_as_unknown():
    """Absent, not invented. A zero stamp rendered literally claims 1970."""
    meta = BeamModel.from_file(FIXTURES / "format6_sparse.beam").metadata(
        lengths=False)
    assert meta["created"] is None
    assert meta["created_unix"] is None
    assert meta["build"]["observe_seconds"] == 0.0
