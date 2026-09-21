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
