"""Every test that existed before the reorganisation still exists after it.

A reorganisation that moves every test id at once is exactly when the gate is
least able to help: the inventory it compares changes wholesale, and the
obvious response -- re-record it -- would absorb a dropped test in silence.
That is anti-goal 1 of this work, and `MOVES.txt` is the answer to it. Each
line is one `old_id -> new_id`, the mapping is checked here rather than
trusted, and an id with no line is taken to map to itself.

What is checked, and why each one matters:

  * TOTAL on the old inventory -- every id that existed before the move has
    somewhere to be. An id with no image is a test that was deleted while
    the diff was too large to read.
  * INJECTIVE -- no two old ids share one new id. Two tests collapsing into
    one is a drop that a count of the new inventory cannot see, because the
    count is right.
  * Every image is COLLECTED TODAY -- the mapping describes the suite that
    exists, not one somebody meant to produce.

Surjectivity -- that every id collected today is accounted for -- is the
gate's own job and it does it on every run: an id that nothing recorded
appears under `added` and fails `pytest inventory` until someone records it
deliberately. Asserting it here as well would freeze this test against the
inventory of one afternoon, and the next test anyone adds would fail it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import check_lib

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "verification" / "baseline"
MOVES = BASELINE / "MOVES.txt"

#: The inventory as it stood BEFORE the move. The recorder preserves the file
#: it replaces under the tag it was given, which is what makes the old side of
#: this mapping something on disk rather than something remembered.
OLD_INVENTORY = BASELINE / "pytest_cases.presuites.txt"


def _mapping():
    pairs = {}
    for number, line in enumerate(
            MOVES.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        old, arrow, new = line.partition(" -> ")
        assert arrow, f"{MOVES.name}:{number}: not an `old -> new` line: {line}"
        pairs[old.strip()] = new.strip()
    return pairs


def _recorded_old():
    return check_lib.collect_ids(OLD_INVENTORY.read_text(encoding="utf-8"))


def _collected_now():
    out = subprocess.run(check_lib.python_collect_argv(ROOT), cwd=ROOT,
                         capture_output=True, text=True,
                         encoding="utf-8").stdout
    return check_lib.collect_ids(out)


def test_every_old_id_maps_to_exactly_one_new_id():
    old = _recorded_old()
    moves = _mapping()

    unknown = sorted(set(moves) - set(old))
    assert unknown == [], "MOVES.txt maps ids that were never in the inventory"

    mapping = {identifier: moves.get(identifier, identifier)
               for identifier in old}
    assert len(mapping) == len(set(old)), "the old inventory repeats an id"
    assert len(set(mapping.values())) == len(mapping), \
        "the mapping is not injective: two tests share one new id"

    lost = sorted(set(mapping.values()) - set(_collected_now()))
    assert lost == [], "ids the mapping promises that nothing collects"


def test_the_mapping_actually_moved_something():
    """A mapping that moves nothing proves nothing, and this file exists
    because a move happened."""
    moves = _mapping()
    assert moves, f"{MOVES.name} has no `old -> new` lines"
    assert any(old != new for old, new in moves.items())
