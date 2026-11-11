"""The C inventory, read from the binaries instead of from their source.

The gate used to find the C test names by grepping `RUN_TEST(` out of
`tests/native/*.c`. That reads the source, not the program. A RUN_TEST line
that `main` never reaches, or one the preprocessor drops, counts there as a
test that ran; so does a name in a file that no longer builds. These tests
pin the other reading -- every binary states its own inventory when asked --
and then require the two to agree, which is the only check that can catch a
test the source advertises and the program does not run.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

import check_lib

ROOT = Path(__file__).resolve().parents[1].parent
BIN_DIR = ROOT / "build" / "tests" / "native"
SRC_DIR = ROOT / "tests" / "native"


def _binaries():
    found = sorted(p for p in BIN_DIR.glob("test_*") if p.is_file())
    if not found:
        pytest.skip(f"no C test binaries under {BIN_DIR}; run `make test` first")
    return found


def _list(binary):
    """The inventory one binary reports, as a list of names in its own order."""
    r = subprocess.run([str(binary), "--list"], capture_output=True, text=True)
    assert r.returncode == 0, f"{binary.name} --list exited {r.returncode}: {r.stderr}"
    return r.stdout.split()


def _list_all_binaries():
    return [name for b in _binaries() for name in _list(b)]


def _all_native_c_text():
    return "\n".join(f.read_text(encoding="utf-8") for f in sorted(SRC_DIR.glob("*.c")))


def test_every_binary_lists_its_tests():
    out = subprocess.run([str(BIN_DIR / "test_alphabet"), "--list"],
                         capture_output=True, text=True, check=True).stdout
    names = out.split()
    assert "test_a_nucleotide_model_builds_and_scores" in names
    assert all(n.startswith("test_") for n in names), out
    assert "PASS" not in out            # it listed, it did not run


def test_a_run_test_line_main_never_calls_is_not_counted():
    """The grep over source could not tell. Execution can."""
    listed = sorted(_list_all_binaries())
    grepped = sorted(re.findall(r"RUN_TEST\((\w+)\)", _all_native_c_text()))
    assert listed == grepped, "source and execution disagree"


def test_one_test_can_be_run_by_name():
    out = subprocess.run([str(BIN_DIR / "test_alphabet"), "--run",
                          "test_lower_case_scores_as_upper_case"],
                         capture_output=True, text=True).stdout
    assert out.count("PASS") == 1
    assert "test_N_is_a_residue_and_other_codes_are_not" not in out


def test_a_name_this_binary_does_not_have_is_an_error():
    """Otherwise a selection typo runs nothing and exits 0, which reads as a
    suite that passed."""
    r = subprocess.run([str(BIN_DIR / "test_alphabet"), "--run",
                        "test_that_was_renamed_last_week"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "NO SUCH TEST: test_that_was_renamed_last_week" in r.stderr


def test_every_binary_contributes_to_the_inventory():
    """What `make list-tests` prints, and what the gate reads.

    The counts are not written down here: the baseline owns the total, and a
    second copy of it would have to be edited by anyone adding a test. What
    this pins is that no binary is silently absent from the listing -- one
    that lists nothing subtracts its whole file from the gate's view while
    still exiting 0.
    """
    built = {b.name for b in _binaries()}
    sources = {f.stem for f in SRC_DIR.glob("test_*.c")}
    assert sources - built == set(), f"never built: {sorted(sources - built)}"
    empty = sorted(b.name for b in _binaries() if not _list(b))
    assert not empty, f"binaries that listed nothing: {empty}"


# ── what the gate does when the binaries cannot answer ─────────────────────
# Reading the inventory from the binaries buys accuracy and costs a
# dependency on the build being current. These pin the costs: every one of
# them used to be impossible (the regex could not fail) and must now be
# reported by name rather than subtracted from the inventory.

def _tree(tmp_path, *, listing="test_one\n", rc=0, stale=False, built=True,
          touched_header=None):
    """A stand-in for the repo: one source, one executable, three headers.

    `touched_header` is a path relative to `tests/`, given a modification
    time in the future so that the binary is older than it.
    """
    src = tmp_path / "tests" / "native"
    src.mkdir(parents=True)
    (src / "test_fake.c").write_text("int main(void) { }\n", encoding="utf-8")
    for header in ("native/test_utils.h", "harness/beam_test.h",
                   "harness/oracles/prior_reference.h"):
        path = tmp_path / "tests" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("/* a header every suite is compiled against */\n",
                        encoding="utf-8")
        if header == touched_header:
            os.utime(path, (2 ** 31, 2 ** 31))
    binaries = tmp_path / "build" / "tests" / "native"
    binaries.mkdir(parents=True)
    if built:
        b = binaries / "test_fake"
        b.write_text(f"#!/bin/sh\nprintf '%s' '{listing}'\nexit {rc}\n",
                     encoding="utf-8")
        b.chmod(0o755)
        if stale:
            os.utime(b, (1, 1))
    return tmp_path


def test_a_healthy_tree_yields_the_names(tmp_path):
    names, problems = check_lib.c_inventory(_tree(tmp_path))
    assert (names, problems) == (["test_one"], [])


def test_a_source_that_was_never_built_is_named(tmp_path):
    names, problems = check_lib.c_inventory(_tree(tmp_path, built=False))
    assert names == []
    assert problems == ["test_fake: not built; run make test"]


def test_a_binary_older_than_its_source_is_named(tmp_path):
    """The hazard the regex did not have: a stale binary answers for source
    that no longer compiles."""
    names, problems = check_lib.c_inventory(_tree(tmp_path, stale=True))
    assert names == []
    assert problems == ["test_fake: binary is older than its source; run make test"]


def test_a_binary_that_cannot_list_is_named(tmp_path):
    names, problems = check_lib.c_inventory(_tree(tmp_path, rc=3))
    assert names == []
    assert problems == ["test_fake: --list exited 3"]


def test_a_binary_that_lists_nothing_is_named(tmp_path):
    names, problems = check_lib.c_inventory(_tree(tmp_path, listing=""))
    assert names == []
    assert problems == ["test_fake: listed no tests"]


def test_recording_is_refused_while_a_binary_cannot_be_asked():
    """--record with a short inventory is the one unrecoverable move: it
    writes the missing tests out of the baseline, and nothing afterwards can
    tell they were ever there."""
    with pytest.raises(SystemExit) as raised:
        check_lib.refuse_short_c_inventory(["test_fake: not built; run make test"])
    assert raised.value.code == 2


def test_recording_proceeds_when_every_binary_answered():
    assert check_lib.refuse_short_c_inventory([]) is None


# ── Which headers count as a source of every binary ────────────────────────
#
# A binary older than a header it was compiled against answers for a
# mechanism it does not have. The check globbed `tests/native/*.h`, which was
# the whole story while the harness lived in `test_utils.h`; the harness is
# now `tests/harness/beam_test.{h,c}` and the C oracle
# `tests/harness/oracles/prior_reference.h`, and neither was looked at.

@pytest.mark.parametrize("header", ["native/test_utils.h",
                                    "harness/beam_test.h",
                                    "harness/oracles/prior_reference.h"])
def test_a_binary_older_than_any_test_header_is_named(tmp_path, header):
    names, problems = check_lib.c_inventory(
        _tree(tmp_path, touched_header=header))
    assert names == []
    assert problems == [
        "test_fake: binary is older than its source; run make test"], header
