"""A change to a test header must rebuild the suites compiled against it.

The Makefile opens by saying header dependency tracking "is not optional":
without it, "changing a struct in a header leaves objects compiled against
the old layout ... That has been mistaken for undefined behaviour once and
for a compiler defect once; both cost hours." The library objects have it
(`-MMD -MP`). The test binaries did not. They are compiled and linked in one
command, no `.d` file is written for them, and their only prerequisites were
the `.c` file, the harness source and the library -- so editing
`tests/native/test_utils.h` rebuilt nothing, and the suites went on running
against the header they were built with.

That was survivable while the gate's staleness check could see the file. It
stopped being survivable when the C oracle moved to `tests/harness/oracles/`,
which neither the Makefile nor the check was looking at: a header with no
witness at all.

`make -q` settles it without building anything -- it exits non-zero when a
target is out of date and runs no command.
"""
import marshal
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every header a C suite is compiled against.
HEADERS = ["tests/native/test_utils.h",
           "tests/harness/beam_test.h",
           "tests/harness/oracles/prior_reference.h"]

#: The cheapest suite to ask about. Any of the 35 would answer the same.
SUITE = "build/tests/native/test_vtrig"


def _out_of_date(target):
    return subprocess.run(["make", "-q", target], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").returncode != 0


@pytest.mark.parametrize("header", HEADERS)
def test_touching_a_test_header_puts_a_suite_out_of_date(header):
    if not (ROOT / SUITE).exists():
        pytest.skip(f"{SUITE} is not built; run make test first")
    if _out_of_date(SUITE):
        pytest.skip("the suite is already out of date, so this proves nothing")
    path = ROOT / header
    before = path.stat()
    try:
        os.utime(path, None)
        assert _out_of_date(SUITE), header
    finally:
        # Restore the timestamp: the gate reads it to decide whether a binary
        # is answering for source it was not built from, and a test that left
        # the tree looking stale would be reporting its own footprint.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


# ── The same hazard on the Python side ─────────────────────────────────────
#
# The C half has a witness for a binary built from source it no longer
# matches: `check_lib.c_inventory` refuses a binary older than its sources.
# The Python half has the same failure in a quieter form. `git mv` renames a
# directory on disk, carrying the untracked `__pycache__` inside it, and a
# `.pyc` is revalidated against its source's mtime and size -- neither of
# which a rename changes. So the interpreter goes on loading bytecode whose
# embedded `co_filename` is the path the file had BEFORE the move, and every
# traceback, skip location and coverage path names a directory that no longer
# exists. That is exactly what happened when tests/python became
# tests/suites, and it was visible only as an odd path in a skip line.

def _stale_bytecode():
    stale = []
    for cache in (ROOT / "tests").rglob("__pycache__"):
        for compiled in sorted(cache.glob("*.pyc")):
            try:
                code = marshal.loads(compiled.read_bytes()[16:])
            except (ValueError, EOFError):     # a truncated or foreign .pyc
                continue
            source = cache.parent / (compiled.name.split(".")[0] + ".py")
            if Path(code.co_filename) != source:
                stale.append(f"{compiled.relative_to(ROOT).as_posix()} was compiled from "
                             f"{code.co_filename}")
    return stale


def test_no_cached_bytecode_remembers_a_path_the_file_no_longer_has():
    stale = _stale_bytecode()
    assert stale == [], (
        "stale bytecode under tests/: delete it with\n"
        "    find tests -name __pycache__ -type d -exec rm -rf {} +\n"
        + "\n".join(stale[:10]))
