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
