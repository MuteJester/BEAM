"""The --wrap targets cannot lose their link flag quietly.

Two suites interpose on the allocator: test_model fails one allocation on
demand, so the out-of-memory rollback paths are exercised rather than
assumed, and test_expected_counts_accounting sums what the allocator handed
out, because nothing else can catch an allocation that is simply never
charged. Both depend on a target-specific `-Wl,--wrap=` in the Makefile.

Drop that flag and the binary still links, still runs and still passes --
with __wrap_malloc sitting there as dead code, testing nothing. That is the
worst shape a defect can take, so it gets two guards: the link itself, and
the Makefile.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
SRC_DIR = ROOT / "tests" / "native"


def _wrap_sources():
    found = []
    for f in sorted(SRC_DIR.glob("test_*.c")):
        text = f.read_text(encoding="utf-8")
        wrapped = set(re.findall(r"^\w[\w \*]*\b__wrap_(\w+)\s*\(", text,
                                 re.MULTILINE))
        if wrapped:
            found.append((f, wrapped))
    assert found, "no --wrap suites found; this guard is watching nothing"
    return found


def test_every_wrapping_suite_names_its_flag_in_the_makefile():
    """A target-specific LDEXTRA per wrapping suite, naming every symbol it
    wraps. --wrap redirects EVERY call in the linked objects, so the flag
    cannot be global: a binary that does not define __wrap_malloc would fail
    to link."""
    # Line continuations joined first: a target-specific LDEXTRA spanning
    # three physical lines is one logical line, and a regex that reads the
    # raw text stops at the first backslash and silently misses the rest.
    makefile = re.sub(r"\\\n\s*", " ",
                      (ROOT / "Makefile").read_text(encoding="utf-8"))
    for source, wrapped in _wrap_sources():
        # Spelled with $(BUILD), not `build`: the directory became a
        # variable so that `make tsan` can build instrumented binaries
        # without overwriting the ones the gate reads.
        target = f"$(BUILD)/tests/native/{source.stem}"
        block = re.search(re.escape(target) + r":[^\n]*?LDEXTRA([^\n]*)",
                          makefile)
        assert block, f"{source.name} wraps {sorted(wrapped)} but no "\
                      f"LDEXTRA line names {target}"
        for symbol in wrapped:
            assert re.search(rf"--wrap=(?:\w+,)*{symbol}\b", block.group(1)), \
                f"{source.name} defines __wrap_{symbol}, which {target}'s "\
                f"LDEXTRA does not pass: {block.group(1).strip()}"


def test_without_the_flag_the_link_fails_rather_than_passing_vacuously(tmp_path):
    """The reason the Makefile guard above has teeth: these suites call
    __real_malloc, which only exists when the flag is given. Verified by
    building one without it."""
    source, _ = _wrap_sources()[0]
    if not (ROOT / "build" / "libbeam.a").exists():
        pytest.skip("build/libbeam.a absent; run make first")
    r = subprocess.run(
        ["cc", "-O0", "-std=c11", "-mpopcnt", "-Iinclude",
         "-Isrc/native/internal", "-Itests/native", "-Itests/harness",
         "-fopenmp", str(source), "tests/harness/beam_test.c",
         "build/libbeam.a", "-lm", "-fopenmp",
         "-o", str(tmp_path / "unwrapped")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode != 0, \
        f"{source.name} linked WITHOUT --wrap: its interposition is dead code"
    assert "__real_" in r.stderr, r.stderr
