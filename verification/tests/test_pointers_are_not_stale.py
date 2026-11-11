"""A path named in a comment has to exist.

This branch renamed `tests/python` to `tests/suites` and swept the references
by grepping for the string. Three times that sweep missed one, each time
because the reference was spelled differently from the grep: `from beam_model
import ...` where the grep said `import beam_model`, `root / "tests" /
"python"` where it said `tests/python`, and four pointers in `.c` and `.h`
files where the grep looked only at `.py`, `.md`, `.ini`, `.toml` and the
Makefile. One of those four is in a PUBLIC header.

A grep for a spelling finds what you thought of. This finds what is there:
every `tests/...` path mentioned in a source file, checked against the disk.
`verification/check.py` already does the same for `.c` pointers in headers
(its `header pointers` check); this covers the other direction.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Where a stale pointer would mislead someone: sources and the files that
#: describe them. Not the baselines or REORGANIZATION.md, which are dated
#: records of states that no longer exist and are supposed to name them.
SEARCHED = ("src", "include", "tests", "tools", "verification", "docs/src")
SKIPPED = ("verification/baseline", "verification/REORGANIZATION.md",
           "__pycache__", ".venv",
           # These tests invent ids like `tests/a.py` as DATA, and the real
           # paths they name -- the file a probe runs, the suite a copy
           # collects -- fail loudly by themselves when they go stale, which
           # is how two of the three misses above were caught.
           "verification/tests")

SUFFIXES = {".c", ".h", ".py", ".md", ".ini", ".toml", ".cfg", ".txt", ".yml"}

#: A path under tests/ with no glob or placeholder in it: `tests/native/*.h`
#: and `tests/suites/<domain>/` are patterns, not pointers.
# Not `\btests/`: that matches the tail of "build/tests/roundtrip.beam"
# and of "verification/tests/x.py", neither of which is a pointer into
# the suite.
POINTER = re.compile(r"(?<![\w/])tests/[A-Za-z0-9_./-]+")


def _files():
    for where in SEARCHED:
        base = ROOT / where
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = str(path.relative_to(ROOT))
            if (path.is_file() and path.suffix in SUFFIXES
                    and not any(skip in rel for skip in SKIPPED)):
                yield path


def test_every_path_under_tests_that_a_source_names_exists():
    stale = []
    for path in sorted(_files()):
        for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for hit in POINTER.findall(line):
                # `tests/harness/beam_test.{h,c}` names two files, not one.
                if f"{hit}{{" in line:
                    continue
                pointer = hit.rstrip(".,;:)`'\"")
                if not (ROOT / pointer).exists():
                    stale.append(f"{path.relative_to(ROOT)}:{number}: {pointer}")
    assert stale == [], "\n".join(stale)
