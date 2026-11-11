"""Pieces of the gate that are worth testing on their own.

`check.py` is a script: it runs its checks at import time, so it cannot be
imported to be tested. What it decides, rather than what it prints, lives
here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def c_inventory(root: Path) -> tuple[list[str], list[str]]:
    """Every C test name, asked of the binaries that run them.

    Returns (names, problems). `names` is a multiset kept as a list, in the
    order the binaries report it; `problems` is empty on a healthy tree.

    The gate used to read these names with a regex over tests/native/*.c.
    That reads the source rather than the program, so a RUN_TEST line main
    never reaches -- or one the preprocessor drops -- counted as a test that
    ran. Asking each binary cannot make that mistake, but it introduces one
    of its own: a binary left over from before an edit answers for source
    that no longer compiles. So a binary older than its source, or than the
    harness header, is a problem rather than an answer.

    Every problem names the file it is about. An unreadable binary must never
    subtract its tests from the inventory quietly -- the baseline would then
    be re-recorded shorter, which is the one failure this whole gate exists
    to prevent.
    """
    src_dir, bin_dir = root / "tests" / "native", root / "build" / "tests" / "native"
    sources = sorted(src_dir.glob("test_*.c"))
    if not sources:
        return [], [f"no C test sources under {src_dir}"]
    # The harness header is a source of every binary: editing it without
    # rebuilding leaves 35 binaries answering for a mechanism they lack.
    header_mtime = max((h.stat().st_mtime for h in src_dir.glob("*.h")), default=0.0)

    names: list[str] = []
    problems: list[str] = []
    for src in sources:
        binary = bin_dir / src.stem
        if not binary.exists():
            problems.append(f"{src.stem}: not built; run make test")
            continue
        if binary.stat().st_mtime < max(src.stat().st_mtime, header_mtime):
            problems.append(f"{src.stem}: binary is older than its source; run make test")
            continue
        r = subprocess.run([str(binary), "--list"], capture_output=True,
                           text=True, encoding="utf-8")
        if r.returncode != 0:
            problems.append(f"{src.stem}: --list exited {r.returncode}")
            continue
        listed = r.stdout.split()
        if not listed:
            problems.append(f"{src.stem}: listed no tests")
            continue
        names.extend(listed)
    return names, problems


def compare(got: list[str], want: list[str]) -> bool:
    """True when the two inventories hold the same names the same number of times.

    A MULTISET, not a set. The gate compared sets, which collapses a name that
    appears in two files into one entry -- so deleting either copy was
    invisible. Four ids are in that position today.

    Ordering still does not matter, and the reason it must not is unchanged:
    GNU sort and Python order underscores differently depending on locale, so
    comparing sequences would report a difference between two identical
    inventories. That hazard belongs to the FILE, which `sorted()` writes;
    comparing sorted copies here is safe because both sides are sorted by the
    same Python.
    """
    return sorted(got) == sorted(want)


def collect_ids(text: str) -> list[str]:
    """Every pytest id in `text`, whole, in order, duplicates kept.

    A LIST and not a set: two files may hold the same test name -- four ids
    are in that position -- and a set comprehension at the call site was
    collapsing them before the comparison could see them, so deleting either
    copy was invisible.

    The id is kept whole, path included, so that a test moving between files
    is a difference. The gate split on `::` and compared bare names, which
    made a file move invisible too.
    """
    return [l.strip() for l in text.splitlines() if "::" in l]


def python_collect_argv(root: Path) -> list[str]:
    """The command the gate collects the Python inventory with.

    `--rootdir` is explicit, and load-bearing. pytest numbers its ids from
    the rootdir, and the rootdir follows the configuration file: with the
    configuration in `tests/pytest.ini`, where it travels with the suite,
    `pytest tests` reports `python/model/test_x.py::test_y`, and run from the
    repository root with the table in `pyproject.toml` it reported
    `tests/python/model/test_x.py::test_y`. The baseline records WHOLE ids,
    so without this flag moving the configuration would have renamed all
    1,216 of them at once -- a diff indistinguishable from a reorganisation,
    arriving for no reason a reader could see.

    Pinning the rootdir does not suppress the travelling configuration: the
    inifile is still found by searching upward from the arguments, so this
    run reads `tests/pytest.ini` and merely numbers its ids from `root`.
    """
    return [sys.executable, "-m", "pytest", str(root / "tests"), "-q",
            "--collect-only", "--rootdir", str(root)]


def python_inventory(want_text: str, collected_text: str) -> tuple[list[str], list[str]]:
    """The two id lists the gate compares: whole ids, duplicates kept.

    WHOLE, path included, so a test moving between files is a difference. The
    gate split on `::` and compared bare names, which made a file move
    invisible -- and a reorganisation that moves every test is exactly when
    the gate most needs to see one.

    One function rather than two comprehensions at the call site, because the
    property worth testing is that both sides are built the same way.
    """
    return collect_ids(want_text), collect_ids(collected_text)


#: Every baseline `--record` can write, by the name `--only` accepts.
RECORDABLE = frozenset({"python_api", "pytest_cases", "c_cases",
                        "numeric", "threads"})


def record_targets(argv: list[str]) -> set[str]:
    """Which baselines this invocation may write.

    `--record TAG` writes all five, which is what it has always done and
    stays the default. `--only NAME[,NAME...]` narrows it.

    The reason it must be narrowable: `python_api.txt` can only be recorded
    correctly by the interpreter CI checks against, and on this machine the
    only interpreter that can import BEAM at all is Python 3.14 -- where
    recording it would write signatures that do not exist on 3.11. Without
    `--only`, re-recording a test inventory meant overwriting the API list
    too, so the only available recorder was the one that must never run.

    `--only` is a separate flag rather than `--record=NAMES` because
    `--record` already takes the TAG that names the preserved previous copy,
    and an `--record=NAMES` spelling would not even match the `in sys.argv`
    test that reaches this code -- it would skip recording silently.
    """
    if "--only" not in argv:
        return set(RECORDABLE)
    if "--record" not in argv:
        print("\nREFUSING: --only selects which baselines --record writes, "
              "and there is no --record here")
        raise SystemExit(2)
    i = argv.index("--only")
    raw = argv[i + 1] if len(argv) > i + 1 else ""
    names = {n.strip() for n in raw.split(",") if n.strip()}
    unknown = sorted(names - RECORDABLE)
    if not names or unknown:
        print(f"\nREFUSING: --only {raw!r} names {unknown or 'nothing'}; "
              f"it accepts {', '.join(sorted(RECORDABLE))}")
        raise SystemExit(2)
    return names


def refuse_short_c_inventory(problems: list[str]) -> None:
    """Refuse to record a C baseline that is missing a whole binary's tests.

    Every other baseline can be re-recorded wrongly and re-recorded again.
    This one cannot: the names it does not contain are the names nothing
    afterwards knows were ever there. So the gate reports the problem and
    exits rather than write it.
    """
    if not problems:
        return None
    print("\n  REFUSED c_cases.txt: the inventory is short by at least one "
          "binary (named above).")
    print("  Recording it now would delete those tests from the baseline, "
          "and nothing afterwards could tell. Build first: make test")
    raise SystemExit(2)
