"""Pieces of the gate that are worth testing on their own.

`check.py` is a script: it runs its checks at import time, so it cannot be
imported to be tested. What it decides, rather than what it prints, lives
here.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
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
    # Every test header is a source of every binary: editing one without
    # rebuilding leaves 35 binaries answering for a mechanism they lack. All
    # three locations, not just tests/native/*.h -- the harness moved to
    # tests/harness/ and the C oracle to tests/harness/oracles/, and a header
    # this check cannot see is a stale binary it cannot report.
    headers = list(src_dir.glob("*.h"))
    headers += list((root / "tests" / "harness").rglob("*.h"))
    header_mtime = max((h.stat().st_mtime for h in headers), default=0.0)

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


def inventory_difference(got: list[str], want: list[str]) -> tuple[list[str], list[str]]:
    """What is missing and what is extra, counting DUPLICATES.

    `compare` is a multiset test, so deleting one of two same-named tests is
    a difference. The report under the FAIL was a membership test -- `[x for
    x in want if x not in got]` -- which for that one case printed nothing:
    the name is still in `got`, once. A FAIL with no `missing` and no `added`
    line beneath it is the thing most likely to end in a re-record, which is
    the failure the whole gate exists to prevent.

    Both lists are sorted, and an id lost twice appears twice.
    """
    missing = Counter(want) - Counter(got)
    extra = Counter(got) - Counter(want)
    return sorted(missing.elements()), sorted(extra.elements())


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
    the rootdir, and the rootdir follows the configuration file. The
    configuration travels inside the suite now, so without this flag
    `pytest tests` reports every id relative to the suite directory and the
    `tests/` prefix disappears from all 1,216 of them at once -- a
    whole-baseline rename, arriving for no reason a reader could see, on a
    branch whose point is that the gate can see a move. The baseline records
    WHOLE ids anchored at the repository root; this keeps them there.

    (This paragraph was itself a casualty of the move: the rename swept the
    example it used, so the "before" and the "after" ended up spelled the
    same way. It now describes the mechanism instead of two sample ids.)

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


def _strip_c_comments(line: str, in_block: bool) -> tuple[str, bool]:
    """The code on one line, with comments removed, tracking `/* */` across lines.

    Needed because the region scan is a text search, and this file's own
    history says why a raw text search is not enough: a prose mention of
    `Py_BEGIN_ALLOW_THREADS` in the doc comment at the top of
    mixture_bindings.c was counted as a region that drops the GIL, exactly as
    `jet_sweep` once matched a comment and was recorded as public API.
    """
    out, i = [], 0
    while i < len(line):
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out), True
            in_block, i = False, j + 2
            continue
        j, k = line.find("/*", i), line.find("//", i)
        if j < 0 and k < 0:
            out.append(line[i:])
            break
        if k >= 0 and (j < 0 or k < j):
            out.append(line[i:k])
            break
        out.append(line[i:j])
        i, in_block = j + 2, True
    return "".join(out), in_block


#: Lines at column 0 that define a function. `#define FOO(x)` is excluded by
#: the leading-# test at the call site; `static PyMethodDef x[] = {` cannot
#: match because the regex requires `(` immediately after the name.
_C_FUNC = re.compile(r"^(?:static\s+)?[A-Za-z_][\w\s\*]*?\**\s*([A-Za-z_]\w*)\s*\(")


def gil_regions(root: Path) -> tuple[list[str], list[str]]:
    """Every region that drops the GIL, and whether it resolves a thread plan.

    Returns (regions, escapes, unbalanced). Each region is one line,
    `<file>::<function>[#N] <planned|unplanned>`, and the pair is what the
    baseline holds.

    WHY THIS READS ALL FOURTEEN BINDING FILES. It used to read two, named in
    a `PLANNED_BINDINGS` tuple, and print `29 analytics regions planned, no
    escapes` -- a sentence a reader takes for a statement about the library.
    It was a statement about analytics_bindings.c and common.c. The other
    twelve files hold 41 regions that drop the GIL and plan nothing, so
    `beam_threads()` falls through to `threads_cap()`, which with no cap set
    is `omp_get_max_threads()`: one thread per core, the default plan.h
    exists to stop asking for. Measured on a 16-core box with 15 free, an
    analytics call plans 8 and `py_observe` runs 16; on the saturated
    48-core node it asks for 48, which the design notes measure at 301x
    worse than serial at half that.
    
    That is a DEFERRAL, not a defect -- DESIGN.md decision 9 reads "No.
    Query-time calls only, for now; fitting keeps today's behaviour until it
    is profiled" -- so the 41 are recorded rather than failed. What the
    recording buys is the thing a count alone cannot: a NEW unplanned region
    fails, so the deferral cannot quietly grow, and wiring one up shows as a
    note that asks to be re-recorded. The lesson is c_symbols.txt, which had
    no recorder at all and drifted by thirty symbols across four merges.

    Identified by function rather than by line, because a line-numbered
    baseline churns on every unrelated edit above it and teaches its reader
    to re-record without reading.
    """
    bind = root / "src" / "native" / "bindings" / "python"
    regions: list[str] = []
    escapes: list[str] = []
    unbalanced: list[str] = []
    for path in sorted(bind.glob("*.c")):
        func, seen = "?", {}
        open_at, open_in = None, ""      # line and function of an unclosed BEGIN
        in_block = False
        for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # Everything below reads `code`, never the raw line.
            code, in_block = _strip_c_comments(raw, in_block)
            if code[:1] not in ("", " ", "\t", "#", "}"):
                m = _C_FUNC.match(code)
                if m:
                    # A region still open at the next function's definition is
                    # never closed. Reported here, and `open_at` cleared, so
                    # that one unbalanced macro implicates its own function
                    # rather than every `return` in the rest of the file -- a
                    # mutation of exactly this shape produced 87 escape lines
                    # and no statement of the actual fault.
                    if open_at is not None:
                        unbalanced.append(
                            f"{path.name}:{open_at}: planned region in "
                            f"{open_in} is never closed")
                        open_at = None
                    func = m.group(1)
            if "BEAM_PY_PLANNED_BEGIN" in code or "Py_BEGIN_ALLOW_THREADS" in code:
                planned = ("planned" if "BEAM_PY_PLANNED_BEGIN" in code
                           else "unplanned")
                n = seen.get(func, 0)
                seen[func] = n + 1
                regions.append(f"{path.name}::{func}{f'#{n}' if n else ''} {planned}")
                if planned == "planned":
                    if open_at is not None:
                        unbalanced.append(
                            f"{path.name}:{open_at}: planned region in "
                            f"{open_in} is never closed")
                    open_at, open_in = i, func
                continue
            if "BEAM_PY_PLANNED_END" in code:
                if open_at is None:
                    unbalanced.append(f"{path.name}:{i}: a planned region is "
                                      f"closed in {func} without being opened")
                open_at = None
                continue
            # A `return` or `goto` out of a planned region skips the leave,
            # which would apply one call's thread count to the next, unrelated
            # call. The macro's safety rests on there being no such path.
            if open_at is not None and ("return" in code or "goto" in code):
                escapes.append(f"{path.name}:{i}: {code.strip()}")
        if open_at is not None:
            unbalanced.append(f"{path.name}:{open_at}: planned region in "
                              f"{open_in} is never closed")
    return sorted(regions), escapes, unbalanced

#: Every baseline `--record` can write, by the name `--only` accepts.
RECORDABLE = frozenset({"python_api", "pytest_cases", "c_cases",
                        "numeric", "threads", "math_values", "gil_regions",
                        })


def missing_baseline_is_fatal(name: str, argv: list[str]) -> bool:
    """Whether a baseline file that is not on disk fails this run.

    It does, for every run but the one writing it. The gate's other checks
    get this for free by reading their baseline unconditionally and dying on
    a missing file, which is correct: a check that cannot find what it
    compares against has not passed, it has not run.

    A check that prints a note instead -- so the first recording of a new
    baseline is not blocked by its own absence -- needs that exemption
    bounded, or a tree where the file was never checked out reports a clean
    gate with the whole comparison silently gone. That is the one failure
    mode a recorded baseline exists to prevent.
    """
    if "--record" not in argv:
        return True
    return name not in record_targets(argv)


def record_targets(argv: list[str]) -> set[str]:
    """Which baselines this invocation may write.

    `--record TAG` writes all of them, which is what it has always done and
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
