"""Pieces of the gate that are worth testing on their own.

`check.py` is a script: it runs its checks at import time, so it cannot be
imported to be tested. What it decides, rather than what it prints, lives
here.
"""
from __future__ import annotations


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
