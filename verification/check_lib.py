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
