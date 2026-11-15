"""Every region that drops the GIL, and whether it resolves a thread plan.

The gate's wiring check used to read two of the fourteen binding files and
print `29 analytics regions planned, no escapes` -- a sentence its reader
takes for a statement about the library, while it was a statement about
`analytics_bindings.c` and `common.c`. The other twelve hold 41 regions that
drop the GIL and plan nothing.

These tests pin the reading that covers all fourteen, and they pin it on
synthetic sources rather than only on the real tree: the real counts move
whenever someone wires a region up, which is the point, so a test that only
asserted them would have to be edited every time it did its job.
"""
from pathlib import Path

import pytest

import check_lib

ROOT = Path(__file__).resolve().parents[2]


def _tree(tmp_path: Path, name: str, body: str) -> Path:
    """A root containing one binding file, laid out where gil_regions looks."""
    d = tmp_path / "src" / "native" / "bindings" / "python"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")
    return tmp_path


# The real tree
# ════════════════════════════════════════════════════════════════════════


def test_the_recorded_baseline_matches_the_source():
    """The baseline is what the gate compares against, so it must be current.

    This is the test that fails when someone adds a GIL-releasing region and
    does not record it -- the same failure the gate reports, caught here
    without a full gate run.
    """
    regions, escapes, unbalanced = check_lib.gil_regions(ROOT)
    recorded = [l for l in (ROOT / "verification" / "baseline" /
                            "gil_regions.txt").read_text(
                                encoding="utf-8").splitlines() if l.strip()]
    assert escapes == []
    assert unbalanced == []
    assert regions == sorted(recorded)


def test_no_region_is_attributed_to_an_unknown_function():
    """`?` means the function-definition scanner lost its place.

    Every identifier in the baseline is `<file>::<function>`, and a `?` there
    would make two unrelated regions share one name -- so one could be
    deleted without the comparison noticing.
    """
    regions, _, _ = check_lib.gil_regions(ROOT)
    assert regions, "no regions found at all; the layout must have moved"
    assert [r for r in regions if "::?" in r] == []


# The parsing rules
# ════════════════════════════════════════════════════════════════════════


def test_a_planned_and_an_unplanned_region_are_told_apart(tmp_path):
    root = _tree(tmp_path, "a.c", """
PyObject *py_planned(PyObject *self) {
    BEAM_PY_PLANNED_BEGIN
    work();
    BEAM_PY_PLANNED_END
}
PyObject *py_bare(PyObject *self) {
    Py_BEGIN_ALLOW_THREADS
    work();
    Py_END_ALLOW_THREADS
}
""")
    regions, escapes, unbalanced = check_lib.gil_regions(root)
    assert regions == ["a.c::py_bare unplanned", "a.c::py_planned planned"]
    assert escapes == [] and unbalanced == []


def test_a_second_region_in_one_function_gets_its_own_identifier(tmp_path):
    """Otherwise two regions collapse into one entry and either can vanish."""
    root = _tree(tmp_path, "a.c", """
PyObject *py_twice(PyObject *self) {
    Py_BEGIN_ALLOW_THREADS
    first();
    Py_END_ALLOW_THREADS
    Py_BEGIN_ALLOW_THREADS
    second();
    Py_END_ALLOW_THREADS
}
""")
    regions, _, _ = check_lib.gil_regions(root)
    assert regions == ["a.c::py_twice unplanned", "a.c::py_twice#1 unplanned"]


def test_a_return_inside_a_planned_region_is_an_escape(tmp_path):
    """A path that skips the leave applies this call's plan to the next one."""
    root = _tree(tmp_path, "a.c", """
PyObject *py_leaky(PyObject *self) {
    BEAM_PY_PLANNED_BEGIN
    if (bad) return NULL;
    BEAM_PY_PLANNED_END
}
""")
    _, escapes, unbalanced = check_lib.gil_regions(root)
    assert len(escapes) == 1
    assert "return NULL" in escapes[0]
    assert unbalanced == []


def test_a_commented_return_is_not_an_escape(tmp_path):
    root = _tree(tmp_path, "a.c", """
PyObject *py_fine(PyObject *self) {
    BEAM_PY_PLANNED_BEGIN
    work();   /* no return here */
    // nor goto here
    BEAM_PY_PLANNED_END
}
""")
    _, escapes, _ = check_lib.gil_regions(root)
    assert escapes == []


def test_an_unclosed_region_is_reported_once_and_blames_its_own_function(
        tmp_path):
    """The regression this test exists for.

    An unbalanced BEGIN used to leave the scanner inside a region for the
    rest of the FILE, so every later `return` became an escape: one mutation
    produced 87 escape lines and never said the macro was unbalanced. The
    fault is one line, and it belongs to the function that opened it.
    """
    root = _tree(tmp_path, "a.c", """
PyObject *py_unclosed(PyObject *self) {
    BEAM_PY_PLANNED_BEGIN
    work();
}
PyObject *py_innocent(PyObject *self) {
    if (a) return NULL;
    if (b) return NULL;
    goto out;
}
""")
    _, escapes, unbalanced = check_lib.gil_regions(root)
    assert len(unbalanced) == 1
    assert "py_unclosed" in unbalanced[0]
    assert "never closed" in unbalanced[0]
    assert escapes == [], f"a later function was blamed: {escapes}"


def test_an_end_without_a_begin_is_reported(tmp_path):
    root = _tree(tmp_path, "a.c", """
PyObject *py_stray(PyObject *self) {
    work();
    BEAM_PY_PLANNED_END
}
""")
    _, _, unbalanced = check_lib.gil_regions(root)
    assert len(unbalanced) == 1
    assert "without being opened" in unbalanced[0]

def test_a_region_named_only_in_a_comment_is_not_a_region(tmp_path):
    """The bug this guard exists for, found by the `::?` test above.

    `mixture_bindings.c` opens with a doc comment that explains what happens
    across "the `Py_BEGIN_ALLOW_THREADS` region". A raw text search counted
    that sentence as a 41st unplanned region, attributed to no function at
    all because it sits above the first one -- the same mistake that once
    recorded `jet_sweep` as public API because a comment mentioned it.
    """
    root = _tree(tmp_path, "a.c", """
/**
 * Prose about the `Py_BEGIN_ALLOW_THREADS` region, and about
 * BEAM_PY_PLANNED_BEGIN as well, spanning several lines.
 */
PyObject *py_real(PyObject *self) {
    Py_BEGIN_ALLOW_THREADS   /* the only real one */
    work();
    Py_END_ALLOW_THREADS
}
// Py_BEGIN_ALLOW_THREADS in a line comment is not one either.
""")
    regions, escapes, unbalanced = check_lib.gil_regions(root)
    assert regions == ["a.c::py_real unplanned"]
    assert escapes == [] and unbalanced == []
