"""What each failure looks like from Python, pinned before it is restructured.

This file exists to be a safety net, not a specification. The error system is
about to grow a real exception hierarchy, and the single most dangerous thing
that can happen is a failure quietly changing its Python class: any
`except ValueError` in a user's pipeline stops catching, and nothing announces
it.

That is not hypothetical. Making `BEAM_ERR_IO_CORRUPT` and
`BEAM_ERR_IO_VERSION` reachable -- they had been dead codes -- silently
converted three `beam_model_load` failures from OSError to ValueError, because
the switch mapping them had never run. Three existing tests caught it. This
file is the generalisation of those three tests to every failure that can be
reached from Python.

WHAT IS PINNED, AND WHY IT IS PINNED PER ENTRY POINT

Not per error code. One code does not imply one class, and a table that
assumed it would would reintroduce exactly the bug above. `BEAM_ERR_NO_LIVE_PATHS`
reaches Python three different ways:

    positional_marginals(clamp=...)   ValueError          (the binding intercepts)
    BeamQuery.marginals(...)          ImpossibleCondition (the facade narrows)
    log_clamp_probability(...)        no exception at all, -inf

All three are deliberate, all three are observable, and a restructure that
routes every code through one table would change two of them. So the unit
pinned here is (entry point -> class), and the special cases below pin the
disagreements explicitly so they cannot be "tidied up" by accident.

Every trigger was derived by running it, not by reading the switch.
"""
from __future__ import annotations

import os
import shutil
import struct

import pytest

import BEAM
from BEAM import _cbeam as _c

SEQS = ["CASSLF", "CASSQF", "CASRLF", "CTTGWF", "CASSLG", "CASSQG"]

#: Arguments to the raw `_cbeam.new`, which yields an UNFINALIZED model --
#: the only way to reach the not-built guards from Python.
NEW_ARGS = (2, 7, 8, (0.6, 0.25, 0.15), (0.0, 0.0, 0.0),
            0.0, 0.0, (0.5, 0.5), True, True, "aa")


@pytest.fixture(scope="module")
def model():
    return BEAM.BeamModel(SEQS, order=2, max_length=8, strict=True)


@pytest.fixture(scope="module")
def saved(model, tmp_path_factory):
    p = str(tmp_path_factory.mktemp("contract") / "mdl")
    model.save(p)
    return p + BEAM.BeamModel.SUFFIX


@pytest.fixture
def starved():
    """Run the body under a budget nothing fits, then restore."""
    b = BEAM.memory_budget()
    BEAM.set_memory_budget(1024)
    yield
    BEAM.set_memory_budget(b)


def _damaged(saved, tmp_path, name, offset, data=None, truncate=None):
    p = str(tmp_path / name)
    shutil.copy(saved, p)
    with open(p, "r+b") as fh:
        if truncate is not None:
            fh.truncate(truncate)
        else:
            fh.seek(offset)
            fh.write(data)
    return p


# ── The contract, one row per reachable entry point ────────────────────────
#
# (label, callable taking (model, saved, tmp_path), expected class, fragment)
#
# The fragment is checked so a row cannot pass by raising the right class for
# the wrong reason -- a bare ValueError from argument parsing would otherwise
# satisfy a ValueError row.

CONTRACT = [
    ("build over budget",
     lambda m, s, t: BEAM.BeamModel(SEQS, order=2, max_length=8, strict=True),
     MemoryError, "memory budget", True),
    ("load over budget",
     lambda m, s, t: _c.load(s),
     MemoryError, "memory budget", True),
    ("positional_marginals over budget",
     lambda m, s, t: m.positional_marginals(6),
     MemoryError, "memory budget", True),

    ("prism omega out of range",
     lambda m, s, t: _c.prism_explain(m._g, "CASSLF", 0),
     ValueError, "omega", False),
    ("finalize with nothing observed",
     lambda m, s, t: _c.finalize(_c.new(*NEW_ARGS)),
     ValueError, "no usable sequence", False),
    ("explain an unreadable sequence",
     lambda m, s, t: _c.explain(m._g, "CXSSLF"),
     ValueError, "the sequence is empty", False),
    ("facade explain, same input",
     lambda m, s, t: m.explain("CXSSLF"),
     ValueError, "does not accept", False),
    ("topology length out of range",
     lambda m, s, t: _c.topology(m._g, 99),
     ValueError, "max_length", False),
    ("sample length not generated",
     lambda m, s, t: _c.sample(m._g, 1, 99, 0),
     ValueError, "not one this model generates", False),

    ("entropy on an unfinalized model",
     lambda m, s, t: _c.entropy(_c.new(*NEW_ARGS)),
     RuntimeError, "not finalized", False),
    ("path_count_total on an unfinalized model",
     lambda m, s, t: _c.path_count_total(_c.new(*NEW_ARGS)),
     RuntimeError, "not finalized", False),
    ("subtract across alphabets",
     lambda m, s, t: _c.subtract(
         m._g, BEAM.BeamModel(["ACGT", "ACGA"], alphabet="nt",
                              order=2, max_length=8)._g),
     RuntimeError, "mismatch", False),

    ("load a missing file",
     lambda m, s, t: _c.load("/no/such/file.beam"),
     OSError, "cannot open", False),
    ("load a truncated file",
     lambda m, s, t: _c.load(_damaged(s, t, "trunc.beam", 0, truncate=200)),
     OSError, "ends before", False),
    ("load a file with a bad checksum",
     lambda m, s, t: _c.load(_damaged(s, t, "crc.beam",
                                      os.path.getsize(s) - 40, b"\xff" * 8)),
     OSError, "checksum", False),
    ("load an unknown format",
     lambda m, s, t: _c.load(_damaged(s, t, "ver.beam", 4,
                                      struct.pack("<I", 99))),
     OSError, "does not read", False),
    ("load a non-model file",
     lambda m, s, t: _c.load(_damaged(s, t, "magic.beam", 0, b"ZZZZ")),
     OSError, "not a BEAM model file", False),
    ("save to a full device",
     lambda m, s, t: _c.save(m._g, "/dev/full"),
     OSError, "incomplete", False),
]


@pytest.mark.parametrize(
    "label,call,cls,fragment,needs_starving",
    CONTRACT, ids=[c[0] for c in CONTRACT])
def test_the_observed_exception_class_is_unchanged(
        model, saved, tmp_path, label, call, cls, fragment, needs_starving):
    """Pin (entry point -> class), the thing a restructure can silently move.

    `type(exc) is cls` is deliberately NOT asserted: the hierarchy will replace
    these with subclasses that keep the built-in as a base, and that is the
    change this file is meant to permit. What must not change is whether an
    existing `except <builtin>` still catches -- which is `isinstance`.
    """
    budget = BEAM.memory_budget()
    if needs_starving:
        BEAM.set_memory_budget(1024)
    try:
        with pytest.raises(cls) as exc:
            call(model, saved, tmp_path)
    finally:
        BEAM.set_memory_budget(budget)

    text = str(exc.value)
    assert text, f"{label}: raised {cls.__name__} with an EMPTY message"
    assert fragment in text, (
        f"{label}: expected {fragment!r} in the message, got {text!r}")


# ── The disagreements, pinned so they cannot be tidied away ────────────────

def test_one_error_code_reaches_python_three_different_ways(model, tmp_path):
    """NO_LIVE_PATHS is a ValueError, an ImpossibleCondition, and a value.

    All three are deliberate. A code-to-class table that assumed one code means
    one class would change two of them, which is precisely how the IO_CORRUPT
    regression happened. If a restructure makes these agree, that has to be a
    decision someone took, not a side effect -- so it fails here first.
    """
    with pytest.raises(ValueError) as a:          # the binding intercepts
        model.positional_marginals(6, clamp={0: "W"})
    assert "cannot produce" in str(a.value)

    q = BEAM.BeamQuery(model)
    from BEAM.query.errors import ImpossibleCondition
    with pytest.raises(ImpossibleCondition):      # the facade narrows
        q.marginals(q.condition(6).at(0, "W"))

    # ...and the same condition asked as a probability is an ANSWER, not a
    # failure: P of an impossible event is zero. This must not start raising.
    assert model.log_clamp_probability(6, {0: "W"}) == float("-inf")


def test_the_query_facade_narrows_builtins_to_its_own_classes(model):
    """`BeamQuery` returns subclasses of the built-ins, not the built-ins.

    So `except MemoryError` and `except QueryTooLarge` both work today. A
    restructure must keep both true; dropping the built-in base is the silent
    break.
    """
    from BEAM.query.errors import QueryTooLarge
    q = BEAM.BeamQuery(model)
    cond = q.condition(6)

    b = BEAM.memory_budget()
    BEAM.set_memory_budget(1024)
    try:
        with pytest.raises(QueryTooLarge) as exc:
            q.marginals(cond)
    finally:
        BEAM.set_memory_budget(b)
    assert isinstance(exc.value, MemoryError), (
        "QueryTooLarge must remain catchable as MemoryError")


def test_a_load_failure_is_not_always_an_OSError(model, saved):
    """Documented because it is surprising, and because I asserted otherwise.

    A previous commit claimed "every `beam_model_load` failure is an OSError"
    while re-pointing two codes to preserve it. Every FILE-shaped failure is --
    but a budget refusal during load is a MemoryError, so the invariant was
    already false. Pinned so the next person reads the truth rather than the
    claim.
    """
    b = BEAM.memory_budget()
    BEAM.set_memory_budget(1024)
    try:
        with pytest.raises(MemoryError) as exc:
            _c.load(saved)
    finally:
        BEAM.set_memory_budget(b)
    assert not isinstance(exc.value, OSError)


def _budget_exceeded():
    from BEAM.errors import BudgetExceeded
    return BudgetExceeded


def test_no_allocation_failure_escapes_the_hierarchy(model):
    """Every allocation failure is a BeamError. This used to be a known gap.

    `PyErr_NoMemory()` raises MemoryError with an empty string, and because the
    hierarchy is reached by name from BEAM.errors, such a failure was NOT a
    BeamError -- `except BeamError` missed it. errors.py carried a note about
    that, and this test used to PIN the gap rather than forbid it: it asserted
    a bare MemoryError, and was moved from site to site as guards closed them.

    There is nowhere left to move it to. Every `return PyErr_NoMemory()` in the
    bindings is gone -- 22 became `beam_py_nomem`, which raises `OutOfMemory`
    with the thing and its size, and the last one was a short-circuit in
    `set_beam_error` that predated `OutOfMemory` existing. So the test now
    asserts the property instead of the gap, and reads the source to say so:
    a reintroduced bare site fails it before anyone has to trigger one.
    """
    from pathlib import Path
    import BEAM as _b
    root = Path(_b.__file__).resolve().parents[2] / "src" / "native" / "bindings"
    if not root.is_dir():                     # installed, not a source tree
        pytest.skip("no binding sources beside the package")
    bare = []
    for f in sorted(root.rglob("*.c")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "PyErr_NoMemory()" in line and not line.lstrip().startswith("*"):
                bare.append(f"{f.name}:{i}")
    assert bare == [], (
        "these raise a bare MemoryError, which is not a BeamError and carries "
        f"no message: {bare}")


def test_the_guarded_sites_now_say_what_went_wrong(model):
    """The counterpart: where a guard exists, the refusal is diagnosable.

    A bare MemoryError names no quantity the caller controls. These name the
    request, the bytes, and the limit -- which is the difference between an
    error a user can act on and one they can only report.
    """
    for call in (lambda: _c.top_k(model._g, 4_000_000_000, 5),
                 lambda: _c.sample(model._g, 4_000_000_000, 5, 1)):
        with pytest.raises(_budget_exceeded()) as exc:
            call()
        msg = str(exc.value)
        assert "4000000000" in msg and "budget" in msg, msg
