"""A failure must not arrive as a value, or as somebody else's diagnosis.

Every case here was a real defect: the library reported a failure in a form the
caller could not distinguish from a real answer, or attached a diagnosis that
belonged to an unrelated earlier call. Two of them were worse than a crash,
because the wrong answer was *specific and actionable* and would have been
acted on:

  - saving to a full disk raised "beam_model_path_count: length 0 is not one
    this model generates", because the write path returned a bare code and the
    thread-local message buffer still held the previous failure;
  - a perfectly good model file that merely did not fit the memory budget was
    reported as "not a model file, truncated, or corrupt", sending the reader
    to rebuild a file that was never damaged.

The tests are written against BEHAVIOUR a user can observe, not against the
implementation, so they keep their meaning if the internals move.
"""
from __future__ import annotations

import os
import shutil

import pytest

import BEAM
from BEAM import _cbeam as _c

SEQS = ["CASSLF", "CASSQF", "CASRLF", "CTTGWF", "CASSLG", "CASSQG"]

#: The prior's strengths as the raw `_cbeam` entry points take them: one per
#: level 0..order, and the length prior's.
ALPHA, ALPHA_LENGTH = (10.0,) * 3, 10.0


@pytest.fixture(scope="module")
def model():
    return BEAM.BeamModel(SEQS, order=2, max_length=8)


@pytest.fixture
def saved(model, tmp_path_factory):
    d = tmp_path_factory.mktemp("io")
    p = str(d / "ok")
    model.save(p)
    return p + BEAM.BeamModel.SUFFIX


@pytest.fixture
def restore_budget():
    b = BEAM.memory_budget()
    yield
    BEAM.set_memory_budget(b)


# ── save ───────────────────────────────────────────────────────────────────

def test_a_write_failure_does_not_report_an_earlier_unrelated_error(model):
    """The message must describe THIS failure.

    `beam_model_save` ended in `return ok ? BEAM_OK : BEAM_ERR_IO_WRITE`, a
    bare code. The binding then read `beam_error_message()`, a thread-local
    buffer nothing had cleared, and served whatever the last failure left
    there. The test primes that buffer deliberately, so a regression restores
    the exact symptom rather than merely a vague one.
    """
    try:                                    # prime the buffer with a real error
        model.path_count(0)
    except Exception:
        pass

    with pytest.raises(Exception) as exc:
        _c.save(model._g, "/dev/full")      # opens fine, cannot be written
    text = str(exc.value)

    assert "path_count" not in text, f"stale diagnosis resurfaced: {text}"
    assert "save" in text
    assert "incomplete" in text, f"must warn the file is unusable: {text}"


# ── load ───────────────────────────────────────────────────────────────────

def test_load_distinguishes_its_failures(saved, tmp_path):
    """Nine causes had one sentence between them.

    Each returned NULL without setting an error, so the binding hard-coded
    "not a model file, truncated, or corrupt" for all of them.
    """
    trunc = str(tmp_path / "trunc.beam")
    shutil.copy(saved, trunc)
    with open(trunc, "r+b") as fh:
        fh.truncate(os.path.getsize(trunc) // 2)

    bad = str(tmp_path / "bad.beam")
    shutil.copy(saved, bad)
    with open(bad, "r+b") as fh:            # corrupt the payload, not the header
        fh.seek(os.path.getsize(bad) - 40)
        fh.write(b"\xff" * 8)

    cases = {
        "/nonexistent/dir/x.beam": "cannot open",
        "/etc/hostname":           "not a BEAM model file",
        trunc:                     "ends before",
        bad:                       "checksum",
    }
    seen = {}
    for path, fragment in cases.items():
        with pytest.raises(Exception) as exc:
            BEAM.BeamModel.from_file(path)
        text = str(exc.value)
        assert fragment in text, f"{path}: expected {fragment!r}, got {text!r}"
        seen[path] = text

    # And they must not all be the same sentence, which was the defect.
    assert len(set(seen.values())) == len(seen)


def test_a_good_file_that_exceeds_the_budget_is_not_called_corrupt(
        saved, restore_budget):
    """The remedy is `set_memory_budget`, so the message must say so.

    `beam_model_new` and `beam_model_finalize` both set precise errors --
    including the budget refusal -- and `load` discarded them, then the binding
    replaced whatever was left with a report about corruption. A user told
    their file is corrupt rebuilds it; the file was never damaged.
    """
    BEAM.set_memory_budget(2048)
    with pytest.raises(Exception) as exc:
        BEAM.BeamModel.from_file(saved)
    text = str(exc.value)

    assert "corrupt" not in text, f"a sound file was called corrupt: {text}"
    assert "memory budget" in text
    assert "set_memory_budget" in text

    BEAM.set_memory_budget(0)                             # 0 restores the default
    assert BEAM.BeamModel.from_file(saved).n_states > 0    # same file, fine


def test_a_budget_refusal_reports_a_figure_in_a_readable_unit(restore_budget):
    """Three refusal sites never adopted the unit scaling and read '0.0 GB'."""
    BEAM.set_memory_budget(2048)
    with pytest.raises(Exception) as exc:
        BEAM.BeamModel(SEQS, order=2, max_length=8)
    text = str(exc.value)
    assert "0.0 GB" not in text, f"unit did not scale: {text}"
    assert any(u in text for u in ("bytes", "kB", "MB", "GB")), text


# ── sentinel-returning functions ───────────────────────────────────────────

def _unfinalized():
    """A model with no sequences observed and no finalize called."""
    return _c.new(2, 8, (0.5, 0.5), ALPHA, ALPHA_LENGTH, True, "aa")


def test_path_count_total_refuses_an_unfinalized_model(model):
    """0.0 is a legitimate total, so it cannot also mean "not built".

    `path_count` set the error on this branch and `path_count_total`, twelve
    lines below in the same file, did not -- so the binding guard could not
    fire and the 0.0 read as "this model generates nothing".
    """
    g = _unfinalized()
    assert _c.is_finalized(g) is False

    with pytest.raises(Exception) as a:
        _c.path_count_total(g)
    with pytest.raises(Exception) as b:
        _c.path_count(g, 5)
    for exc in (a, b):
        assert "not finalized" in str(exc.value)


def test_the_length_accessors_separate_an_error_from_an_answer(model):
    """-inf is the true worth of a length the model does not generate.

    So the error conditions have to be carried by the error context, not by
    the same value. Only the out-of-range case changed; both -inf answers
    below must survive. The prior gives every length from 1 to max_length a
    share, so the empty sequence is the one in-range length left to ask.
    """
    g = model._g
    assert _c.log_partition(g, 0) == float("-inf")        # real answer
    assert _c.log_length_prior(g, 0) == float("-inf")     # real answer
    assert _c.log_length_prior(g, 3) > float("-inf")      # never observed

    for L in (200, 4_000_000_000):
        for fn in (_c.log_partition, _c.log_length_prior):
            with pytest.raises(ValueError, match="outside"):
                fn(g, L)


# ── argument validation at the boundary ────────────────────────────────────

def test_a_build_parameter_too_large_is_refused_not_wrapped(model):
    """`order=258` wrapped to a LEGAL order 2 and built a working model.

    That is the worst shape of this defect: the value is silently replaced by
    a plausible one, validation passes, and every score afterwards is wrong
    with nothing to indicate it.
    """
    def build(order=2, maxlen=8):
        # One strength per level, so the only thing wrong is the one under
        # test; a mis-sized alpha is refused with a message of its own.
        return _c.build(SEQS, None, order, maxlen, (0.5, 0.5),
                        (10.0,) * max(order + 1, 0), ALPHA_LENGTH, True, "aa")

    assert _c.config(build())["order"] == 2               # the sane case still works

    for kw in (dict(order=258), dict(maxlen=300), dict(order=-1)):
        with pytest.raises(ValueError, match="0..255"):
            build(**kw)


def test_an_out_of_range_length_is_refused_before_anything_is_sized(model):
    """Sized first, a nonsense length raised an EMPTY MemoryError.

    The allocator's verdict on a multi-gigabyte request replaced the C layer's
    precise "length N is not one this model generates".
    """
    for call in (lambda L: _c.topology(model._g, L),
                 lambda L: _c.positional_marginals(model._g, L, None)):
        with pytest.raises(ValueError, match="max_length"):
            call(2 ** 29)
        # and a length within range still works
    assert _c.topology(model._g, 6) is not None
