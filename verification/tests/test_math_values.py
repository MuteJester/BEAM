"""The math-value baseline, and whether it is actually a net.

A recorded baseline that cannot fail is worse than none: it reports OK for
ever and licenses the change that broke something. So these tests do not only
check that the capture runs. They check that it notices a change of one unit
in the last place, that it says the same thing twice, that it says the same
thing at one thread and at four, and that it covers every public math method
rather than the subset somebody remembered to list.

Everything here uses ONE order-2 model, because the mechanism under test is
the capture, not the model: determinism, one-ULP sensitivity, coverage and
thread invariance hold on any model that is not degenerate. The baseline
itself is captured from the orders in `MODEL_ORDERS`, and an order-3 capture
costs twelve times an order-2 one, which would buy this file nothing.

The three model-shape tests at the end pin behaviour that the baseline cannot
see, because the baseline is captured from one freshly built model: a
compacted model, a model read back from disk, and one through `pickle`.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from BEAM import BeamModel

from math_values import (_encode, build_model, capture, capture_model,
                         captured_method_names, differences,
                         public_math_methods)

#: Built once. Rebuilding per test would dominate the file's runtime and
#: prove nothing a single model does not.
MODEL = build_model(2)
TAG = "t"
BEFORE = capture_model(MODEL, TAG)


def test_the_encoding_distinguishes_one_ulp():
    """The property every other guarantee rests on, at its smallest."""
    x = 1.2345678901234567
    assert _encode(x) != _encode(float(np.nextafter(x, np.inf)))


def test_the_encoding_distinguishes_one_ulp_inside_an_array():
    """Arrays are hashed, so one changed element must change the hash."""
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    b[1] = np.nextafter(b[1], np.inf)
    assert _encode(a) != _encode(b)


def test_the_encoding_separates_values_a_tolerance_would_merge():
    """A near-equal pair must not compare equal: this is not a tolerance."""
    assert _encode(1.0) != _encode(1.0 + 2.220446049250313e-16)


def test_capture_is_deterministic():
    """Two captures of the same model agree exactly."""
    assert capture_model(MODEL, TAG) == BEFORE


def test_capture_notices_a_one_ulp_change(monkeypatch):
    """End to end: a single `nextafter` inside a method must not pass.

    A tolerance-based comparison would wave this through, which is the whole
    reason the baseline stores exact bytes.
    """
    original = BeamModel.entropy
    monkeypatch.setattr(
        BeamModel, "entropy",
        lambda self, **kw: float(np.nextafter(original(self, **kw), np.inf)))
    after = capture_model(MODEL, TAG)

    moved = differences(after, BEFORE)
    assert moved, "a one-ULP change produced an identical capture"
    assert any(k.rpartition("/")[2].startswith("entropy") for k in moved)


def test_capture_covers_every_public_math_method():
    """A math method added to the facade and never captured fails here."""
    assert public_math_methods() - captured_method_names() == set()


def test_capture_names_no_method_that_does_not_exist():
    """And a captured name that stops existing fails here."""
    assert captured_method_names() - public_math_methods() == set()


def test_capture_is_thread_invariant():
    """One thread and four must agree, or the net false-alarms under load."""
    assert capture_model(MODEL, TAG, threads=1) == \
           capture_model(MODEL, TAG, threads=4)


def test_a_compacted_model_reports_the_same_math(tmp_path):
    """`compact()` releases the raw counts; the tables are what math reads."""
    clone = BeamModel.from_file(_saved(MODEL, tmp_path, "compact_src"))
    clone.compact()
    assert capture_model(clone, TAG) == BEFORE


def test_a_model_read_back_from_disk_reports_the_same_math(tmp_path):
    """A loaded model is a different capsule and must answer identically."""
    reloaded = BeamModel.from_file(_saved(MODEL, tmp_path, "roundtrip"))
    assert capture_model(reloaded, TAG) == BEFORE


def test_a_model_cannot_be_pickled():
    """Pinning what is true, which is not what you would assume.

    A model holds its tables in a C capsule and defines no `__reduce__`, so
    `pickle` refuses it outright. That matters to the extraction only as the
    answer to "can moving these functions break pickling" -- it cannot,
    because there is nothing working to break -- but the fact is worth a
    test, so that making models picklable is a deliberate act with a visible
    failure here rather than a silent change of behaviour.

    `save`/`from_file` is the supported round-trip, and the two tests above
    check that it preserves every math value.
    """
    # Matched on `PyCapsule` rather than on CPython's phrasing of "cannot
    # pickle": the capsule is the fact about BEAM, the wording around it is
    # the interpreter's and has changed before.
    with pytest.raises(TypeError, match="PyCapsule"):
        pickle.dumps(MODEL)


def _saved(model, tmp_path, name):
    """`model` on disk, returned as the path `from_file` wants."""
    stem = tmp_path / name
    model.save(str(stem))
    return str(stem.with_suffix(".beam"))
