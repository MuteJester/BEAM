#!/usr/bin/env python3
"""Exact values for every public math method, recorded as a gate baseline.

The numeric baseline in `numeric.json` pins four of these quantities. The
other twenty-two were covered only by tests that assert a property -- that a
Hill profile decreases, that a transform agrees with a sum over the support --
and a property holds either side of a change that moves the number. Moving
code between modules can move a number: a different import order, a different
accumulation order, a `self` that became a `model`. This file is what makes
that visible.

Every float is recorded as `float.hex()`, the same exact spelling the numeric
block uses, so the comparison is bit-for-bit and a one-ULP drift fails. Arrays
are hashed over their raw bytes rather than listed, because a positional
spectrum at 1,024 bins would otherwise be most of the baseline.

The arguments below are part of the recorded contract. Changing one changes
every value it produces, which is a re-record, not a bug fix.

Run directly:
    verification/math_values.py --check            compare against the baseline
    verification/math_values.py --record TAG       write it, keeping the old copy
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "verification" / "baseline" / "math_values.json"
CORPUS = ROOT / "verification" / "baseline" / "corpus.json"

#: The corpus the gate already records, reused rather than reinvented: one
#: fixture, one source of truth, and no seeded RNG whose stream could change
#: with the interpreter.
#:
#: Orders 2 and 3, not 2 and 4. Order 4 over the 20-residue alphabet is
#: 160,000 contexts, and capturing it costs 212 s against 0.9 s for order 2
#: and 11.3 s for order 3 (measured, threads pinned). That is a gate run
#: nobody would wait for, and the fourth order pins nothing the third does
#: not: both exercise the backoff path that order 2 never reaches.
MODEL_ORDERS = (2, 3)
MAX_LENGTH = 15

#: Threads to hold the capture at.
#:
#: Not a performance preference -- a correctness-of-measurement one. On a
#: machine whose load exceeds its core count, running with threads = cores
#: makes every parallel region wait for its whole team to be scheduled:
#: measured here, one `spectrum_pair` on an 80-sequence model took 1.5 s at
#: 16 threads and 0.00 s at 8, five times each, which turned a 29-call
#: capture into 13 s of scheduling latency. Values do not depend on this --
#: `set_threads` documents bit-for-bit agreement across thread counts, and
#: `test_capture_is_thread_invariant` checks it rather than trusting it.
CAPTURE_THREADS = 4

ORDERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
COMPLEX_ORDERS = (0.5 + 0.0j, 1.0 + 0.25j, 2.0 - 0.5j)
BINS = 64
SPECTRA_MAX_LENGTH = 8

#: Every call the baseline holds. A name before `:` is the method it
#: exercises; the part after it distinguishes two calls of the same method.
CALL_NAMES = (
    "entropy", "entropy:base2",
    "hill_number:q1", "hill_number:q2",
    "hill_numbers", "hill_profile", "hill_curve",
    "diversity_profile", "effective_diversity",
    "log_mellin:q1", "log_mellin:q2", "log_mellin_many",
    "log_mellin_complex", "log_mellin_line",
    "cgf", "cgf_jet", "length_moments",
    "surprisal_moments", "surprisal_cumulants", "surprisal_stats",
    "surprisal_range", "surprisal_by_depth", "surprisal_distribution",
    "surprisal_pdf", "surprisal_cdf",
    "saddlepoint",
    "spectrum", "spectrum_pair", "spectra_by_length",
)

_MATH = re.compile(
    r"mellin|saddle|moment|cumulant|cgf|spectr|surprisal|entropy|hill|diversity",
    re.I)


def public_math_methods() -> set[str]:
    """The public math methods `BeamModel` actually has, by introspection.

    Asked of the class rather than listed here, so a method added to the
    facade and never captured is a test failure rather than a silent gap.
    """
    from BEAM import BeamModel
    return {n for n in dir(BeamModel)
            if not n.startswith("_") and _MATH.search(n)
            and callable(getattr(BeamModel, n, None))}


def captured_method_names() -> set[str]:
    """The methods `CALL_NAMES` exercises."""
    return {n.partition(":")[0] for n in CALL_NAMES}


def _encode(value):
    """A value as something exact, comparable and JSON-serialisable."""
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {"shape": list(a.shape), "dtype": str(a.dtype),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return repr(value)
    if isinstance(value, float):
        return float(value).hex()
    if isinstance(value, complex):
        return [float(value.real).hex(), float(value.imag).hex()]
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return f"<unencodable {type(value).__name__}>"


def model_calls(model) -> dict:
    """`CALL_NAMES` mapped to thunks over `model`.

    `x` for the three pointwise calls is taken from this model's own
    surprisal range, so the points are always inside the support whatever
    the model.
    """
    low, high = model.surprisal_range()
    xs = [low + (high - low) * f for f in (0.1, 0.5, 0.9)]
    calls = {
        "entropy": lambda: model.entropy(),
        "entropy:base2": lambda: model.entropy(base=2.0),
        "hill_number:q1": lambda: model.hill_number(1.0),
        "hill_number:q2": lambda: model.hill_number(2.0),
        "hill_numbers": lambda: model.hill_numbers(ORDERS),
        "hill_profile": lambda: model.hill_profile(ORDERS),
        "hill_curve": lambda: model.hill_curve(ORDERS),
        "diversity_profile": lambda: model.diversity_profile(),
        "effective_diversity": lambda: model.effective_diversity(),
        "log_mellin:q1": lambda: model.log_mellin(1.0),
        "log_mellin:q2": lambda: model.log_mellin(2.0),
        "log_mellin_many": lambda: model.log_mellin_many(ORDERS),
        "log_mellin_complex": lambda: model.log_mellin_complex(COMPLEX_ORDERS),
        "log_mellin_line": lambda: model.log_mellin_line(1.0, 0.0, 0.25, 8),
        "cgf": lambda: model.cgf(0.5),
        "cgf_jet": lambda: model.cgf_jet(0.5),
        "length_moments": lambda: model.length_moments(q=1.0, order=4),
        "surprisal_moments": lambda: model.surprisal_moments(q=1.0, order=4),
        "surprisal_cumulants": lambda: model.surprisal_cumulants(q=1.0, order=4),
        "surprisal_stats": lambda: model.surprisal_stats(q=1.0),
        "surprisal_range": lambda: model.surprisal_range(),
        "surprisal_by_depth": lambda: model.surprisal_by_depth(q=1.0),
        "surprisal_distribution": lambda: model.surprisal_distribution(),
        "surprisal_pdf": lambda: model.surprisal_pdf(xs),
        "surprisal_cdf": lambda: model.surprisal_cdf(xs),
        "saddlepoint": lambda: model.saddlepoint(xs[1]),
        "spectrum": lambda: model.spectrum(bins=BINS),
        "spectrum_pair": lambda: model.spectrum_pair(bins=BINS),
        "spectra_by_length": lambda: model.spectra_by_length(
            bins=BINS, max_length=SPECTRA_MAX_LENGTH),
    }
    assert set(calls) == set(CALL_NAMES), "CALL_NAMES and model_calls disagree"
    return calls


@contextlib.contextmanager
def pinned_threads(n: int = CAPTURE_THREADS):
    """Hold the process-wide thread cap at `n`, then put it back.

    `set_threads` is global, and this runs inside a gate that checks thread
    independence afterwards, so the previous setting is restored even if the
    body raises. `threads()` reports 0 for "whatever OpenMP chooses", and
    `set_threads(0)` restores exactly that.
    """
    import BEAM
    before = BEAM.threads()
    BEAM.set_threads(n)
    try:
        yield
    finally:
        BEAM.set_threads(before)


def _corpus(root: Path):
    path = (CORPUS if root == ROOT
            else root / "verification" / "baseline" / "corpus.json")
    c = json.loads(path.read_text(encoding="utf-8"))
    return c["seqs"], c["ab"]


def build_model(order: int, root: Path = ROOT):
    """One model of `order` from the recorded corpus."""
    import BEAM
    seqs, ab = _corpus(root)
    with pinned_threads():
        return BEAM.BeamModel(seqs, abundances=ab, order=order,
                              max_length=MAX_LENGTH)


def build_models(root: Path = ROOT):
    """`[(tag, model)]` from the recorded corpus, one per order."""
    return [(f"o{order}", build_model(order, root)) for order in MODEL_ORDERS]


def capture_model(model, tag: str, threads: int = CAPTURE_THREADS) -> dict:
    """Every call against one model, keyed `tag/call`.

    A call that raises is recorded as the exception it raised rather than
    skipped: "this raises here" is a fact about the library worth pinning,
    and a silently dropped key would shorten the baseline.
    """
    out = {}
    with pinned_threads(threads):
        for name, thunk in model_calls(model).items():
            try:
                out[f"{tag}/{name}"] = _encode(thunk())
            except Exception as exc:                   # noqa: BLE001 - recorded
                out[f"{tag}/{name}"] = f"<raised {type(exc).__name__}: {exc}>"
    return out


def capture(root: Path = ROOT, models=None,
            threads: int = CAPTURE_THREADS) -> dict:
    """The whole baseline: every call against every model."""
    out = {}
    for tag, model in (models if models is not None else build_models(root)):
        out.update(capture_model(model, tag, threads))
    return out


def differences(got: dict, want: dict) -> list[str]:
    """Every key whose value moved, missing and extra keys included."""
    return sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))


def _serialise(values: dict) -> str:
    return json.dumps(values, indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if "--record" in argv:
        i = argv.index("--record")
        tag = argv[i + 1] if len(argv) > i + 1 else ""
        if not tag or tag.startswith("-"):
            print("REFUSING: --record needs a TAG naming the previous copy")
            return 2
        previous = BASELINE.with_name(f"math_values.{tag}.json")
        if BASELINE.exists():
            if previous.exists():
                print(f"REFUSED: {previous.name} already exists")
                return 2
            previous.write_bytes(BASELINE.read_bytes())
        BASELINE.write_text(_serialise(capture()), encoding="utf-8")
        n = len(json.loads(BASELINE.read_text(encoding="utf-8")))
        print(f"recorded math_values.json ({n} values)"
              + (f"  (previous kept as {previous.name})" if previous.exists() else ""))
        return 0

    if "--check" in argv:
        if not BASELINE.exists():
            print("no math_values.json baseline; record one first")
            return 2
        want = json.loads(BASELINE.read_text(encoding="utf-8"))
        got = capture()
        bad = differences(got, want)
        if bad:
            print(f"MATH VALUES CHANGED: {len(bad)}/{len(got)} keys differ")
            for k in bad[:8]:
                print(f"  {k}")
            return 1
        print(f"math values unchanged ({len(got)} values)")
        return 0

    print(__doc__.strip().splitlines()[-3].strip())
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
