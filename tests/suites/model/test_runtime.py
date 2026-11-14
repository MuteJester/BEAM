"""Thread controls and parallel determinism.
"""
from __future__ import annotations

import importlib.util
import math
import os
import random
import warnings

import numpy as np
import pytest

from BEAM import BeamModel

from _engineering.corpora import AA, J_MOTIFS, make_corpus, ANALYSIS_BINS

# Parallelism
#
# The cap is process-wide, so every test here restores it. The property that
# makes it safe to use is that results do not depend on it: every reduction
# is order-independent or summed in a fixed index order.
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def restore_threads():
    from BEAM import threads, set_threads
    before = threads()
    yield
    set_threads(before)


def test_thread_cap_round_trips(model, restore_threads):
    from BEAM import effective_threads, threads, set_threads
    set_threads(2)
    assert threads() == 2
    assert model.threads == 2
    assert 1 <= effective_threads() <= 2
    assert model.effective_threads == effective_threads()

    model.threads = 1
    assert threads() == 1
    assert model.effective_threads == 1

    set_threads(0)
    assert threads() == 0
    assert model.effective_threads >= 1


def test_thread_cap_treats_negative_as_default(model, restore_threads):
    from BEAM import threads
    model.threads = -4
    assert threads() == 0


def test_results_do_not_depend_on_the_thread_count(model, corpus,
                                                   restore_threads):
    """The whole point of a cap being safe to set: pinning it for a benchmark
    must change the timing and not the answer. Both the batch scorer and an
    analytic that reduces over the trellis are checked."""
    seqs, _ = corpus
    probe = seqs[:300]

    model.threads = 1
    one_scores = model.log_pseq_batch(probe)
    one_entropy = model.entropy()
    one_mellin = model.log_mellin(2.0)

    model.threads = 4
    assert np.array_equal(model.log_pseq_batch(probe), one_scores)
    assert model.entropy() == one_entropy
    assert model.log_mellin(2.0) == one_mellin


def test_atoms_feed_the_publicness_model(small):
    """The atom pair is what a sharing prediction consumes, so it has to be
    accepted by `PublicnessModel.expected_counts` as it comes off the
    spectrum — no reshaping, no dtype surgery at the call site."""
    from BEAM import PublicnessModel

    p, c = small.pseq_analysis().atoms(ANALYSIS_BINS).as_arrays()
    depths = np.arange(100, 400, dtype=np.int64)
    pb = PublicnessModel(depths, depth_bins=8)
    out = pb.expected_counts(p, c, levels=np.array([1, 5, len(depths) + 1]))
    counts = np.asarray(out["expected_counts"], dtype=np.float64)
    assert counts.size == 2
    assert np.all(np.isfinite(counts)) and np.all(counts >= 0)


# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# The thread plan, and being able to ask why
# ════════════════════════════════════════════════════════════════════════


def test_the_plan_reports_every_input_it_used():
    """A user who sees an unexpectedly slow call needs to be able to ask.

    The plan's ANSWER on its own does not help: the surprises in this
    subsystem were never the thread count, they were which of the three
    sources produced it. So every input is reported beside it.
    """
    import BEAM

    plan = BEAM.parallel_plan()
    for key in ("plan", "user_cap", "omp_num_threads", "cores",
                "available", "max_width"):
        assert key in plan, f"{key} missing from {sorted(plan)}"
    assert plan["plan"] >= 1
    assert plan["cores"] >= 1
    assert 1 <= plan["available"] <= plan["cores"]


def test_an_explicit_cap_is_what_the_plan_reports():
    """`set_threads(n)` means n, and the probe has to say so, because this is
    the contract a pinned benchmark and the paper's reproduction rely on."""
    import BEAM

    before = BEAM.threads()
    try:
        BEAM.set_threads(3)
        plan = BEAM.parallel_plan()
        assert plan["user_cap"] == 3
        assert plan["plan"] == 3
    finally:
        BEAM.set_threads(before)


def test_the_plan_is_capped_by_the_scaling_ceiling():
    """Auto mode never plans wider than where the work stops scaling, which
    is the measured upper limit rather than the machine's size."""
    import BEAM

    before = BEAM.threads()
    try:
        BEAM.set_threads(0)
        plan = BEAM.parallel_plan()
        assert plan["plan"] <= plan["max_width"]
        assert plan["plan"] <= max(plan["available"], 1)
    finally:
        BEAM.set_threads(before)


def test_a_pinned_availability_is_visible_in_the_plan(monkeypatch):
    """`BEAM_AVAILABLE_CORES` is how a benchmark or a CI run buys back the
    reproducibility that a load-aware plan costs, so it must be observable."""
    import BEAM

    monkeypatch.setenv("BEAM_AVAILABLE_CORES", "2")
    before = BEAM.threads()
    try:
        BEAM.set_threads(0)
        plan = BEAM.parallel_plan()
        assert plan["available"] == 2
        assert plan["plan"] == 2
    finally:
        BEAM.set_threads(before)


def test_available_cores_never_exceeds_the_cores_present():
    """The estimate narrows what the machine has; it can never widen it."""
    import BEAM

    assert 1 <= BEAM.available_cores() <= BEAM.parallel_plan()["cores"]


# Forking
#
# The guard exists because the failure it reports is a hang: no exception, no
# traceback, nothing for a user to search for. These tests never let a child
# reach a parallel region, which is the thing that would hang -- each one
# exits immediately after reading what the handler did.
# ════════════════════════════════════════════════════════════════════════


def test_the_fork_warning_names_the_hang_and_the_remedy():
    """The message has to answer "why has my program stopped" by itself.

    Asserted on content rather than on being non-empty: a warning that fires
    in the right place and does not say `spawn` leaves the reader exactly
    where they were.
    """
    from BEAM import runtime
    before = runtime._FORK_THREADS
    runtime._FORK_THREADS = 8
    try:
        with pytest.warns(RuntimeWarning) as caught:
            runtime._after_fork_in_child()
    finally:
        runtime._FORK_THREADS = before
    message = str(caught[0].message)
    assert "hang" in message
    assert "spawn" in message
    assert "set_threads" in message
    assert "8" in message              # the count it actually observed


def test_a_fork_with_nothing_running_is_silent():
    """A fork before any parallel work is safe, and must not be warned about.

    This is what keeps the guard usable: a library that warns on every fork
    teaches people to filter the category, which silences it for the case it
    was written for.
    """
    from BEAM import runtime
    before = runtime._FORK_THREADS
    runtime._FORK_THREADS = 1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runtime._after_fork_in_child()
    finally:
        runtime._FORK_THREADS = before
    # Asserted on the RECORDED list, not by turning warnings into errors:
    # `_after_fork_in_child` swallows exceptions on purpose, so an error
    # raised by a filter would be caught there and this test would pass
    # against a guard that warns on every fork. Mutating the predicate to
    # `> 0` is what showed it.
    assert caught == [], f"warned with nothing running: {[str(c.message) for c in caught]}"


def test_an_unreadable_thread_count_is_one_rather_than_a_guess(monkeypatch):
    """Reporting 1 makes the guard silent on a platform without /proc.

    The alternative -- assuming the worst -- would warn on every fork of every
    process on such a platform, which is the failure above.
    """
    from BEAM import runtime

    def _no_proc(*a, **k):
        raise OSError("no /proc here")

    monkeypatch.setattr("builtins.open", _no_proc)
    assert runtime._native_thread_count() == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_a_real_fork_after_parallel_work_warns_in_the_child(model,
                                                            restore_threads):
    """End to end, which is the only way to prove the handler is REGISTERED.

    The three tests above call `_after_fork_in_child` directly, so all of them
    would still pass if `os.register_at_fork` were never reached -- and that
    one line is the whole mechanism.
    """
    from BEAM import runtime, set_threads
    set_threads(4)
    model.entropy()          # any analytic starts a team, which is the hazard
    if runtime._native_thread_count() < 2:
        pytest.skip("no worker pool on this build, so there is nothing to warn "
                    "about and the test would pass without testing anything")

    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pid = os.fork()
        if pid == 0:
            # Forked from a threaded process: anything that takes a lock may
            # deadlock here, so this does one syscall and leaves. It never
            # returns into pytest.
            try:
                os.write(write_fd, b"1" if caught else b"0")
            finally:
                os._exit(0)
        os.close(write_fd)
        answer = os.read(read_fd, 1)
        os.waitpid(pid, 0)
    os.close(read_fd)
    assert answer == b"1", "the child was not warned that it had forked"
