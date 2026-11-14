"""Process-wide thread and memory budget controls.
"""
from __future__ import annotations

import os
import warnings

from . import _cbeam as _c
from ._simulation_result import SimulationResult


def set_threads(n: int) -> None:
    """Cap the threads BeamModel uses, **process-wide**.

    A library that silently grabs every core is unusable inside a job
    scheduler or alongside another parallel stage, so this is a hard cap
    rather than a hint: every parallel region in the BeamModel implementation runs
    with at most ``n`` threads.

    ``n = 0`` restores the default, which is whatever the OpenMP runtime would
    choose on its own. Negative values are treated as 0. A build without
    OpenMP ignores the setting and always reports 1.

    Results do NOT depend on it: every reduction in the implementation is
    either order-independent or summed in a fixed index order, so a
    one-thread run and a sixteen-thread run agree bit for bit. That is what
    makes pinning threads for a runtime comparison a fair thing to do.

    The setting is global and not synchronised. Set it before starting work,
    not from one thread while another is inside a computation.
    """
    _c.set_threads(int(n))


def threads() -> int:
    """The cap in force, or 0 if none is. The CONFIGURED value."""
    return int(_c.threads())


def effective_threads() -> int:
    """The thread cap in force. The value a run can be REPRODUCED with.

    The cap if one is set, otherwise the OpenMP runtime's own maximum, and
    always 1 in a build without OpenMP.

    This used to be called "threads a parallel region would actually start",
    and it is deliberately not that: a planned region may run narrower than
    this on a busy machine, because BEAM decides thread counts per call. What
    this reports does not move with the machine, which is why it is the figure
    to pass to :func:`set_threads` to reproduce a timing.
    """
    return int(_c.effective_threads())


def parallel_plan() -> dict:
    """How many threads the next call will use, and every input behind it.

    Keys: ``plan`` (threads a call would start now), ``user_cap`` (0 when
    unset), ``omp_num_threads`` (whether the caller set it), ``cores``,
    ``available`` (free cores, estimated) and ``max_width`` (the widest team
    auto mode will plan, where the work stops scaling).

    The inputs matter more than the answer. A surprising thread count is
    almost always surprising because of WHICH of them decided it: an explicit
    cap, an ``OMP_NUM_THREADS`` in the environment, a busy machine, or the
    ceiling. This reports all four so the question is answerable rather than
    guessable.

    Pin ``BEAM_AVAILABLE_CORES`` to make the plan reproducible, which is what
    a benchmark or a CI run wants.
    """
    return dict(_c.parallel_plan())


def available_cores() -> int:
    """Free cores right now, as the thread policy estimates them.

    Affinity and any cgroup CPU quota, narrowed by the count of runnable
    tasks. At least 1, never more than the cores present. ``BEAM_AVAILABLE_CORES``
    overrides it.
    """
    return int(_c.available_cores())


def set_memory_budget(nbytes: int) -> None:
    """The most scratch BEAM will hold at once for one operation.

    Enforced two ways. Lengths that run concurrently **divide** it, so the
    thread count cannot multiply the footprint: one length's histogram grids
    run to 1.6 GB at order 4 with 512 bins, and running one per core would ask
    for twenty-five gigabytes. And a single program that does not fit is
    **refused before it allocates**, naming itself, the figure it needed and
    this setter.

    So lowering this **can** turn a working call into a failing one — that is
    what makes it a budget rather than a suggestion. One forward-backward
    sweep is 6.6 GB on a nucleotide model at the alphabet's ceiling order, and
    a limit that cannot refuse it protects nothing.

    The default is 4 GB; ``0`` restores it. **Results never depend on it**: it
    changes what is refused and how much runs at once, never an answer.
    """
    _c.set_memory_budget(int(nbytes))


def memory_budget() -> int:
    """The scratch budget in force, in bytes."""
    return int(_c.memory_budget())


def file_provenance_span() -> tuple[int, int]:
    """Where a saved model keeps its build timestamps: ``(offset, bytes)``.

    Every other byte of a ``.beam`` file is decided by the sequences, the
    weights and the config, so building the same model twice writes the same
    file. The three provenance fields are the exception by construction, since
    they read a clock, and the trailing four-byte checksum covers them and so
    moves with them.

    This exists so a byte-for-byte comparison of two saved models can skip
    exactly those and nothing else, taking the window from the format rather
    than hardcoding an offset that a later field would move. It is a property
    of the file format, not of any particular model.
    """
    off, nbytes = _c.file_provenance_span()
    return int(off), int(nbytes)


# ── Forking ────────────────────────────────────────────────────────────────
#
# BEAM's OpenMP runtime does not survive `fork`. Once any parallel region has
# run, the runtime holds a pool of worker threads, and `fork` copies only the
# thread that called it: the child's first parallel computation then waits on
# workers that do not exist there and hangs -- no exception, no traceback, no
# exit. That is the worst failure mode a library has, and the documentation
# that says to use `spawn` gets read after the hang rather than before it.
#
# So the child says so itself, on the way past. The condition is "this process
# had more than one native thread when it forked", read from /proc rather than
# from `threading.active_count()`, which cannot see an OpenMP worker because
# it is not a Python thread.
#
# That condition is deliberately broader than "BEAM had a pool": another
# library's threads satisfy it too, and this will occasionally warn a child
# that would have been fine. Forking a multithreaded process is the hazard
# whoever started the threads, the warning is suppressible by category, and a
# guard that stays quiet through the case it exists for is worth less than one
# that is sometimes early.
_FORK_THREADS = 1

_FORK_WARNING = (
    "this process forked while {n} native threads were running. BEAM's "
    "OpenMP runtime does not survive fork: the first parallel computation in "
    "this child will hang, with no error raised. Start workers with "
    'multiprocessing.get_context("spawn") and call BEAM.set_threads(T) in '
    "each one."
)


def _native_thread_count() -> int:
    """Threads in this process, including the ones Python cannot see.

    1 when the count cannot be read, which makes the fork warning silent on a
    platform without /proc rather than wrong about it.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 1


def _before_fork() -> None:
    # Read in the parent: by the time the child runs it has one thread, so the
    # count that matters is gone.
    global _FORK_THREADS
    _FORK_THREADS = _native_thread_count()


def _after_fork_in_child() -> None:
    if _FORK_THREADS > 1:
        # Wrapped, because a child that cannot warn must still run. The point
        # is to explain a hang, never to become a different failure.
        try:
            warnings.warn(_FORK_WARNING.format(n=_FORK_THREADS),
                          RuntimeWarning, stacklevel=2)
        except Exception:                                    # pragma: no cover
            pass


# Unix only: Windows has no fork, and no register_at_fork to hang it on.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork,
                        after_in_child=_after_fork_in_child)
