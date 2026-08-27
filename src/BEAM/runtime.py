"""Process-wide thread and memory budget controls.
"""
from __future__ import annotations



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
    """Threads a parallel region would actually start right now.

    The cap if one is set, otherwise the OpenMP runtime's own maximum, and
    always 1 in a build without OpenMP.
    """
    return int(_c.effective_threads())


def set_memory_budget(nbytes: int) -> None:
    """Cap the scratch memory concurrent per-length spectra may hold together.

    Only :meth:`BeamModel.spectra_by_length` is large enough to care: one length's
    histogram grids run to 1.6 GB at order 4 with 512 bins, so running one
    length per core would ask for twenty-five gigabytes. It divides this
    budget by what one length needs and runs that many at a time.

    The default is 4 GB; ``0`` restores it. This caps **concurrency only** — a
    single length always runs whatever its size — so lowering it can never
    turn a working call into a failing one, only a slower one.

    It is a resource knob and nothing else. **Results do not depend on it**,
    which the C unit tests pin at a budget of one byte.
    """
    _c.set_memory_budget(int(nbytes))


def memory_budget() -> int:
    """The scratch budget in force, in bytes."""
    return int(_c.memory_budget())
