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
