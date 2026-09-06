"""The exceptions BEAM raises.

Every class here inherits from :class:`BeamError` **and** from the built-in it
replaces. That second base is not decoration: it is what makes this module
additive rather than a breaking change. Code that already writes

    try:
        model.entropy()
    except ValueError:
        ...

keeps working unchanged, and code that wants to be specific can now write
``except BeamValueError`` or ``except BudgetExceeded`` instead of matching on
the text of a message.

Why the second base is per-class rather than uniform
----------------------------------------------------
It would be tidier to make everything a ``BeamValueError``. It would also
break eight tests, because BEAM raises ``TypeError`` in eight places where the
argument's *type* is wrong rather than its value. The built-in each class
carries is the one that entry point produces today, established by running
every reachable failure rather than by reading the C switch -- the switch and
the observed behaviour disagree in several places, and
``tests/python/test_error_contract.py`` pins the observed side.

Why there is no class per error code
------------------------------------
One C error code does not imply one Python class, and assuming it did is how a
regression shipped earlier: ``BEAM_ERR_IO_CORRUPT`` had never been produced,
so nobody noticed its switch arm said ``ValueError`` while every
``beam_model_load`` failure was a hard-coded ``OSError``. Making the code
reachable silently converted three failures.

``BEAM_ERR_NO_LIVE_PATHS`` is the standing example: it surfaces as a
``ValueError`` from ``positional_marginals``, as an ``ImpossibleCondition``
from :class:`~BEAM.BeamQuery`, and as a plain ``-inf`` return from
``log_clamp_probability`` -- all three deliberate. So the code supplies a
default and entry points that mean something more specific keep saying it.

A gap worth knowing about
-------------------------
A few allocation sites in the bindings raise a bare ``MemoryError`` with an
empty message and no error code at all. ``except BeamError`` will **not**
catch those. They are pinned in ``test_error_contract.py`` so the gap is
visible rather than discovered later.
"""
from __future__ import annotations

__all__ = [
    "BeamError",
    "BeamValueError",
    "BeamTypeError",
    "InvalidSequence",
    "NotFinalized",
    "NoLivePaths",
    "ModelMismatch",
    "BudgetExceeded",
    "OutOfMemory",
    "ModelFileError",
    "InternalError",
]


class BeamError(Exception):
    """Base of every exception BEAM raises deliberately.

    Catch this to mean "BEAM refused", regardless of which built-in the
    specific failure also happens to be. It carries no built-in base of its
    own, which is what lets the subclasses each choose their own.
    """


class BeamValueError(BeamError, ValueError):
    """An argument BEAM cannot use: out of range, unusable, or inconsistent."""


class BeamTypeError(BeamError, TypeError):
    """An argument of the wrong type.

    Separate from :class:`BeamValueError` because eight entry points raise
    ``TypeError`` today and collapsing them would break every
    ``pytest.raises(TypeError)`` that guards them.
    """


class InvalidSequence(BeamValueError):
    """A sequence the model cannot read: bad characters, empty, or too long.

    Distinct from a sequence the model can read but never generates, which is
    not an error at all -- that is a probability of zero, and
    :meth:`~BEAM.BeamModel.log_pseq` returns ``-inf`` for it.
    """


class NotFinalized(BeamError, RuntimeError):
    """The model has not been finalized, or has been compacted past this use."""


class NoLivePaths(BeamError, RuntimeError):
    """The question has no answer because nothing satisfies it.

    Raised where a conditional would be undefined -- there is no distribution
    to report. Where the quantity asked for is a probability rather than a
    conditional, an impossible condition is an ANSWER of zero and no exception
    is raised.
    """


class ModelMismatch(BeamError, RuntimeError):
    """Two models cannot be combined: different alphabets, orders, or shapes.

    ``RuntimeError`` rather than ``ValueError``, which is arguably the better
    fit, because that is what this failure has always raised. It reached Python
    through the mapping's ``default:`` arm -- the only live error code with no
    case of its own -- and was therefore indistinguishable from an unfinalized
    model. Naming it is the fix; re-typing it would be a separate, breaking
    decision.
    """


class BudgetExceeded(BeamError, MemoryError):
    """Refused by :func:`~BEAM.set_memory_budget`, not by the allocator.

    The remedy is in the caller's hands: raise the budget, ask at a lower
    order, or ask a smaller question. The message carries the figure needed and
    the budget in force, in a unit scaled to their size.

    Distinct from :class:`OutOfMemory`, which is the machine saying no and
    which raising the budget will not fix. They shared an error code until
    ``BEAM_ERR_BUDGET`` was split out, which is why code that wanted to tell
    them apart had to match on the message text.
    """


class OutOfMemory(BeamError, MemoryError):
    """A genuine allocation failure: the machine could not supply the memory."""


class ModelFileError(BeamError, OSError):
    """A model file could not be read or written.

    Covers the whole family: missing, unreadable, truncated, checksum
    mismatch, and a format this build does not know. ``OSError`` is the base
    because that is what ``beam_model_load`` and ``beam_model_save`` have
    always raised.

    Note that not every failure of a load is one of these -- a sound file that
    does not fit the memory budget raises :class:`BudgetExceeded`, which is a
    ``MemoryError`` and deliberately not an ``OSError``.
    """


class InternalError(BeamError, RuntimeError):
    """An invariant inside BEAM was violated. This is a bug in BEAM.

    Never raised in the ordinary course of using the library. If you see one,
    the engine has produced something it knows to be impossible -- the message
    names which invariant -- and it is worth reporting rather than working
    around.
    """
