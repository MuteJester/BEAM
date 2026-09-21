"""Exceptions the mixture layer raises, and what each one means.

Two of the three narrow a refusal the C layer already makes, the way
:mod:`BEAM.query.errors` does and for the same reason: ``MemoryError`` and
``RuntimeError`` are too broad to catch selectively around one analytic. Each
keeps the library-wide class from :mod:`BEAM.errors` as a second base, so
``except BEAM.errors.BudgetExceeded`` still catches a union sweep that outgrew
its ceiling.

The third, :class:`UnsupportedOrder`, has no C counterpart. It exists because
the mixture is the first object in this library with a quantity that is not
merely expensive but *not computable in closed form at all*, and that has to
be a refusal rather than a number.
"""
from __future__ import annotations

from BEAM.errors import (BeamError, BeamValueError, BudgetExceeded,
                         ModelMismatch)

__all__ = ["MixtureError", "IncompatibleMembers", "UnsupportedOrder",
           "MixtureTooLarge"]


class MixtureError(BeamError):
    """Base for every refusal from the mixture layer.

    A direct child of ``BeamError``, which carries no built-in base, so that
    catching this does not silently widen an ``except ValueError`` elsewhere.
    """


class IncompatibleMembers(MixtureError, ModelMismatch):
    """The members cannot share a trellis, so there is no mixture to form.

    A mixture reads all its members at ONE state index. That is sound only
    because a state index is a function of the alphabet and the order and never
    of the data, which in turn requires the members to agree on alphabet,
    order and ``max_length``. Disagree on any of them and the
    sweeps would be reading different trellises through the same subscript.

    Raised at construction rather than at the first analytic: the mixture is
    ill-defined from the moment it is assembled, and finding that out three
    minutes into a Gram matrix helps nobody.
    """


class UnsupportedOrder(MixtureError, BeamValueError):
    """That Hill order has no exact answer for a mixture.

    Only ``q = 0`` and integer ``q >= 2`` are exactly computable here. Integer
    ``q >= 2`` expands into a sum over q-tuples of members, each an exact
    forward sweep; ``q = 0`` is the union of the supports, also exact. Every
    other order -- ``q = 1`` above all -- requires ``log sum_i w_i P_i(s)``,
    which does not factor over the edges of the trellis, so no forward
    recursion computes it.

    ``q = 1`` is refused rather than estimated ON PURPOSE, and the alternative
    is not hypothetical caution. A number that comes back as a bare float gets
    read once, pasted into a table, and separated from its caveat, which lives
    in prose and does not travel with it; this project has already lost four
    days to exactly that. Refusing costs the caller one line and removes the
    possibility. :meth:`~BEAM.mixture.Mixture.entropy_bounds` and
    :meth:`~BEAM.mixture.Mixture.entropy_estimate` are the two honest paths,
    and both return objects that carry their own exactness.
    """


class MixtureTooLarge(MixtureError, BudgetExceeded):
    """Exactly answerable, and too large to answer within the memory budget.

    Almost always the support union, whose cost is set by how many distinct
    member-sets survive per trellis state rather than by the number of members.
    The message carries the length, the layer and the ceiling reached; raise
    the budget, or take
    :meth:`~BEAM.mixture.Mixture.support_size` with ``exact=False`` for bounds
    that cost nothing.
    """
