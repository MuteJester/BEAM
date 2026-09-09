"""Exceptions the query engine raises, and what each one means.

Three failures are worth telling apart, because a caller does something
different about each. A condition that selects nothing is a question with no
answer, and asking it again differently is the remedy. A question outside what
decomposes over the trellis has no exact answer at all, and no rephrasing
helps. A question that is exactly answerable but too large to run is a budget
decision.

The C layer distinguishes these already, and the bindings map them onto
built-in Python types -- ``RuntimeError``, ``ValueError``, ``MemoryError``.
Those are too broad to catch selectively, so this layer narrows them while
keeping the built-in as a base class: existing code that catches ``ValueError``
around a query keeps working.

Each class also inherits the library-wide equivalent from :mod:`BEAM.errors`,
so ``except BEAM.errors.BudgetExceeded`` catches a query refused by the memory
budget just as it catches a model that will not fit. ``QueryError`` itself
descends directly from ``BeamError``, which carries no built-in of its own --
deliberately, because re-basing it onto one would make every query refusal a
subclass of that built-in and silently widen what an existing
``except ValueError`` around a query swallows.
"""
from __future__ import annotations

from BEAM.errors import (BeamError, BeamValueError, BudgetExceeded,
                         NoLivePaths)

__all__ = ["QueryError", "ImpossibleCondition", "UnsupportedQuery",
           "QueryTooLarge"]


class QueryError(BeamError):
    """Base for every refusal from the query engine.

    A direct child of ``BeamError``, which carries no built-in base. Anything
    else here would push a built-in onto every query refusal at once.
    """


class ImpossibleCondition(QueryError, NoLivePaths):
    """The condition selects no sequence, so the conditional does not exist.

    Distinct from a probability of zero, which is an ANSWER and is returned as
    ``-inf`` rather than raised: ``P(A)`` is well defined when ``A`` cannot
    happen, and ``P(. | A)`` is not.
    """


class UnsupportedQuery(QueryError, BeamValueError):
    """The question has no exact answer, or is malformed.

    Only quantities that decompose over the trellis are computable in closed
    form. This engine refuses the rest rather than estimating them, which is
    the promise the rest of the library makes.
    """


class QueryTooLarge(QueryError, BudgetExceeded):
    """Exactly answerable, and too large to answer within the memory budget.

    The message carries the figure and the knob: raise the budget, ask at a
    lower order, or simplify the condition.
    """
