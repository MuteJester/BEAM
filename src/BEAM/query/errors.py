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
"""
from __future__ import annotations

__all__ = ["QueryError", "ImpossibleCondition", "UnsupportedQuery",
           "QueryTooLarge"]


class QueryError(Exception):
    """Base for every refusal from the query engine."""


class ImpossibleCondition(QueryError, RuntimeError):
    """The condition selects no sequence, so the conditional does not exist.

    Distinct from a probability of zero, which is an ANSWER and is returned as
    ``-inf`` rather than raised: ``P(A)`` is well defined when ``A`` cannot
    happen, and ``P(. | A)`` is not.
    """


class UnsupportedQuery(QueryError, ValueError):
    """The question has no exact answer, or is malformed.

    Only quantities that decompose over the trellis are computable in closed
    form. This engine refuses the rest rather than estimating them, which is
    the promise the rest of the library makes.
    """


class QueryTooLarge(QueryError, MemoryError):
    """Exactly answerable, and too large to answer within the memory budget.

    The message carries the figure and the knob: raise the budget, ask at a
    lower order, or simplify the condition.
    """
