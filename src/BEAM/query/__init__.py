"""Exact probabilistic queries over a fitted BEAM model.

    >>> from BEAM.query import BeamQuery, Condition
    >>> q = BeamQuery(model)
    >>> c = q.condition(14).at(4, "G").contains("CASS")
    >>> q.probability(c), q.entropy(c)

Everything here is exact. A question with no exact answer is refused, never
estimated; see :mod:`BEAM.query.errors` for what the refusals mean.
"""
from .conditions import Condition
from .engine import BeamQuery
from .errors import (ImpossibleCondition, QueryError, QueryTooLarge,
                     UnsupportedQuery)

__all__ = ["BeamQuery", "Condition", "QueryError", "ImpossibleCondition",
           "UnsupportedQuery", "QueryTooLarge"]
