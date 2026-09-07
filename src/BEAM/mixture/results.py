"""Return types that carry their own exactness.

Every quantity in this package is one of three things -- exact, bracketed, or
sampled -- and which one it is has to survive the trip to wherever the number
is written down. Prose in a docstring does not survive that trip. A frozen
object with an ``exact`` field does, and it cannot be pasted into a table by
accident because it is not a float and does not pretend to be one: none of
these define ``__float__``.

That is deliberate friction. The reader who wants the bare number has to name
the field, and naming ``lower`` or ``estimate`` is the moment they learn which
one they have.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["EntropyBounds", "EntropyEstimate", "SupportSize", "BetaDiversity"]


@dataclass(frozen=True)
class EntropyBounds:
    """Exact bounds on ``H(pbar)``, in nats. Both sides are free.

    ``sum_i w_i H_i <= H(pbar) <= sum_i w_i H_i + H(w)``

    The lower bound is the entropy already inside the members; the gap is the
    entropy of the choice of member, which is the most that choice can add. The
    two bounds are attained: identical members sit exactly on the lower one,
    members with disjoint supports exactly on the upper one. So the width is
    not slack in the method, it is the range the answer genuinely occupies.

    Restated as diversity this is ``1 <= D_beta,1 <= exp(H(w))``, which for
    uniform weights is ``1 <= D_beta,1 <= M``.

    Attributes:
        lower: ``sum_i w_i H_i``, in nats.
        upper: ``lower + weight_entropy``, in nats.
        weight_entropy: ``H(w)``, the entropy of the mixing weights.
        exact: Always ``True``. Both bounds are exact; the QUANTITY they
            bracket is what has no closed form.
    """
    lower: float
    upper: float
    weight_entropy: float
    exact: bool = True

    @property
    def width(self) -> float:
        """``upper - lower``, which is exactly ``H(w)``."""
        return self.upper - self.lower

    def brackets(self, value: float) -> bool:
        """Whether ``value`` lies inside the bracket."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class EntropyEstimate:
    """A sampled ``H(pbar)`` with its standard error. NOT exact.

    ``log pbar(s)`` is exactly computable for any single ``s`` -- score it
    under every member and log-sum-exp with the weights -- even though the sum
    over all ``s`` is not. So drawing from the mixture and averaging
    ``-log pbar(s)`` is unbiased, and the only error is Monte Carlo, which is
    quantifiable and shrinks as ``1/sqrt(n)``.

    Attributes:
        value: The sample mean of ``-log pbar(s)``, in nats.
        stderr: Its standard error, in nats. Roughly 68% of the time the true
            entropy is within one of these of ``value``, 95% within two.
        n_samples: Draws behind the estimate.
        seed: The seed used, so the draw can be repeated exactly.
        bounds: The exact bracket from :class:`EntropyBounds`. The estimate
            must lie inside it up to Monte Carlo error; if it does not, that is
            a bug and not a small sample.
        exact: Always ``False``.
    """
    value: float
    stderr: float
    n_samples: int
    seed: int | None
    bounds: EntropyBounds
    exact: bool = False

    @property
    def diversity(self) -> float:
        """``exp(value)``, the estimated ``D1`` of the mixture. Also not exact."""
        return math.exp(self.value)


@dataclass(frozen=True)
class SupportSize:
    """``|union of the members' supports|``, exactly or as a bracket.

    This is ``D0`` of the mixture, and it is NOT ``path_count`` of a model
    built by adding the members' counts. A pooled model can splice an edge
    learned from one member onto an edge learned from another and so spell
    sequences no member spells; its support strictly contains this one.

    Attributes:
        value: The exact count, or ``None`` when only bounds were computed.
        lower: ``max_i D0(i)``. The union contains every member's support.
        upper: ``sum_i D0(i)``. Attained exactly when the supports are
            pairwise disjoint. ``D0`` of the pooled model is also an upper
            bound and is sometimes tighter, but computing it means refinalizing
            the pooled counts, which this package deliberately never does.
        exact: Whether ``value`` is filled in.
        peak_states: The largest number of live ``(state, member-set)`` pairs
            any layer held. This, not the 64-member structural limit, is what
            decides how many members an exact union can take on real data.
    """
    value: float | None
    lower: float
    upper: float
    exact: bool
    peak_states: int = 0


@dataclass(frozen=True)
class BetaDiversity:
    """``D_beta = D_gamma / D_alpha`` at one order, and its two parts.

    ``D_alpha`` is the POWER MEAN OF ORDER ``1 - q`` of the members' own Hill
    numbers -- arithmetic at ``q = 0``, geometric at ``q = 1``, harmonic at
    ``q = 2``. The naive arithmetic mean at every order is the trap: it is
    right only at ``q = 0``, and elsewhere it silently UNDER-reports beta. On a
    measured two-donor case with no shared sequence, where beta must be exactly
    2, the arithmetic version returns 1.325.

    ``1 <= D_beta <= M`` always, with 1 for identical members and M for members
    that share nothing. That bound is a property of THE MIXTURE. It does not
    constrain a model trained on the pooled counts, whose diversity can exceed
    the sum of the members' because it spells sequences none of them do.

    Attributes:
        order: The Hill order ``q``.
        gamma: ``D_q`` of the mixture, or ``None`` when it has no closed form.
        alpha: The power mean of the members' ``D_q``. Always exact.
        beta: ``gamma / alpha``, or ``None`` when ``gamma`` is ``None``.
        lower: Exact lower bound on beta.
        upper: Exact upper bound on beta. Equal to ``beta`` when exact.
        estimate: A sampled beta, filled in only when sampling was asked for.
        stderr: The standard error of ``estimate``.
        exact: Whether ``beta`` is filled in.
    """
    order: float
    gamma: float | None
    alpha: float
    beta: float | None
    lower: float
    upper: float
    exact: bool
    estimate: float | None = None
    stderr: float | None = None
