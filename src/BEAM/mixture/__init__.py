"""The probability mixture of several fitted models.

    >>> from BEAM.mixture import Mixture
    >>> mix = Mixture([donor_a, donor_b, donor_c])
    >>> mix.hill_number(2.0)              # exact
    >>> mix.beta_diversity(2.0).beta      # exact, 1 <= beta <= M
    >>> mix.entropy_bounds()              # exact bracket; q=1 has no closed form

``pbar(s) = sum_i w_i P_i(s)`` is "pick a donor by w, then pick a sequence from
that donor". It is the distribution beta diversity between individuals is
defined on, and it is the object this package provides.

## It is NOT what adding counts gives you

:meth:`BeamModel.merged` sums edge counts and refinalizes, which produces the
model you would get by TRAINING on the pooled corpus. That is a different
distribution with a strictly larger support: a pooled path may splice an edge
learned from one donor onto an edge learned from another and so spell a
sequence no single donor spells. Pooling 64 donors at a fixed 30,000-sequence
budget raises D0 by 24.3% while the number of transitions rises 0.57%, so
almost all of the excess is recombination rather than new evidence.

Nothing here is implemented in terms of the count algebra. ``D0(union)`` from
:meth:`Mixture.support_size` is always ``<= D0(merged)``, and the gap is the
whole reason this package exists.

## What is exact, and what is not

Say it once, plainly, because the three Hill orders are not equally tractable
and every method below is labelled by which case it is in.

===========  ==========================================================
``q = 0``    EXACT, by a forward sweep carrying the set of members still
             consistent with the path. Refuses rather than degrades when
             the surviving member-sets outgrow the budget, and
             :meth:`Mixture.support_size` will hand back free bounds
             instead.
``q = 1``    NOT COMPUTABLE in closed form, at all. ``log sum_i w_i
             P_i(s)`` does not factor over the trellis. What you get is
             an exact bracket (:meth:`Mixture.entropy_bounds`) and an
             unbiased sampled estimate with a standard error
             (:meth:`Mixture.entropy_estimate`).
             :meth:`Mixture.hill_number` at ``q = 1`` RAISES.
``q >= 2``   EXACT for integer orders, and this is the main deliverable.
             ``sum_s pbar^q`` expands into a sum over q-tuples of
             members, each an exact forward sweep over the shared
             trellis. ``q = 2`` costs ``M(M+1)/2`` sweeps.
===========  ==========================================================

## Cost, because it decides what M you can use

Every ``q = 2`` quantity is ``M(M+1)/2`` inner products, each one sweep per
length. What that costs is set by how much of the trellis is LIVE, not by the
model's nominal size, and the spread is eighty-fold:

  - donor-sized models (30,000 sequences, order 4, ``max_length`` 27):
    **0.03-0.05 s per pair**, so ``M = 64`` is about seventy seconds;
  - the 927M-sequence foundation model, same configuration, whose trellis is
    nearly all live: **3.3 s per pair**, so ``M = 64`` is about two hours.

Measure one :meth:`Mixture.log_inner_product` on your own models before
building a Gram matrix. There is no shortcut to look for: ``sum_i w_i P_i(s)``
does not factor over edges, which is precisely why the mixture needs this
machinery at all.

At full amino-acid configuration what runs out first is not the sweep but
holding the models -- 218 MB each at order 4, so 32 of them is 7 GB. See
https://mutejester.github.io/BEAM/reference/mixture/.
"""
from .errors import (IncompatibleMembers, MixtureError, MixtureTooLarge,
                     UnsupportedOrder)
from .mixture import Mixture
from .results import (BetaDiversity, EntropyBounds, EntropyEstimate,
                      SupportSize)

__all__ = ["Mixture", "MixtureError", "IncompatibleMembers",
           "UnsupportedOrder", "MixtureTooLarge",
           "EntropyBounds", "EntropyEstimate", "SupportSize",
           "BetaDiversity"]
