"""PRISM: is a sequence better explained as a model draw, or as a repaired one?

A prism splits a beam into components; PRISM splits one BEAM score into
per-position evidence, in the same pass.

    from BEAM import PrismScorer
    p = PrismScorer(model)
    p.factor(seq)          # log B for one sequence
    p.factor(seqs)         # ... or many, in parallel
    p.explain(seq)         # log B plus the PRISM posterior over sites
    p.flag(seqs)           # B >= 100, a 1% false-positive BOUND by Markov
"""
from .index import Prism, PrismScorer

__all__ = ["Prism", "PrismScorer"]
