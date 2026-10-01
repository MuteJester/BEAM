"""The probability of a whole repertoire under a BEAM model.

    >>> from BEAM import repertoire_likelihood, pair_likelihood
    >>> r = repertoire_likelihood(model, seqs, templates)
    >>> r.log_likelihood                  # exact
    >>> [(p.name, p.log_prob) for p in r.pieces]
    >>> pair = pair_likelihood(union, seqs_a, templates_a, seqs_b, templates_b)
    >>> pair.piece("sharing").observed, pair.piece("sharing").expected

The total is the multinomial: N reads, each an independent draw from the
model's ``P(s)``. The pieces factor that same total, each given the N reads
and the ones before it -- how many unique, which ones, the abundance pattern,
which sequence has which count -- and for two repertoires the number they
share. The totals are exact; the pieces are labelled.

See ``docs/api/python/likelihood.md``.
"""
from .chain import (DEFAULT_PROFILE_EDGES, abundance_spectrum,
                    auto_profile_edges, pair_likelihood, repertoire_likelihood)
from .results import (AbundanceSpectrum, PairLikelihood, Piece,
                      RepertoireLikelihood)

__all__ = ["repertoire_likelihood", "pair_likelihood", "RepertoireLikelihood",
           "PairLikelihood", "Piece", "DEFAULT_PROFILE_EDGES",
           "abundance_spectrum", "auto_profile_edges", "AbundanceSpectrum"]
