"""The Dirichlet-multinomial posterior.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Dirichlet-multinomial posterior
# ════════════════════════════════════════════════════════════════════════


def test_posterior_at_kappa_zero_is_a_no_op(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    before = donor.log_pseq_batch(seqs[:40])
    donor.posterior(prior, kappa=0.0)
    assert np.array_equal(donor.log_pseq_batch(seqs[:40]), before)


def test_posterior_moves_the_model_toward_the_prior(corpus):
    """The point of the update: with a large kappa the individual's scores sit
    closer to the prior's than they did on the individual's own counts."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    prior = BeamModel(seqs[200:], abundances=abund[200:], order=2, max_length=20)
    probe = seqs[900:1100]

    own = donor.log_pseq_batch(probe)
    ref = prior.log_pseq_batch(probe)
    near = donor.with_prior(prior, kappa=1e6).log_pseq_batch(probe)

    assert np.mean(np.abs(near - ref)) < np.mean(np.abs(own - ref))


def test_posterior_closes_on_the_prior_but_never_reaches_it(corpus):
    """Raising kappa pulls the individual toward the prior, and then stops.

    It stops because `posterior` is not a blend of two models: it adds
    pseudo-counts to ROWS, and three things survive any kappa -- rows the
    prior never observed, this model's own hierarchical prior reapplied to
    the blended counts, and its own Z(L) and pi(L). Anyone reading
    `posterior` as "become the prior at large kappa" would be wrong, so the
    floor is asserted rather than left to be discovered.
    """
    seqs, abund = corpus
    donor = BeamModel(seqs[:200], abundances=abund[:200], order=2, max_length=20)
    prior = BeamModel(seqs[200:], abundances=abund[200:], order=2, max_length=20)
    probe = seqs[900:1100]
    ref = prior.log_pseq_batch(probe)

    def gap(k):
        return float(np.mean(np.abs(
            donor.with_prior(prior, kappa=k).log_pseq_batch(probe) - ref)))

    closing = [gap(k) for k in (0.0, 1.0, 10.0, 100.0)]
    assert all(b < a for a, b in zip(closing, closing[1:])), closing
    assert closing[-1] < 0.1 * closing[0]
    assert gap(1e6) > 0.5, "a huge kappa must not collapse onto the prior"


def test_posterior_leaves_the_sequence_counters_alone(corpus):
    """A pseudo-count is not an observation, so a leave-one-out corpus size
    must not shift when the smoothing strength does."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    w, n = donor.total_weight, donor.n_observed
    donor.posterior(prior, kappa=5_000.0)
    assert donor.total_weight == w
    assert donor.n_observed == n


def test_posterior_leaves_the_model_scoreable(corpus):
    """The C clears `finalized` rather than rebuilding, so the wrapper has to
    finalize. If it did not, every score below would be -inf or an error."""
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    donor.posterior(prior, kappa=100.0)
    lp = donor.log_pseq_batch(seqs[:50])
    assert np.all(np.isfinite(lp))
    assert donor.pgen_diagnostics()["is_proper"] is True


def test_with_prior_does_not_mutate_either_operand(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    d0 = donor.log_pseq_batch(seqs[:40])
    p0 = prior.log_pseq_batch(seqs[:40])

    out = donor.with_prior(prior, kappa=250.0)
    assert np.array_equal(donor.log_pseq_batch(seqs[:40]), d0)
    assert np.array_equal(prior.log_pseq_batch(seqs[:40]), p0)
    assert not np.array_equal(out.log_pseq_batch(seqs[:40]), d0)

    inplace = donor.copy()
    inplace.posterior(prior, kappa=250.0)
    assert np.array_equal(out.log_pseq_batch(seqs[:40]),
                          inplace.log_pseq_batch(seqs[:40]))


def test_posterior_rejects_bad_operands(corpus):
    seqs, abund = corpus
    donor = BeamModel(seqs[:300], abundances=abund[:300], order=2, max_length=20)
    prior = BeamModel(seqs[300:], abundances=abund[300:], order=2, max_length=20)
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        donor.posterior(prior, kappa=-1.0)
    with pytest.raises(TypeError, match="expects a BeamModel"):
        donor.posterior(object(), kappa=1.0)
    with pytest.raises(ValueError, match="with the model itself"):
        donor.posterior(donor, kappa=1.0)
