"""Tempering: ``P_t(s) proportional to P(s)^t``, returned as a model.

The tempered law is already the workhorse of this library's analytics -- the
surprisal moments, the spectrum and the saddlepoint all run under it -- but it
has never been materialised as something you can score, sample and re-analyse.
These tests pin it against the quantities that already compute it, so nothing
here needs new ground truth.

Writing ``m(q)`` for ``log_mellin``, the two identities that pin the whole
family are

    log_mellin_{P_t}(q) == m(t*q) - q*m(t)
    pi_t(L)             == length_moments(q=t) share

and the sharpest regression guard is that ``t = 1`` is the identity.

The fixture is a short nucleotide model because the identities are checked
over the WHOLE support, and under the prior that is every string of A, C, G,
T and N up to max_length: 19,530 of them at max_length 6, enumerated directly
rather than asked of the model.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from BEAM import BeamModel

SEQS = ["TGTGCC", "TGTGCA", "TGCGCC", "TGTAC", "TGTG", "GCC", "TGTGC", "TTC"]
AB = [40.0, 25.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0]
MAX_LENGTH = 6


@pytest.fixture(scope="module")
def model():
    return BeamModel(SEQS, abundances=AB, alphabet="nt", order=3,
                     max_length=MAX_LENGTH)


def test_tempering_at_one_is_the_identity(model):
    """``P_1`` is ``P``. The sharpest regression guard there is."""
    one = model.tempered(1.0)
    for s in SEQS:
        assert one.log_pseq(s) == pytest.approx(model.log_pseq(s), rel=1e-12)


TS = (0.3, 0.5, 0.8, 1.0, 1.7, 3.0)


#: Every string over the alphabet up to max_length: the whole support.
SUPPORT = ["".join(p) for L in range(1, MAX_LENGTH + 1)
           for p in itertools.product("ACGTN", repeat=L)]


def test_the_support_is_what_the_model_counts(model):
    """The enumeration below stands in for the support, so it must be it."""
    assert len(SUPPORT) == model.path_count()
    assert np.isfinite(model.log_pseq_batch(SUPPORT)).all()


def test_the_defining_identity(model):
    """``ln P_t(s) == t ln P(s) - ln sum_s P(s)^t``, over the whole support.

    The second term is ``log_mellin(t)``, which this model already computes
    exactly, so the identity is checked against arithmetic rather than against
    a second implementation of it.
    """
    base = model.log_pseq_batch(SUPPORT)
    for t in TS:
        hot = model.tempered(t)
        z = model.log_mellin(t)
        np.testing.assert_allclose(hot.log_pseq_batch(SUPPORT), t * base - z,
                                   rtol=1e-12, atol=1e-12)


def test_the_tempered_mellin_transform_is_the_original_reparametrised(model):
    """``m_t(q) == m(t*q) - q*m(t)``, which pins the whole family at once."""
    for t in TS:
        hot = model.tempered(t)
        for q in (0.4, 0.9, 1.0, 1.5, 2.0, 3.0):
            assert hot.log_mellin(q) == pytest.approx(
                model.log_mellin(t * q) - q * model.log_mellin(t),
                rel=1e-10, abs=1e-10)


def test_the_tempered_model_is_a_distribution(model):
    """Its mass over the whole support is exactly 1, not approximately."""
    for t in TS:
        hot = model.tempered(t)
        total = float(np.exp(hot.log_pseq_batch(SUPPORT)).sum())
        assert total == pytest.approx(1.0, rel=1e-12)
        assert hot.pgen_diagnostics()["is_proper"]


def test_the_length_prior_is_the_tilted_share(model):
    """``pi_t(L)`` is the share ``length_moments(q=t)`` already reports.

    The trap this closes: ``pi_t`` is NOT ``pi(L)**t`` normalised. It picks up
    the within-length power sum, and getting that wrong still leaves a
    normalised distribution and plausible numbers.
    """
    for t in TS:
        hot = model.tempered(t)
        lm = model.length_moments(q=t, order=1)
        want = dict(zip(lm["length"].tolist(), lm["share"].tolist()))
        got = hot.length_distribution
        for L, share in want.items():
            if share > 0.0:
                assert got[L] == pytest.approx(share, rel=1e-10, abs=1e-12)


def test_tempering_reweights_the_support_without_widening_it(model):
    """``t > 0`` keeps a positive cell positive and cannot make a zero one
    live, so ``D0`` must not move: it stays every string up to max_length."""
    for t in TS:
        assert model.tempered(t).path_count() == model.path_count()


def test_the_entropy_follows_the_mellin_derivative(model):
    """``H(P_t) == m(t) - t m'(t)``, with ``m'(t) = -E_t[-ln P]``.

    Both right-hand terms are quantities this model already reports, the
    second being the mean surprisal under the very tilt being materialised.
    """
    for t in TS:
        hot = model.tempered(t)
        mean_surprisal_t = model.surprisal_moments(q=t, order=1)[1]
        assert hot.entropy() == pytest.approx(
            model.log_mellin(t) + t * mean_surprisal_t, rel=1e-10, abs=1e-10)


def test_raising_the_exponent_lowers_the_entropy(model):
    """Tempering concentrates. Strictly, on a model with unequal masses."""
    h = [model.tempered(t).entropy() for t in TS]
    assert all(a > b for a, b in zip(h, h[1:])), h


def test_within_one_length_the_ranking_survives(model):
    """A monotone reparametrisation cannot reorder a fixed length: not the
    top of it, which is what `top_k_sequences` reports, and not any of the
    15,625 strings below it either."""
    L = MAX_LENGTH
    base = [s for s, _ in model.top_k_sequences(3, length=L)]
    strings = [s for s in SUPPORT if len(s) == L]
    by_base = np.argsort(-model.log_pseq_batch(strings), kind="stable")
    for t in TS:
        hot = model.tempered(t)
        assert [s for s, _ in hot.top_k_sequences(3, length=L)] == base
        ranked = hot.log_pseq_batch(strings)[by_base]
        assert np.all(np.diff(ranked) <= 1e-9), t


def test_what_tempering_refuses(model):
    """``t <= 0`` and a non-finite ``t``, each named rather than clamped."""
    for bad in (0.0, -1.0, -0.5):
        with pytest.raises(Exception, match="strictly positive"):
            model.tempered(bad)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(Exception, match="finite"):
            model.tempered(bad)


def test_the_result_has_no_counts_left(model):
    """The guard that stops a refinalize silently reverting the tempering."""
    hot = model.tempered(2.0)
    with pytest.raises(Exception, match="compact"):
        hot.copy()
    with pytest.raises(Exception, match="compact|counts"):
        hot.add(model)


def test_the_original_is_untouched(model):
    """``tempered`` is non-destructive, like ``merged`` and ``without``."""
    before = [model.log_pseq(s) for s in SEQS]
    weight, count = model.total_weight, model.path_count()
    model.tempered(2.5)
    assert [model.log_pseq(s) for s in SEQS] == before
    assert model.total_weight == weight and model.path_count() == count


@pytest.mark.parametrize("sparse", [True, False])
def test_the_identity_holds_for_both_storages(sparse):
    """Two code paths scale the tables, and both are load-bearing: the sparse
    one must reach the ancestor rows an unobserved context reads as well as
    the rows it stores."""
    m = BeamModel(SEQS, abundances=AB, alphabet="nt", order=3,
                  max_length=MAX_LENGTH, sparse=sparse)
    t = 1.9
    hot, z = m.tempered(t), m.log_mellin(t)
    for s in SEQS + ["NNNNNN", "A", "GGGCCC"]:
        assert hot.log_pseq(s) == pytest.approx(
            t * m.log_pseq(s) - z, rel=1e-10, abs=1e-10)
    assert hot.pgen_diagnostics()["is_proper"]
