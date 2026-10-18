"""Scoring behaviour, and what the hierarchical prior makes possible.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Scoring behaviour ────────────────────────────────────────────

def test_batch_matches_scalar(model, corpus):
    seqs, _ = corpus
    batch = model.log_pseq_batch(seqs[:200])
    for i in range(200):
        assert batch[i] == pytest.approx(model.log_pseq(seqs[i]), abs=1e-12)


def test_explain_sums_to_the_score(model, corpus):
    """The per-residue decomposition must reconstruct the score exactly, so it
    can be trusted as an explanation rather than an illustration."""
    seqs, _ = corpus
    cfg = model.config
    wf, wb = cfg["weights"]
    for s in seqs[:20]:
        e = model.explain(s)
        raw = sum(wf * f + wb * b for f, b in zip(e["forward"], e["backward"]))
        want = raw + model.log_length_prior(len(s)) - model.log_partition(len(s))
        assert want == pytest.approx(e["log_pseq"], abs=1e-9)


def test_explain_returns_each_residues_combined_factor(model, corpus):
    """`combined` is log psi_i = w_F log q_F + w_B log q_B, the residue's factor
    of Phi_L(s); with the length prior and the normaliser it gives the score."""
    seqs, _ = corpus
    wf, wb = model.config["weights"]
    for s in seqs[:20]:
        e = model.explain(s)
        np.testing.assert_allclose(e["combined"], wf * e["forward"] + wb * e["backward"],
                                   rtol=0, atol=1e-12)
        assert e["log_length_prior"] == model.log_length_prior(len(s))
        assert e["log_partition"] == model.log_partition(len(s))
        total = float(np.sum(e["combined"])) + e["log_length_prior"] - e["log_partition"]
        assert total == pytest.approx(e["log_pseq"], abs=1e-9)


def test_c_terminus_expert_is_doing_work(model):
    """Swapping the terminal residue for one no J gene produces must cost
    orders of magnitude. A single left-to-right chain cannot see this."""
    good = model.log_pseq("CASSLGQGAEQYF")
    bad = model.log_pseq("CASSLGQGAEQYW")
    assert math.isfinite(good) and math.isfinite(bad)
    assert good - bad > math.log(1000), (good, bad)


def test_rejects_non_amino_acid(model):
    """The three things the prior cannot price, and the only three -inf
    scores left: a symbol outside the alphabet, the empty string, and a
    length past max_length."""
    assert model.log_pseq("CASSXQYF") == -math.inf
    assert model.log_pseq("") == -math.inf
    assert model.log_pseq("C" * 40) == -math.inf


def test_an_unobserved_length_takes_its_share_of_the_length_prior(model,
                                                                   corpus):
    """A length the corpus never had is possible, not ruled out: the length
    prior gives it `alpha_length / max_length` pseudo-observations out of
    `N + alpha_length`. The share is recomputed here from the rule."""
    seqs, abund = corpus
    assert 5 not in {len(s) for s in seqs}
    cfg = model.config
    want = math.log((cfg["alpha_length"] / cfg["max_length"])
                    / (sum(abund) + cfg["alpha_length"]))
    assert model.log_length_prior(5) == pytest.approx(want, abs=1e-12)
    assert math.isfinite(model.log_pseq("CASSF"))


# ── What the prior does to sequences the corpus never spelled ────────────

TINY = ["CASSF", "CASSY", "CTTGW"]


def test_an_unseen_sequence_is_possible_and_ranked_below_what_was_seen():
    """Three training sequences, so almost everything is prior. A near miss
    that keeps the corpus's context but emits a residue it never followed is
    finite and below the sequences it resembles; junk that shares nothing
    with the corpus is below that, because every row it reads is unobserved
    and falls back to its shorter context."""
    m = BeamModel(TINY, max_length=6)
    seen, near, junk = (m.log_pseq(s) for s in ("CASSF", "CASSW", "WWWWW"))
    assert all(math.isfinite(v) for v in (seen, near, junk))
    assert seen > near > junk


def test_the_support_is_every_string_whatever_the_data():
    """D0 counts every string over the alphabet of length 1..max_length, so
    it is the same for any corpus and says nothing about the one it was built
    from: the reason D0 is not reported as a repertoire's richness."""
    want = float(sum(20 ** L for L in range(1, 7)))
    for seqs in (TINY, ["CASRD", "GQ", "YYYYYY"]):
        m = BeamModel(seqs, max_length=6)
        assert m.path_count() == want
        assert m.hill_number(0) == pytest.approx(want, rel=1e-12)


def test_abundance_is_replication():
    """An abundance is a count, so weight 9 is nine copies to the prior as to
    everything else -- the pseudo-counts are added to the SAME n."""
    weighted = BeamModel(["CASSF", "CASSY"], abundances=[9.0, 1.0],
                         max_length=6)
    copied = BeamModel(["CASSF"] * 9 + ["CASSY"], max_length=6)
    probe = ["CASSF", "CASSY", "CASSW", "WW", "CTTGWF"]
    assert np.array_equal(weighted.log_pseq_batch(probe),
                          copied.log_pseq_batch(probe))
    assert weighted.log_pseq("CASSF") > weighted.log_pseq("CASSY")
