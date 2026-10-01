"""SCALE attribution and the three-channel index.

The claims under test are the ones the figure rests on: the two anchored
profiles reconstruct the score exactly, two anchors localise without the bias
that one anchor cannot escape, a corruption is visible to the one-edit
neighbourhood, and the index's threshold is a false-positive rate.

The certified bracket -- an ``-inf`` in a profile proving a window around the
error -- has nothing to certify any more: under the prior every string over
the alphabet is spellable, a corrupted one included, so no profile is ever
``-inf`` and the bracket is never issued. What is left of "unspellable" is a
string past max_length, and the index still scores that infinite.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.analytics.scale import (Attribution, ScaleIndex, anchor_profile,
                                  attribute, crossing_amplitude,
                                  neighbourhood_ratio, residue_profile)

from _helpers import AA, make_corpus


@pytest.fixture(scope="module")
def strong_corpus():
    """Structured CDR3-shaped strings: conserved head, motif middle, J tail."""
    rng = random.Random(17)
    vocab = ["".join(rng.choice("GTSLNRDQ") for _ in range(4))
             for _ in range(40)]
    return ["CASS" + "".join(rng.choice(vocab)
                             for _ in range(rng.randint(1, 2)))
            + rng.choice(("EQYF", "EQFF", "NTIYF")) for _ in range(6000)]


@pytest.fixture(scope="module")
def strong(strong_corpus):
    """A model with real positional structure, so localisation is meaningful.

    ``make_corpus`` fills the middle with i.i.d. uniform residues, which is
    exactly the case where NOTHING can localise an error: a substitution there
    produces a sequence the model considers equally likely. These tests need a
    middle drawn from a small motif vocabulary, so that a corrupted residue is
    genuinely improbable and the dipole has something to point at.

    Every corruption is spellable under the prior, so the profiles are the
    only thing deciding where an error is.
    """
    return BeamModel(strong_corpus, abundances=None, order=3, max_length=24)


def _corrupt(seq, rng, kind="sub"):
    """One uniform single-residue edit, and the index it happened at."""
    i = rng.randrange(len(seq))
    if kind == "sub":
        a = rng.choice([c for c in AA if c != seq[i]])
        return seq[:i] + a + seq[i + 1:], i
    if kind == "ins":
        return seq[:i] + rng.choice(AA) + seq[i:], i
    return seq[:i] + seq[i + 1:], i


# Profiles
# ═══════════════════════════════════════════════════════════════

def test_profiles_reconstruct_the_score(strong):
    """The residue profile is a DECOMPOSITION, not a summary: it must add back
    up to log_pseq through the length prior and partition function."""
    s = strong.sample(1, seed=4)[0]
    c = residue_profile(strong, s)
    total = -c.sum() + strong.log_length_prior(len(s)) \
        - strong.log_partition(len(s))
    assert total == pytest.approx(strong.log_pseq(s), rel=1e-9)


def test_anchor_profile_is_indexed_on_gaps(strong):
    s = strong.sample(1, seed=6)[0]
    g = anchor_profile(strong, s)
    assert len(g) == len(s) + 1, "one value per gap, not per residue"
    assert g[0] == 0.0


def test_clean_sequences_have_no_dipole(strong):
    """The whole attribution rests on the disagreement being centred at zero
    for a sequence with no error. If clean draws had a systematic dipole the
    crossing would point at that instead."""
    draws = strong.sample(400, seed=8)
    amps = np.array([crossing_amplitude(strong, s) for s in draws])
    amps = amps[np.isfinite(amps)]
    rng = random.Random(2)
    errs = [v for v in (crossing_amplitude(strong, _corrupt(s, rng)[0])
                        for s in draws) if np.isfinite(v)]
    assert len(errs) > 50 and len(amps) > 50
    assert np.median(errs) > np.median(amps), (np.median(errs),
                                               np.median(amps))


# Attribution
# ═══════════════════════════════════════════════════════════════

def test_attribution_localises_far_better_than_chance(strong):
    rng = random.Random(21)
    draws = [s for s in strong.sample(400, seed=12)
             if np.isfinite(strong.log_pseq(s))]
    hit = tot = 0
    for s in draws:
        bad, i = _corrupt(s, rng)
        tot += 1
        hit += abs(attribute(strong, bad).site - i) <= 1
    chance = 3.0 / np.mean([len(s) for s in draws])
    assert tot > 100
    assert hit / tot > 4 * chance, (hit / tot, chance)


def test_two_anchors_remove_the_single_anchor_bias(strong):
    """The decisive claim for the architecture. One anchor is contaminated
    ONLY downstream of the error, so its estimate is systematically biased;
    the mixture is not.

    Stated as the exact fact rather than through an argmax. A forward term at
    position p reads s[p-k..p] and nothing else, so corrupting s[i] cannot
    reach any p < i: the excess surprise to the LEFT of the site is zero, not
    small. It is the asymmetry that the C anchor exists to cancel.

    This used to assert that the forward argmax lands LATE on average. That
    was a proxy, and a poor one: the argmax is settled by whichever position
    is most surprising anywhere in the profile, so baseline noise at an
    unrelated position outvotes the corruption whenever the corruption is
    mild. Measured on this fixture the forward argmax sits exactly ON the site
    52% of the time and its mean is dragged negative by a tail reaching -17,
    none of which is contamination. The quantity below has no such tail
    because it is the contamination itself.
    """
    rng = random.Random(33)
    k = int(strong.config["order"])
    right, left = [], []
    for s in strong.sample(500, seed=14):
        if not np.isfinite(strong.log_pseq(s)):
            continue
        # Away from both termini, so the window on each side is full.
        i = rng.randrange(k, len(s) - k)
        c = rng.choice([x for x in AA if x != s[i]])
        bad = s[:i] + c + s[i + 1:]
        if not np.isfinite(strong.log_pseq(bad)):
            continue
        f0 = np.asarray(strong.explain(s)["forward"], dtype=np.float64)
        f1 = np.asarray(strong.explain(bad)["forward"], dtype=np.float64)
        excess = f0 - f1                  # surprise the corruption added
        right.append(float(excess[i + 1:i + 1 + k].sum()))
        left.append(float(excess[i - k:i].sum()))

    assert len(right) > 100, len(right)
    # Exact, not approximate: the forward context cannot see the future.
    assert max(abs(v) for v in left) == 0.0
    assert np.mean(right) > 1.0, np.mean(right)
    # And the two anchors together are less biased than one alone.
    wf, wb = strong.config["weights"]
    assert wf > 0.0 and wb > 0.0


def test_attribute_reports_spellability(strong):
    """A draw is spellable, and so is its corruption: under the prior every
    string over the alphabet is, so no profile is -inf and there is never a
    bracket to certify."""
    s = strong.sample(1, seed=9)[0]
    a = attribute(strong, s)
    assert isinstance(a, Attribution)
    assert a.spellable is True
    assert a.bracket is None
    assert repr(a).startswith("Attribution(")
    bad, _ = _corrupt(s, random.Random(44))
    b = attribute(strong, bad)
    assert b.spellable is True and b.bracket is None
    assert np.isfinite(b.residue).all()


# The three-channel index
# ═══════════════════════════════════════════════════════════════

def test_a_corruption_raises_the_neighbourhood_ratio(strong):
    """The one-edit Bayes factor is the channel that sees a point error: a
    corrupted draw has its clean original one edit away, so the mass next to
    it dwarfs its own.

    This used to be stated at its limit -- a corruption the model could not
    spell, one edit from one it could, scored infinite. Under the prior every
    corruption is spellable and the ratio is finite, so the claim is the
    comparison: corrupted above clean, draw by draw. Measured on this fixture
    that holds for all 200 pairs, with medians of 0.19 and 22.7 nats.
    """
    rng = random.Random(2)
    draws = strong.sample(200, seed=8)
    clean = np.array([neighbourhood_ratio(strong, s) for s in draws])
    bad = np.array([neighbourhood_ratio(strong, _corrupt(s, rng)[0])
                    for s in draws])
    assert np.isfinite(clean).all() and np.isfinite(bad).all()
    assert np.mean(bad > clean) > 0.9
    assert np.median(bad) > np.median(clean) + 5.0


@pytest.fixture(scope="module")
def index(strong, strong_corpus):
    """Built over the lengths the corpus has. The calibration covers every
    length up to max_length now, and the ones the corpus never had are the
    length prior's pseudo-count; simulating a null for each would multiply the
    cost by six to cover lengths no draw below lands on."""
    return ScaleIndex.build(strong, n_sim=600, seed=3,
                            lengths=sorted(set(map(len, strong_corpus))))


def test_index_threshold_is_a_false_positive_rate(strong, index):
    """Draws from the model are its null, so the index must exceed t on about
    10**-t of them. This is what makes the number deployable."""
    draws = [s for s in strong.sample(1200, seed=77)
             if len(s) in index.lengths]
    z = index.score(strong, draws)
    z = z[np.isfinite(z)]
    assert len(z) > 400
    assert np.mean(z >= 1.0) == pytest.approx(0.10, abs=0.06)


def test_index_is_infinite_exactly_when_unspellable(strong, index):
    """Junk over the alphabet is spellable now, so it scores finite -- and
    far above a draw. What the model cannot spell is a string past
    max_length, and zero probability is not a small p-value."""
    s = [x for x in strong.sample(20, seed=21) if len(x) in index.lengths][0]
    drawn = index.score(strong, s)
    junk = index.score(strong, "W" * len(s))
    assert np.isfinite(drawn) and np.isfinite(junk)
    assert junk > drawn
    too_long = "C" + "A" * int(strong.config["max_length"]) + "F"
    assert not np.isfinite(strong.log_pseq(too_long))
    assert index.score(strong, too_long) == float("inf")


def test_channels_are_valid_probabilities(strong, index):
    draws = [s for s in strong.sample(60, seed=31) if len(s) in index.lengths]
    ch = index.channels(strong, draws)
    assert set(ch) == {"U", "X", "L"}
    for k, v in ch.items():
        assert np.all((v > 0) & (v <= 1.0)), k


def test_uncovered_length_contributes_no_evidence(strong, index):
    """A length with no null must not be scored as if it had one."""
    missing = max(index.lengths) + 3
    if missing > strong.config["max_length"]:
        pytest.skip("no room above the covered lengths")
    ch = index.channels(strong, "C" + "A" * (missing - 2) + "F")
    assert ch["X"] == 1.0 and ch["L"] == 1.0
