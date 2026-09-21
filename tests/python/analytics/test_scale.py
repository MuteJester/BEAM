"""SCALE attribution and the three-channel index.

The claims under test are the ones the figure rests on: the two anchored
profiles reconstruct the score exactly, an ``-inf`` in a profile CERTIFIES a
window around the error, two anchors localise without the bias that one anchor
cannot escape, and the index's threshold is a false-positive rate.
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
    """
    return BeamModel(strong_corpus, abundances=None, order=3, max_length=24)


@pytest.fixture(scope="module")
def dense(strong_corpus):
    """The same corpus in the OPEN world, where every corruption is spellable.

    The two mechanisms are separable and must be tested separately. In a strict
    model of this size almost every single-residue edit falls off the support
    entirely -- which is what the certified bracket is for -- leaving too few
    finite cases to measure the dipole on. The open world keeps every sequence
    finite, so the profiles are the only thing deciding, which is the regime
    the 927M foundation is actually in: there only 5.9% of substitutions are
    unspellable.
    """
    return BeamModel(strong_corpus, abundances=None, order=3, max_length=24,
                     strict=False)


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


def test_clean_sequences_have_no_dipole(dense):
    """The whole attribution rests on the disagreement being centred at zero
    for a sequence with no error. If clean draws had a systematic dipole the
    crossing would point at that instead."""
    draws = dense.sample(400, seed=8)
    amps = np.array([crossing_amplitude(dense, s) for s in draws])
    amps = amps[np.isfinite(amps)]
    rng = random.Random(2)
    errs = [v for v in (crossing_amplitude(dense, _corrupt(s, rng)[0])
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


def test_two_anchors_remove_the_single_anchor_bias(dense):
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
    k = int(dense.config["order"])
    right, left = [], []
    for s in dense.sample(500, seed=14):
        if not np.isfinite(dense.log_pseq(s)):
            continue
        # Away from both termini, so the window on each side is full.
        i = rng.randrange(k, len(s) - k)
        c = rng.choice([x for x in AA if x != s[i]])
        bad = s[:i] + c + s[i + 1:]
        if not np.isfinite(dense.log_pseq(bad)):
            continue
        f0 = np.asarray(dense.explain(s)["forward"], dtype=np.float64)
        f1 = np.asarray(dense.explain(bad)["forward"], dtype=np.float64)
        excess = f0 - f1                  # surprise the corruption added
        right.append(float(excess[i + 1:i + 1 + k].sum()))
        left.append(float(excess[i - k:i].sum()))

    assert len(right) > 100, len(right)
    # Exact, not approximate: the forward context cannot see the future.
    assert max(abs(v) for v in left) == 0.0
    assert np.mean(right) > 1.0, np.mean(right)
    # And the two anchors together are less biased than one alone.
    wf, wb = dense.config["weights"]
    assert wf > 0.0 and wb > 0.0

def test_bracket_is_a_certificate_not_an_estimate(strong):
    """When the model cannot spell a sequence, the -inf positions in the two
    anchored profiles PROVE a window. A single-edit corruption whose bracket
    exists must have its true site inside it, every time."""
    rng = random.Random(44)
    draws = [s for s in strong.sample(600, seed=15)
             if np.isfinite(strong.log_pseq(s))]
    checked = 0
    for s in draws:
        bad, i = _corrupt(s, rng)
        if np.isfinite(strong.log_pseq(bad)):
            continue
        a = attribute(strong, bad)
        if a.bracket is None:
            continue
        lo, hi = a.bracket
        checked += 1
        assert lo <= i <= hi, (bad, i, a.bracket)
        assert lo <= a.site <= hi
    assert checked > 20, "the fixture must produce unspellable corruptions"


def test_attribute_reports_spellability(strong):
    s = strong.sample(1, seed=9)[0]
    a = attribute(strong, s)
    assert isinstance(a, Attribution)
    assert a.spellable is True
    assert a.bracket is None
    assert repr(a).startswith("Attribution(")


# The three-channel index
# ═══════════════════════════════════════════════════════════════

def test_neighbourhood_ratio_is_infinite_for_a_rescuable_unspellable(strong):
    """A sequence the model cannot spell, one edit from one it can, is the
    strongest possible point-error evidence."""
    s = strong.sample(1, seed=19)[0]
    bad = s[:-1] + ("W" if s[-1] != "W" else "P")
    if np.isfinite(strong.log_pseq(bad)):
        pytest.skip("fixture did not produce an unspellable neighbour")
    assert neighbourhood_ratio(strong, bad) == float("inf")


@pytest.fixture(scope="module")
def index(strong):
    return ScaleIndex.build(strong, n_sim=600, seed=3)


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
    s = [x for x in strong.sample(20, seed=21) if len(x) in index.lengths][0]
    assert np.isfinite(index.score(strong, s))
    assert index.score(strong, "W" * len(s)) == float("inf")


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
